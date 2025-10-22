"""
모듈: inference_pipeline.py
설명:
- 프레임 시퀀스 수집과 랜드마크 추출/모델 추론을 수행하는 공통 파이프라인 구현을 제공합니다.
- 주요 구성 요소:
  - run_inference(predictor, frames): 랜드마크 추출 및 Predictor를 이용한 추론 수행
  - SequenceCollector: 실시간 스트림에서 프레임을 수집하고 타이밍을 관리하는 유틸
  - schedule_quiz_save: 추론 결과를 비동기적으로 저장하도록 스케줄

since: 2025.10.17
author: 백승현
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, cast

from configs import settings

# 중앙 설정에서 값을 가져옵니다.
TARGET_FRAME_COUNT = settings.TARGET_FRAME_COUNT
COLLECTION_DURATION_SECONDS = settings.COLLECTION_DURATION_SECONDS
MAX_FRAMES_TO_COLLECT = settings.MAX_FRAMES_TO_COLLECT

# 안전한(옵션) heavy deps import
HEAVY_IMPORTS_AVAILABLE = True
try:
    import torch
    import numpy as np
    from processing.landmark_extractor import extract_sequence_from_frames, FRAME_FEATURE_DIM
    from processing.predictor import Predictor
except Exception as _e:
    HEAVY_IMPORTS_AVAILABLE = False
    torch = None
    np = None
    extract_sequence_from_frames = None
    FRAME_FEATURE_DIM = 0
    Predictor = None


def run_inference(predictor: "Predictor", landmark_sequence: Optional[Any]) -> Dict[str, Any]:
    """
    이미 추출된 랜드마크 시퀀스(또는 None)를 받아 모델 추론을 수행합니다.
    이 함수는 더 이상 원시 프레임을 받아 랜드마크를 추출하지 않습니다.
    """
    start = time.time()
    landmarks_info: Dict[str, Any] = {}
    predicted_label = "오류: 추론 실패"
    score = 0.0

    try:
        # 입력으로 이미 랜드마크 시퀀스를 받는다고 가정합니다.
        # len()가 지원되지 않는 타입도 있을 수 있으므로 안전하게 처리합니다.
        empty_seq = False
        try:
            seq_len = len(landmark_sequence)  # may raise TypeError
            empty_seq = (seq_len == 0)
        except Exception:
            seq_len = None
            empty_seq = False

        if landmark_sequence is None or empty_seq:
             landmarks_info = {"enabled": True, "error": "landmark_sequence is None or empty"}
             predicted_label = "오류: 랜드마크를 감지하지 못했습니다."
        elif np is not None and (hasattr(landmark_sequence, '__array__') or hasattr(landmark_sequence, 'dtype')) and np.all(landmark_sequence == 0):
            # numpy가 설치되어 있고, 배열 형태라면 전체 0 체크
            landmarks_info = {"enabled": True, "error": "All landmarks are zero (No detection)"}
            predicted_label = "오류: 랜드마크를 감지하지 못했습니다. (데이터 없음)"
        else:
            landmarks_info = {
                "enabled": True,
                "seq_shape": list(landmark_sequence.shape) if hasattr(landmark_sequence, 'shape') else None,
                "feature_dim": FRAME_FEATURE_DIM,
            }
            if predictor is None:
                raise RuntimeError("Predictor not available (missing heavy dependencies)")
            landmark_sequence_float32 = landmark_sequence.astype(np.float32) if np is not None and hasattr(landmark_sequence, 'astype') else landmark_sequence
            # static-analyzer friendly: cast predictor to Any before calling predict
            p_any = cast(Any, predictor)
            if not hasattr(p_any, "predict"):
                raise RuntimeError("Predictor does not implement 'predict' method")
            predicted_label, score = p_any.predict(landmark_sequence_float32)

    except Exception as e:
        print(f"[ERROR][inference_pipeline] Exception during inference: {e}")
        traceback.print_exc()
        error_message = str(e)
        landmarks_info = {"enabled": True, "error": error_message}
        predicted_label = f"오류: 추론 중 예외 발생 ({error_message})"

    end = time.time()
    # frames_used는 landmark_sequence의 첫 차원(시퀀스 길이)을 사용하거나 0으로 둡니다.
    frames_used = 0
    try:
        if landmark_sequence is not None:
            # prefer shape[0] for numpy-like objects, fallback to len()
            if hasattr(landmark_sequence, 'shape') and getattr(landmark_sequence, 'shape'):
                frames_used = int(landmark_sequence.shape[0])
            else:
                try:
                    frames_used = int(len(landmark_sequence))
                except Exception:
                    frames_used = 0
    except Exception:
        frames_used = 0

    return {
        "predicted": predicted_label,
        "score": score,
        "inference_start": start,
        "inference_end": end,
        "inference_ms": int((end - start) * 1000),
        "frames_used": frames_used,
        "landmarks": landmarks_info,
    }


@dataclass
class SequenceCollector:
    """세션별 프레임 수집기.

    역할/정의:
    - WebSocket/DataChannel 등에서 들어오는 바이트 프레임을 임시 보관하고,
      수집 시작/종료 타이밍, 수집 완료 판단(is_full) 등을 제공합니다.

    주요 메서드:
    - start_collection(): 수집 시작 타이머 설정
    - add_frame(data): 프레임 추가(조건에 따라 무시)
    - is_full(): 시간/프레임 수 기준으로 수집 완료 여부 판단
    - build_timings(): 수집 타이밍 정보 반환

    since: 2025.10.17
    author: 백승현
    """
    frames: List[bytes] = field(default_factory=list)
    start_ts: Optional[float] = None
    processed: bool = False

    def start_collection(self):
        """수집 타이머를 시작합니다."""
        if self.start_ts is None:
            self.start_ts = time.time()

    def add_frame(self, data: bytes):
        """
        수집 기간 내에 있고 최대 프레임 수를 초과하지 않은 경우에만 프레임을 추가합니다.
        """
        if self.processed:
            return

        if self.start_ts is not None and not self.is_full():
            if len(self.frames) < MAX_FRAMES_TO_COLLECT:
                self.frames.append(data)

    def is_full(self) -> bool:
        """시간 조건 또는 프레임 수 조건으로 수집이 완료되었는지 판단합니다.

        변경: TARGET_FRAME_COUNT 기준 검사 제거 — 이제 수집 완료 여부는 시간 경과 또는 MAX_FRAMES_TO_COLLECT만으로 결정됩니다.
        """
        if self.start_ts is None:
            return False

        time_elapsed = time.time() - self.start_ts
        if time_elapsed >= COLLECTION_DURATION_SECONDS:
            return True

        # TARGET_FRAME_COUNT 검사 제거: 클라이언트에서 오는대로 수집하고, 필요 시 MAX_FRAMES_TO_COLLECT에서 중단
        if len(self.frames) >= MAX_FRAMES_TO_COLLECT:
            return True

        return False

    def build_timings(self) -> Dict[str, Any]:
        end_ts = time.time()
        return {
            "frame_count": len(self.frames),
            "receive_first_ts": self.start_ts,
            "receive_last_ts": end_ts,
            "receive_duration_ms": None if self.start_ts is None else int((end_ts - self.start_ts) * 1000),
        }


# ------------------ Background saving scheduler ------------------
try:
    # `storage` 패키지에서 통일된 인터페이스를 가져옵니다. (현재는 local_file_saver로 연결되어 있음)
    from storage import save_quiz, save_learning
except Exception:
    # 테스트/개발 환경에서 저장 모듈이 없을 경우 더미 구현을 사용합니다.
    async def save_quiz(*args, **kwargs):
        print("[inference_pipeline] save_quiz dummy called")
        return {"ok": False, "reason": "no_storage"}

    async def save_learning(*args, **kwargs):
        print("[inference_pipeline] save_learning dummy called")
        return {"ok": False, "reason": "no_storage"}


async def schedule_quiz_save(landmark_sequence: Optional[object], inference_result: Dict[str, Any], session_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
    """
    추론 완료 후 퀴즈 결과를 비동기적으로 저장하도록 스케줄합니다.
    - frames에서 랜드마크 시퀀스를 추출하고, 저장 인터페이스로 전달합니다.
    - meta는 웹소켓 세션에서 수집한 메타데이터(예: user_id, word_pk 등)를 전달합니다.
    """
    if meta is None:
        meta = {}

    try:
        # 이미 랜드마크 시퀀스를 받는다고 가정합니다. (None이면 저장 로직에서 처리)
        res = await save_quiz(
            landmark_sequence=landmark_sequence,
            inference_result=inference_result,
            session_id=session_id or (inference_result.get("session_id") if isinstance(inference_result, dict) else "unknown"),
            meta=meta,
        )
        print(f"[inference_pipeline] Quiz save result: {res}")

    except Exception as e:
        print(f"[inference_pipeline][ERROR] Quiz save failed: {e}")
        traceback.print_exc()


async def schedule_learning_save(landmark_sequence: Optional[object], session_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
    """
    학습 데이터 저장을 비동기적으로 스케줄합니다.
    - frames에서 랜드마크 시퀀스를 추출하여 저장 인터페이스로 전달합니다.
    """
    if meta is None:
        meta = {}

    try:
        # 이미 랜드마크 시퀀스를 받는다고 가정합니다.
        res = await save_learning(
            landmark_sequence=landmark_sequence,
            session_id=session_id or meta.get("session_id", "unknown"),
            meta=meta,
        )
        print(f"[inference_pipeline] Learning save result: {res}")

    except Exception as e:
        print(f"[inference_pipeline][ERROR] Learning save failed: {e}")
        traceback.print_exc()


# 유틸: synchronous run_inference를 다른 쓰레드/태스크에서 호출할 때 사용하는 래퍼
def run_inference_sync(predictor, landmark_sequence):
    return run_inference(predictor, landmark_sequence)
