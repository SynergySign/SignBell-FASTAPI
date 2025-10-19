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
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

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


def run_inference(predictor: "Predictor", frames: List[bytes]) -> Dict[str, Any]:
    """
    수집된 프레임에 대해 랜드마크 추출 및 모델 추론을 수행합니다.
    (원래 main.py에 있던 내용을 여기로 옮겼습니다.)
    변경: 이제 TARGET_FRAME_COUNT로 강제 패딩/절단하지 않습니다. 클라이언트에서 넘어온 프레임 시퀀스 그대로 추출하여 모델에 전달합니다.
    """
    start = time.time()
    landmarks_info: Dict[str, Any] = {}
    predicted_label = "오류: 추론 실패"
    score = 0.0

    try:
        # 1. 랜드마크 추출
        if extract_sequence_from_frames is None:
            raise RuntimeError("Landmark extractor not available (missing heavy dependencies)")

        # NOTE: target_len=None 으로 호출하여 클라이언트에서 넘어온 길이를 유지합니다.
        landmark_sequence = extract_sequence_from_frames(frames, target_len=None, skip_missing=False)

        # 2. 추출된 데이터 유효성 검사
        if landmark_sequence is None or (hasattr(landmark_sequence, '__len__') and len(landmark_sequence) == 0):
            landmarks_info = {"enabled": True, "error": "landmark_sequence is None or empty"}
            predicted_label = "오류: 랜드마크를 감지하지 못했습니다."
        elif np is not None and np.all(landmark_sequence == 0):
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
            landmark_sequence_float32 = landmark_sequence.astype(np.float32) if np is not None else landmark_sequence
            # Static-analysis-safe local guard: ensure predictor is not None before calling
            p = predictor
            if p is None:
                raise RuntimeError("Predictor not available (missing heavy dependencies)")
            predicted_label, score = p.predict(landmark_sequence_float32)

    except Exception as e:
        print(f"[ERROR][inference_pipeline] Exception during inference: {e}")
        traceback.print_exc()
        error_message = str(e)
        landmarks_info = {"enabled": True, "error": error_message}
        predicted_label = f"오류: 추론 중 예외 발생 ({error_message})"

    end = time.time()
    return {
        "predicted": predicted_label,
        "score": score,
        "inference_start": start,
        "inference_end": end,
        "inference_ms": int((end - start) * 1000),
        "frames_used": len(frames),
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


async def schedule_quiz_save(frames: List[bytes], inference_result: Dict[str, Any], session_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
    """
    추론 완료 후 퀴즈 결과를 비동기적으로 저장하도록 스케줄합니다.
    - frames에서 랜드마크 시퀀스를 추출하고, 저장 인터페이스로 전달합니다.
    - meta는 웹소켓 세션에서 수집한 메타데이터(예: user_id, word_pk 등)를 전달합니다.
    """
    if meta is None:
        meta = {}

    try:
        if extract_sequence_from_frames is None:
            landmark_sequence = None
        else:
            landmark_sequence = extract_sequence_from_frames(frames, target_len=None, skip_missing=False)

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


async def schedule_learning_save(frames: List[bytes], session_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
    """
    학습 데이터 저장을 비동기적으로 스케줄합니다.
    - frames에서 랜드마크 시퀀스를 추출하여 저장 인터페이스로 전달합니다.
    """
    if meta is None:
        meta = {}

    try:
        if extract_sequence_from_frames is None:
            landmark_sequence = None
        else:
            landmark_sequence = extract_sequence_from_frames(frames, target_len=None, skip_missing=False)

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
def run_inference_sync(predictor, frames):
    return run_inference(predictor, frames)
