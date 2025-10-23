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

TARGET_FRAME_COUNT = settings.TARGET_FRAME_COUNT
# COLLECTION_DURATION_SECONDS 제거 (타이머 로직 없음)
MAX_FRAMES_TO_COLLECT = settings.MAX_FRAMES_TO_COLLECT

# ... (import 부분 동일) ...
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
    # ... (함수 내용 동일) ...
    start = time.time()
    landmarks_info: Dict[str, Any] = {}
    predicted_label = "오류: 추론 실패"
    score = 0.0

    try:
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
            p_any = cast(Any, predictor)
            if not hasattr(p_any, "predict"):
                raise RuntimeError("Predictor does not implement 'predict' method")
            predicted_label, score = p_any.predict(landmark_sequence_float32)

    except Exception as e:
        # ... (예외 처리 동일) ...
        pass
    end = time.time()
    frames_used = 0
    try:
        if landmark_sequence is not None:
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
    """세션별 프레임 수집기."""
    frames: List[bytes] = field(default_factory=list)

    # --- ⬇️ 수정: start_ts 와 processed 필드 제거 ⬇️ ---
    # start_ts: Optional[float] = None
    # processed: bool = False
    # --- ⬆️ 수정 완료 ⬆️ ---

    # --- ⬇️ 수정: start_collection 메서드 제거 ⬇️ ---
    # def start_collection(self):
    #     ...
    # --- ⬆️ 수정 완료 ⬆️ ---

    def add_frame(self, data: bytes):
        """
        최대 프레임 수를 초과하지 않은 경우에만 프레임을 추가합니다.
        (시간제한 및 processed 로직 제거)
        """
        # --- ⬇️ 수정: if self.processed: return 제거 ⬇️ ---
        # if self.processed:
        #     return
        # --- ⬆️ 수정 완료 ⬆️ ---

        # --- ⬇️ 수정: 타이머(start_ts) 체크 로직 제거 ⬇️ ---
        if not self.is_full():
            if len(self.frames) < MAX_FRAMES_TO_COLLECT:
                self.frames.append(data)
        # --- ⬆️ 수정 완료 ⬆️ ---

    def is_full(self) -> bool:
        """프레임 수 조건으로 수집이 완료되었는지 판단합니다."""

        # --- ⬇️ 수정: 타이머(start_ts) 체크 로직 제거 ⬇️ ---
        # if self.start_ts is None:
        #     return False
        # ... (시간 관련 if 문 제거) ...
        # --- ⬆️ 수정 완료 ⬆️ ---

        if len(self.frames) >= MAX_FRAMES_TO_COLLECT:
            return True
        return False

    def build_timings(self) -> Dict[str, Any]:
        end_ts = time.time()
        # --- ⬇️ 수정: start_ts 관련 로직 제거 ⬇️ ---
        return {
            "frame_count": len(self.frames),
            "receive_first_ts": None,
            "receive_last_ts": end_ts,
            "receive_duration_ms": None,
        }
        # --- ⬆️ 수정 완료 ⬆️ ---

# ------------------ Background saving scheduler ------------------
# ... (이하 저장 로직 동일) ...
try:
    from storage import save_quiz, save_learning
except Exception:
    async def save_quiz(*args, **kwargs):
        print("[inference_pipeline] save_quiz dummy called")
        return {"ok": False, "reason": "no_storage"}

    async def save_learning(*args, **kwargs):
        print("[inference_pipeline] save_learning dummy called")
        return {"ok": False, "reason": "no_storage"}


async def schedule_quiz_save(landmark_sequence: Optional[object], inference_result: Dict[str, Any], session_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
    # ... (함수 내용 동일) ...
    pass

async def schedule_learning_save(landmark_sequence: Optional[object], session_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
    # ... (함수 내용 동일) ...
    pass

def run_inference_sync(predictor, landmark_sequence):
    return run_inference(predictor, landmark_sequence)