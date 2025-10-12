"""
Inference pipeline module
- 분리된 run_inference 함수와 SequenceCollector 구현
- 퀴즈 결과를 비동기 백그라운드로 저장하는 스케줄러 제공

이 모듈은 heavy deps(mediapipe, torch 등)가 없을 경우에도 안전하게 동작하도록 예외 처리를 포함합니다.
"""
from __future__ import annotations

import os
import time
import traceback
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# 환경 변수와 동일한 기본값을 사용합니다.
TARGET_FRAME_COUNT = int(os.getenv("SIGN_SEQUENCE_TARGET_FRAMES", "300"))
COLLECTION_DURATION_SECONDS = float(os.getenv("SIGN_SEQUENCE_COLLECTION_SECONDS", "5.0"))
MAX_FRAMES_TO_COLLECT = 300

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
    """
    start = time.time()
    landmarks_info: Dict[str, Any] = {}
    predicted_label = "오류: 추론 실패"
    score = 0.0

    try:
        # 1. 랜드마크 추출
        if extract_sequence_from_frames is None:
            raise RuntimeError("Landmark extractor not available (missing heavy dependencies)")

        landmark_sequence = extract_sequence_from_frames(frames, target_len=TARGET_FRAME_COUNT, skip_missing=False)

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
            predicted_label, score = predictor.predict(landmark_sequence_float32)

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
        """시간 조건 또는 프레임 수 조건으로 수집이 완료되었는지 판단합니다."""
        if self.start_ts is None:
            return False

        time_elapsed = time.time() - self.start_ts
        if time_elapsed >= COLLECTION_DURATION_SECONDS:
            return True

        if len(self.frames) >= TARGET_FRAME_COUNT:
            return True

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
async def schedule_quiz_save(frames: List[bytes], inference_result: Dict[str, Any], session_id: Optional[str] = None):
    """
    추론 완료 후 퀴즈 결과를 비동기적으로 저장하도록 스케줄합니다.
    실제 저장 모듈이 없으면 더미 동작으로 로그만 남깁니다.
    """
    try:
        # 상대 경로 import로 강결합을 피하고, 없을 경우 예외를 잡습니다.
        from storage.s3_db_saver import save_quiz

        # save_quiz는 async 함수로 구현되어 있어야 합니다.
        res = await save_quiz(frames=frames, inference_result=inference_result, session_id=session_id)
        print(f"[inference_pipeline] Quiz save result: {res}")

    except Exception as e:
        print(f"[inference_pipeline][WARN] save_quiz not available or failed: {e}")
        # 간단한 대체 동작: 로컬에 임시 파일을 저장하거나 로그 처리(여기서는 로그)
        await asyncio.sleep(0.01)
        print("[inference_pipeline] Quiz save skipped (dummy).")


# 유틸: synchronous run_inference를 다른 쓰레드/태스크에서 호출할 때 사용하는 래퍼
def run_inference_sync(predictor, frames):
    return run_inference(predictor, frames)

