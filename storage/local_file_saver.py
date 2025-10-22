"""
storage/local_file_saver.py
로컬 파일 시스템에 좌표(.npy)와 메타데이터(.csv)를 저장하는 유틸.
이 모듈은 비동기 인터페이스를 제공하며, 추후 S3/DB 구현으로 교체될 수 있습니다.

since: 2025.10.17
author: assistant
"""

from __future__ import annotations

import asyncio
import time
import numpy as np
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

# 로컬 저장 기본 경로 (프로젝트 루트의 data 디렉토리 사용)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


async def _save_metadata_csv(meta_data: Dict[str, Any], file_path: Path):
    """메타데이터를 CSV 파일에 추가합니다 (비동기 래퍼)."""
    await asyncio.to_thread(_save_meta_sync, meta_data, file_path)


def _save_meta_sync(meta_data: Dict[str, Any], file_path: Path):
    """동기 CSV 저장 로직"""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # 필드 정렬을 위해 key 순서를 고정합니다.
    fieldnames = list(meta_data.keys())

    file_exists = file_path.exists()
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: v for k, v in meta_data.items()})


async def _save_npy_sequence(sequence: np.ndarray, file_path: Path):
    """NumPy 배열을 .npy 파일로 저장합니다 (비동기 래퍼)."""
    await asyncio.to_thread(_save_npy_sync, sequence, file_path)


def _save_npy_sync(sequence: np.ndarray, file_path: Path):
    """동기 NumPy 저장 로직"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # numpy.save은 확장자를 자동으로 붙이므로 .npy 확장자 없이도 동작하지만
    # 일관성을 위해 파일명에 .npy를 포함하여 호출하도록 합니다.
    np.save(file_path, sequence)


# --- 퀴즈/추론 결과 저장 함수 ---
async def save_quiz_data(
    landmark_sequence: Optional[np.ndarray],
    inference_result: Dict[str, Any],
    session_id: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """퀴즈(추론) 결과와 랜드마크 시퀀스를 저장합니다.

    반환: 저장 위치 정보를 담은 dict
    """
    quiz_dir = DATA_DIR / "quiz"

    try:
        ts = int(time.time() * 1000)
    except Exception:
        ts = int(time.time() * 1000)

    # 메타데이터 준비
    quiz_meta = {
        "session_id": session_id,
        "timestamp": ts,
        **(meta or {}),
        "predicted_label": inference_result.get("predicted") if isinstance(inference_result, dict) else None,
        "score": inference_result.get("score") if isinstance(inference_result, dict) else None,
        "frames_used": inference_result.get("frames_used") if isinstance(inference_result, dict) else None,
    }

    meta_path = quiz_dir / "quiz_meta.csv"
    await _save_metadata_csv(quiz_meta, meta_path)

    if landmark_sequence is not None:
        npy_file_name = f"{session_id}_{ts}.npy"
        npy_path = quiz_dir / npy_file_name
        await _save_npy_sequence(landmark_sequence, npy_path)
        return {"ok": True, "type": "quiz", "session_id": session_id, "npy_path": str(npy_path), "meta_path": str(meta_path)}

    return {"ok": True, "type": "quiz", "session_id": session_id, "meta_path": str(meta_path), "message": "no_sequence"}


# --- 학습 데이터 저장 함수 ---
async def save_learning_data(
    landmark_sequence: Optional[np.ndarray],
    session_id: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """학습용 랜드마크 시퀀스와 메타데이터를 저장합니다."""
    learning_dir = DATA_DIR / "learning"

    try:
        ts = int(time.time() * 1000)
    except Exception:
        ts = int(time.time() * 1000)

    learning_meta = {
        "session_id": session_id,
        "timestamp": ts,
        **(meta or {}),
        "seq_shape": list(landmark_sequence.shape) if (landmark_sequence is not None and hasattr(landmark_sequence, 'shape')) else "N/A",
    }

    meta_path = learning_dir / "learning_meta.csv"
    await _save_metadata_csv(learning_meta, meta_path)

    if landmark_sequence is not None:
        npy_file_name = f"{session_id}_{ts}.npy"
        npy_path = learning_dir / npy_file_name
        await _save_npy_sequence(landmark_sequence, npy_path)
        return {"ok": True, "type": "learning", "session_id": session_id, "npy_path": str(npy_path), "meta_path": str(meta_path)}

    return {"ok": True, "type": "learning", "session_id": session_id, "meta_path": str(meta_path), "message": "no_sequence"}

