"""
모듈: storage/s3_db_saver.py
설명:
- 개발/테스트용으로 로컬 파일 시스템에 퀴즈 및 학습 데이터를 저장하는 더미 구현입니다.
- 애플리케이션에서 기대하는 비동기 `save_learning` 및 `save_quiz` 인터페이스를 제공합니다.
- 실제 운영 환경 대신 `data/storage_exports/`에 결과를 기록합니다 (프로덕션용 S3/DB와는 다름).

since: 2025.10.17
author: 백승현
"""

from __future__ import annotations

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXPORT_DIR = BASE_DIR / "data" / "storage_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


async def _write_file(path: Path, data: bytes):
    """이벤트 루프를 블로킹하지 않도록 스레드에서 파일 바이트를 기록합니다."""
    def _sync_write():
        with open(path, "wb") as f:
            f.write(data)
    await asyncio.to_thread(_sync_write)


async def _write_json(path: Path, obj: Dict[str, Any]):
    """JSON 객체를 파일에 비동기적으로 저장합니다."""
    def _sync_write():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    await asyncio.to_thread(_sync_write)


async def save_learning(frames: List[bytes], meta: Dict[str, Any]) -> Dict[str, Any]:
    """학습용 데이터(frames + meta)를 로컬에 저장하는 더미 구현.

    반환값: 저장 결과를 설명하는 dict({"ok": bool, "type": "learning", "session_id": ..., "path": ...})
    """
    try:
        ts = int(asyncio.get_event_loop().time() * 1000)
    except Exception:
        import time
        ts = int(time.time() * 1000)

    session_id = meta.get("session_id", meta.get("session", "unknown"))
    label = meta.get("label", "unlabeled")

    out_dir = EXPORT_DIR / f"learning_{session_id}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 프레임을 순차적인 바이너리 파일로 저장
    for i, b in enumerate(frames):
        p = out_dir / f"frame_{i:04d}.bin"
        await _write_file(p, b if isinstance(b, (bytes, bytearray)) else bytes(b))

    # 메타데이터 저장
    meta_path = out_dir / "meta.json"
    await _write_json(meta_path, {**meta, "frame_count": len(frames)})

    return {"ok": True, "type": "learning", "session_id": session_id, "path": str(out_dir)}


async def save_quiz(frames: List[bytes], inference_result: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    """퀴즈(추론) 결과와 일부 프레임을 로컬에 저장하는 더미 구현.

    반환값: 저장 결과를 설명하는 dict({"ok": bool, "type": "quiz", "session_id": ..., "path": ...})
    """
    try:
        ts = int(asyncio.get_event_loop().time() * 1000)
    except Exception:
        import time
        ts = int(time.time() * 1000)

    sid = session_id or inference_result.get("session_id", "unknown")
    out_dir = EXPORT_DIR / f"quiz_{sid}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 요약 JSON 저장
    summary = {
        "session_id": sid,
        "inference": inference_result,
        "frame_count": len(frames),
    }
    summary_path = out_dir / "summary.json"
    await _write_json(summary_path, summary)

    # 디스크 사용을 줄이기 위해 일부 프레임만 저장
    for i, b in enumerate(frames[:10]):
        p = out_dir / f"frame_{i:04d}.bin"
        await _write_file(p, b if isinstance(b, (bytes, bytearray)) else bytes(b))

    return {"ok": True, "type": "quiz", "session_id": sid, "path": str(out_dir)}
