"""
Dummy S3/DB saver implementation.
- Provides async `save_learning` and `save_quiz` functions expected by the rest of the app.
- Functions perform local, non-production saves into `data/storage_exports/` for inspection.
- Uses minimal dependencies and falls back to simple file I/O if numpy is not available.
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
    """Helper to write bytes to a file using a thread to avoid blocking event loop."""
    def _sync_write():
        with open(path, "wb") as f:
            f.write(data)
    await asyncio.to_thread(_sync_write)


async def _write_json(path: Path, obj: Dict[str, Any]):
    def _sync_write():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    await asyncio.to_thread(_sync_write)


async def save_learning(frames: List[bytes], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Save frames + meta as learning data (dummy).
    Returns information about the saved artifacts.
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

    # Save frames as sequential binary files
    for i, b in enumerate(frames):
        p = out_dir / f"frame_{i:04d}.bin"
        await _write_file(p, b if isinstance(b, (bytes, bytearray)) else bytes(b))

    # Save metadata
    meta_path = out_dir / "meta.json"
    await _write_json(meta_path, {**meta, "frame_count": len(frames)})

    return {"ok": True, "type": "learning", "session_id": session_id, "path": str(out_dir)}


async def save_quiz(frames: List[bytes], inference_result: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    """Save quiz result and optional frames (dummy).
    Returns a dict describing saved artifacts.
    """
    try:
        ts = int(asyncio.get_event_loop().time() * 1000)
    except Exception:
        import time
        ts = int(time.time() * 1000)

    sid = session_id or inference_result.get("session_id", "unknown")
    out_dir = EXPORT_DIR / f"quiz_{sid}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save a small summary json
    summary = {
        "session_id": sid,
        "inference": inference_result,
        "frame_count": len(frames),
    }
    summary_path = out_dir / "summary.json"
    await _write_json(summary_path, summary)

    # Optionally save a few frames for inspection (not all to avoid huge disk usage)
    for i, b in enumerate(frames[:10]):
        p = out_dir / f"frame_{i:04d}.bin"
        await _write_file(p, b if isinstance(b, (bytes, bytearray)) else bytes(b))

    return {"ok": True, "type": "quiz", "session_id": sid, "path": str(out_dir)}

