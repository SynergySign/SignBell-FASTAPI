"""
Storage interface abstractions for S3/DB saving.
This module defines async function signatures expected by the rest of the app.
Concrete implementations (e.g. `s3_db_saver.py`) should provide these functions.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional


async def save_learning(frames: List[bytes], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Save learning data (frames + meta) to persistent storage.

    This is an abstract placeholder. Implementations must return a dict with
    at least {"ok": bool, "path": str}.
    """
    raise NotImplementedError("save_learning must be implemented by a storage backend")


async def save_quiz(frames: List[bytes], inference_result: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    """Save quiz inference result and associated frames.

    Returns a dict describing the saved artifact.
    """
    raise NotImplementedError("save_quiz must be implemented by a storage backend")

