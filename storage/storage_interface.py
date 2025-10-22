"""
모듈: storage/storage_interface.py
설명:
- S3 또는 데이터베이스 저장을 위한 저장소 인터페이스 추상 정의를 제공합니다.
- 애플리케이션의 나머지 부분은 이 모듈에 정의된 비동기 함수 시그니처를 기대합니다.
- 실제 구현(s3_db_saver 등)은 이 인터페이스를 구현해야 합니다.

since: 2025.10.17
author: 백승현
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional


async def save_learning(frames: List[bytes], meta: Dict[str, Any]) -> Dict[str, Any]:
    """학습용 데이터(frames + meta)를 영구 저장소에 저장합니다.

    이 함수는 추상적 플레이스홀더입니다. 실제 구현체는 최소한 {"ok": bool, "path": str} 형태의 dict를 반환해야 합니다.
    """
    raise NotImplementedError("save_learning must be implemented by a storage backend")


async def save_quiz(frames: List[bytes], inference_result: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    """퀴즈(추론) 결과와 관련 프레임을 저장합니다.

    저장 결과를 설명하는 dict를 반환해야 합니다.
    """
    raise NotImplementedError("save_quiz must be implemented by a storage backend")
