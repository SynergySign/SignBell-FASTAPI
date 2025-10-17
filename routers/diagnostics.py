"""
모듈: routers/diagnostics.py
설명:
- 진단용 라우터로 간단한 상태 확인 및 에코 엔드포인트를 제공합니다.
- 주요 엔드포인트:
  - GET  /api/diagnostics/status : 라우터 상태 확인
  - POST /api/diagnostics/echo   : 요청 페이로드를 그대로 반환 (스모크 테스트용)

since: 2025.10.17
author: 백승현
"""

from fastapi import APIRouter
from typing import Any

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])


@router.get("/status")
async def diagnostics_status() -> Any:
    return {"ok": True, "message": "diagnostics router alive"}


@router.post("/echo")
async def diagnostics_echo(payload: dict) -> Any:
    """Simple echo endpoint used in smoke tests."""
    return {"echo": payload}
