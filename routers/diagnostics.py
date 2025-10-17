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

