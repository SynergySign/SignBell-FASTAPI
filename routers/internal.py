"""
모듈: routers/internal.py
설명:
- 내부 API 라우터로 인증된 서비스 간 통신용 엔드포인트를 제공합니다.
- 주요 엔드포인트:
  - POST /api/internal/save-quiz     : 퀴즈(추론) 결과를 비동기로 저장하도록 스케줄
  - POST /api/internal/save-learning : 학습용 데이터 저장을 비동기로 스케줄

since: 2025.10.17
author: 백승현
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Any

import asyncio

from security.jwt_validator import get_current_user_id
from schemas import SaveQuizRequest, SaveLearningRequest
from storage.s3_db_saver import save_quiz, save_learning

router = APIRouter(prefix="/api/internal", tags=["internal"])


def _model_to_dict(m):
    """Pydantic 모델을 dict로 변환합니다. pydantic v2의 model_dump를 우선 사용하고, 없으면 dict() 사용.

    since: 2025.10.17
    author: 백승현
    """
    if hasattr(m, "model_dump"):
        return m.model_dump()
    return m.dict()


@router.post("/save-quiz")
async def save_quiz_endpoint(payload: SaveQuizRequest, user_id: Any = Depends(get_current_user_id)):
    """Schedule quiz save in background. Returns accepted status."""
    try:
        # Schedule background task to save quiz; pass empty frames (caller may include frames separately)
        asyncio.create_task(save_quiz(frames=[], inference_result=_model_to_dict(payload), session_id=payload.session_id))
        return {"status": "accepted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-learning")
async def save_learning_endpoint(payload: SaveLearningRequest, user_id: Any = Depends(get_current_user_id)):
    """Schedule learning save in background. Returns accepted status."""
    try:
        # Schedule background task to save learning data; frames may be attached separately in real flow
        asyncio.create_task(save_learning(frames=[], meta=_model_to_dict(payload)))
        return {"status": "accepted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
