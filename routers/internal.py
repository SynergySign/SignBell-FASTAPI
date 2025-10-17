from fastapi import APIRouter, Depends, HTTPException
from typing import Any

import asyncio

from security.jwt_validator import get_current_user_id
from schemas import SaveQuizRequest, SaveLearningRequest
from storage.s3_db_saver import save_quiz, save_learning

router = APIRouter(prefix="/api/internal", tags=["internal"])


@router.post("/save-quiz")
async def save_quiz_endpoint(payload: SaveQuizRequest, user_id: Any = Depends(get_current_user_id)):
    """Schedule quiz save in background. Returns accepted status."""
    try:
        # Schedule background task to save quiz; pass empty frames (caller may include frames separately)
        asyncio.create_task(save_quiz(frames=[], inference_result=payload.dict(), session_id=payload.session_id))
        return {"status": "accepted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-learning")
async def save_learning_endpoint(payload: SaveLearningRequest, user_id: Any = Depends(get_current_user_id)):
    """Schedule learning save in background. Returns accepted status."""
    try:
        # Schedule background task to save learning data; frames may be attached separately in real flow
        asyncio.create_task(save_learning(frames=[], meta=payload.dict()))
        return {"status": "accepted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
