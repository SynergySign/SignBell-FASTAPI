"""
모듈: routers/internal.py
설명:
- 이 라우터는 WebSocket 기반 흐름으로 대체되어 현재 사용되지 않습니다.
- 원래 엔드포인트들은 주석 처리되어 보관됩니다. 필요 시 주석을 해제하거나 삭제하세요.

since: 2025.10.17
author: 백승현
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/internal", tags=["internal"])

# NOTE: internal API endpoints were replaced by WebSocket flows in main.py.
# The original implementations are intentionally commented out to avoid duplicate code paths.

# Example (commented):
# @router.post("/save-quiz")
# async def save_quiz_endpoint(...):
#     pass

# @router.post("/save-learning")
# async def save_learning_endpoint(...):
#     pass
