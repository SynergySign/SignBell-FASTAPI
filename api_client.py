"""
Dummy internal API client for Spring ↔ FastAPI integration.
- 현재는 더미(모의) 구현입니다. 실제 Spring 통합 시 내부 호출(URL, 인증, payload 형식)에 맞춰 구현하세요.
"""
from typing import Optional, Dict, Any
import asyncio


class APIClient:
    """간단한 더미 클라이언트.

    목적:
    - Spring 서버가 FastAPI 내부 엔드포인트를 호출해야 할 때 사용할 인터페이스를 미리 정의합니다.
    - 실제 네트워크 호출을 하지 않고 내부 호출/백그라운드 작업 트리거를 모의합니다.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def post_save_learning(self, session_id: str, word: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """더미 저장 호출. 실제 구현에서는 HTTP 요청을 보냅니다.

        Returns a mock response dict synchronously to simulate success.
        """
        # 여기는 네트워크가 아닌 내부 테스트용 더미 응답입니다.
        await asyncio.sleep(0)  # 비동기 문맥을 유지
        return {
            "status": "ok",
            "action": "save_learning",
            "session_id": session_id,
            "word": word,
            "metadata": metadata,
        }

    async def post_save_quiz(self, session_id: str, predicted: str, score: float, timings: Optional[Dict[str, Any]] = None, landmarks_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """더미 퀴즈 저장 호출. 실제 구현에서는 HTTP 요청 또는 내부 함수 호출을 수행합니다."""
        await asyncio.sleep(0)
        return {
            "status": "ok",
            "action": "save_quiz",
            "session_id": session_id,
            "predicted": predicted,
            "score": score,
            "timings": timings,
            "landmarks_info": landmarks_info,
        }


# 모듈 수준의 간단한 싱글톤 접근자
_default_client: Optional[APIClient] = None


def get_default_client() -> APIClient:
    global _default_client
    if _default_client is None:
        _default_client = APIClient()
    return _default_client


if __name__ == "__main__":
    # 간단한 동작 확인
    async def _test():
        client = get_default_client()
        r1 = await client.post_save_learning("session123", "안녕", {"example": True})
        r2 = await client.post_save_quiz("session123", "안녕", 0.95, {"dur_ms": 1200}, {"enabled": True})
        print(r1)
        print(r2)

    asyncio.run(_test())

