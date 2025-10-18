"""
모듈: configs/settings.py
설명:
- 애플리케이션 설정을 로드하고 캐시된 Settings 인스턴스를 제공하는 경량 설정 모듈입니다.
- pydantic에 의존하지 않도록 구현되어 있어, pydantic 미설치 환경에서도 프로젝트가 실행될 수 있습니다.

주요 항목:
- JWT 관련 설정
- 추론/수집 관련 튜닝 파라미터 (TARGET_FRAME_COUNT 등)
- SSL 경로 설정

since: 2025.10.17
author: 백승현
"""

# Lightweight settings implementation that does not depend on pydantic.
# This keeps the project runnable even when pydantic v2/pydantic-settings are not installed.
from functools import lru_cache
import os


class Settings:
    def __init__(self):
        # JWT
        self.JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
        self.JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")

        # Inference / collector tuning
        self.TARGET_FRAME_COUNT: int = int(os.getenv("TARGET_FRAME_COUNT", "24"))
        self.COLLECTION_DURATION_SECONDS: float = float(os.getenv("COLLECTION_DURATION_SECONDS", "1.0"))
        self.MAX_FRAMES_TO_COLLECT: int = int(os.getenv("MAX_FRAMES_TO_COLLECT", "1024"))

        # SSL (optional)
        self.SSL_CERT_PATH: str = os.getenv("SSL_CERT_PATH", "certs/cert.pem")
        self.SSL_KEY_PATH: str = os.getenv("SSL_KEY_PATH", "certs/key.pem")

        # Cookie settings (used for cookie-based token extraction)
        # Default name kept as ACCESS_TOKEN to match Spring's suggested CookieProperties
        self.COOKIE_ACCESS_TOKEN_NAME: str = os.getenv("COOKIE_ACCESS_TOKEN_NAME", "ACCESS_TOKEN")
        # Optional max age for the access token cookie (seconds). Not used by validator but available to other modules.
        self.COOKIE_ACCESS_TOKEN_MAX_AGE: int = int(os.getenv("COOKIE_ACCESS_TOKEN_MAX_AGE", "3600"))


@lru_cache()
def get_settings() -> Settings:
    """캐시된 Settings 인스턴스를 반환합니다.

    사용 예:
        from configs.settings import get_settings
        settings = get_settings()
        secret = settings.JWT_SECRET_KEY
    """
    return Settings()


# Module-level 편의 별칭(기존 코드 호환성 유지)
_settings = get_settings()
JWT_SECRET_KEY = _settings.JWT_SECRET_KEY
JWT_ALGORITHM = _settings.JWT_ALGORITHM
TARGET_FRAME_COUNT = _settings.TARGET_FRAME_COUNT
COLLECTION_DURATION_SECONDS = _settings.COLLECTION_DURATION_SECONDS
MAX_FRAMES_TO_COLLECT = _settings.MAX_FRAMES_TO_COLLECT
SSL_CERT_PATH = _settings.SSL_CERT_PATH
SSL_KEY_PATH = _settings.SSL_KEY_PATH

# Cookie aliases
COOKIE_ACCESS_TOKEN_NAME = _settings.COOKIE_ACCESS_TOKEN_NAME
COOKIE_ACCESS_TOKEN_MAX_AGE = _settings.COOKIE_ACCESS_TOKEN_MAX_AGE

# get_settings 함수 노출
get_settings = get_settings
