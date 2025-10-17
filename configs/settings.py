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


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Usage:
        from configs.settings import get_settings
        settings = get_settings()
        secret = settings.JWT_SECRET_KEY
    """
    return Settings()


# Module-level convenience aliases for backward compatibility with code that imported
# `from configs import settings; settings.TARGET_FRAME_COUNT`.
_settings = get_settings()
JWT_SECRET_KEY = _settings.JWT_SECRET_KEY
JWT_ALGORITHM = _settings.JWT_ALGORITHM
TARGET_FRAME_COUNT = _settings.TARGET_FRAME_COUNT
COLLECTION_DURATION_SECONDS = _settings.COLLECTION_DURATION_SECONDS
MAX_FRAMES_TO_COLLECT = _settings.MAX_FRAMES_TO_COLLECT
SSL_CERT_PATH = _settings.SSL_CERT_PATH
SSL_KEY_PATH = _settings.SSL_KEY_PATH

# Expose function
get_settings = get_settings
