"""
모듈: security/jwt_validator.py
설명:
- JWT 디코딩/검증 유틸리티를 제공합니다. FastAPI Depends 및 WebSocket 검증 흐름에서 사용됩니다.
- 제공 함수:
  - get_current_user_id(token: str = Depends(oauth2_scheme)) : REST 엔드포인트용 의존성, 토큰에서 user id 반환
  - validate_token_and_get_user_id(token: str) : WebSocket 핸드셰이크 등에서 토큰을 검증하고 user id 반환

주의: `python-jose`가 설치되어 있지 않으면 데코딩 시 예외를 발생시키는 더미 구현이 사용됩니다.

since: 2025.10.17
author: 백승현
"""

from typing import Any, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

# Try to import python-jose; if unavailable provide a dummy implementation that raises
try:
    from jose import JWTError, jwt  # type: ignore
    _HAVE_JOSE = True
except Exception:
    # Provide fallbacks so module import does not fail when `python-jose` is not installed.
    JWTError = Exception  # type: ignore
    _HAVE_JOSE = False

    class _DummyJWT:
        @staticmethod
        def decode(token: str, key: str, algorithms: list[str]):
            raise RuntimeError("python-jose is not installed. Install with `pip install python-jose[cryptography]` to enable JWT decoding.")

    jwt = _DummyJWT()  # type: ignore

from configs import settings

# OAuth2 scheme for REST endpoints (token endpoint not implemented here; used for Depends)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


def _decode_jwt(token: str) -> dict[str, Any]:
    """Decode and verify a JWT using project settings.

    Raises HTTPException(401) on any verification error.
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except Exception as exc:
        # Catch any error from jwt.decode (including missing dependency) and translate to HTTPException
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(exc)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _extract_token_from_request(request: Request) -> Optional[str]:
    """Extract JWT from request using the strategy: Authorization: Bearer <token> -> cookie.

    This mirrors the Spring `resolveToken` behaviour: prefer the Authorization header, fall back to
    an HTTP-only cookie named by `settings.COOKIE_ACCESS_TOKEN_NAME`.

    Returns the token string or None if not found.
    """
    # 1) Authorization header (Bearer)
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]

    # 2) Cookie fallback
    try:
        cookie_name = getattr(settings, "COOKIE_ACCESS_TOKEN_NAME", "ACCESS_TOKEN")
        token = request.cookies.get(cookie_name)
        if token:
            return token
    except Exception:
        # Defensive: request.cookies may not behave as dict-like in some test doubles
        pass

    return None


def _extract_user_id_from_payload(payload: dict[str, Any]) -> Any:
    """Normalize user id extraction from JWT payload (sub or user_id)."""
    return payload.get("sub") or payload.get("user_id")


def get_current_user_id(request: Request) -> Any:
    """FastAPI dependency to obtain current user id from the incoming Request.

    This supports tokens provided in either the Authorization Bearer header or an HTTP cookie
    (see `_extract_token_from_request`). Use as `Depends(get_current_user_id)` in REST endpoints.
    """
    token = _extract_token_from_request(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token (expected Authorization header or cookie)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = _decode_jwt(token)
    user_id = _extract_user_id_from_payload(payload)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required user identifier",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


def validate_token_and_get_user_id(token: str) -> Any:
    """Utility for WebSocket handlers: validate raw token (query param) and return user id.

    This decodes the provided token directly (WebSocket handshakes may only send a query param).
    Caller (e.g., WebSocket handshake) should catch HTTPException and close the connection if needed.
    """
    payload = _decode_jwt(token)
    user_id = _extract_user_id_from_payload(payload)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required user identifier",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id
