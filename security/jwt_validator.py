from typing import Any

from fastapi import Depends, HTTPException, status
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


def get_current_user_id(token: str = Depends(oauth2_scheme)) -> Any:
    """FastAPI dependency to obtain current user id from a Bearer token.

    This is convenient for REST endpoints that use Depends(get_current_user_id).
    It returns the value of `sub` or `user_id` claim (as-is).
    """
    payload = _decode_jwt(token)
    user_id = payload.get("sub") or payload.get("user_id")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required user identifier",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


def validate_token_and_get_user_id(token: str) -> Any:
    """Utility for WebSocket handlers: validate raw token (query param) and return user id.

    Caller (e.g., WebSocket handshake) should catch HTTPException and close the connection if needed.
    """
    return get_current_user_id(token)
