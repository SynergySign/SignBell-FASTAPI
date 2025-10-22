"""WebSocket handler 모듈

이 모듈은 `main.py`에서 사용되는 WebSocket 연결 처리 로직을 포함합니다.
핸드셰이크, 토큰 검증, 텍스트/바이너리 수신 루프, collector 관리 등을 수행합니다.
"""
from __future__ import annotations

import json
import traceback
import asyncio
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect, status

from security.jwt_validator import validate_token_and_get_user_id
from configs import settings
from inference_pipeline import run_inference, schedule_quiz_save, schedule_learning_save


async def websocket_handler(websocket: WebSocket, session_id: str, app_state: Any):
    """WebSocket 연결을 처리하는 공통 핸들러.

    Args:
        websocket: FastAPI WebSocket 객체
        session_id: URL 경로로 전달된 세션 아이디
        app_state: main.py의 app.state.ss 객체 (AppState 인스턴스)
    """
    # 핸드셰이크와 토큰 검증
    try:
        try:
            headers_dict = dict(websocket.headers)
        except Exception:
            headers_dict = {}
        try:
            cookies_dict = websocket.cookies or {}
        except Exception:
            cookies_dict = {}
        try:
            query_dict = dict(websocket.query_params)
        except Exception:
            query_dict = {}

        print(f"[WS HANDSHAKE] session_id={session_id} path={getattr(websocket, 'url', None)}")
        print("[WS HANDSHAKE] headers:", headers_dict)
        print("[WS HANDSHAKE] cookies:", cookies_dict)
        print("[WS HANDSHAKE] query_params:", query_dict)

        # Token extraction: cookie -> Authorization header -> query param
        token = None
        token_source = None
        try:
            token = websocket.cookies.get(settings.COOKIE_ACCESS_TOKEN_NAME)
            if token:
                token_source = f"cookie({settings.COOKIE_ACCESS_TOKEN_NAME})"
        except Exception:
            token = None

        if not token:
            auth_header = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
            if auth_header:
                parts = auth_header.split()
                if len(parts) == 2 and parts[0].lower() == "bearer":
                    token = parts[1]
                    token_source = "authorization_header"

        if not token:
            token = websocket.query_params.get("token")
            if token:
                token_source = "query_param"

        print("[WS HANDSHAKE] resolved token source:", token_source is not None, token_source)

        if not token:
            print(f"[WS AUTH] No token found for session {session_id} - rejecting handshake")
            await websocket.close(code=status.HTTP_401_UNAUTHORIZED)
            return

        # Token debug 로그 (unverified)
        try:
            try:
                from jose import jwt as _jose_jwt
                try:
                    hdr = _jose_jwt.get_unverified_header(token)
                except Exception:
                    hdr = None
                try:
                    claims = _jose_jwt.get_unverified_claims(token)
                except Exception:
                    claims = None
                print("[WS DEBUG] unverified token header:", hdr)
                print("[WS DEBUG] unverified token payload:", claims)
            except Exception:
                try:
                    import jwt as _pyjwt
                    try:
                        hdr = _pyjwt.get_unverified_header(token)
                    except Exception:
                        hdr = None
                    try:
                        claims = _pyjwt.decode(token, options={"verify_signature": False})
                    except Exception:
                        claims = None
                    print("[WS DEBUG] unverified token header:", hdr)
                    print("[WS DEBUG] unverified token payload:", claims)
                except Exception as _e:
                    print("[WS DEBUG] failed to parse token unverified:", _e)
        except Exception as _e:
            print("[WS DEBUG] unexpected error while logging token unverified:", _e)

        # Validate token and get user_id
        try:
            user_id = validate_token_and_get_user_id(token)
            print(f"[WS AUTH] token validated for user_id={user_id}")
        except Exception as e:
            print(f"[WS AUTH] token validation failed: {e}")
            await websocket.close(code=status.HTTP_401_UNAUTHORIZED)
            return
    except Exception as _e:
        print("[WS HANDSHAKE][ERROR] Unexpected handshake error:", _e)
        await websocket.close(code=status.HTTP_401_UNAUTHORIZED)
        return

    await websocket.accept()

    # 세션용 collector 준비
    collector = app_state.collectors.get(session_id) or app_state.new_collector(session_id)

    try:
        while True:
            data = await websocket.receive()

            # ASGI 에서의 disconnect 이벤트 처리
            if isinstance(data, dict) and data.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(code=data.get("code"))

            # 바이너리 프레임 수신
            if "bytes" in data and data.get("bytes") is not None:
                frame_bytes = data["bytes"]
                try:
                    collector.start_collection()
                except Exception:
                    pass
                try:
                    collector.add_frame(frame_bytes)
                except Exception as e:
                    print(f"[WS][WARN] Failed to add frame for session {session_id}: {e}")
                continue

            # 텍스트(JSON) 신호 처리
            if "text" in data and data.get("text") is not None:
                raw = data["text"]
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                mtype = msg.get("type")
                if mtype == "meta":
                    collector.meta = {
                        "word_pk": msg.get("word_pk"),
                        "word_name": msg.get("word_name"),
                        "user_id": user_id,
                    }
                    try:
                        collector.frames = []
                        collector.processed = False
                        collector.start_collection()
                    except Exception:
                        pass
                    await websocket.send_text(json.dumps({"type": "meta_ack"}))

                elif mtype == "save_learning":
                    frames = getattr(collector, "frames", [])
                    session_meta = getattr(collector, "meta", {})
                    asyncio.create_task(
                        schedule_learning_save(frames=frames, session_id=session_id, meta=session_meta)
                    )
                    await websocket.send_text(json.dumps({"type": "learning_ack", "status": "accepted"}))

                elif mtype == "flush":
                    predictor = app_state.predictor
                    frames = getattr(collector, "frames", [])
                    result = run_inference(predictor, frames)
                    session_meta = getattr(collector, "meta", {})
                    asyncio.create_task(
                        schedule_quiz_save(frames=frames, inference_result=result, session_id=session_id, meta=session_meta)
                    )
                    await websocket.send_text(json.dumps({"type": "inference_result", "result": result}))

                else:
                    await websocket.send_text(json.dumps({"type": "noop"}))

    except WebSocketDisconnect:
        app_state.collectors.pop(session_id, None)
        return

    except Exception as e:
        print(f"[WS][ERROR] Unexpected error in websocket handler for session {session_id}: {e}")
        traceback.print_exc()
        try:
            await websocket.close()
        except Exception:
            pass
        app_state.collectors.pop(session_id, None)
        return

