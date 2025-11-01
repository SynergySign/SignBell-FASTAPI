"""WebSocket handler 모듈 (토큰 로직 복원, Lazy Import/Init 적용)

[수정 사항]
- (1) 원본의 토큰 검증 로직을 복원하여 'NameError'로 인한 1006 즉시 크래시 문제를 해결합니다.
- (2) "Lazy Import" + "Lazy Init"을 적용하여 Mediapipe 메모리 누수를 해결합니다.
"""
from __future__ import annotations
import json
import traceback
import asyncio
from typing import Any, Optional, Type # 👈 Type 임포트 추가

from fastapi import WebSocket, WebSocketDisconnect, status
from security.jwt_validator import validate_token_and_get_user_id
from configs import settings
from inference_pipeline import run_inference, schedule_quiz_save, schedule_learning_save

# --- ⬇️ 클래스 임포트 제거 ⬇️ ---
# (파일 상단에서 mediapipe 관련 모듈을 임포트하지 않습니다)
# --- ⬆️ 임포트 제거 완료 ⬆️ ---

# 타입 힌팅을 위해 클래스 변수를 미리 선언 (값은 None)
RealtimeLandmarkExtractor: Optional[Type] = None
SequenceBuilder: Optional[Type] = None


async def websocket_handler(websocket: WebSocket, session_id: str, app_state: Any):

    # --- ⬇️ (1) 원본 토큰 검증 로직 복원 (1006 크래시 해결) ⬇️ ---
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
        print("[WS HANDSHAKE] query_params:", query_dict) # 로그 순서 변경

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

        user_id = validate_token_and_get_user_id(token)
        print(f"[WS AUTH] token validated for user_id={user_id}")
    except Exception as e:
        print(f"[WS AUTH] token validation failed: {e}")
        await websocket.close(code=status.HTTP_401_UNAUTHORIZED)
        return
    # --- ⬆️ 토큰 검증 로직 복원 완료 ⬆️ ---

    await websocket.accept()
    collector = app_state.collectors.get(session_id) or app_state.new_collector(session_id)

    # --- ⬇️ (2) Lazy Initialization (메모리 누수 해결) ⬇️ ---
    extractor: Optional[Any] = None # 타입을 Any로 변경
    # --- ⬆️ 수정 완료 ⬆️ ---

    # --- ⬇️ 전역 변수 참조 (Lazy Import용) ⬇️ ---
    global RealtimeLandmarkExtractor, SequenceBuilder
    # --- ⬆️ 추가 ⬆️ ---

    print(f"[WS HANDLER {session_id}] Handler function started. Extractor will be imported & initialized on first use.")

    try:
        print(f"[WS HANDLER {session_id}] Entering main receive loop...")
        while True:
            data = await websocket.receive()

            if "bytes" in data and data.get("bytes") is not None:
                try:
                    collector.add_frame(data["bytes"])
                except Exception as e:
                    print(f"[WS][WARN] Failed to add frame: {e}")
                continue

            if "text" in data and data.get("text") is not None:
                raw = data["text"]
                try:
                    msg = json.loads(raw)
                except Exception:
                    print(f"[WS][WARN] Failed to parse JSON: {raw}")
                    continue

                mtype = msg.get("type")

                if mtype == "meta":
                    collector.meta = { "word_pk": msg.get("word_pk"), "word_name": msg.get("word_name"), "user_id": user_id }
                    try:
                        collector.frames = []
                    except Exception:
                        pass
                    await websocket.send_text(json.dumps({"type": "meta_ack"}))

                elif mtype == "save_learning" or mtype == "flush":
                    # --- ⬇️ "진짜 Lazy Import + Init" 로직 (메모리 누수 해결) ⬇️ ---
                    if extractor is None:
                        # 1. 클래스가 아직 로드되지 않았다면(전역 변수가 None이면) 임포트
                        if RealtimeLandmarkExtractor is None or SequenceBuilder is None:
                            print(f"[WS HANDLER {session_id}] Lazily *importing* Extractor/Builder...")
                            try:
                                from processing.landmark_extractor import RealtimeLandmarkExtractor as ExtractorCls
                                from processing.landmark_extractor import SequenceBuilder as BuilderCls
                                RealtimeLandmarkExtractor = ExtractorCls
                                SequenceBuilder = BuilderCls
                                print(f"[WS HANDLER {session_id}] Import successful.")
                            except Exception as e_import:
                                print(f"[WS HANDLER {session_id}] [FATAL] Lazy import failed: {e_import}")
                                traceback.print_exc()
                                await websocket.send_text(json.dumps({"type": "error", "message": "Import failed"}))
                                raise # 핸들러 종료

                        # 2. 클래스 생성 (Lazy Init)
                        print(f"[WS HANDLER {session_id}] Lazily *initializing* RealtimeLandmarkExtractor...")
                        try:
                            extractor = RealtimeLandmarkExtractor(skip_missing=False) # type: ignore
                            if not extractor.available():
                                print(f"[WS HANDLER {session_id}] [ERROR] Extractor lazy init failed (not available)!")
                                extractor = None
                        except Exception as e:
                            print(f"[WS HANDLER {session_id}] [ERROR] RealtimeLandmarkExtractor lazy init crash: {e}")
                            traceback.print_exc()
                            extractor = None
                            await websocket.send_text(json.dumps({"type": "error", "message": "Extractor init failed"}))
                            raise # 핸들러 종료

                    # --- ⬆️ 지연 초기화/임포트 완료 ⬆️ ---

                    frames = getattr(collector, "frames", [])
                    session_meta = getattr(collector, "meta", {})
                    landmark_sequence = None

                    if extractor is not None and SequenceBuilder is not None:
                        try:
                            print(f"[WS HANDLER {session_id}] Calling SequenceBuilder with *reused* extractor (for {mtype})...")
                            builder = SequenceBuilder(extractor)
                            for fb in frames:
                                builder.add_frame(fb)
                            landmark_sequence = builder.build(target_len=None)
                            print(f"[WS HANDLER {session_id}] Sequence built. Shape: {landmark_sequence.shape if landmark_sequence is not None else 'None'}")
                        except Exception as e:
                            print(f"[WS][ERROR] SequenceBuilder failed: {e}")
                            traceback.print_exc()
                    else:
                         print(f"[WS][ERROR] Extractor or Builder is still None. Cannot process frames.")

                    # --- 분기 처리 (save_learning / flush) ---
                    if mtype == "save_learning":
                        if landmark_sequence is None:
                            await websocket.send_text(json.dumps({"type": "learning_ack", "status": "failed", "reason": "landmark_extraction_failed"}))
                        else:
                            asyncio.create_task(schedule_learning_save(landmark_sequence, session_id, session_meta))
                            await websocket.send_text(json.dumps({"type": "learning_ack", "status": "accepted"}))

                    elif mtype == "flush":
                        result = run_inference(app_state.predictor, landmark_sequence)
                        result["frames_used"] = len(frames)
                        asyncio.create_task(schedule_quiz_save(landmark_sequence, result, session_id, session_meta))
                        await websocket.send_text(json.dumps({"type": "inference_result", "result": result}))

                    try:
                        collector.frames = []
                    except Exception:
                        pass

                else:
                    await websocket.send_text(json.dumps({"type": "noop"}))

    except WebSocketDisconnect:
        print(f"[WS DISCONNECT {session_id}] Connection closed.")
    except Exception as e:
        print(f"[WS][ERROR] Unexpected error in websocket handler for session {session_id}: {e}")
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": "Internal server error"}))
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        # --- ⬇️ 종료 시 Extractor 닫기 (중요) ⬇️ ---
        if extractor:
            try:
                extractor.close()
                print(f"[WS CLOSE {session_id}] Reusable extractor closed.")
            except Exception as e_close:
                print(f"[WS CLOSE {session_id}] Error closing extractor: {e_close}")
        # --- ⬆️ 수정 완료 ⬆️ ---
        app_state.collectors.pop(session_id, None)
        return