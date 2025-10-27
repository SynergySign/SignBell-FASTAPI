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
try:
    from processing.landmark_extractor import extract_sequence_from_frames
except Exception:
    extract_sequence_from_frames = None

async def websocket_handler(websocket: WebSocket, session_id: str, app_state: Any):
    # --- 핸드셰이크 및 토큰 검증 로직 시작 (변경 없음) ---
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
        try:
            try:
                from jose import jwt as _jose_jwt
                # ... (unverified token 로그 동일) ...
            except Exception:
                try:
                    import jwt as _pyjwt
                    # ... (unverified token 로그 동일) ...
                except Exception as _e:
                    print("[WS DEBUG] failed to parse token unverified:", _e)
        except Exception as _e:
            print("[WS DEBUG] unexpected error while logging token unverified:", _e)
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
    # --- 핸드셰이크 및 토큰 검증 로직 끝 ---

    await websocket.accept()
    collector = app_state.collectors.get(session_id) or app_state.new_collector(session_id)

    # --- ⬇️ 여기 로그 추가 ⬇️ --- (이전 추가 로그)
    print(f"[WS HANDLER {session_id}] Handler function started successfully.")
    print(f"[WS HANDLER {session_id}] Type of 'extract_sequence_from_frames' in this scope: {type(extract_sequence_from_frames)}")
    # --- ⬆️ 로그 추가 완료 ⬆️ ---

    try:
        # --- ⬇️ 여기 로그 추가 ⬇️ --- (이번 추가 로그 1)
        print(f"[WS HANDLER {session_id}] Entering main receive loop...")
        # --- ⬆️ 로그 추가 완료 ⬆️ ---
        while True:
            # --- ⬇️ 여기 로그 추가 ⬇️ --- (이번 추가 로그 2)
            print(f"[WS HANDLER {session_id}] Waiting to receive data...")
            # --- ⬆️ 로그 추가 완료 ⬆️ ---

            data = await websocket.receive()

            # --- ⬇️ 여기 로그 추가 ⬇️ --- (이번 추가 로그 3)
            print(f"[WS HANDLER {session_id}] Received data type: {type(data)}")
            # --- ⬆️ 로그 추가 완료 ⬆️ ---

            # --- ⬇️ 여기 로그 추가 ⬇️ --- (이번 추가 로그 4)
            # 수신된 데이터의 상세 내용 (텍스트/바이트 구분)
            if "text" in data:
                print(f"[WS HANDLER {session_id}] Received TEXT content: {data.get('text')}")
            elif "bytes" in data:
                print(f"[WS HANDLER {session_id}] Received BYTES content length: {len(data.get('bytes', b''))}")
            # --- ⬆️ 로그 추가 완료 ⬆️ ---

            if isinstance(data, dict) and data.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(code=data.get("code"))

            # 바이너리 프레임 수신
            if "bytes" in data and data.get("bytes") is not None:
                # --- ⬇️ 여기 로그 추가 ⬇️ --- (이번 추가 로그 5)
                print(f"[WS HANDLER {session_id}] Handling BYTES data...")
                # --- ⬆️ 로그 추가 완료 ⬆️ ---
                frame_bytes = data["bytes"]
                try:
                    collector.add_frame(frame_bytes)
                    # --- ⬇️ 여기 로그 추가 ⬇️ --- (이번 추가 로그 6)
                    print(f"[WS HANDLER {session_id}] Frame added. Total frames: {len(getattr(collector, 'frames', []))}")
                    # --- ⬆️ 로그 추가 완료 ⬆️ ---
                except Exception as e:
                    print(f"[WS][WARN] Failed to add frame: {e}")
                continue # 다음 메시지 기다림

            # 텍스트(JSON) 신호 처리
            if "text" in data and data.get("text") is not None:
                # --- ⬇️ 여기 로그 추가 ⬇️ --- (이번 추가 로그 7)
                print(f"[WS HANDLER {session_id}] Handling TEXT data...")
                # --- ⬆️ 로그 추가 완료 ⬆️ ---
                raw = data["text"]
                try:
                    msg = json.loads(raw)
                except Exception:
                    print(f"[WS][WARN] Failed to parse JSON: {raw}") # JSON 파싱 실패 로그 추가
                    continue

                mtype = msg.get("type")
                if mtype == "meta":
                    collector.meta = {
                        "word_pk": msg.get("word_pk"),
                        "word_name": msg.get("word_name"),
                        "user_id": user_id,
                    }
                    try:
                        collector.frames = [] # 새 작업 시작 시 프레임 비우기
                    except Exception:
                        pass
                    await websocket.send_text(json.dumps({"type": "meta_ack"}))

                elif mtype == "save_learning":
                    frames = getattr(collector, "frames", [])
                    session_meta = getattr(collector, "meta", {})
                    if extract_sequence_from_frames is None:
                        landmark_sequence = None
                    else:
                        try:
                            # --- ⬇️ 여기 로그 추가 ⬇️ --- (이번 추가 로그 8)
                            print(f"[WS HANDLER {session_id}] Calling extract_sequence_from_frames for save_learning...")
                            # --- ⬆️ 로그 추가 완료 ⬆️ ---
                            landmark_sequence = extract_sequence_from_frames(frames, target_len=None, skip_missing=False)
                            # --- ⬇️ 여기 로그 추가 ⬇️ --- (이번 추가 로그 9)
                            print(f"[WS HANDLER {session_id}] extract_sequence_from_frames finished. Result type: {type(landmark_sequence)}")
                            # --- ⬆️ 로그 추가 완료 ⬆️ ---
                        except Exception as e:
                            print(f"[WS][ERROR] extract_sequence_from_frames failed: {e}") # 오류 로그 레벨 변경
                            traceback.print_exc() # 상세 트레이스백 추가
                            landmark_sequence = None

                    if landmark_sequence is None:
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "learning_ack",
                                "status": "failed",
                                "reason": "landmark_extraction_failed",
                            }))
                        except Exception as _e:
                            print(f"[WS][WARN] Failed to send learning failure ack: {_e}")
                    else:
                        asyncio.create_task(
                            schedule_learning_save(landmark_sequence=landmark_sequence, session_id=session_id, meta=session_meta)
                        )
                        try:
                            await websocket.send_text(json.dumps({"type": "learning_ack", "status": "accepted"}))
                        except Exception as _e:
                            print(f"[WS][WARN] Failed to send learning accepted ack: {_e}")

                    try:
                        collector.frames = [] # 프레임 비우기
                    except Exception:
                        pass

                elif mtype == "flush":
                    predictor = app_state.predictor
                    frames = getattr(collector, "frames", [])
                    session_meta = getattr(collector, "meta", {})
                    if extract_sequence_from_frames is None:
                        landmark_sequence = None
                    else:
                        try:
                            # --- ⬇️ 여기 로그 추가 ⬇️ --- (이번 추가 로그 10)
                            print(f"[WS HANDLER {session_id}] Calling extract_sequence_from_frames for flush...")
                            # --- ⬆️ 로그 추가 완료 ⬆️ ---
                            landmark_sequence = extract_sequence_from_frames(frames, target_len=None, skip_missing=False)
                            # --- ⬇️ 여기 로그 추가 ⬇️ --- (이번 추가 로그 11)
                            print(f"[WS HANDLER {session_id}] extract_sequence_from_frames finished. Result type: {type(landmark_sequence)}")
                            # --- ⬆️ 로그 추가 완료 ⬆️ ---
                        except Exception as e:
                            print(f"[WS][ERROR] extract_sequence_from_frames failed: {e}") # 오류 로그 레벨 변경
                            traceback.print_exc() # 상세 트레이스백 추가
                            landmark_sequence = None

                    # --- ⬇️ 여기 로그 추가 ⬇️ --- (이번 추가 로그 12)
                    print(f"[WS HANDLER {session_id}] Calling run_inference...")
                    # --- ⬆️ 로그 추가 완료 ⬆️ ---
                    result = run_inference(predictor, landmark_sequence)
                    # --- ⬇️ 여기 로그 추가 ⬇️ --- (이번 추가 로그 13)
                    print(f"[WS HANDLER {session_id}] run_inference finished. Result: {result.get('predicted')}")
                    # --- ⬆️ 로그 추가 완료 ⬆️ ---

                    try:
                        result["frames_used"] = len(frames)
                    except Exception:
                        pass
                    asyncio.create_task(
                        schedule_quiz_save(landmark_sequence=landmark_sequence, inference_result=result, session_id=session_id, meta=session_meta)
                    )
                    await websocket.send_text(json.dumps({"type": "inference_result", "result": result}))

                    try:
                        collector.frames = [] # 프레임 비우기
                    except Exception:
                        pass
                else:
                    await websocket.send_text(json.dumps({"type": "noop"}))

    except WebSocketDisconnect:
        app_state.collectors.pop(session_id, None)
        return
    except Exception as e:
        print(f"[WS][ERROR] Unexpected error in websocket handler for session {session_id}: {e}")
        traceback.print_exc() # 상세 트레이스백 추가
        try:
            # 오류 발생 시 클라이언트에게 알림 시도 (선택 사항)
            await websocket.send_text(json.dumps({"type": "error", "message": "Internal server error"}))
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass # 이미 닫혔거나 보낼 수 없는 상태면 무시
        app_state.collectors.pop(session_id, None)
        return