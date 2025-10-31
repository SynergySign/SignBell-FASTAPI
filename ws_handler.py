"""WebSocket handler 모듈 (메모리 누수 수정됨)

이 모듈은 `main.py`에서 사용되는 WebSocket 연결 처리 로직을 포함합니다.
핸드셰이크, 토큰 검증, 텍스트/바이너리 수신 루프, collector 관리 등을 수행합니다.

[수정 사항]
- RealtimeLandmarkExtractor를 세션마다 한 번만 생성하고 재사용하여
  MediaPipe 모델 반복 생성으로 인한 메모리 누수를 해결합니다.
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

# --- ⬇️ 메모리 누수 수정 1: 클래스 직접 임포트 ⬇️ ---
try:
    # 기존 함수 대신, Extractor와 Builder 클래스를 직접 가져옵니다.
    from processing.landmark_extractor import RealtimeLandmarkExtractor, SequenceBuilder

    # 오래된 방식(extract_sequence_from_frames)은 비상시(fallback)에만 사용
    from processing.landmark_extractor import extract_sequence_from_frames as leaky_extract_sequence

    EXTRACTOR_AVAILABLE = True
except Exception:
    RealtimeLandmarkExtractor = None
    SequenceBuilder = None
    EXTRACTOR_AVAILABLE = False

    # (대체) extract_sequence_from_frames만 임포트 시도
    try:
        from processing.landmark_extractor import extract_sequence_from_frames as leaky_extract_sequence
    except Exception:
        leaky_extract_sequence = None
# --- ⬆️ 수정 완료 ⬆️ ---


async def websocket_handler(websocket: WebSocket, session_id: str, app_state: Any):
    # --- 핸드셰이크 및 토큰 검증 로직 시작 (변경 없음) ---
    try:
        # ... (핸드셰이크 로직 동일) ...
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

    # --- ⬇️ 메모리 누수 수정 2: Extractor 세션당 1회 생성 ⬇️ ---
    extractor: RealtimeLandmarkExtractor | None = None
    if RealtimeLandmarkExtractor is not None:
        try:
            # skip_missing=False: 학습 데이터와 동일하게 누락된 프레임도 0으로 채움
            extractor = RealtimeLandmarkExtractor(skip_missing=False)
            if not extractor.available():
                print(f"[WS HANDLER {session_id}] [ERROR] RealtimeLandmarkExtractor init failed!")
                extractor = None # 사용 불가능 처리
        except Exception as e:
            print(f"[WS HANDLER {session_id}] [ERROR] RealtimeLandmarkExtractor init error: {e}")
            traceback.print_exc()
            extractor = None
    # --- ⬆️ 수정 완료 ⬆️ ---

    print(f"[WS HANDLER {session_id}] Handler function started.")
    print(f"[WS HANDLER {session_id}] Reusable Extractor initialized: {extractor is not None}")

    try:
        print(f"[WS HANDLER {session_id}] Entering main receive loop...")
        while True:
            print(f"[WS HANDLER {session_id}] Waiting to receive data...")
            data = await websocket.receive()

            print(f"[WS HANDLER {session_id}] Received data type: {type(data)}")

            if "text" in data:
                print(f"[WS HANDLER {session_id}] Received TEXT content: {data.get('text')}")
            elif "bytes" in data:
                print(f"[WS HANDLER {session_id}] Received BYTES content length: {len(data.get('bytes', b''))}")

            if isinstance(data, dict) and data.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(code=data.get("code"))

            # 바이너리 프레임 수신 (변경 없음)
            if "bytes" in data and data.get("bytes") is not None:
                print(f"[WS HANDLER {session_id}] Handling BYTES data...")
                frame_bytes = data["bytes"]
                try:
                    collector.add_frame(frame_bytes)
                    print(f"[WS HANDLER {session_id}] Frame added. Total frames: {len(getattr(collector, 'frames', []))}")
                except Exception as e:
                    print(f"[WS][WARN] Failed to add frame: {e}")
                continue

            # 텍스트(JSON) 신호 처리
            if "text" in data and data.get("text") is not None:
                print(f"[WS HANDLER {session_id}] Handling TEXT data...")
                raw = data["text"]
                try:
                    msg = json.loads(raw)
                except Exception:
                    print(f"[WS][WARN] Failed to parse JSON: {raw}")
                    continue

                mtype = msg.get("type")

                # --- meta (변경 없음) ---
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

                # --- save_learning (수정됨) ---
                elif mtype == "save_learning":
                    frames = getattr(collector, "frames", [])
                    session_meta = getattr(collector, "meta", {})
                    landmark_sequence = None

                    # --- ⬇️ 메모리 누수 수정 3: 재사용 Extractor로 시퀀스 빌드 ⬇️ ---
                    if extractor is not None and SequenceBuilder is not None:
                        try:
                            print(f"[WS HANDLER {session_id}] Calling SequenceBuilder with *reused* extractor (for save_learning)...")
                            builder = SequenceBuilder(extractor)
                            for fb in frames:
                                builder.add_frame(fb) # extractor.extract()가 내부적으로 호출됨
                            landmark_sequence = builder.build(target_len=None)
                            print(f"[WS HANDLER {session_id}] Sequence built. Shape: {landmark_sequence.shape if landmark_sequence is not None else 'None'}")
                        except Exception as e:
                            print(f"[WS][ERROR] SequenceBuilder with *reused* extractor failed: {e}")
                            traceback.print_exc()
                            landmark_sequence = None
                    else:
                        # (대체) Extractor가 없는 경우, 기존의 메모리 누수 방식이라도 시도
                        print(f"[WS HANDLER {session_id}] [WARN] Falling back to *leaky* extract_sequence_from_frames (for save_learning)...")
                        if leaky_extract_sequence is not None:
                            try:
                                landmark_sequence = leaky_extract_sequence(frames, target_len=None, skip_missing=False)
                            except Exception as e:
                                print(f"[WS][ERROR] *Leaky* extract_sequence_from_frames failed: {e}")
                                traceback.print_exc()
                        else:
                            print(f"[WS][ERROR] No extractor available at all.")
                    # --- ⬆️ 수정 완료 ⬆️ ---

                    if landmark_sequence is None:
                        # ... (실패 전송 로직 동일) ...
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "learning_ack", "status": "failed", "reason": "landmark_extraction_failed",
                            }))
                        except Exception as _e:
                            print(f"[WS][WARN] Failed to send learning failure ack: {_e}")
                    else:
                        # ... (성공 및 저장 로직 동일) ...
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

                # --- flush (수정됨) ---
                elif mtype == "flush":
                    predictor = app_state.predictor
                    frames = getattr(collector, "frames", [])
                    session_meta = getattr(collector, "meta", {})
                    landmark_sequence = None

                    # --- ⬇️ 메모리 누수 수정 3: 재사용 Extractor로 시퀀스 빌드 ⬇️ ---
                    if extractor is not None and SequenceBuilder is not None:
                        try:
                            print(f"[WS HANDLER {session_id}] Calling SequenceBuilder with *reused* extractor (for flush)...")
                            builder = SequenceBuilder(extractor)
                            for fb in frames:
                                builder.add_frame(fb) # extractor.extract()가 내부적으로 호출됨
                            landmark_sequence = builder.build(target_len=None)
                            print(f"[WS HANDLER {session_id}] Sequence built. Shape: {landmark_sequence.shape if landmark_sequence is not None else 'None'}")
                        except Exception as e:
                            print(f"[WS][ERROR] SequenceBuilder with *reused* extractor failed: {e}")
                            traceback.print_exc()
                            landmark_sequence = None
                    else:
                        # (대체) Extractor가 없는 경우, 기존의 메모리 누수 방식이라도 시도
                        print(f"[WS HANDLER {session_id}] [WARN] Falling back to *leaky* extract_sequence_from_frames (for flush)...")
                        if leaky_extract_sequence is not None:
                            try:
                                landmark_sequence = leaky_extract_sequence(frames, target_len=None, skip_missing=False)
                            except Exception as e:
                                print(f"[WS][ERROR] *Leaky* extract_sequence_from_frames failed: {e}")
                                traceback.print_exc()
                        else:
                            print(f"[WS][ERROR] No extractor available at all.")
                    # --- ⬆️ 수정 완료 ⬆️ ---

                    print(f"[WS HANDLER {session_id}] Calling run_inference...")
                    result = run_inference(predictor, landmark_sequence)
                    print(f"[WS HANDLER {session_id}] run_inference finished. Result: {result.get('predicted')}")

                    try:
                        result["frames_used"] = len(frames)
                    except Exception:
                        pass

                    # ... (저장 및 전송 로직 동일) ...
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
        print(f"[WS DISCONNECT {session_id}] Connection closed.")
        # --- ⬇️ 메모리 누수 수정 4: 종료 시 Extractor 닫기 ⬇️ ---
        if extractor:
            try:
                extractor.close()
                print(f"[WS DISCONNECT {session_id}] Reusable extractor closed.")
            except Exception as e_close:
                print(f"[WS DISCONNECT {session_id}] Error closing extractor: {e_close}")
        # --- ⬆️ 수정 완료 ⬆️ ---
        app_state.collectors.pop(session_id, None)
        return

    except Exception as e:
        print(f"[WS][ERROR] Unexpected error in websocket handler for session {session_id}: {e}")
        traceback.print_exc() # 상세 트레이스백 추가

        # --- ⬇️ 메모리 누수 수정 4: 종료 시 Extractor 닫기 ⬇️ ---
        if extractor:
            try:
                extractor.close()
                print(f"[WS ERROR {session_id}] Reusable extractor closed on error.")
            except Exception as e_close:
                print(f"[WS ERROR {session_id}] Error closing extractor: {e_close}")
        # --- ⬆️ 수정 완료 ⬆️ ---

        try:
            await websocket.send_text(json.dumps({"type": "error", "message": "Internal server error"}))
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass # 이미 닫혔거나 보낼 수 없는 상태면 무시

        app_state.collectors.pop(session_id, None)
        return