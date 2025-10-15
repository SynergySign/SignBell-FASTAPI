"""
SignSense Test Server
- FastAPI 기반 WebSocket 시그널링 + WebRTC DataChannel 프레임 수신 프로토타입
- 모델 로딩 및 프레임 시퀀스 수집 후 추론 결과 반환
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse

# --- Core Dependencies ---
# 일부 라이브러리는 개발환경에 설치되어 있지 않을 수 있으므로 안전하게 임포트합니다.
HEAVY_IMPORTS_AVAILABLE = True
try:
    import torch
    import numpy as np
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate

    from processing.predictor import Predictor, get_predictor, CNN_BiLSTM_Attention, PositionalEncoding
    from processing.landmark_extractor import extract_sequence_from_frames, FRAME_FEATURE_DIM
except Exception as _e:
    # 의존성이 없는 환경에서도 기본 REST 엔드포인트를 테스트할 수 있도록 예외를 무시하고
    # 런타임에서 필요한 경우 명확한 에러를 발생시키도록 처리합니다.
    print(f"[WARN] Optional heavy imports failed: {_e}")
    HEAVY_IMPORTS_AVAILABLE = False

    torch = None
    np = None
    # aiortc 관련 객체는 None으로 대체
    RTCPeerConnection = None
    RTCSessionDescription = None
    RTCIceCandidate = None

    # Predictor 등은 None / 더미로 대체 (get_predictor은 None을 반환하도록 설정)
    Predictor = None

    def get_predictor():
        return None

    CNN_BiLSTM_Attention = None
    PositionalEncoding = None

    def extract_sequence_from_frames(frames, target_len=None, skip_missing=False):
        # 더미 구현: 실제 추론을 위해서는 mediapipe 등 의존성이 필요합니다.
        return None

    FRAME_FEATURE_DIM = 0


# --- 상수 정의 ---
BASE_DIR = Path(__file__).resolve().parent
TARGET_FRAME_COUNT = int(os.getenv("SIGN_SEQUENCE_TARGET_FRAMES", "300")) # 모델 입력 크기, 수집과 무관
COLLECTION_DURATION_SECONDS = float(os.getenv("SIGN_SEQUENCE_COLLECTION_SECONDS", "5.0"))
MAX_FRAMES_TO_COLLECT = 300  # 메모리 보호를 위한 안전장치

app = FastAPI(title="SignSense Inference Server", version="0.2.0")


# ----------------------------- Model & Inference Logic -----------------------------

# Use the shared inference pipeline implementation (separated module)
from inference_pipeline import run_inference, SequenceCollector, schedule_quiz_save


# ----------------------------- Global State -----------------------------

class AppState:
    def __init__(self):
        # Predictor는 heavy deps가 없으면 None이 됩니다.
        # --- 모델 로드 오류 해결 ---
        # torch.load가 unpickle 시 특정 모듈명 아래에서 클래스를 찾는 경우가 있어,
        # 필요한 클래스들을 가능한 모듈 이름에 미리 주입한 뒤 모델을 로드합니다.
        import types

        def _inject_class_to_module(mod_name: str, cls_name: str, cls_obj):
            if cls_obj is None:
                return
            mod = sys.modules.get(mod_name)
            if mod is None:
                # 가상 모듈을 만들어 sys.modules에 등록합니다.
                mod = types.ModuleType(mod_name)
                sys.modules[mod_name] = mod
            setattr(mod, cls_name, cls_obj)

        candidate_module_names = [
            '__main__',
            'uvicorn.__main__',
            __name__,
            'main',
        ]

        for mname in candidate_module_names:
            _inject_class_to_module(mname, 'CNN_BiLSTM_Attention', CNN_BiLSTM_Attention)
            _inject_class_to_module(mname, 'PositionalEncoding', PositionalEncoding)

        # 위 주입이 완료된 이후에 predictor를 로드하도록 합니다.
        self.predictor: Optional["Predictor"] = get_predictor() if HEAVY_IMPORTS_AVAILABLE else None

        self.active_peers: Dict[str, "RTCPeerConnection"] = {}
        self.collectors: Dict[str, SequenceCollector] = {}

    def new_collector(self, session_id: str) -> SequenceCollector:
        collector = SequenceCollector()
        self.collectors[session_id] = collector
        return collector


app.state.ss = AppState()


# ----------------------------- Startup Event -----------------------------

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실제 추론 모델(Predictor)을 로드하려 시도합니다.
    운영환경에서 aiortc/mediapipe/torch 등이 설치되어 있다면 `processing.predictor.get_predictor()`를 호출해
    Predictor 인스턴스를 초기화합니다. 실패하면 에러를 로깅하고 predictor는 None으로 남겨둡니다.
    """
    print("[STARTUP] Attempting to load Predictor...")
    # 이미 로드되어 있으면 재사용
    if getattr(app.state.ss, 'predictor', None):
        print("[STARTUP] Predictor already initialized.")
        return

    try:
        # 동적으로 Import 시도하여 실제 운영환경의 종속성을 사용할 수 있게 함
        from processing.predictor import get_predictor as _get_predictor
        predictor_instance = _get_predictor()
        app.state.ss.predictor = predictor_instance
        print("[STARTUP] Predictor successfully loaded.")
    except Exception as e:
        # 자세한 예외 정보 로그
        print(f"[STARTUP][ERROR] Failed to load Predictor dynamically: {e}")
        traceback.print_exc()
        app.state.ss.predictor = None
        print("[STARTUP] Running without Predictor. Install required packages and restart the server for real inference.")


# ----------------------------- REST Endpoints -----------------------------

@app.get("/")
async def root():
    return {"status": "ok", "message": "SignSense Inference Server is running."}


@app.get("/health")
async def health():
    # predictor가 성공적으로 로드되었는지 확인
    is_healthy = app.state.ss.predictor is not None
    return {"ok": is_healthy}


@app.get("/model/status")
async def model_status():
    predictor = app.state.ss.predictor
    return {
        "predictor_loaded": predictor is not None,
        "model_path": str(predictor.model_path) if predictor else "N/A",
        "device": str(predictor.device) if predictor else "N/A",
    }


@app.get("/config")
async def get_config():
    return {
        "target_frame_count": TARGET_FRAME_COUNT,
        "collection_duration_seconds": COLLECTION_DURATION_SECONDS,
        "max_frames_to_collect": MAX_FRAMES_TO_COLLECT,
    }


@app.get("/client", response_class=HTMLResponse)
async def serve_client_test_page():
    """테스트용 HTML 클라이언트 페이지를 서빙합니다."""
    client_path = BASE_DIR / "client_test.html"
    if not client_path.is_file():
        return HTMLResponse(
            status_code=404,
            content=f"<h1>404 Not Found</h1><p>client_test.html not found at {client_path}</p>"
        )
    return HTMLResponse(content=client_path.read_text(encoding="utf-8"), status_code=200)


# === 시뮬레이터 엔드포인트: 스모크 테스트용 ===
@app.post("/simulate/predict")
async def simulate_predict():
    """더미 프레임을 사용하여 run_inference 흐름을 검증하는 간단한 엔드포인트.
    실제 프레임은 브라우저 DataChannel을 통해 전달되므로, 여기서는 고정 길이의 바이트 블록을 전달합니다.
    """
    predictor = app.state.ss.predictor
    # 더미 프레임(바이트)을 생성합니다. 실제 환경에서는 JPEG 바이트 배열이 들어옵니다.
    dummy_frames = [b"\x00"] * 10
    result = run_inference(predictor, dummy_frames)
    return result


@app.post("/simulate/create-collector")
async def simulate_create_collector(request: Request):
    """Test helper: create an empty SequenceCollector for given session_id.
    Body JSON: {"session_id": str}
    This endpoint is only intended for local testing (smoke_test).
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    # Create a collector (empty) for session
    collector = app.state.ss.new_collector(session_id)
    return {"ok": True, "session_id": session_id, "collector_frames": len(collector.frames)}


# ----------------------------- Internal storage endpoints -----------------------------

def _check_internal_access(request: Request):
    """Optional internal access guard.
    If ENV INTERNAL_API_TOKEN is set, require header 'x-internal-token' to match.
    Otherwise allow requests from localhost addresses.
    """
    token = os.getenv("INTERNAL_API_TOKEN")
    if token:
        hdr = request.headers.get("x-internal-token")
        if hdr != token:
            raise HTTPException(status_code=403, detail="Forbidden: invalid internal token")
    else:
        # If no token configured, allow only local requests as a reasonable default.
        client = request.client
        if client is None or client.host not in ("127.0.0.1", "::1", "localhost"):
            # Not strictly secure in all deployments, but suitable for local/intra-host calls.
            raise HTTPException(status_code=403, detail="Forbidden: internal endpoint only")


@app.post("/internal/save-learning")
async def internal_save_learning(request: Request):
    """Internal endpoint to save collected frames for a session as 'learning' data.
    Body JSON: { "session_id": str, "meta": { ... } }
    This schedules an async save and returns immediately.
    """
    _check_internal_access(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    session_id = payload.get("session_id")
    meta = payload.get("meta", {})

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    collector = app.state.ss.collectors.get(session_id)
    if collector is None:
        raise HTTPException(status_code=404, detail="collector not found for session_id")

    frames = collector.frames

    try:
        # schedule async save_learning
        from storage.s3_db_saver import save_learning
        import asyncio
        asyncio.create_task(save_learning(frames=frames, meta={**meta, "session_id": session_id}))
        return {"ok": True, "scheduled": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule save_learning: {e}")


@app.post("/internal/save-quiz")
async def internal_save_quiz(request: Request):
    """Internal endpoint to save quiz inference results. Body JSON: { "session_id": str, "inference_result": {...} }
    The endpoint will look up frames for session_id (if available) and schedule async save_quiz.
    """
    _check_internal_access(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    session_id = payload.get("session_id")
    inference_result = payload.get("inference_result")

    if not session_id or inference_result is None:
        raise HTTPException(status_code=400, detail="session_id and inference_result are required")

    collector = app.state.ss.collectors.get(session_id)
    frames = collector.frames if collector is not None else []

    try:
        from storage.s3_db_saver import save_quiz
        import asyncio
        asyncio.create_task(save_quiz(frames=frames, inference_result=inference_result, session_id=session_id))
        return {"ok": True, "scheduled": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule save_quiz: {e}")

# ----------------------------- WebSocket Signaling -----------------------------

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    peer_id = f"peer_{id(websocket)}"
    print(f"[WS] Connected: {peer_id}")
    pc = RTCPeerConnection()
    app.state.ss.active_peers[peer_id] = pc

    async def safe_send_json(payload: Dict[str, Any]):
        try:
            await websocket.send_text(json.dumps(payload))
        except WebSocketDisconnect:
            print(f"[WS][SEND][WARN] Peer {peer_id} disconnected before sending.")
        except Exception as e:
            print(f"[WS][SEND][ERR] for {peer_id}: {e}")

    # --- 데이터 채널 콜백 정의 ---
    @pc.on("datachannel")
    def on_datachannel(channel):
        print(f"[{session_id}] DataChannel '{channel.label}' created")
        # collector를 직접 생성하지 않고, 래퍼 딕셔너리를 사용해 관리
        collector_wrapper = {'collector': app.state.ss.new_collector(session_id)}

        async def run_inference_once():
            """추론을 한 번만 실행하는 헬퍼 함수"""
            collector = collector_wrapper['collector']
            if collector.processed:
                return
            collector.processed = True

            print(f"[{session_id}] Flush signal received. Running inference...")

            result = run_inference(app.state.ss.predictor, collector.frames)
            result["timings"] = collector.build_timings()

            await safe_send_json({
                "type": "inference_result",
                "data": result
            })

            # 비동기 백그라운드로 퀴즈 저장 스케줄 (외부 호출에 영향을 주지 않도록 비동기 처리)
            try:
                import asyncio
                asyncio.create_task(schedule_quiz_save(frames=collector.frames, inference_result=result, session_id=session_id))
            except Exception as e:
                print(f"[WS][WARN] Failed to schedule quiz save: {e}")

        @channel.on("message")
        async def on_message(message):
            if isinstance(message, str):
                if message == "flush":
                    await run_inference_once()
                elif message == "reset":
                    print(f"[{session_id}] Resetting collector for new capture.")
                    # 새 collector로 교체
                    collector_wrapper['collector'] = app.state.ss.new_collector(session_id)
                return

            collector = collector_wrapper['collector']
            # 추론이 시작되기 전까지(processed=False) 프레임을 계속 수집합니다.
            if not collector.processed:
                # 첫 프레임 수신 시 로그를 남깁니다.
                if collector.start_ts is None:
                    collector.start_collection()
                    print(f"[{session_id}] First frame received. Collecting frames...")
                collector.add_frame(message)

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            action = msg.get("action")

            if action == "offer":
                sdp = msg.get("sdp")
                offer = RTCSessionDescription(sdp=sdp, type=msg.get("type", "offer"))

                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                await safe_send_json({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp,
                })

            elif action == "ice-candidate":
                candidate_info = msg.get("candidate")
                if candidate_info:
                    candidate = RTCIceCandidate(
                        sdpMid=candidate_info.get("sdpMid"),
                        sdpMLineIndex=candidate_info.get("sdpMLineIndex"),
                        candidate=candidate_info.get("candidate"),
                    )
                    await pc.addIceCandidate(candidate)

    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {peer_id}")
    except Exception as e:
        print(f"[WS][FATAL] Error in WebSocket handler for {peer_id}: {e}")
        traceback.print_exc()
    finally:
        if peer_id in app.state.ss.active_peers:
            pc_to_close = app.state.ss.active_peers.pop(peer_id)
            await pc_to_close.close()
        if session_id in app.state.ss.collectors:
            del app.state.ss.collectors[session_id]
        print(f"[WS] Cleaned up resources for {peer_id} (session: {session_id})")


# ----------------------------- Diagnostic Endpoints -----------------------------

@app.post('/model/test-predict')
async def model_test_predict():
    """Diagnostic endpoint to verify Predictor.predict works with a random input.
    Returns {'predicted': label, 'score': float} when predictor is available, otherwise 503.
    """
    predictor = app.state.ss.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    # require numpy available
    try:
        import numpy as _np
    except Exception:
        raise HTTPException(status_code=503, detail="numpy not available on server")

    # build dummy sequence matching expected feature dim
    try:
        feat_dim = FRAME_FEATURE_DIM if 'FRAME_FEATURE_DIM' in globals() else getattr(predictor, 'input_size', None)
        if not feat_dim:
            raise RuntimeError('Unknown feature dim')
        seq_len = min(60, TARGET_FRAME_COUNT)
        dummy = _np.random.rand(seq_len, int(feat_dim)).astype(_np.float32)
        label, score = predictor.predict(dummy)
        return { 'predicted': label, 'score': float(score) }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict failed: {e}")


if __name__ == "__main__":
    import uvicorn

    ssl_cert_path = BASE_DIR / "certs" / "cert.pem"
    ssl_key_path = BASE_DIR / "certs" / "key.pem"

    if not ssl_cert_path.is_file() or not ssl_key_path.is_file():
        print("[WARN] SSL certificates not found. Running without HTTPS.")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("[INFO] Starting server with HTTPS.")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            ssl_keyfile=str(ssl_key_path),
            ssl_certfile=str(ssl_cert_path),
        )
