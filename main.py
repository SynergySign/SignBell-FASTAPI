"""
SignSense Test Server
- FastAPI 기반 WebSocket 시그널링 + WebRTC DataChannel 프레임 수신 프로토타입
- 모델 로딩 및 프레임 시퀀스 수집 후 추론 결과 반환
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# --- Core Dependencies ---
# 이 라이브러리들은 서버의 핵심 기능이므로, 없는 경우 즉시 에러를 발생시키는 것이 타당합니다.
import torch
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate

from processing.predictor import Predictor, get_predictor, CNN_BiLSTM_Attention, PositionalEncoding
from processing.landmark_extractor import extract_sequence_from_frames, FRAME_FEATURE_DIM


# --- 상수 정의 ---
BASE_DIR = Path(__file__).resolve().parent
TARGET_FRAME_COUNT = int(os.getenv("SIGN_SEQUENCE_TARGET_FRAMES", "300"))

app = FastAPI(title="SignSense Inference Server", version="0.2.0")


# ----------------------------- Model & Inference Logic -----------------------------

def run_inference(predictor: Predictor, frames: List[bytes]) -> Dict[str, Any]:
    """
    수집된 프레임에 대해 랜드마크 추출 및 모델 추론을 수행합니다.
    """
    start = time.time()
    landmarks_info: Dict[str, Any] = {}
    predicted_label = "오류: 추론 실패"
    score = 0.0

    try:
        # 1. 랜드마크 추출
        landmark_sequence = extract_sequence_from_frames(frames, target_len=TARGET_FRAME_COUNT, skip_missing=False)

        # 2. 추출된 데이터 유효성 검사
        if landmark_sequence is None or len(landmark_sequence) == 0:
            landmarks_info = {"enabled": True, "error": "landmark_sequence is None or empty"}
            predicted_label = "오류: 랜드마크를 감지하지 못했습니다."
        elif np.all(landmark_sequence == 0):
            landmarks_info = {"enabled": True, "error": "All landmarks are zero (No detection)"}
            predicted_label = "오류: 랜드마크를 감지하지 못했습니다. (데이터 없음)"
        else:
            # 3. 모델 추론
            landmarks_info = {
                "enabled": True,
                "seq_shape": list(landmark_sequence.shape),
                "feature_dim": FRAME_FEATURE_DIM,
            }
            predicted_label, score = predictor.predict(landmark_sequence)

    except Exception as e:
        print(f"[ERROR] Exception during inference: {e}")
        traceback.print_exc()
        error_message = str(e)
        landmarks_info = {"enabled": True, "error": error_message}
        predicted_label = f"오류: 추론 중 예외 발생 ({error_message})"

    end = time.time()
    return {
        "predicted": predicted_label,
        "score": score,
        "inference_start": start,
        "inference_end": end,
        "inference_ms": int((end - start) * 1000),
        "frames_used": len(frames),
        "landmarks": landmarks_info,
    }


# ----------------------------- Sequence Collector -----------------------------

@dataclass
class SequenceCollector:
    target_count: int
    frames: List[bytes] = field(default_factory=list)
    first_receive_ts: Optional[float] = None
    last_receive_ts: Optional[float] = None
    processed: bool = False

    def add_frame(self, data: bytes):
        now = time.time()
        if self.first_receive_ts is None:
            self.first_receive_ts = now
        self.last_receive_ts = now
        self.frames.append(data)

    def is_ready(self) -> bool:
        return len(self.frames) >= self.target_count and not self.processed

    def build_timings(self) -> Dict[str, Any]:
        return {
            "frame_count": len(self.frames),
            "receive_first_ts": self.first_receive_ts,
            "receive_last_ts": self.last_receive_ts,
            "receive_duration_ms": None if (self.first_receive_ts is None or self.last_receive_ts is None) else int(
                (self.last_receive_ts - self.first_receive_ts) * 1000
            ),
        }


# ----------------------------- Global State -----------------------------

class AppState:
    def __init__(self):
        self.predictor: Predictor = get_predictor()
        # --- 모델 로드 오류 해결 ---
        # torch.load가 __main__에서 클래스를 찾으므로, 수동으로 매핑해줍니다.
        sys.modules['__main__'].CNN_BiLSTM_Attention = CNN_BiLSTM_Attention
        sys.modules['__main__'].PositionalEncoding = PositionalEncoding
        self.active_peers: Dict[str, RTCPeerConnection] = {}
        self.collectors: Dict[str, SequenceCollector] = {}

    def new_collector(self, session_id: str) -> SequenceCollector:
        collector = SequenceCollector(target_count=TARGET_FRAME_COUNT)
        self.collectors[session_id] = collector
        return collector


app.state.ss = AppState()


# ----------------------------- Startup Event -----------------------------

@app.on_event("startup")
async def startup_event():
    """��버 시작 시 실제 추론 모델(Predictor)을 로드합니다."""
    print("[STARTUP] Loading predictor...")
    try:
        # 상태 초기화 시 이미 로드되었으므로, 여기서는 확인만 합니다.
        if app.state.ss.predictor:
            print("[STARTUP] Predictor ready.")
        else:
            # 이 경우는 get_predictor() 실패 시 발생
            raise RuntimeError("Predictor could not be loaded.")
    except Exception as e:
        print(f"[STARTUP][FATAL] Failed to load predictor: {e}")
        # 필수 의존성이므로, 로드 실패 시 프로세스를 종료하거나 상태를 명확히 할 수 있습니다.
        # 여기서는 에러 로그를 남기고, health check 등에서 감지되도록 합니다.
        app.state.ss.predictor = None


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
    return {"target_frame_count": TARGET_FRAME_COUNT}

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
        collector = app.state.ss.new_collector(session_id)

        @channel.on("message")
        async def on_message(message):
            collector.add_frame(message)

            if collector.is_ready():
                collector.processed = True
                print(f"[{session_id}] Sequence ready. Running inference...")

                result = run_inference(app.state.ss.predictor, collector.frames)
                result["timings"] = collector.build_timings()

                await safe_send_json({
                    "type": "inference_result",
                    "data": result
                })

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
            await app.state.ss.active_peers[peer_id].close()
            del app.state.ss.active_peers[peer_id]
        if peer_id in app.state.ss.collectors:
            del app.state.ss.collectors[peer_id]
        print(f"[WS] Cleaned up resources for {peer_id}")


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
