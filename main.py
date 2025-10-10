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
TARGET_FRAME_COUNT = int(os.getenv("SIGN_SEQUENCE_TARGET_FRAMES", "300")) # 모델 입력 크기, 수집과 무관
COLLECTION_DURATION_SECONDS = float(os.getenv("SIGN_SEQUENCE_COLLECTION_SECONDS", "5.0"))
MAX_FRAMES_TO_COLLECT = 300  # 메모리 보호를 위한 안전장치

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
            # --- 데이터 타입 일치 오류 수정 ---
            # predictor.predict에 전달하기 직전에 데이터 타입을 float32로 명시적으로 변환합니다.
            landmark_sequence_float32 = landmark_sequence.astype(np.float32)
            predicted_label, score = predictor.predict(landmark_sequence_float32)

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
    frames: List[bytes] = field(default_factory=list)
    start_ts: Optional[float] = None
    processed: bool = False

    def start_collection(self):
        """수집 타이머를 시작합니다."""
        if self.start_ts is None:
            self.start_ts = time.time()

    def add_frame(self, data: bytes):
        """
        수집 기간 내에 있고 최대 프레임 수를 초과하지 않은 경우에만 프레임을 추가합니다.
        """
        # 추론이 시작되었다면 더 이상 프레임을 추가하지 않습니다.
        if self.processed:
            return

        if self.start_ts is not None and not self.is_full():
            # 메모리 폭주를 방지하기 위한 안전장치
            if len(self.frames) < MAX_FRAMES_TO_COLLECT:
                self.frames.append(data)

    def is_full(self) -> bool:
        """수집 시간이 다 되었는지 확인합니다."""
        if self.start_ts is None:
            return False

        time_elapsed = time.time() - self.start_ts
        # 시간 조건만으로 수집 종료를 결정합니다.
        return time_elapsed >= COLLECTION_DURATION_SECONDS

    def build_timings(self) -> Dict[str, Any]:
        end_ts = time.time()
        return {
            "frame_count": len(self.frames),
            "receive_first_ts": self.start_ts,
            "receive_last_ts": end_ts,
            "receive_duration_ms": None if self.start_ts is None else int(
                (end_ts - self.start_ts) * 1000
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
        collector = SequenceCollector()
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
