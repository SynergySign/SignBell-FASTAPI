"""
SignSense Test Server
- FastAPI 기반 WebSocket 시그널링 + WebRTC DataChannel 프레임 수신 프로토타입
- 모델 로딩 (stub) 및 프레임 시퀀스 수집 후 더미 추론 결과 반환

실제 운영 전 해야 할 TODO
1. Torch / ONNX 실제 모델 로드 및 전처리/후처리 코드 연결
2. MediaPipe landmark 추출 파이프라인 통합
3. Chunking 전송 규약(헤더/순번/종료 플래그) 확정 후 재조립 안정화
4. ICE 후보(candidate) 완전한 교환 로직 + TURN/STUN 설정
5. 에러 / 타임아웃 / 메모리 사용량 모니터링 고도화
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse

# --- Predictor 모듈 임포트 ---
try:
    from processing.predictor import Predictor, get_predictor, CNN_BiLSTM_Attention, PositionalEncoding
except ImportError as e:
    print(f"[WARN] Predictor import failed: {e}. Inference will not be available.")
    Predictor = None
    get_predictor = None
    CNN_BiLSTM_Attention = None
    PositionalEncoding = None

# 선택적 landmark 추출 모듈 (학습 coordinate_extractor 재현)
SIGN_EXTRACT_LANDMARKS = os.getenv("SIGN_EXTRACT_LANDMARKS", "0") == "1"
try:
    if SIGN_EXTRACT_LANDMARKS:
        from processing.landmark_extractor import extract_sequence_from_frames, FRAME_FEATURE_DIM  # type: ignore
    else:
        extract_sequence_from_frames = None  # type: ignore
        FRAME_FEATURE_DIM = None  # type: ignore
except Exception:  # noqa
    extract_sequence_from_frames = None  # type: ignore
    FRAME_FEATURE_DIM = None  # type: ignore

# 외부 라이브러리 (선택적) import 시 실패해도 서버가 기동되도록 예외 처리
try:
    import torch  # type: ignore
except Exception:  # noqa
    torch = None  # fallback

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription  # type: ignore
    from aiortc import RTCIceCandidate  # type: ignore
except Exception:  # noqa
    RTCPeerConnection = None  # type: ignore
    RTCSessionDescription = None  # type: ignore
    RTCIceCandidate = None  # type: ignore

# --- 상수 정의 ---
BASE_DIR = Path(__file__).resolve().parent
TARGET_FRAME_COUNT = int(os.getenv("SIGN_SEQUENCE_TARGET_FRAMES", "300"))  # predictor.py와 일치시킴

app = FastAPI(title="SignSense Test Server", version="0.1.0")


# ----------------------------- Model & Inference Logic -----------------------------

def run_inference(predictor: Predictor, frames: List[bytes]) -> Dict[str, Any]:
    """
    수집된 프레임에 대해 랜드마크 추출 및 모델 추론을 수행합니다.
    :param predictor: 초기화된 Predictor 객체
    :param frames: WebRTC로 수신된 프레임 데이터 리스트
    :return: 추론 결과 딕셔너리
    """
    start = time.time()

    landmarks_info: Dict[str, Any] = {}
    predicted_label = "오류: 랜드마크 추출 실패"
    score = 0.0

    if not predictor:
        predicted_label = "오류: Predictor가 로드되지 않았습니다."
    elif SIGN_EXTRACT_LANDMARKS and extract_sequence_from_frames:
        try:
            # 1. 랜드마크 추출
            landmark_sequence = extract_sequence_from_frames(frames, target_len=TARGET_FRAME_COUNT, skip_missing=False)

            if landmark_sequence is not None and len(landmark_sequence) > 0:
                landmarks_info = {
                    "enabled": True,
                    "seq_shape": list(np.array(landmark_sequence).shape),
                    "feature_dim": FRAME_FEATURE_DIM,
                }
                # 2. 모델 추론
                predicted_label = predictor.predict(landmark_sequence)
                # TODO: 실제 score 값도 predictor에서 반환하도록 수정 필요
                score = 0.95  # 임시 score
            else:
                landmarks_info = {"enabled": True, "error": "landmark_sequence is None or empty"}
                predicted_label = "오류: 랜드마크를 감지하지 못했습니다."

        except Exception as e:
            landmarks_info = {"enabled": True, "error": str(e)}
            predicted_label = f"오류: 추론 중 예외 발생 ({e})"
    else:
        landmarks_info = {"enabled": False}
        predicted_label = "오류: 랜드마크 추출 기능이 비활성화되었습니다."

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
    start_send_ts: Optional[float] = None  # 클라이언트에서 첫 전송 시작(클라 기준) -> 아직은 서버 추정
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
        self.predictor: Optional[Predictor] = None
        self.active_peers: Dict[str, RTCPeerConnection] = {}
        self.collectors: Dict[str, SequenceCollector] = {}
        self.lock = asyncio.Lock()

    def new_collector(self, session_id: str) -> SequenceCollector:
        collector = SequenceCollector(target_count=TARGET_FRAME_COUNT)
        self.collectors[session_id] = collector
        return collector


app.state.ss = AppState()  # type: ignore[attr-defined]


# ----------------------------- Startup Event -----------------------------

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실제 추론 모델(Predictor)을 로드합니다."""
    if get_predictor and CNN_BiLSTM_Attention and PositionalEncoding:
        print("[STARTUP] Loading predictor...")
        try:
            # --- 모델 로드 오류 해결 ---
            # torch.load가 __main__에서 클래스를 찾으므로, 수동으로 매핑해줍니다.
            sys.modules['__main__'].CNN_BiLSTM_Attention = CNN_BiLSTM_Attention
            sys.modules['__main__'].PositionalEncoding = PositionalEncoding

            # get_predictor 함수를 호출하여 predictor 인스턴스를 로드하고 상태에 저장
            app.state.ss.predictor = get_predictor()
            print("[STARTUP] Predictor ready.")
        except Exception as e:
            print(f"[STARTUP][ERROR] Failed to load predictor: {e}")
            app.state.ss.predictor = None
    else:
        print("[STARTUP][WARN] Predictor module not available. Running without inference.")


# ----------------------------- REST Endpoints -----------------------------

@app.get("/")
async def root():
    return {"status": "ok", "frames_target": TARGET_FRAME_COUNT}


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/model/status")
async def model_status():
    predictor = app.state.ss.predictor
    return {
        "predictor_loaded": predictor is not None,
        "model_path": str(predictor.model_path) if predictor else None,
        "device": str(predictor.device) if predictor else None,
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

    with open(client_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


# ----------------------------- WebSocket Signaling -----------------------------

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    peer_id = f"peer_{id(websocket)}"
    print(f"[WS] Connected: {peer_id}")
    pc: Optional[RTCPeerConnection] = None
    collector: Optional[SequenceCollector] = None

    async def safe_send_json(payload: Dict[str, Any]):
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception as e:  # noqa
            print(f"[WS][SEND][ERR] {e}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await safe_send_json({"type": "error", "error": "invalid_json"})
                continue

            action = msg.get("action")

            # --- Offer 처리 ---
            if action == "offer":
                if RTCPeerConnection is None:
                    await safe_send_json({"type": "error", "error": "aiortc_not_installed"})
                    continue
                if pc is not None:
                    await safe_send_json({"type": "error", "error": "peer_already_initialized"})
                    continue

                sdp = msg.get("sdp")
                offer_type = msg.get("type", "offer")
                pc = RTCPeerConnection()
                app.state.ss.active_peers[peer_id] = pc
                collector = app.state.ss.new_collector(peer_id)

                # (aiortc 기본 예제: trickle ICE 생략. icecandidate 이벤트 데코레이터는 존재하지 않으므로
                #  ICE 후보 전송이 필요하면 iceGatheringState 변화를 감시하고 localDescription 재전송 방식 적용 예정)

                # --- 데이터 채널 콜백 정의 ---
                @pc.on("datachannel")
                def on_datachannel(channel):
                    print(f"[{session_id}] DataChannel '{channel.label}' created")
                    collector = app.state.ss.new_collector(session_id)

                    @channel.on("message")
                    async def on_message(message):
                        # 프레임 데이터 수신 및 수집
                        collector.add_frame(message)

                        # 목표 프레임 수에 도달하면 추론 실행
                        if collector.is_ready():
                            collector.processed = True
                            print(f"[{session_id}] Sequence ready. Running inference...")

                            # 실제 추론 함수 호출
                            result = run_inference(app.state.ss.predictor, collector.frames)

                            # 타이밍 정보 추가
                            result["timings"] = collector.build_timings()

                            # 추론 결과를 클라이언트에 전송
                            await safe_send_json({
                                "type": "inference_result",
                                "data": result
                            })

                # Offer SDP 적용 및 Answer 생성
                await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=offer_type))
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                # 생성된 Answer를 클라이언트에 전송
                await safe_send_json({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp,
                })

            # --- ICE Candidate 처리 ---
            elif action == "ice-candidate":
                if pc is None:
                    await safe_send_json({"type": "error", "error": "peer_not_initialized"})
                    continue

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
    finally:
        if pc and peer_id in app.state.ss.active_peers:
            await pc.close()
            del app.state.ss.active_peers[peer_id]
        if peer_id in app.state.ss.collectors:
            del app.state.ss.collectors[peer_id]
        print(f"[WS] Cleaned up resources for {peer_id}")


if __name__ == "__main__":
    import uvicorn

    # SSL/TLS 설정
    # certs/cert.pem 및 certs/key.pem 파일이 필요합니다.
    ssl_cert_path = BASE_DIR / "certs" / "cert.pem"
    ssl_key_path = BASE_DIR / "certs" / "key.pem"

    if not ssl_cert_path.is_file() or not ssl_key_path.is_file():
        print("[WARN] SSL certificates not found. Running without HTTPS.")
        print(f"         - Searched for: {ssl_cert_path}")
        print(f"         - Searched for: {ssl_key_path}")
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
