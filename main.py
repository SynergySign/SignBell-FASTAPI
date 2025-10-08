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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse

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

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "cnn_bilstm_attention_model.pth"

TARGET_FRAME_COUNT = int(os.getenv("SIGN_SEQUENCE_TARGET_FRAMES", "60"))  # 테스트용 (실서비스: 9~10초 구간 프레임 수)
INFERENCE_LABELS = ["안녕하세요", "감사합니다", "학교", "사람", "테스트"]  # Dummy labels

app = FastAPI(title="SignSense Test Server", version="0.1.0")


# ----------------------------- Model Loader (Stub) -----------------------------

def load_model() -> Any:
    """실제 추론 모델 로드 (현재는 더미). 실패해도 서버는 기동.
    Returns: 모델 객체 또는 더미 식별자
    """
    if MODEL_PATH.exists() and torch is not None:
        try:
            model = torch.load(MODEL_PATH, map_location="cpu")
            if hasattr(model, "eval"):
                model.eval()
            print(f"[MODEL] Loaded torch model: {MODEL_PATH.name}")
            return model
        except Exception as e:  # noqa
            print(f"[MODEL][WARN] Torch model load failed: {e}. Fallback to dummy model.")
    else:
        if not MODEL_PATH.exists():
            print(f"[MODEL][WARN] Model file not found: {MODEL_PATH}")
        if torch is None:
            print("[MODEL][WARN] torch not available in environment.")
    return "DUMMY_MODEL"


def dummy_infer(frames: List[bytes]) -> Dict[str, Any]:
    """수신된 프레임 더미 추론 -> 랜덤 라벨 선택
    실제 구현 시: MediaPipe -> Keypoint 추출 -> 시퀀스 전처리 -> 모델 추론 -> 후처리
    """
    import random

    start = time.time()

    landmarks_info: Dict[str, Any] = {}
    if SIGN_EXTRACT_LANDMARKS and extract_sequence_from_frames is not None:
        try:
            seq = extract_sequence_from_frames(frames, target_len=TARGET_FRAME_COUNT, skip_missing=False)  # type: ignore
            if seq is not None:
                landmarks_info = {
                    "enabled": True,
                    "seq_shape": list(seq.shape),
                    "feature_dim": FRAME_FEATURE_DIM,
                }
            else:
                landmarks_info = {"enabled": True, "error": "seq_none"}
        except Exception as e:  # noqa
            landmarks_info = {"enabled": True, "error": str(e)}
    else:
        landmarks_info = {"enabled": False}

    time.sleep(0.05)  # 모의 지연
    label = random.choice(INFERENCE_LABELS)
    score = round(random.uniform(0.75, 0.99), 3)
    end = time.time()
    return {
        "predicted": label,
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
        self.model = None
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
    print("[STARTUP] Loading model...")
    app.state.ss.model = load_model()
    print("[STARTUP] Model ready.")


# ----------------------------- REST Endpoints -----------------------------

@app.get("/")
async def root():
    return {"status": "ok", "frames_target": TARGET_FRAME_COUNT}


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/model/status")
async def model_status():
    model_obj = app.state.ss.model
    return {"model": str(type(model_obj)), "is_dummy": model_obj == "DUMMY_MODEL"}


@app.get("/config")
async def config():
    return {"target_frame_count": TARGET_FRAME_COUNT, "labels": INFERENCE_LABELS}


@app.get("/client")
async def client_page():
    """테스트용 WebRTC + DataChannel 클라이언트 HTML 제공."""
    html_file = BASE_DIR / "client_test.html"
    if not html_file.exists():
        return HTMLResponse("<h1>client_test.html not found</h1>", status_code=500)
    try:
        content = html_file.read_text(encoding="utf-8")
    except Exception as e:  # noqa
        return HTMLResponse(f"<h1>Failed to read client_test.html: {e}</h1>", status_code=500)
    return HTMLResponse(content)


# ----------------------------- WebSocket (Signaling & Result) -----------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    peer_id = f"peer_{id(ws)}"
    print(f"[WS] Connected: {peer_id}")
    pc: Optional[RTCPeerConnection] = None
    collector: Optional[SequenceCollector] = None

    async def safe_send_json(payload: Dict[str, Any]):
        try:
            await ws.send_text(json.dumps(payload))
        except Exception as e:  # noqa
            print(f"[WS][SEND][ERR] {e}")

    try:
        while True:
            raw = await ws.receive_text()
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

                @pc.on("datachannel")
                def on_datachannel(channel):  # noqa
                    print(f"[RTC] DataChannel opened: {channel.label}")

                    @channel.on("message")
                    def on_message(data):  # noqa
                        nonlocal collector
                        if isinstance(data, (bytes, bytearray)):
                            if collector is None:
                                collector = app.state.ss.new_collector(peer_id)
                            collector.add_frame(bytes(data))
                            if collector.is_ready():
                                asyncio.create_task(process_sequence_and_respond())
                        else:
                            # 텍스트 제어 메시지 (예: 시작/끝 플래그) 처리 가능
                            if data == "flush" and collector and not collector.is_ready():
                                # 강제 처리 (디버그용)
                                asyncio.create_task(process_sequence_and_respond(force=True))

                async def process_sequence_and_respond(force: bool = False):
                    nonlocal collector
                    if collector is None:
                        return
                    if not collector.is_ready() and not force:
                        return
                    if collector.processed:
                        return
                    collector.processed = True
                    timings = collector.build_timings()
                    inference_result = dummy_infer(collector.frames)
                    result_payload = {
                        "type": "result",
                        "peer_id": peer_id,
                        "timings": timings,
                        "inference": inference_result,
                    }
                    await safe_send_json(result_payload)

                # SDP 처리
                await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=offer_type))
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await safe_send_json({
                    "action": "answer",
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type,
                })
                print(f"[RTC] Answer sent to {peer_id}")

            # --- ICE Candidate (클라이언트 -> 서버) ---
            elif action == "candidate":
                if pc is None or RTCIceCandidate is None:
                    continue
                candidate = msg.get("candidate")
                sdp_mid = msg.get("sdpMid")
                sdp_index = msg.get("sdpMLineIndex")
                if candidate:
                    ice_obj = RTCIceCandidate(candidate=candidate, sdpMid=sdp_mid, sdpMLineIndex=sdp_index)
                    await pc.addIceCandidate(ice_obj)

            # --- Ping ---
            elif action == "ping":
                await safe_send_json({"action": "pong", "ts": time.time()})

            else:
                await safe_send_json({"type": "error", "error": f"unknown_action:{action}"})

    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {peer_id}")
    except Exception as e:  # noqa
        print(f"[WS][ERR] {peer_id}: {e}")
    finally:
        # 정리
        if pc:
            await pc.close()
        if peer_id in app.state.ss.active_peers:
            del app.state.ss.active_peers[peer_id]
        if peer_id in app.state.ss.collectors:
            del app.state.ss.collectors[peer_id]


# ----------------------------- Utility for tests -----------------------------

@app.post("/simulate/predict")
async def simulate_predict():
    fake_frames = [b"frame"] * TARGET_FRAME_COUNT
    result = dummy_infer(fake_frames)
    return JSONResponse(result)


# ----------------------------- Run Helper -----------------------------
if __name__ == "__main__":
    # 개발 편의를 위한 직접 실행
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
