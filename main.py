"""
모듈: main.py
설명:
- SignSense Inference Server의 FastAPI 진입점입니다.
- WebSocket 시그널링 엔드포인트와 간단한 REST 엔드포인트들을 제공하며,
  모델 로드와 전역 상태(AppState) 관리를 담당합니다.
- 주요 엔드포인트:
  - GET  /                 : 서버 상태 확인
  - GET  /health           : 예측기(모델) 로드 상태 확인
  - GET  /model/status     : 모델 로드 상태 및 경로 정보 반환
  - GET  /config           : 서버 구성값 반환
  - GET  /client           : 테스트용 HTML 클라이언트 제공
  - POST /simulate/predict: 스모크 테스트용 더미 추론 엔드포인트
  - WS   /ws/{session_id}  : WebSocket 시그널링 엔드포인트 (token 쿼리 파라미터로 JWT 검증)

since: 2025.10.17
author: 백승현
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict
from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from security.jwt_validator import validate_token_and_get_user_id

from configs import settings
from routers import internal as internal_router
from routers import diagnostics as diagnostics_router

# --- 중요 의존성 ---
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

# 중앙 설정에서 값을 가져옵니다 (환경변수는 configs/settings.py에서 처리).
TARGET_FRAME_COUNT = settings.TARGET_FRAME_COUNT
COLLECTION_DURATION_SECONDS = settings.COLLECTION_DURATION_SECONDS
MAX_FRAMES_TO_COLLECT = settings.MAX_FRAMES_TO_COLLECT

BASE_DIR = Path(__file__).resolve().parent

# --- FastAPI 앱 초기화 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 수명주기: 시작 시 Predictor 로드 시도 (기존 on_event('startup') 대체)."""
    print("[STARTUP] Attempting to load Predictor...")
    # 이미 로드되어 있으면 재사용
    if getattr(app.state, 'ss', None) and getattr(app.state.ss, 'predictor', None):
        print("[STARTUP] Predictor already initialized.")
        yield
        return

    try:
        # 동적으로 Import 시도하여 실제 운영환경의 종속성을 사용할 수 있게 함
        from processing.predictor import get_predictor as _get_predictor
        predictor_instance = _get_predictor()
        # app.state.ss 가 없을 수 있으므로 안전하게 설정
        if not getattr(app.state, 'ss', None):
            app.state.ss = AppState()
        app.state.ss.predictor = predictor_instance
        print("[STARTUP] Predictor successfully loaded.")
    except Exception as e:
        # 자세한 예외 정보 로그
        print(f"[STARTUP][ERROR] Failed to load Predictor dynamically: {e}")
        traceback.print_exc()
        if not getattr(app.state, 'ss', None):
            app.state.ss = AppState()
        app.state.ss.predictor = None
        print("[STARTUP] Running without Predictor. Install required packages and restart the server for real inference.")

    yield
    # (선택) 종료 시 정리 로직을 여기에 추가할 수 있습니다.


app = FastAPI(title="SignSense Inference Server", version="0.2.0", lifespan=lifespan)

# 명시적으로 app.state를 초기화하여 정적 분석기의 'state' 관련 경고를 줄입니다.
app.state = SimpleNamespace()

# 라우터 등록
app.include_router(internal_router.router)
app.include_router(diagnostics_router.router)


# ----------------------------- 모델 및 추론 로직 -----------------------------

# 공통 추론 파이프라인 사용
from inference_pipeline import run_inference, SequenceCollector


# ----------------------------- 전역 상태 -----------------------------

class AppState:
    """애플리케이션 전역 상태를 보관하는 컨테이너.

    역할/정의:
    - Predictor 인스턴스, 활성 피어 연결, 세션별 SequenceCollector를 관리합니다.

    since: 2025.10.17
    author: 백승현
    """
    def __init__(self):
        # Predictor는 heavy deps가 없으면 None이 됩니다.
        # ---모델 로드 오류 해결---
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


# ----------------------------- 스타트업 이벤트 -----------------------------

# @app.on_event("startup") 블록은 lifespan으로 대체되어 제거되었습니다.


# ----------------------------- REST 엔드포인트 -----------------------------

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
        "target_frame_count": settings.TARGET_FRAME_COUNT,
        "collection_duration_seconds": settings.COLLECTION_DURATION_SECONDS,
        "max_frames_to_collect": settings.MAX_FRAMES_TO_COLLECT,
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


# === WebSocket 시그널링 엔드포인트 ===
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """간단한 WebSocket 시그널링 엔드포인트.

    동작:
    - 쿼리 파라미터 `token`을 받아 JWT를 검증합니다. 실패 시 연결을 닫습니다.
    - 텍스트 메시지로 `{"type": "meta", "word_pk": .., "word_name": ..}`를 수신하면
      app.state.ss.collectors[session_id]에 해당 메타를 저장합니다.
    - 클라이언트가 연결을 유지하면서 DataChannel으로 프레임을 전송한다고 가정합니다.
    """
    # 먼저 토큰 검증
    params = websocket.query_params
    token = params.get("token")
    try:
        if not token:
            await websocket.close(code=1008)
            return
        user_id = validate_token_and_get_user_id(token)
    except Exception:
        # 검증 실패 시 연결을 거부
        await websocket.close(code=1008)
        return

    await websocket.accept()

    # 세션용 collector가 없으면 새로 생성
    collector = app.state.ss.collectors.get(session_id) or app.state.ss.new_collector(session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                # 텍스트가 JSON이 아닌 경우 무시
                continue

            mtype = msg.get("type")
            if mtype == "meta":
                # 단어 메타데이터를 세션 상태에 저장
                collector.meta = {
                    "word_pk": msg.get("word_pk"),
                    "word_name": msg.get("word_name"),
                    "user_id": user_id,
                }
                await websocket.send_text(json.dumps({"type": "meta_ack"}))
            elif mtype == "flush":
                # 클라이언트가 flush 시그널을 보내면 추론 실행
                # 실제 구현에서는 바이트 프레임 리스트를 수집하여 run_inference 호출
                predictor = app.state.ss.predictor
                frames = getattr(collector, "frames", [])
                result = run_inference(predictor, frames)
                await websocket.send_text(json.dumps({"type": "inference_result", "result": result}))
            else:
                # 기타 메시지: 무시 또는 에코
                await websocket.send_text(json.dumps({"type": "noop"}))
    except WebSocketDisconnect:
        # 연결 종료 시 리소스 정리
        app.state.ss.collectors.pop(session_id, None)
        return


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
