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

import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from configs import settings
from routers import diagnostics as diagnostics_router
from ws_handler import websocket_handler

# --- 중요 의존성 ---
# 일부 라이브러리는 개발환경에 설치되어 있지 않을 수 있으므로 안전하게 임포트합니다.
HEAVY_IMPORTS_AVAILABLE = True
try:
    import torch
    import numpy as np

    from processing.predictor import Predictor, get_predictor, CNN_BiLSTM_Attention, PositionalEncoding
    from processing.landmark_extractor import extract_sequence_from_frames, FRAME_FEATURE_DIM
except Exception as _e:
    # 의존성이 없는 환경에서도 기본 REST 엔드포인트를 테스트할 수 있도록 예외를 무시하고
    # 런타임에서 필요한 경우 명확한 에러를 발생시키도록 처리합니다.
    print(f"[WARN] Optional heavy imports failed: {_e}")
    HEAVY_IMPORTS_AVAILABLE = False

    torch = None
    np = None

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
    try:
        # Log JWT config (masked secret) for debugging environment loading
        alg = getattr(settings, 'JWT_ALGORITHM', None)
        secret = getattr(settings, 'JWT_SECRET_KEY', None) or ''
        masked = (secret[:4] + '...' + secret[-4:]) if len(secret) > 8 else ('*' * len(secret))
        print(f"[STARTUP] JWT_ALGORITHM={alg} JWT_SECRET_KEY={masked}")
    except Exception:
        pass

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
# internal router is deprecated in favor of WebSocket-based flow; keep commented to avoid exposing REST save endpoints
# app.include_router(internal_router.router)
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

        # aiortc를 사용하지 않으므로 Any로 타입 지정
        self.active_peers: Dict[str, Any] = {}
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
    """WebSocket 엔드포인트 래퍼: 실제 로직은 `ws_handler.websocket_handler`에 위임합니다."""
    try:
        await websocket_handler(websocket, session_id, app.state.ss)
    except WebSocketDisconnect:
        app.state.ss.collectors.pop(session_id, None)
    except Exception as e:
        print("[WS][ERROR] websocket_endpoint wrapper error:", e)
        try:
            await websocket.close()
        except Exception:
            pass
        app.state.ss.collectors.pop(session_id, None)




if __name__ == "__main__":
    import uvicorn
    # from pathlib import Path  # pathlib를 import합니다.

    # 1. mkcert로 생성한 인증서의 절대 경로를 지정합니다.
    # (Python에서는 C:\certs\... 보다 C:/certs/... (슬래시)를 쓰는 것이 편합니다)
    # ssl_cert_path = Path("C:/certs/localhost+1.pem")
    # ssl_key_path = Path("C:/certs/localhost+1-key.pem")

    # 2. 해당 경로에 mkcert 인증서 파일이 있는지 확인합니다.
    # if not ssl_cert_path.is_file() or not ssl_key_path.is_file():
    #     print(f"[ERROR] mkcert SSL certificates not found at C:/certs/")
    #     print("Check if 'localhost+1.pem' and 'localhost+1-key.pem' exist.")
    #     print("Server cannot start with HTTPS.")
    # else:
    #     print("[INFO] Starting server with mkcert HTTPS.")
    #     uvicorn.run(
    #         app,  # 'app' 변수는 이 파일 상단 어딘가에 정의되어 있어야 합니다.
    #
    #         # 3. host를 127.0.0.1로 변경합니다.
    #         # (Vite 프록시가 127.0.0.1을 바라보고, mkcert 인증서도 localhost/127.0.0.1용입니다)
    #         host="127.0.0.1",
    #         port=8000,
    #
    #         # 4. mkcert 파일 경로를 문자열(str)로 전달합니다.
    #         ssl_keyfile=str(ssl_key_path),
    #         ssl_certfile=str(ssl_cert_path),
    #     )

    # 1. 이전 코드의 로컬 인증서 경로 설정 및 확인 로직을 제거합니다.
    # 2. 서버를 HTTP 모드로 실행합니다.
    print("[INFO] Starting server with HTTP (for Docker/ALB environment).")

    # [수정] host="127.0.0.1" -> host="0.0.0.0" 으로 변경하여 외부 접근을 허용합니다.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # ssl_keyfile과 ssl_certfile 인수를 제거합니다.
        # (제거된 부분: ssl_keyfile=str(ssl_key_path), ssl_certfile=str(ssl_cert_path))
    )
