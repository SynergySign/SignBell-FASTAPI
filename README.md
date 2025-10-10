# SignSense Test Server

FastAPI + WebSocket(WebRTC 시그널링) + aiortc(DataChannel) 기반 수어 인식 프로토타입 서버입니다.
브라우저에서 DataChannel로 전송한 프레임 시퀀스를 수집하여 (현재는 예비 더미 추론 또는 모델 추론)을 수행합니다.

요약
- 서버 엔트리: `main.py` (WebSocket 시그널링 + DataChannel 프레임 수집, Predictor 로드)
- 테스트 클라이언트: `client_test.html` (브라우저에서 카메라 캡처 → JPEG → DataChannel 전송)
- 모델 위치: `models/` (예: `cnn_bilstm_attention_model.pth` 포함)
- 전처리/추론 유틸: `processing/` (`landmark_extractor.py`, `predictor.py`)
- 의존성: `requirements.txt`

중요 변경/현황 (코드 기준)
- 기본 수집/모델 입력 프레임 길이(TARGET_FRAME_COUNT): 300 (환경변수 `SIGN_SEQUENCE_TARGET_FRAMES`로 오버라이드 가능)
- 수집 기준 시간: 환경변수 `SIGN_SEQUENCE_COLLECTION_SECONDS` 또는 기본 5.0초
- 모델 파일 샘플: `models/cnn_bilstm_attention_model.pth` (레포에 포함되어 있음)

설치 및 실행 (Windows - cmd.exe 예시)
1) 가상환경 생성 및 활성화
```
python -m venv .venv
.venv\Scripts\activate
```
2) 의존성 설치
```
pip install --upgrade pip
pip install -r requirements.txt
```
참고: `torch`는 플랫폼/하드웨어(예: CUDA 버전)에 따라 별도로 설치 권장합니다. 예:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
3) 서버 실행 (HTTP, 개발용)
```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
HTTPS 로컬 테스트(권장: WebRTC/카메라 관련 기능 테스트 시)
- 프로젝트에 인증서가 있는 경우(`certs/cert.pem`, `certs/key.pem`), `main.py`는 이를 자동으로 감지하고 HTTPS로 실행합니다.
- 수동 실행 예 (cmd.exe에서 줄바꿈 시 \로 연결하지 않아도 괜찮음):
```
uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs\key.pem --ssl-certfile certs\cert.pem
```

주의: `mkcert` 등으로 로컬 인증서를 발급해 `certs/`에 넣어 사용하세요.

핵심 엔드포인트
- GET /              : 상태 확인
- GET /health        : 헬스 체크 (predictor 로드 상태 확인)
- GET /model/status  : 모델 로드 상태, 경로, 디바이스 정보
- GET /config        : 서버 기본 설정(TARGET_FRAME_COUNT, collection duration 등)
- GET /client        : 테스트용 HTML 페이지 (`client_test.html`) 반환
- WS  /ws/{session}  : WebSocket 시그널링(Offer/Answer) — 브라우저와 서버 간 시그널링 통신
- POST /simulate/predict : 더미 프레임으로 동작하는 REST 추론 시뮬레이터 (성능 측정용)

WebRTC 시그널링 & DataChannel 동작 요약
1. 브라우저(`client_test.html`)에서 RTCPeerConnection 생성 후 Offer 생성
2. ICE Gathering 완료 시 WebSocket `/ws/{session}`에 `{action: 'offer', sdp, type}` 전송
3. 서버는 Answer 생성하여 `{type: 'answer', sdp}` 반환 (현재 트리클 ICE 미사용: Answer에 후보 포함)
4. DataChannel(label='frames') 오픈되면 브라우저가 JPEG Blob → ArrayBuffer로 변환하여 전송
5. 서버는 `SequenceCollector`로 프레임을 수집. 기본 기준은 collection duration(기본 5s) 또는 수집된 프레임 개수(안전 상한)
6. 수집 완료 혹은 클라이언트의 `flush` 명령 수신 시 `run_inference`를 호출하여 랜드마크 추출 → (모델)추론 → WebSocket으로 `inference_result` 전송

클라이언트 테스트 사용법
1. 서버 실행
2. 브라우저에서 `https://localhost:8000/client` 접속
3. Connect → 브라우저 권한으로 카메라 접근 허용 → DataChannel이 열리면 Start Capture 클릭
4. Capture가 끝나면 서버로부터 추론 결과를 수신
5. 필요 시 Flush(강제추론) 또는 Reset(다음 캡처 준비) 사용

프레임/랜드마크 전처리
- `processing/landmark_extractor.py`에 실시간 랜드마크 추출 로직이 구현되어 있습니다. MediaPipe Holistic을 사용하여 다음 특징 벡터(프레임당 147차원)를 생성합니다.
  - Pose 6개(인덱스 11~16), 좌/우 손 21포인트, 손목 간 상대 벡터 등 → 총 49 포인트 × 3 = 147 차원
- `SIGN_EXTRACT_LANDMARKS=1` 등의 환경변수로 활성화하여 서버 측에서 프레임 → 랜드마크 처리를 수행할 수 있습니다.

모델 & Predictor
- 모델 로드: `processing/predictor.py`의 `get_predictor()`가 `models/cnn_bilstm_attention_model.pth`를 로드합니다.
- 모델이 없으면 `get_predictor()`는 FileNotFoundError를 발생시키므로 배포/테스트 전에 모델 파일 존재를 확인하세요.
- 샘플 모델 구조: `CNN_BiLSTM_Attention` (predictor에서 정의)

의존성 (요약)
- 필수: fastapi, uvicorn, aiortc, python-multipart
- ML/영상: torch, numpy, mediapipe, opencv-python, Pillow
- 테스트: httpx (smoke_test.py)
자세한 버전은 `requirements.txt`를 확인하세요.

간단 스모크 테스트
- REST 엔드포인트 동작 확인(비-RTC 환경에서도 가능):
```
python smoke_test.py
```
- WebRTC(DataChannel) 기능을 테스트하려면 브라우저에서 `/client`를 사용하세요. aiortc가 설치되어 있지 않으면 WebRTC 시그널링/데이터채널은 동작하지 않습니다.

디버깅 / 자주 묻는 문제
- SSL 인증서가 없으면 `main.py`가 자동으로 HTTP로 실행됩니다. WebRTC의 카메라 권한/HTTPS 동작을 테스트하려면 로컬 인증서를 준비하세요.
- 모델 로드 에러: `processing/predictor.py`에서 모델 경로(`models/cnn_bilstm_attention_model.pth`)가 존재하는지, checkpoint의 키(`model_state_dict` 등)가 적절한지 확인하세요.
- MediaPipe/Opencv import 오류: 해당 패키지 설치 필요. 실시간 랜드마크 추출은 추가 패키지 설치가 필요합니다.

TODO / 향후 작업
- 학습 원본 코드와 coordinate_extractor 비교 후 정밀 보정
- Chunking(프레임 헤더/순번) 프로토콜 안정화
- TURN 서버(coturn) 연동
- 성능 모니터링 (latency, 메모리, fps)

라이선스
- 현재 PoC 단계(라이선스 미지정). 배포 전 라이선스 명시 필요.

참고 문서
- `docs/개발진행현황.md`, `docs/테스트가이드.md`, `docs/모델로드방식.md` 등을 참고하세요.
