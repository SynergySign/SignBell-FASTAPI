# SignSense FAST-API Server

본 문서는 SignSense 수어 인식 FastAPI 서버의 현재 프로젝트 구조, 핵심 엔드포인트, 그리고 React(브라우저) 클라이언트가 HTTPS/WSS 환경에서 서버에 연결해 프레임을 전송하고 결과를 받는 방법을 집중적으로 정리한 문서입니다.

작성자: 백승현
문서 버전: v1.1
최종 수정일: 2025-10-21

요약
- 서버 엔트리: `main.py` (WebSocket 시그널링 + DataChannel 프레임 수집, Predictor 로드)
- 테스트 클라이언트: `client_test.html` (브라우저에서 카메라 캡처 → JPEG → DataChannel 전송)
- 모델 위치: `models/` (예: `cnn_bilstm_attention_model.pth`)
- 전처리/추론 유틸: `processing/` (`landmark_extractor.py`, `predictor.py`)
- 저장 인터페이스/구현: `storage/` (`local_file_saver.py`, `s3_db_saver.py`, `storage_interface.py`)
- 의존성: `requirements.txt`

중요 변경 요지
- React(프론트엔드)와의 연동 가이드를 보강했습니다(시그널링 메시지 형식, DataChannel 사용법, flush/meta 메시지 예시 포함).
- 로컬 HTTPS(wss) 연결 방법(로컬 인증서, uvicorn 실행 예, 브라우저에서 인증서 신뢰 방법)을 상세히 추가했습니다.
- 내부 저장 엔드포인트(`/api/internal/save-learning`, `/api/internal/save-quiz`) 사용 방식(서버의 백그라운드 작업 또는 내부 호출)을 명확히 기술했습니다.
- 오래되어 혼동을 주는 일부 문단(예: 너무 상세한 Spring 예제의 반복)은 간결화/제거했습니다.

목차
- 설치 및 실행(간단)
- [프로젝트 구조](#프로젝트-구조)
- [핵심 엔드포인트 요약](#핵심-엔드포인트-요약)
- React(프론트엔드)에서 연결하는 방법 — 자세한 단계
- HTTPS / wss(로컬) 설정 및 인증서 신뢰 방법
- 내부 저장 엔드포인트 사용 방식(save-learning / save-quiz)
- [간단한 테스트 체크리스트](#간단한-테스트-체크리스트)
- [향후 권장 작업(요약)](#향후-권장-작업요약)


## 설치 및 실행(간단)

공통 전제
- Python 3.10 이상 권장
- 프로젝트 루트에 `main.py`, `requirements.txt`, `models/`, `certs/`(선택)이 있어야 함

Windows(cmd.exe) 예
- 가상환경 생성 및 활성화:
```
python -m venv .venv
.venv\Scripts\activate
```
- 의존성 설치:
```
pip install -r requirements.txt
```
- 개발용 HTTP 실행:
```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- 로컬 HTTPS(wss) 실행 (로컬 인증서 사용 시):
```
uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs\key.pem --ssl-certfile certs\cert.pem --reload
```

참고: 브라우저에서 카메라 접근(getUserMedia)은 대부분 HTTPS 환경에서만 허용되므로 WebRTC/DataChannel 기반 테스트는 HTTPS 환경을 권장합니다.


## 프로젝트 구조
(주요 파일/디렉터리)
- `main.py` - FastAPI 앱 진입점: WebSocket 시그널링, DataChannel 프레임 수집, SequenceCollector 및 run_inference 호출
- `client_test.html` - 간단한 브라우저 테스트 페이지
- `models/` - 학습된 모델 파일 위치
- `processing/` - `landmark_extractor.py`, `predictor.py` 등 전처리 및 추론 유틸
- `inference_pipeline.py` - 추론 파이프라인 유틸(분리된 로직)
- `storage/` - 저장 인터페이스 및 구현(`local_file_saver.py`, `s3_db_saver.py` 등)
- `configs/settings.py` - 서버 설정(프레임 길이, 수집시간 등)
- `certs/` - 로컬 테스트용 인증서(선택)


## 핵심 엔드포인트 요약
- GET  /                 : 상태 확인 (루트)
- GET  /health           : 헬스 체크(예: predictor 로드 상태)
- GET  /model/status     : 모델 로드 상태, 경로, 디바이스 정보
- GET  /config           : 서버 기본 설정(TARGET_FRAME_COUNT 등)
- GET  /client           : `client_test.html` 반환(테스트용)
- WS   /ws/{session}     : WebSocket 시그널링(Offer/Answer) — WebRTC 시그널링 및 DataChannel 통신
- POST /simulate/predict : 더미 프레임 기반 REST 추론(테스트용)
- POST /api/internal/save-learning : 내부 호출(또는 서버 백그라운드 작업)으로 학습 데이터 저장 스케줄
- POST /api/internal/save-quiz     : 내부 호출(또는 서버 백그라운드 작업)으로 퀴즈(시험) 데이터 저장 스케줄

설명: `save-learning`과 `save-quiz` 엔드포인트는 보통 외부 공개 API가 아니라 서버 내부(백그라운드 작업) 또는 내부 시스템(예: 신뢰된 백엔드)이 호출하도록 설계되어 있습니다. 운영 환경에서 직접 노출할 때는 인증 토큰(`INTERNAL_API_TOKEN`)로 보호하십시오.


## React(프론트엔드)에서 연결하는 방법 — 자세한 단계
목표: 브라우저(React)가 카메라를 캡처하여 서버의 DataChannel로 JPEG(또는 바이너리 프레임)를 전송하고, 서버에서 추론 결과를 받도록 구현합니다.

- 사전 요구
  - 브라우저에서 HTTPS로 접근(또는 localhost에서 개발시 로컬 인증서로 신뢰된 HTTPS)
  - 서버가 HTTPS로 실행되어 있어야 함(wss 사용 가능)
  - DataChannel label: 'frames' (서버가 동일하게 수신하도록 설계됨)

- 인증(토큰) 전송 방식 요약
  - 본 프로젝트의 WebSocket 핸드셰이크에서 서버는 토큰을 다음 우선순위로 검사합니다:
    1) HTTP 쿠키 (key: `configs.settings.COOKIE_ACCESS_TOKEN_NAME`, 기본값 `ACCESS_TOKEN`) — 브라우저 클라이언트 권장
    2) Authorization 헤더: `Authorization: Bearer <token>` — 서버사이드/스크립트 클라이언트 권장
    3) 쿼리 파라미터: `?token=<token>` (fallback)

  따라서 외부(서버사이드) 클라이언트는 Authorization 헤더를 사용해 WebSocket을 열고, 브라우저는 로그인 응답에서 `Set-Cookie`로 토큰을 발급하거나 개발용 `GET /debug/set-cookie?token=...` 엔드포인트로 테스트용 쿠키를 설정하면 됩니다.

  간단한 예시
  - Node.js (서버사이드, Authorization 헤더로 연결)
    - wss://your.server/ws/<session_id>로 연결 시 요청 헤더에 `Authorization: Bearer <token>`을 포함하세요.
  - Python (서버사이드, websockets 라이브러리 사용)
    - websockets.connect(..., extra_headers=[("Authorization","Bearer <token>")]) 형태로 연결하세요.
  - 브라우저(React) 방식
    - 브라우저는 WebSocket API로 임의 헤더를 설정할 수 없으므로, 로그인 시 서버가 `Set-Cookie`로 토큰을 발급하거나 테스트 시 `/debug/set-cookie?token=...`를 사용해 쿠키를 설정한 뒤 일반적인 `new WebSocket('wss://.../ws/<session_id>')`로 연결하세요.

- 설정(환경변수 및 defaults) 정리
  - 서버 설정은 `configs/settings.py`에서 관리됩니다. 주요 항목과 기본값은 다음과 같습니다:
    - JWT_SECRET_KEY: (환경변수) - JWT 서명 키
    - JWT_ALGORITHM: HS256
    - COOKIE_ACCESS_TOKEN_NAME: ACCESS_TOKEN
    - COOKIE_ACCESS_TOKEN_MAX_AGE: 3600
    - SSL_CERT_PATH / SSL_KEY_PATH: certs/cert.pem / certs/key.pem (로컬 테스트용)
    - TARGET_FRAME_COUNT, COLLECTION_DURATION_SECONDS, MAX_FRAMES_TO_COLLECT: 프레임 수집/추론 튜닝 파라미터

- YAML 파일 관련
  - 현재 저장소에는 `*.yml` 또는 `*.yaml` 설정 파일이 없습니다. 외부 클라이언트와 맞춰야 할 설정은 환경변수(또는 `configs/settings.py`)를 통해 전달하시기 바랍니다. 필요하시면 예시 `docker-compose.yml` 또는 `appsettings.yml`을 제가 만들어 드릴 수 있습니다.

간단한 단계 요약
1) React에서 `RTCPeerConnection` 생성
2) `pc.createDataChannel('frames')`로 DataChannel 생성 및 이벤트 바인딩
3) 로컬 비디오 트랙을 preview 용도로 `pc.addTrack()`로 추가(선택)
4) Offer 생성 후 시그널링 WebSocket(`wss://<host>:<port>/ws/{session}`)으로 전송
5) 서버로부터 Answer 수신 → `pc.setRemoteDescription(answer)`
6) DataChannel이 open 상태가 되면 캡처한 JPEG Blob을 ArrayBuffer로 변환하여 `dc.send(arrayBuffer)`로 전송
7) 필요한 경우 시그널링 WebSocket에 `{ action: 'flush' }` 메시지를 보내 서버에 즉시 추론(및 저장) 트리거
8) 메타데이터 제공이 필요하면 `{ action: 'meta', meta: { user_id, word_pk, word_name, ... } }` 형식으로 전송

시그널링 메시지(예시)
- 클라이언트 → 서버
  - Offer 전송: `{ action: 'offer', type: 'offer', sdp: <offer.sdp> }`
  - Flush 트리거: `{ action: 'flush' }` (서버에 즉시 추론 요청)
  - Meta 전송: `{ action: 'meta', meta: { user_id: 'u1', word_pk: 123, word_name: '안녕하세요' } }`
- 서버 → 클라이언트
  - Answer 응답: `{ type: 'answer', sdp: <answer.sdp> }`
  - 추론 결과 알림: `{ action: 'inference_result', result: { label: '...', score: 0.95, ... } }`

React 내부 구현 팁
- DataChannel 전송은 binary (ArrayBuffer)를 사용하세요. JPEG Blob을 `await blob.arrayBuffer()`로 변환해 전송하면 서버가 받기 쉽습니다.
- 프레임을 너무 자주 보내면 네트워크/서버에 무리가 생깁니다. 클라이언트에서 쓰로틀링(예: 20~25fps) 또는 프레임 드롭을 적용하세요.
- 긴 시퀀스(예: 10초 이상)를 보낼 경우, 서버의 `SIGN_SEQUENCE_TARGET_FRAMES` 한계를 확인하세요(기본값은 `configs/settings.py`).
- 메타데이터는 `meta` action으로 먼저 보내두고, `flush`를 보내 추론/저장을 트리거하는 패턴을 권장합니다.

간단한 동작 플로우
1. 페이지 진입 → WS 연결(wss://.../ws/session123)
2. Offer 교환 → DataChannel open
3. 캡처(또는 canvas.toBlob) → Blob->ArrayBuffer -> dc.send()
4. 원하는 시점에 WS로 `{action:'flush'}` 전송 → 서버가 추론 실행 및 내부 저장 스케줄
5. 서버가 결과를 WS로 전송 → React에서 UI 업데이트


## HTTPS / wss(로컬) 설정 및 인증서 신뢰 방법
웹캠/마이크 접근을 테스트하려면 브라우저에서 HTTPS 또는 localhost 환경이 필요합니다. 로컬에서 HTTPS를 사용하는 방법(권장)

1) mkcert 사용(간단하고 권장)
- mkcert 설치: https://github.com/FiloSottile/mkcert
- 로컬용 인증서 생성(예: macOS/Windows):
```
mkcert -install
mkcert localhost 127.0.0.1 ::1
```
- 생성된 cert/key 파일을 `certs/`에 복사(예: `certs/localhost.pem`, `certs/localhost-key.pem`)

2) uvicorn을 SSL로 실행 (Windows 예)
```
uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs\localhost-key.pem --ssl-certfile certs\localhost.pem --reload
```

3) 브라우저에서 인증서 신뢰
- mkcert로 만든 인증서는 시스템 신뢰 체인에 등록되므로 별도 조치가 필요없음(대부분의 경우)
- 자체 서명(self-signed) 인증서를 사용했다면 브라우저에서 수동으로 예외를 추가하거나 인증서를 신뢰하도록 설정해야 함

주의사항
- HTTPS/SSL 설정을 하지 않으면 클라이언트는 getUserMedia를 차단할 수 있으며, 브라우저에서 wss 연결이 실패할 수 있습니다.
- 운영환경에서는 반드시 정식 인증서(예: Let's Encrypt)를 사용하고 포트/방화벽을 적절히 설정하세요.


## 내부 저장 엔드포인트 사용 방식(save-learning, save-quiz)

요약
- `/api/internal/save-learning` 및 `/api/internal/save-quiz`는 서버 내부에서 추론 후 데이터를 영구 저장하거나(예: S3/로컬) DB에 메타를 남기기 위해 사용되는 엔드포인트입니다.
- 일반적으로 WebRTC 핸들러가 추론 완료 시 백그라운드 작업으로 해당 엔드포인트를 호출하거나 직접 저장 로직(`storage` 모듈)을 호출합니다.
- 외부에서 직접 호출 시에는 `INTERNAL_API_TOKEN` 같은 인증 토큰을 사용해 호출을 보호해야 합니다.

요청 예(내부 호출 또는 신뢰된 서버에서 호출)
- POST /api/internal/save-learning
  - body: JSON { session_id: string, frames_npy_base64?: string, meta: { user_id, label, ... } }
- POST /api/internal/save-quiz
  - body: JSON { session_id: string, frames_npy_base64?: string, meta: { user_id, word_pk, word_name, ... } }

(구현 세부는 `storage/` 내 구현체에 따라 다르므로 실제 저장 포맷과 키 형식은 해당 모듈을 확인하세요.)


## 간단한 테스트 체크리스트
- [ ] `.venv` 활성화 후 `pip install -r requirements.txt`
- [ ] `models/`에 모델 파일이 있는지 확인
- [ ] (로컬) `mkcert`로 인증서 생성 후 `uvicorn`을 SSL 옵션으로 실행
- [ ] 브라우저에서 `https://localhost:8443/client` 접속 → 카메라 권한 확인
- [ ] React에서 WS(wss://...)로 시그널링 연결 후 DataChannel로 프레임 전송 및 `flush` 확인


## 향후 권장 작업(요약)
- 시그널링/데이터 전송 프로토콜 문서화(메시지 스펙, chunking, 순서 보장)
- TURN 서버 연동(브라우저-서버 직접 연결이 어려운 환경 대비)
- `storage/s3_db_saver.py`에 S3 업로드/DB 기록 로직 강화
- 내부 엔드포인트 인증/권한 정책 정리

---

필요하시면 이 문서를 `docs/리액트 작업순서.md`로 별도 복제하여 React 개발팀 전용 가이드로 더 간단하게 만든 후, 예제 컴포넌트 코드(완전 동작하는 작은 컴포넌트)도 추가해 드리겠습니다.
