## 🚀 클라이언트 $\leftrightarrow$ FastAPI 실시간 영상 추론 워크플로우

### 1단계: 초기 설정 및 인증 (JWT & WebSocket 핸드셰이크)

클라이언트가 유효한 JWT 토큰을 사용하여 WebSocket 시그널링 채널을 확보하는 단계입니다.

| 단계 | 주체 | 동작 | 핵심 기술/정보 |
| :--- | :--- | :--- | :--- |
| **1.1. 토큰 확보** | 클라이언트 (React) | 로그인 후 확보한 **유효한 JWT**를 메모리에 보관합니다. | JWT |
| **1.2. 토큰 첨부** | 클라이언트 (React) | 웹 소켓 연결을 시작할 때, **토큰만** 쿼리 파라미터로 첨부합니다. | `wss://.../ws/signaling?token=...` |
| **1.3. 핸드셰이크 요청** | 클라이언트 (React) | FastAPI 서버에 웹 소켓 연결을 요청합니다. | Web Socket Handshake (HTTP GET) |
| **1.4. 토큰 파싱** | **FastAPI** | 쿼리 파라미터에서 **토큰**을 추출합니다. | FastAPI `Query` / `Depends` |
| **1.5. JWT 검증** | **FastAPI** | `security/jwt_validator.py` 모듈을 사용하여 토큰의 서명 및 만료 시간을 **독립적으로 검증**합니다. | JWT 디코딩 (e.g., `python-jose`) |
| **1.6. 연결 수락/거부** | **FastAPI** | **검증 통과 시:** `await websocket.accept()` 호출. **실패 시:** `websocket.close()` 호출 후 연결 즉시 거부. | Web Socket `accept()` |

---

### 2단계: 메타데이터 전달 및 준비 (WebSocket 내부 메시지)

WebSocket 연결이 수립된 직후, 클라이언트가 추론에 필요한 단어 정보를 서버에 전달하고 서버가 이를 세션 상태에 저장하는 단계입니다. **외부 API 조회 단계가 제거됩니다.**

| 단계 | 주체 | 동작 | 핵심 기술/정보 |
| :--- | :--- | :--- | :--- |
| **2.1. 메타데이터 패키징** | 클라이언트 (React) | 분석 대상인 **단어 PK**와 **단어 이름**을 포함하는 JSON 메시지를 생성합니다. | `{"type": "meta", "word_pk": 42, "word_name": "사랑"}` (JSON) |
| **2.2. 메타데이터 전송** | 클라이언트 (React) | 웹 소켓을 통해 이 JSON 메시지를 **텍스트**로 FastAPI에 전송합니다. | Web Socket `send()` (텍스트) |
| **2.3. 메시지 수신 및 파싱** | **FastAPI** (`main.py` WS 핸들러) | WebSocket 루프에서 텍스트 메시지를 수신하고, `msg.get("type")`이 `"meta"`인지 확인하여 `word_pk`와 `word_name`을 추출합니다. | `msg = json.loads(raw)` |
| **2.4. 세션 상태 저장** | **FastAPI** (`main.py` WS 핸들러) | 수신한 단어 정보(이름, PK)를 `app.state.ss.collectors` 등 **인메모리 세션 상태**에 저장하여 이후 추론 로직에서 사용합니다. | `app.state.ss.collectors[session_id]['word_name'] = name` |

---

### 3단계: WebRTC 시그널링 (실시간 영상 채널 구축)

WebSocket 연결을 통해 제어 정보(SDP)를 교환하여 고성능 WebRTC 영상 채널을 엽니다.

| 단계 | 주체 | 동작 | 핵심 기술/정보 |
| :--- | :--- | :--- | :--- |
| **3.1. SDP Offer 생성** | 클라이언트 (React) | 웹캠에 접근하여 미디어 스트림을 확보한 후, **SDP Offer**를 생성합니다. | WebRTC API |
| **3.2. Offer 전송** | 클라이언트 (React) | 생성된 SDP Offer를 **웹 소켓**을 통해 **텍스트 메시지**로 FastAPI에 전송합니다. | Web Socket `send()` (시그널링) |
| **3.3. Answer 생성** | **FastAPI** | 수신한 SDP Offer를 `aiortc`에 전달하고 **SDP Answer**를 생성합니다. | `RTCSessionDescription`, `pc.createAnswer()` |
| **3.4. Answer 전송** | **FastAPI** | SDP Answer를 다시 **웹 소켓**을 통해 클라이언트에게 푸시합니다. | Web Socket `send()` (시그널링) |
| **3.5. 연결 확정** | 클라이언트 (React) | SDP Answer를 받아 WebRTC 연결을 최종적으로 확정하고, **WebRTC DataChannel**을 통해 프레임 전송을 시작합니다. | WebRTC Peer-to-Peer 연결 |

---

### 4단계: 실시간 영상 스트리밍 및 추론

WebRTC DataChannel을 통해 영상 프레임이 전송되고, 서버가 이를 수집/추론하는 핵심 단계입니다.

| 단계 | 주체 | 동작 | 핵심 기술/정보 |
| :--- | :--- | :--- | :--- |
| **4.1. 영상 스트리밍** | 클라이언트 (React) | 웹캠 영상 프레임 바이트를 **WebRTC DataChannel**을 통해 FastAPI 서버로 **지속적으로** 전송합니다. | DataChannel `send()` |
| **4.2. 프레임 수신 및 수집** | **FastAPI** (`main.py` DataChannel 리스너) | `on_message` 콜백에서 바이트 데이터(프레임)를 수신하고 **`SequenceCollector`**에 저장합니다. | `collector.add_frame(message)` |
| **4.3. 추론 실행** | **FastAPI** (`on_message` 콜백) | 클라이언트가 `"flush"` 시그널을 보내면, `inference_pipeline.py`의 **`run_inference`** 함수가 호출됩니다. | `run_inference(predictor, collector.frames)` |
| **4.3. (세부 로직)** | **`processing/`** | **`landmark_extractor.py`**가 프레임에서 좌표를 추출하고, **`predictor.py`**가 최종 추론(예측된 단어 이름, 점수)을 실행합니다. | ML 모델 예측 |
| **4.4. 결과 푸시** | **FastAPI** (`run_inference_once` 헬퍼) | 추론 결과(`predicted`, `score`)를 JSON 형태로 **웹 소켓 채널**을 통해 클라이언트에게 즉시 푸시합니다. | `await safe_send_json({"type": "inference_result", ...})` |
| **4.5. 백그라운드 저장** | **FastAPI** | `run_inference_once`가 끝난 후, `storage/s3_db_saver.py`의 **`schedule_quiz_save`**를 비동기로 스케줄링합니다. | `asyncio.create_task(schedule_quiz_save(...))` |