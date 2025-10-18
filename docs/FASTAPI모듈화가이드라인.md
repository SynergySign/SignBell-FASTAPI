# FAST API 모듈화 가이드라인


### 1. 설정 및 보안 모듈 구현 가이드

이 영역은 **1단계: 초기 설정 및 인증**을 지원하며, 서버의 모든 주요 상수와 보안 비밀을 관리합니다.

#### 1.1. `configs/settings.py` (신규 파일)

**역할:** 환경 변수(ENV)로부터 서버 설정과 JWT 비밀 키를 로드하여 중앙에서 관리합니다.

| 설정 항목 | 소스 | 필요 이유 |
| :--- | :--- | :--- |
| `JWT_SECRET_KEY` | ENV | **1.5. JWT 검증**을 위한 핵심 비밀 키. |
| `JWT_ALGORITHM` | ENV (기본값: HS256) | JWT 인코딩/디코딩 알고리즘. |
| `TARGET_FRAME_COUNT` | ENV | `inference_pipeline.py`와 `main.py`의 상수를 대체합니다. |
| `COLLECTION_DURATION_SECONDS` | ENV | `inference_pipeline.py`와 `main.py`의 상수를 대체합니다. |
+| `COOKIE_ACCESS_TOKEN_NAME` | ENV (기본값: ACCESS_TOKEN) | REST 요청에서 쿠키 기반 토큰을 찾을 때 사용하는 쿠키 이름. |
+| `COOKIE_ACCESS_TOKEN_MAX_AGE` | ENV (기본값: 3600) | 액세스 토큰 쿠키의 권장 만료(초). |

#### 1.2. `security/jwt_validator.py` (신규 파일)

**역할:** **1.5. JWT 검증** 로직을 캡슐화합니다.

| 구현 내용 | 설명 |
| :--- | :--- |
| **의존성** | `configs.settings`, `python-jose` 라이브러리. |
| **핵심 함수** | `def get_current_user_id(request: Request) -> Any` | REST 엔드포인트용 FastAPI 의존성으로, Authorization 헤더의 Bearer 토큰을 우선 확인하고(Authorization: Bearer <token>), 없으면 HTTP 쿠키(`COOKIE_ACCESS_TOKEN_NAME`)에서 토큰을 찾아 디코딩/검증합니다. 유효하면 사용자 ID(또는 세션에 필요한 클레임)를 반환하고, 실패 시 `HTTPException(401)`을 발생시켜 접근을 거부합니다. |
| **WebSocket 사용** | WebSocket 핸드셰이크 흐름은 쿼리파라미터 `token`을 사용합니다. `/ws/{session_id}` 핸들러는 쿼리 토큰을 받아 `validate_token_and_get_user_id(token)`으로 검증합니다(쿼리 토큰은 `jwt_validator`의 WS 유틸을 재사용). |

---

### 2. 라우팅 및 상태 관리 구현 가이드

이 영역은 FastAPI의 진입점과 REST API 엔드포인트의 책임을 분리합니다.

#### 2.1. `main.py` (최종)

**역할:** 앱 인스턴스 생성, 라우터 등록, 전역 상태(`AppState`) 초기화, **WebSocket 시그널링** (3단계) 및 **데이터 수신** (4단계)을 처리합니다.

| 수정 내용 | 설명 |
| :--- | :--- |
| **상수 제거** | `TARGET_FRAME_COUNT`, `COLLECTION_DURATION_SECONDS` 등은 `configs.settings` 참조로 변경합니다. |
| **라우터 등록** | `app.include_router(diagnostics.router, ...)`와 같이 **`routers/`** 폴더에서 정의된 라우터들을 등록합니다. |
| **WS 핸들러 변경** | **1단계 인증 통합:** 쿼리 파라미터 토큰을 받아 `security/jwt_validator.py`를 호출하여 인증합니다. |
| **WS 핸들러 변경** | **2단계 메타데이터 수신:** WebSocket `while True` 루프 내에서 `"meta"` 타입의 JSON 메시지를 처리하고, `session_id`에 연결된 `collector` 객체에 단어 이름, PK를 저장합니다. |

#### 2.2. `routers/internal.py` (신규 파일)

**역할:** `APIRouter`를 사용하여 **내부 통신용** 엔드포인트 (`/internal/*`)를 정의합니다.

| 구현 내용 | 설명 |
| :--- | :--- |
| **엔드포인트 정의** | `@router.post("/save-quiz")`, `@router.post("/save-learning")` 구현. |
| **인증 적용** | `security.jwt_validator.get_current_user_id`를 `Depends`로 사용하여, **내부 API 토큰 검증 로직(`_check_internal_access`)을 대체**하고 강화합니다. |
| **저장 로직** | `storage.s3_db_saver`의 `save_quiz` 및 `save_learning` 함수를 비동기 태스크로 스케줄링합니다. |

---

### 3. 추론 파이프라인 모듈 구현 가이드

이 영역은 **4단계: 실시간 영상 스트리밍 및 추론**을 담당하며, 이미 제공된 파일들이 잘 분리되어 있습니다.

#### 3.1. `inference_pipeline.py` (제공됨)

**역할:** 프레임 수집 및 추론 오케스트레이션을 담당합니다.

| 주요 구성요소 | 역할 |
| :--- | :--- |
| **`SequenceCollector`** | DataChannel에서 수신된 프레임 바이트를 시간 및 프레임 수 제한에 맞춰 수집합니다. |
| **`run_inference`** | 수집된 프레임 리스트를 받아 `extract_sequence_from_frames`와 `predictor.predict`를 순차적으로 호출합니다. |
| **`schedule_quiz_save`** | 추론 완료 후 `storage.s3_db_saver.save_quiz`를 호출하여 백그라운드 저장을 스케줄링합니다. |

#### 3.2. `processing/landmark_extractor.py` (제공됨)

**역할:** 프레임 바이트를 **ML 모델이 사용할 수 있는 랜드마크 시퀀스 (T, 147)**로 변환합니다.

| 주요 구성요소 | 역할 |
| :--- | :--- |
| **`RealtimeLandmarkExtractor`** | `mediapipe.Holistic` 모델을 사용하여 단일 프레임에서 포즈, 손 랜드마크를 추출하고 정규화합니다. |
| **`extract_sequence_from_frames`** | 여러 프레임 바이트를 연속적으로 처리하여 최종 시퀀스를 생성하고 패딩/자르기를 수행합니다. |

#### 3.3. `processing/predictor.py` (제공됨)

**역할:** 로드된 `CNN_BiLSTM_Attention` 모델을 사용하여 랜드마크 시퀀스로부터 최종 예측(`predicted_label`, `score`)을 수행합니다.

---

### 4. 저장소 모듈 구현 가이드

이 영역은 **4.5. 백그라운드 저장**을 지원합니다.

#### 4.1. `storage/s3_db_saver.py` (제공됨)

**역할:** `storage_interface.py`에서 정의된 인터페이스를 구현하며, 실제로는 S3와 DB에 데이터를 저장하는 로직이 들어갑니다.

| 주요 구성요소 | 역할 |
| :--- | :--- |
| **`async def save_learning(...)`** | 학습용 데이터를 저장합니다. |
| **`async def save_quiz(...)`** | 퀴즈 결과를 저장합니다. |

> **참고:** `storage/storage_interface.py`는 추상적인 정의를 제공하며, `inference_pipeline.py`와 `routers/internal.py`는 이 인터페이스를 참조하여 결합도를 낮춥니다.

---

### 5. 제거 및 유지 대상 파일

| 파일 | 처리 | 이유 |
| :--- | :--- | :--- |
| **`api_client.py`** | **제거** | 2단계 워크플로우 변경으로 Spring API 조회(REST GET)가 제거되었으므로, 이 더미 클라이언트는 더 이상 필요하지 않습니다. |
| **`smoke_test.py`** | **유지/업데이트** | `internal/*` 엔드포인트의 경로를 `/api/internal/*`로 변경하고, `/simulate/*` 엔드포인트를 `routers/diagnostics.py`로 이동했다면 해당 경로를 반영하여 업데이트해야 합니다. |
| **`requirements.txt`** | **유지** | 모든 모듈의 의존성(PyTorch, mediapipe, aiortc 등)이 명시되어 있으므로 그대로 유지합니다. |
| **`schemas.py`** | **유지** | Pydantic 모델 정의는 REST 엔드포인트의 입력/출력 유효성 검사에 필수적이므로 유지합니다. |
















제공해주신 모든 파일과 워크플로우를 분석하여, 프로젝트의 목표(실시간 영상 추론 서버) 달성에 **필수적인 모듈**만 남기고, 이전 논의에서 제안된 모듈화 구조를 반영한 **최종 프로젝트 파일 목록**을 정리해 드리겠습니다.

불필요하거나 내용이 없는 파일(`frame_config.py`, `check_toc.py`, `api_client.py`)은 제거하고, 나머지 파일들은 역할에 맞게 새 디렉토리 구조에 배치했습니다.

-----

## 🗑️ 제거된 파일 (불필요/대체됨)

| 파일 이름 | 제거 이유 | 대체 모듈 |
| :--- | :--- | :--- |
| `api_client.py` | **워크플로우 변경:** Spring API를 통한 데이터 조회 단계가 제거되어 더미 클라이언트가 불필요합니다. | N/A (클라이언트가 WS로 직접 데이터 전달) |
| `frame_config.py` | **기능 없음:** 내용이 없으며, 설정은 `configs/settings.py`로 통합됩니다. | `configs/settings.py` |
| `check_toc.py` | **기능 없음:** 내용이 없으므로 제거합니다. | N/A |

## 🎯 최종 프로젝트 디렉토리 구조 및 파일 목록

프로젝트의 핵심 기능인 **WebRTC, AI 추론, JWT 인증, 비동기 저장**을 수행하는 데 필요한 파일 목록입니다.

```
signsense-server/
├── main.py                    # 🗄️ FastAPI 앱 인스턴스, 라우터 등록, WS/WebRTC 시그널링 루프
├── configs/                   # ⚙️ 서버 설정 관리
│   └── settings.py            # 환경 변수, JWT 비밀 키, 상수(TARGET_FRAME_COUNT 등)
├── security/                  # 🔑 인증 및 권한 로직
│   └── jwt_validator.py       # JWT 검증 및 FastAPI Depends 함수
├── routers/                   # 🧭 REST API 엔드포인트 모듈
│   ├── __init__.py            # 라우터 패키지 초기화
│   ├── internal.py            # /api/internal/* (비동기 저장 요청)
│   └── diagnostics.py         # /api/diagnostics/* (상태, 테스트, 시뮬레이션 엔드포인트)
├── processing/                # 🧠 AI 추론 코어 로직
│   ├── __init__.py            # ML/처리 패키지 초기화
│   ├── predictor.py           # ML 모델 구조(`CNN_BiLSTM_Attention`), 로딩(`Predictor`), 예측
│   └── landmark_extractor.py  # `mediapipe` 기반 랜드마크 추출 및 시퀀스 생성
├── storage/                   # 💾 데이터 저장/I/O 로직
│   ├── __init__.py            # 저장소 패키지 초기화
│   ├── storage_interface.py   # 저장 함수들의 인터페이스(추상) 정의
│   └── s3_db_saver.py         # 인터페이스의 더미 구현체 (로컬 파일 저장)
├── inference_pipeline.py      # 🏃‍♂️ 추론 오케스트레이션: `run_inference`, `SequenceCollector`, `schedule_quiz_save`
├── schemas.py                 # 🧾 Pydantic 데이터 모델 (요청/응답/내부 모델)
├── requirements.txt           # 📦 프로젝트 종속성 (fastapi, aiortc, torch, mediapipe 등)
└── smoke_test.py              # 🧪 REST 엔드포인트 통합 테스트 스크립트
```

<br>

-----

## 📑 기존 파일 재배치 및 역할 분할 가이드

| 기존 파일 이름 | 새 경로 / 역할 | 핵심 모듈화 변경 사항 |
| :--- | :--- | :--- |
| **`main.py`** | `main.py` (최소화된 진입점) | **`/internal/*`, `/simulate/*`, `/config` 엔드포인트 로직을 `routers/`로 이동.** WS 핸들러에서 \*\*`security/jwt_validator.py`\*\*를 사용하여 토큰을 검증하고, **2단계 메타데이터 수신** 로직을 추가해야 합니다. |
| **`s3_db_saver.py`** | `storage/s3_db_saver.py` | `storage` 디렉토리로 이동. 이 파일은 `storage_interface.py`의 구현체 역할을 명확히 합니다. |
| **`storage_interface.py`** | `storage/storage_interface.py` | `save_learning`, `save_quiz` 함수의 **추상 인터페이스 정의**를 제공합니다. |
| **`schemas.py`** | `schemas.py` | Pydantic 데이터 모델을 담으며, `routers/internal.py`에서 요청 본문 유효성 검사용으로 사용됩니다. |
| **`predictor.py`** | `processing/predictor.py` | `processing` 디렉토리로 이동. AI 모델 정의 및 예측 핵심 로직을 캡슐화합니다. |
| **`landmark_extractor.py`** | `processing/landmark_extractor.py` | `processing` 디렉토리로 이동. 영상 처리 및 랜드마크 추출 로직을 전담합니다. |
| **`inference_pipeline.py`** | `inference_pipeline.py` | `run_inference`, `SequenceCollector`, `schedule_quiz_save` 등 추론 전체 흐름을 담당하며, `storage` 및 `processing` 모듈들을 연결하는 오케스트레이터 역할을 합니다. |