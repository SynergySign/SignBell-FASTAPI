# SignSense FAST-API Server

본 문서는 수어 인식 FAST API 서버의 환경 설정, 핵심 모듈, REST/WebRTC 엔드포인트 및 프론트엔드/백엔드 연동을 위한 종합 가이드를 제공합니다.

**작성자**: [백승현](https://github.com/sirosho)

**문서 버전:** v1.0

**최종 수정일:** 2025.10.12


FastAPI + WebSocket(WebRTC 시그널링) + aiortc(DataChannel) 기반 수어 인식 서버입니다.
브라우저에서 DataChannel로 전송한 프레임 시퀀스를 수집하여 랜드마크 추출 및 추론을 수행합니다.

요약
- 서버 엔트리: `main.py` (WebSocket 시그널링 + DataChannel 프레임 수집, Predictor 로드)
- 테스트 클라이언트: `client_test.html` (브라우저에서 카메라 캡처 → JPEG → DataChannel 전송)
- 모델 위치: `models/` (예: `cnn_bilstm_attention_model.pth`)
- 전처리/추론 유틸: `processing/` (`landmark_extractor.py`, `predictor.py`)
- 저장 더미/인터페이스: `storage/` (`s3_db_saver.py`, `storage_interface.py`)
- 의존성: `requirements.txt`

주요 동작/구성
- 기본 수집 프레임 길이(TARGET_FRAME_COUNT): 300 (환경변수 `SIGN_SEQUENCE_TARGET_FRAMES`로 오버라이드 가능)
- 수집 기준 시간: 환경변수 `SIGN_SEQUENCE_COLLECTION_SECONDS` (기본 5.0 초)
- DataChannel label: `frames` (브라우저 ↔ 서버가 프레임을 주고받는 채널)

목표 독자
- 프론트엔드(React) 개발자: WebRTC(DataChannel)로 프레임 전송하는 방법
- 백엔드(Spring Java) 개발자: 내부 REST로 학습/저장 호출하는 방법
- 인프라/DevOps: S3 연동 포인트와 환경변수

<!-- TOC -->
## 목차
- [설치 및 실행](#설치-및-실행)
- [핵심 엔드포인트](#핵심-엔드포인트)
- [모듈별 설명 (파일/디렉터리)](#모듈별-설명-파일디렉터리)
- [React(프론트엔드)에서 연결하는 방법](#react프론트엔드에서-연결하는-방법)
- [Spring(Java) 서버에서 연동하는 방법](#springjava-서버에서-연동하는-방법)
- [AWS S3 연동(저장)](#aws-s3-연동저장)
- [테스트 및 검증](#테스트-및-검증)
- [디버깅 / 자주 묻는 문제](#디버깅-자주-묻는-문제)
- [향후 작업(권장)](#향후-작업권장)
- [라이선스 & 참고 문서](#라이선스-및-참고-문서)
<!-- /TOC -->

## 설치 및 실행

---

공통 전제
- Python 3.10 이상 권장
- 프로젝트 루트에 다음 항목이 있어야 함: `main.py`, `requirements.txt`, `models/` (모델 파일), `certs/`(선택)
- 권한: 로컬에서 카메라를 사용하는 브라우저 테스트의 경우 HTTPS가 필요할 수 있습니다.

1) Windows (cmd.exe) — 권장 개발 절차

- 가상환경 생성 및 활성화 (cmd.exe):
```cmd
python -m venv .venv
.venv\Scripts\activate
```

- 의존성 설치:
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- (선택) GPU용 torch는 시스템/CUDA 버전에 따라 별도 설치 권장:
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

- 서버 실행 (HTTP, 개발용):
```cmd
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- HTTPS(로컬 인증서 사용, WebRTC 테스트 권장):
```cmd
uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs\key.pem --ssl-certfile certs\cert.pem
```

2) macOS / Linux (bash / zsh)

- 가상환경 생성 및 활성화:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

- 의존성 설치:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- GPU/torch 설치 예(필요할 경우):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

- 서버 실행 (HTTP):
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- HTTPS(로컬 인증서 사용):
```bash
uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
```

주: macOS에서 브라우저가 카메라 권한을 HTTPS로만 허용하는 경우가 많으므로 로컬 테스트 시 인증서 준비를 권장합니다(`mkcert` 사용 가능).

환경변수 요약
- SIGN_SEQUENCE_TARGET_FRAMES (int): 모델 입력 시퀀스 길이. 기본 300
- SIGN_SEQUENCE_COLLECTION_SECONDS (float): 수집 지속시간(초). 기본 5.0
- SIGN_EXTRACT_LANDMARKS (0/1): 서버 측에서 프레임→랜드마크 추출 수행 여부
- INTERNAL_API_TOKEN: 내부 엔드포인트 보호용 토큰(운영 시 권장)

---

## 핵심 엔드포인트

---

- GET  /                 : 상태 확인
- GET  /health           : 헬스 체크(predictor 로드 상태)
- GET  /model/status     : 모델 로드 상태, 경로, 디바이스 정보
- GET  /config           : 서버 기본 설정 (TARGET_FRAME_COUNT 등)
- GET  /client           : `client_test.html` 반환 (테스트용)
- WS   /ws/{session}     : WebSocket 시그널링(Offer/Answer)
- POST /simulate/predict: 더미 프레임으로 동작하는 REST 추론 시뮬레이터
- POST /internal/save-learning: 내부(개발용) 학습 데이터 저장 스케줄 엔드포인트
- POST /internal/save-quiz    : 내부(개발용) 퀴즈 결과 저장 스케줄 엔드포인트

---

## 모듈별 설명 (파일/디렉터리)

---

- `main.py`:
  - FastAPI 앱 엔트리포인트입니다. WebSocket 시그널링(offer/answer) 및 DataChannel을 통해 들어오는 프레임을 수집합니다.
  - `SequenceCollector`와 `run_inference` 호출 지점이 있으며, 추론 완료 시 WebSocket으로 결과를 전송합니다.
  - 내부 저장 호출(`/internal/save-learning`, `/internal/save-quiz`)을 백그라운드 작업으로 스케줄합니다.

- `processing/landmark_extractor.py`:
  - MediaPipe Holistic 기반의 랜드마크 추출기입니다. 프레임(이미지) 시퀀스를 받아 프레임 당 147차원 특징 벡터로 변환합니다.
  - 서버에서 `SIGN_EXTRACT_LANDMARKS=1`로 활성화할 수 있습니다.

- `processing/predictor.py`:
  - 모델 로드 및 추론을 담당합니다. `get_predictor()`가 모델을 로드하며, 기본 모델 파일은 `models/cnn_bilstm_attention_model.pth`입니다.
  - 모델이 없으면 FileNotFoundError를 발생시킵니다.

- `inference_pipeline.py`:
  - `main.py`의 `run_inference` 및 `SequenceCollector` 일부를 별도 모듈로 분리하기 위한 설계 파일입니다(유닛테스트/재사용성 향상).

- `storage/`:
  - `storage_interface.py`: 저장 인터페이스(추상) 정의. S3/DB 연동을 교체 가능한 형태로 구현하도록 설계되어야 합니다.
  - `s3_db_saver.py`: 개발용 더미 저장 구현. 로컬 `data/storage_exports/`에 학습/퀴즈 데이터를 저장합니다.

- `api_client.py`:
  - 프로젝트 외부(예: Spring 서버)와 통신할 때 사용할 내부 REST 클라이언트 모듈(더미 구현 또는 실제 구현 확장 가능).

- `client_test.html`:
  - 브라우저에서 카메라를 캡쳐하고 JPEG를 DataChannel로 전송하는 테스트 페이지입니다.

- `smoke_test.py`:
  - REST 엔드포인트(비-RTC 부분)를 검증하는 간단한 스모크 테스트 스크립트입니다.

---

## React(프론트엔드)에서 연결하는 방법

---
목적: React 앱(브라우저)이 카메라를 캡처하여 DataChannel을 통해 서버에 프레임을 전송하고, 서버의 추론 결과를 수신합니다.

사전 준비
- HTTPS로 서비스 중(로컬의 경우 `uvicorn`에 `--ssl-*` 옵션 사용 권장)
- 브라우저: 최신 Chrome/Firefox 권장
- DataChannel label: 'frames' (서버가 동일하게 수신합니다)

핵심 로직 요약
1. RTCPeerConnection 생성
2. DataChannel 생성(`pc.createDataChannel('frames')`)
3. (선택) 로컬 비디오 트랙을 pc에 추가(예: preview용)
4. Offer 생성 및 WebSocket(`/ws/{session}`)으로 전송
5. 서버로부터 Answer 수신 후 setRemoteDescription
6. DataChannel open 이벤트에서 캡처한 JPEG를 ArrayBuffer로 변환해 전송
7. 필요 시 signaling WebSocket에 `{action: 'flush'}`를 전송하여 서버에 즉시 추론 트리거

React 예제 (핵심 부분만, 컴포넌트에 맞게 조정)

```javascript
// ... React 컴포넌트 내부 예
const pc = new RTCPeerConnection();
const ws = new WebSocket('wss://localhost:8443/ws/demo-session');
const dc = pc.createDataChannel('frames');

ws.onopen = () => console.log('signaling WS open');
ws.onmessage = async (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === 'answer') {
    await pc.setRemoteDescription(msg);
  } else if (msg.action === 'inference_result') {
    console.log('inference_result', msg.result);
  }
};

pc.onicecandidate = ({candidate}) => {
  // 트리클 ICE 사용하지 않을 경우 후보 포함된 offer만 보내도 됩니다
  if (candidate) return; // 트리클을 사용하지 않는 간단한 구현
};

async function startOffer() {
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  // ICE가 수집된 후(간단 구현: 바로 전송)
  ws.send(JSON.stringify({ action: 'offer', sdp: offer.sdp, type: offer.type }));
}

// DataChannel에서 프레임 전송 예 (canvas.toBlob 사용 시)
async function sendFrame(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  dc.send(arrayBuffer);
}

// flush 요청 예
function flush() {
  ws.send(JSON.stringify({ action: 'flush' }));
}
```

실전 팁
- JPEG Quality: 0.6~0.8 권장(네트워크/서버 부하 고려)
- 프레임 속도: 최소 초당 20fps 권장(권장 전송 범위 20~25fps). 클라이언트에서 캡처를 30fps로 설정하더라도 전송은 20~25fps로 쓰로틀링(throttling)하거나 필요시 프레임 드롭을 적용하는 것을 권장합니다(실운영에서는 네트워크/서버 상황에 따라 실제 전송률이 21.xfps처럼 낮게 측정될 수 있습니다).
- 긴 시퀀스: 프레임을 chunking/프레임 넘버를 붙여 전송하면 순서 보장이 쉬워집니다

프레임률 관련 권장 설정 및 계산 예시
- 권장 전송 FPS 범위: 20 ~ 30 fps (실전에서는 네트워크/서버 상태에 따라 20~30fps 사이에서 조절)
- 최대 프레임(제한): `SIGN_SEQUENCE_TARGET_FRAMES` 기본값은 300입니다. 수집 동안 전송되는 총 프레임 수는 이 값을 넘지 않아야 합니다.

기본 계산식
- 목표 전송 FPS를 F (fps), 수집 유지시간을 T (초), 최대 허용 프레임을 M (`SIGN_SEQUENCE_TARGET_FRAMES`, 기본 300)이라 하면
  - 전송할 총 프레임 수 = F * T
  - 반드시 F * T <= M 를 만족하도록 설정하세요.

주요 유도식
- 주어진 F에서 허용 가능한 최대 수집시간: T_max = floor(M / F)
- 주어진 T에서 허용 가능한 최대 FPS: F_max = floor(M / T)

간단 예시
- M=300, F=20fps → T_max = floor(300 / 20) = 15초 (즉 15초 이상으로 설정하면 300프레임을 넘음)
- M=300, F=30fps → T_max = floor(300 / 30) = 10초
- T=5초로 고정하고 F를 선택하면: F_max = floor(300 / 5) = 60fps (그러나 권장 범위는 20~30fps이므로 실전에서는 20~30fps로 선택; 예: 20fps → 총 100프레임, 30fps → 총 150프레임)

권장 운영 지침
- 지연(레이턴시) 우선: T(수집시간)를 작게 유지(예: 5초)하고 F는 20~30fps 범위에서 선택 → 총 프레임은 보통 100~150 범위
- 정확도(시퀀스 길이) 우선: M=300을 유지하려면 T를 늘려야 함(예: F=20fps일 때 T=15초)
- 캡처 vs 전송 분리: 클라이언트에서 캡처는 30fps로 유지해도 되지만, 전송은 네트워크 상황에 맞춰 20~30fps로 쓰로틀링하고 `F * T <= M` 조건을 지키세요

실전 팁 (요약)
- 항상 F * T <= M를 확인하여 과다 전송을 방지하세요.
- 권장 전송 FPS 범위는 20~30fps입니다. (이 범위를 기본으로 테스트/조정하세요.)
- 운영 시에는 수집시간(`SIGN_SEQUENCE_COLLECTION_SECONDS`) 또는 목표 프레임(`SIGN_SEQUENCE_TARGET_FRAMES`) 중 하나를 조정해 시스템 요구(지연/정확도)에 맞춥니다.

---

## Spring(Java) 서버에서 연동하는 방법

---

목적: Spring 애플리케이션이 FastAPI 내부 API를 호출해 특정 세션의 수집 데이터 저장(학습/퀴즈) 또는 상태 조회를 수행합니다.

권장 방식
- 내부 엔드포인트는 `INTERNAL_API_TOKEN`으로 보호하세요.
- HTTPS 및 네트워크 레벨 접근 제어(방화벽)를 구성하세요.

간단한 Java(RestTemplate) 예제 - `/internal/save-learning` 호출

```java
// RestTemplate 예제
RestTemplate rest = new RestTemplate();
String url = "https://fastapi-host:8443/internal/save-learning";
HttpHeaders headers = new HttpHeaders();
headers.setContentType(MediaType.APPLICATION_JSON);
headers.setBearerAuth(System.getenv("INTERNAL_API_TOKEN"));
Map<String, Object> body = new HashMap<>();
body.put("session_id", "testsesson_12345");
body.put("meta", Map.of("user_id", "u1", "label", "hello"));
HttpEntity<Map<String, Object>> req = new HttpEntity<>(body, headers);
ResponseEntity<String> resp = rest.postForEntity(url, req, String.class);
System.out.println(resp.getStatusCode() + ": " + resp.getBody());
```

상태 조회 예 - `/model/status`

```java
ResponseEntity<String> status = rest.getForEntity("https://fastapi-host:8443/model/status", String.class);
System.out.println(status.getBody());
```

연동 팁
- 호출 타임아웃/재시도 정책을 적용하세요 (Spring Retry 또는 WebClient 사용 권장)
- 내부 API 응답을 폴링하지 말고 비동기 콜백/메시지 큐로 결과를 처리하도록 설계하면 확장성이 좋습니다

---

## AWS S3 연동(저장)

---

목표: 저장 파이프라인()을 S3에 업로드하고 메타정보를 DB에 기록합니다.

접점 모듈
- `storage/storage_interface.py` — 저장 인터페이스(추상)
- `storage/s3_db_saver.py` — 현재 더미 구현(로컬 저장). 운영 시 S3 업로드 로직으로 교체합니다.

권장 구현 순서
1. AWS 자격증명 준비(IAM 사용자/역할, 액세스 키 또는 인스턴스 역할)
2. `boto3` 설치: `pip install boto3` (requirements에 추가 권장)
3. `storage/s3_db_saver.py`에 S3 업로드 코드 추가(예: `upload_fileobj` 또는 `put_object`)
4. 업로드 후 DB(RDS 등)에 메타정보(세션ID, user_id, label, s3_key, timestamp) 기록

파이썬 boto3 예제 (README용 참고 코드)

```python
import os
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

BUCKET = os.getenv('S3_BUCKET_NAME')

def upload_bytes(data_bytes: bytes, key: str, content_type: str = 'application/octet-stream') -> str:
    try:
        s3.put_object(Bucket=BUCKET, Key=key, Body=data_bytes, ContentType=content_type)
        return f's3://{BUCKET}/{key}'
    except ClientError as e:
        raise

# 사용 예
# s3_key = f'learning/{session_id}/frames.npy'
# upload_bytes(numpy_bytes, s3_key, 'application/octet-stream')
```

환경변수 예시 (설정 필요)
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION
- S3_BUCKET_NAME

보안 주의
- 액세스 키를 코드/레포지토리에 커밋하지 마세요. 권한이 최소가 되도록 IAM 정책을 구성하세요.
- 운영 환경에서는 인스턴스 프로파일(EC2/ECS task role) 또는 AWS Secrets Manager 사용을 권장합니다.

---

## 테스트 및 검증

---
- REST 스모크 테스트: `python smoke_test.py` — non-RTC REST 엔드포인트 검증
- 클라이언트 테스트: 브라우저에서 `https://localhost:8443/client` 또는 `https://localhost:8000/client` 접속
- HTTPS가 준비되지 않았으면 REST 부분만으로 검증 가능

간단 체크리스트
- [ ] `.venv` 활성화 후 `pip install -r requirements.txt` 실행
- [ ] (옵션) `pip install boto3` — S3 연동 필요 시
- [ ] `models/`에 모델 파일 배치(배포/모델 추론 테스트용)
- [ ] `certs/`에 로컬 인증서(HTTPS 테스트용) 배치

---

## 디버깅 / 자주 묻는 문제

---
- SSL 인증서가 없으면 `main.py`가 자동으로 HTTP로 실행됩니다. WebRTC/카메라 동작 테스트는 HTTPS 환경을 권장합니다.
- 모델 로드 에러: `processing/predictor.py`에서 모델 경로(`models/cnn_bilstm_attention_model.pth`)와 checkpoint 키(예: `model_state_dict`)를 확인하세요.
- MediaPipe/Opencv import 오류: 해당 패키지 설치 필요. MediaPipe는 플랫폼별 의존성이 있으므로 문서 확인 후 설치하세요.
- Internal endpoints 보호: `INTERNAL_API_TOKEN`을 설정해 내부 엔드포인트 접근을 제한하세요.

---

## 향후 작업(권장)

---
- `inference_pipeline.py`로 `run_inference` 분리 및 유닛테스트 추가
- DataChannel chunking / 순번 프로토콜 개선
- TURN 서버(coturn) 연동(브라우저-서버 직접 통신이 어려운 환경 대비)
- S3/DB 실제 연동 구현 및 권한 정책 정리

---

## 라이선스 및 참고 문서

라이선스
- 현재 PoC 단계(라이선스 미지정). 배포 시 적절한 라이선스를 추가하세요.

참고 문서
- `docs/개발진행현황.md`, `docs/테스트가이드.md`, `docs/모델로드방식.md`, `docs/프로젝트 모듈화 및 연동 계획.md`

문의 및 기여
- 이 저장소의 이슈 트래커에 버그/요청을 남겨주세요. 작은 개선(문서, 테스트, 버그픽스)은 PR 환영합니다.




[목차로 돌아가기](#목차)