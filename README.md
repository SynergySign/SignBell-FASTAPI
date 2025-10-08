# SignSense Test Server

FastAPI + WebSocket(WebRTC 시그널링) + aiortc(DataChannel) 기반 수어 인식 프로토타입 서버.
수신한 프레임 시퀀스를 더미 추론(`dummy_infer`)으로 처리하여 랜덤 라벨을 반환합니다.

## 개발 진행 현황
상세 진행 상황 / 로드맵은 `docs/개발진행현황.md` 문서를 참조하세요.

▶ 테스트/검증 절차와 명령어는 `docs/테스트가이드.md` 에 정리되어 있습니다.

## 구성 요소
- `main.py`: 서버 엔트리. 모델 로딩 스텁 + WebSocket 시그널링 + DataChannel 프레임 수집.
- `client_test.html`: 브라우저(WebRTC) 테스트용 프론트엔드. 카메라 캡처 → JPEG 변환 → DataChannel 전송.
- `models/`: (선택) PyTorch / ONNX 모델 파일 위치.
- `docs/`: 모델 로딩 방식, 반응 속도 테스트 설계 문서.
- `requirements.txt`: 필수/선택 패키지 목록.

## 설치 & 실행 (Windows 예시)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# (GPU / 실제 추론 필요시) pip install torch --index-url https://download.pytorch.org/whl/cu121
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 엔드포인트
| 경로 | 메서드 | 설명 |
|------|--------|------|
| `/` | GET | 상태 확인 + 목표 프레임 수 |
| `/health` | GET | 헬스 체크 |
| `/model/status` | GET | 모델 로딩 상태 (더미/실제) |
| `/config` | GET | (클라이언트 구동 참고) 목표 프레임 / 라벨 목록 |
| `/client` | GET | 테스트 클라이언트 HTML 로드 |
| `/ws` | WS | WebRTC 시그널링 (Offer/Answer, DataChannel 프레임 처리) |
| `/simulate/predict` | POST | 더미 프레임으로 즉시 추론 (성능 측정 용이) |

## WebRTC 시그널링 & DataChannel 흐름
1. 브라우저 `client_test.html`에서 RTCPeerConnection 생성 후 Offer SDP 로컬 생성.
2. ICE Gathering complete 시 `/ws` WebSocket에 `{action:'offer', sdp}` 전송.
3. 서버가 Answer 생성 후 `{action:'answer', sdp}` 반환.
4. DataChannel(`frames`) 오픈.
5. 브라우저가 주기적(Frame Interval)으로 JPEG Blob → ArrayBuffer 변환 → DataChannel 전송.
6. 서버는 `TARGET_FRAME_COUNT` 도달하면 더미 추론 후 WebSocket 으로 `{type:'result'}` JSON 응답.

(현재 Trickle ICE 미사용: Answer 전송 시 SDP 에 후보 ICE 포함)

## 테스트 클라이언트 사용
1. 서버 실행 후 브라우저에서: `http://localhost:8000/client` 접속.
2. `Connect` 클릭 → WebSocket + WebRTC 연결 → DataChannel open 로그 확인.
3. `Start Capture` 클릭 → 프레임 전송 시작.
4. 목표 프레임(기본 60) 도달 후 서버 추론 결과 수신.
5. 필요 시 `Flush(강제결과)` 버튼으로 조기 처리.

### 파라미터
- Frame Interval(ms): 전송 간격 (Latency vs Bandwidth 트레이드오프 조절)
- JPEG Quality: 전송 크기 최적화 (0.6~0.8 권장)
- Target Frames: 시퀀스 길이 (실 운용 계획 9~10초 ≈ FPS * seconds)

## HTTPS (로컬 개발)
WebRTC 영상/카메라 접근은 HTTPS 가 권장됩니다.
`mkcert` 로 인증서 발급 후:
```bash
uvicorn main:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile .cert/key.pem --ssl-certfile .cert/cert.pem
```

## 실제 모델 통합 TODO
| 항목 | 설명 | 우선순위 |
|------|------|----------|
| MediaPipe 파이프라인 | Landmark 추출 + 좌표 normalization | High |
| 전처리 유닛 | Landmark → 시퀀스 텐서 변환 | High |
| Torch / ONNX 런타임 | CPU/GPU 선택, 세션 캐싱 | High |
| Chunking 프로토콜 | 헤더(순번, 길이, 종료) 정의 → 재조립 안정화 | Medium |
| 성능 모니터링 | 처리 시간(추론, 수신), 메모리, fps | Medium |
| 에러/타임아웃 | 프레임 누락, DataChannel 끊김 처리 | Medium |
| TURN 서버 | NAT traversal 안정화 (coturn) | Medium |

## 실시간 랜드마크 전처리 (훈련 coordinate_extractor 재현)
환경 변수 `SIGN_EXTRACT_LANDMARKS=1` 로 활성화할 수 있는 선택 기능입니다. 활성화 시 `dummy_infer` 이전에 프레임 바이트 배열을 MediaPipe Holistic 으로 처리하여 학습 시 사용한 특징 구조(추정)를 재현하려 합니다.

### 특징 구성 (Frame Feature Vector)
- Pose 상체 관절 6개: indices (11,12,13,14,15,16)
- Left Hand: MediaPipe Hands 21 포인트 (wrist 포함)
- Right Hand: MediaPipe Hands 21 포인트 (wrist 포함)
- 양 손목 간 상대 3D 벡터 1개 (Right - Left)
- 총 포인트: 49 → 각 (x,y,z) → 49 * 3 = 147 차원

### 정규화 / 스케일링 규칙
1. 안정 중심(stable_center) = ((좌/우 어깨 평균) + 코) / 2
2. 모든 pose/손목 좌표: `(coord - stable_center) / 0.3`
3. Pose Z, 손목 Z: 추가 감쇠 계수 0.7
4. 손가락 포인트: 손목 기준 상대좌표 `(finger - wrist)/0.3`, Z 감쇠 0.6
5. 양 손목 모두 검출 시 손목 간 (dx,dy,dz) 추가, 아니면 (0,0,0)
6. 누락된 손 또는 포인트는 0 패딩

> 주의: 실제 학습에 사용된 원본 `coordinate_extractor` 와 완전 일치 보장을 위해서는 학습 스크립트의 원본 코드를 대조해야 합니다. 현재 구현은 일반적인 수어 포즈 정규화 패턴과 제공된 설명을 기반으로 한 추정입니다.

### skip_missing 옵션
`RealtimeLandmarkExtractor(skip_missing=...)`
- `True`: Pose 가 검출되지 않는 프레임은 특징에서 제외 (학습 때 동일 동작이면 선택)
- `False`(기본): 0 벡터를 유지하여 시퀀스 길이 고정에 유리

### Enable 방법 (Windows 예)
```bash
set SIGN_EXTRACT_LANDMARKS=1
uvicorn main:app --reload
```

### 추가 설치 필요 패키지
`requirements.txt` 에 주석 처리된 항목을 설치:
```bash
pip install mediapipe==0.10.11 opencv-python numpy
```
(GPU 추론이 필요하다면 별도로 torch 설치)

### 응답 예시 (`/simulate/predict` 또는 WebSocket 결과 내 inference.landmarks)
```json
{
  "predicted": "안녕하세요",
  "score": 0.91,
  "landmarks": {
    "enabled": true,
    "seq_shape": [60,147],
    "feature_dim": 147
  }
}
```
오류/비활성화 시:
```json
"landmarks": { "enabled": false }
```
또는 `{ "enabled": true, "error": "..." }` 형식.

### 차후 개선 포인트
- 학습 원본 코드 비교 후 damping 계수/scale 재조정
- visibility / handedness 추가 여부 결정
- outlier smoothing (EMA) / temporal interpolation
- 성능 최적화를 위한 MediaPipe GPU delegate 고려

## 더미 추론 구조
`dummy_infer(frames)`:
- 50ms sleep 모의 지연
- 라벨 랜덤 선택 + 0.75~0.99 점수
- 사용 프레임 수 / 시간 메타데이터 포함

## 간단 스모크 테스트
환경: aiortc 미설치 상태에서도 비추론 REST 는 동작.
```bash
python smoke_test.py
```

## 개발 편의
- 환경 변수 `SIGN_SEQUENCE_TARGET_FRAMES` 로 수집 프레임 수 변경 가능.
- 에디터 경고(IDE) 중 aiortc 관련 객체 호출 불가 경고는 패키지 설치 시 해소.

## 라이선스
내부 PoC 단계 (라이선스 미지정). 배포 전 명시 필요.

## 차후 확장 아이디어
- DataChannel 과 별개로 WebRTC 영상 Track 직접 수신 + 서버 측 프레임 디코딩
- Frame Delta 전송(압축률 향상)
- gRPC 스트리밍 대안 비교 벤치마킹
