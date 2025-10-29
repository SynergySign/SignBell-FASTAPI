# SignBell-FASTAPI

SignBell 프로젝트의 FastAPI 백엔드(모노-레포형) 코드 베이스입니다. 이 README는 로컬 개발 환경 설정과 HTTPS(mkcert) 적용 방법을 포함한 빠른 시작 가이드를 제공합니다.

## 작성자
- [백승현](https://github.com/sirosho)

## 대상 독자
- 로컬에서 개발/테스트용으로 이 FastAPI 서버를 실행하려는 개발자
- 프로젝트에 새로 합류한 팀원

## 저장소 구조(주요 파일)
- `main.py` — FastAPI 앱 진입점 (uvicorn으로 실행)
- `ws_handler.py` — WebSocket 처리기
- `inference_pipeline.py` — 모델 추론 파이프라인 관련 유틸
- `schemas.py` — Pydantic 스키마 정의
- `requirements.txt` — Python 의존성
- `client_test.html` — 간단한 클라이언트 테스트 페이지
- `certs/` — (선택) mkcert로 생성한 인증서를 보관하는 디렉토리
- `README.md` — 이 파일

## 사전 준비
- Python 3.10+ (권장)
- Git
- (선택) mkcert — 로컬에서 신뢰 가능한 개발용 인증서 생성

## 빠른 시작

### 1) 저장소 클론

```cmd
git clone <your-repo-url>
cd SignBell-FASTAPI
```

### 2) 가상환경 생성 및 의존성 설치

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3) 애플리케이션 실행(HTTP, 개발용)

```cmd
# 개발용(HTTPS 없이) - 디버깅용
python main.py
# 또는
uvicorn main:app --reload --port 8000
```

## 로컬 HTTPS(권장: 브라우저에서 WebRTC/WebSocket/카메라 권한 테스트를 위해) — mkcert 사용

개요: mkcert로 생성한 PEM 인증서를 Uvicorn에 전달하면 로컬에서 https/wss를 사용해 서버를 실행할 수 있습니다. 아래는 Windows와 macOS/Linux에서의 단계입니다.

### Windows (PowerShell 관리자 권한 권장)

#### 1. 필수 도구 설치 (한 번만)

```powershell
# PowerShell(관리자)에서
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
choco install mkcert -y
```

#### 2. 로컬 CA 설치 및 인증서 생성

```powershell
mkcert -install
mkdir C:\certs
mkcert -cert-file C:\certs\localhost+1.pem -key-file C:\certs\localhost+1-key.pem localhost 127.0.0.1 ::1
```

#### 3. Uvicorn으로 HTTPS 실행 (cmd.exe에서 실행할 경우)

```cmd
# 예: 포트 8000에서 HTTPS로 실행
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --ssl-keyfile "C:\certs\localhost+1-key.pem" --ssl-certfile "C:\certs\localhost+1.pem" --reload
```

참고: mkcert가 생성한 PEM 파일은 FastAPI(uvicorn)에 그대로 전달하면 됩니다. Spring/Java처럼 keystore 변환은 필요하지 않습니다.

### macOS / Linux

#### 1. mkcert 설치 (Homebrew 예시)

```bash
brew install mkcert
mkcert -install
mkdir -p ~/certs
mkcert -cert-file ~/certs/localhost+1.pem -key-file ~/certs/localhost+1-key.pem localhost 127.0.0.1 ::1
```

#### 2. Uvicorn으로 HTTPS 실행

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --ssl-keyfile "~/certs/localhost+1-key.pem" --ssl-certfile "~/certs/localhost+1.pem" --reload
```

## 운영(또는 다른 개발 서버와의 통합)에서의 팁
- 개발 중 프론트엔드(예: Vite)를 함께 사용하는 경우, 프론트엔드가 `https://localhost:5173` 등에서 동작하도록 mkcert로 인증서를 생성하고 Vite 설정에 해당 경로를 넣어 프록시를 HTTPS 백엔드로 연결하면 원활합니다.
- 프론트엔드-백엔드 간 WebSocket을 사용할 때는 `wss://`를 사용하고, 포트/경로가 일치하는지 확인하세요 (`ws_handler.py`에서 설정 확인).

## 환경 변수 설정
- 이 프로젝트는 루트에 `.env` 파일을 두고 환경변수를 설정하는 것을 권장합니다. `python-dotenv`가 설치되어 있으면 `configs/settings.py`가 자동으로 `.env`를 로드합니다. 설치되어 있지 않으면 OS 환경변수에서 값을 읽습니다.

아래는 `configs/settings.py`에서 사용되는 주요 환경 변수와 예시값입니다 (.env 파일에 추가):

```env
# JWT
JWT_SECRET_KEY=your_strong_jwt_secret_here(백엔드 key와 동일)
JWT_ALGORITHM=HS256

# 추론 / 수집 관련 튜닝
MAX_FRAMES_TO_COLLECT=300 # 최대 수집 프레임 수

# SSL 경로 (mkcert로 생성한 PEM 파일 경로를 지정)
# Windows 기본값(프로젝트 기본 설정과 일치): C:/certs
SSL_CERT_PATH=C:/certs/localhost+1.pem
SSL_KEY_PATH=C:/certs/localhost+1-key.pem

# Cookie 설정
COOKIE_ACCESS_TOKEN_NAME=ACCESS_TOKEN
COOKIE_ACCESS_TOKEN_MAX_AGE=3600

# (선택) smoke test용 환경변수
SMOKE_BASE_URL=http://127.0.0.1:8000
INTERNAL_API_TOKEN=your_internal_api_token_here
```

## 설명(요약)
- `JWT_SECRET_KEY`: JWT 서명에 사용되는 비밀키(중요). 실제 배포 환경에서는 길고 안전한 문자열 사용. 빈 값이면 JWT 사용이 제한될 수 있습니다.
- `JWT_ALGORITHM`: JWT 서명 알고리즘(기본: HS256)
- `TARGET_FRAME_COUNT`, `COLLECTION_DURATION_SECONDS`, `MAX_FRAMES_TO_COLLECT`: 수집 및 추론 관련 파라미터로, 프레임 샘플링/버퍼 크기와 관련됩니다.
- `SSL_CERT_PATH`, `SSL_KEY_PATH`: 로컬 HTTPS 실행 시 사용하는 PEM 파일 경로. mkcert로 생성한 파일 경로를 지정하세요. `configs/settings.py`의 기본값은 Windows의 `C:/certs/localhost+1.pem` 경로를 사용하도록 설정되어 있습니다.
- `COOKIE_ACCESS_TOKEN_NAME`, `COOKIE_ACCESS_TOKEN_MAX_AGE`: 쿠키 기반 토큰 이름과 만료시간(초)

## 주의
- `.env` 파일과 로컬 인증서(예: `C:/certs/*`)는 민감 정보를 포함할 수 있으니 절대 공개 저장소에 커밋하지 마세요. `.gitignore`에 `certs/`와 `.env`를 추가하는 것을 권장합니다.

## 문제 해결
- 브라우저에서 "보안 연결이 신뢰되지 않음" 메시지: mkcert로 생성한 인증서를 신뢰하지 않았거나 시스템 신뢰 저장소에 설치되지 않았습니다. `mkcert -install`을 다시 실행하세요.
- 포트 충돌: 이미 같은 포트를 쓰는 프로세스가 있다면 종료하거나 `--port`로 다른 포트를 지정하세요.
- 인증서 파일을 찾을 수 없음: `--ssl-keyfile`, `--ssl-certfile`에 지정한 경로가 정확한지 확인하세요.

## 테스트
- HTTPS 동작 확인: 브라우저에서 https://127.0.0.1:8000 또는 https://localhost:8000 접속
- WebSocket 테스트: `client_test.html` 파일을 사용하여 wss 연결 시도
- 간단한 smoke 테스트 파일이 있는 경우: `smoke_test.py`를 확인하고 사용

## 보안 주의사항
- mkcert로 생성한 인증서는 로컬 개발 전용입니다. 절대 프로덕션 트래픽에 사용하지 마세요.
- 인증서/키 파일을 저장소에 커밋하지 마세요(`certs/`는 .gitignore에 추가 권장).

## 기여하기
1. 저장소를 Fork
2. 기능 브랜치 생성: `git checkout -b feature/your-feature`
3. 변경 커밋: `git commit -m "Add feature"`
4. Push 및 Pull Request 생성

## 추가 자료
- mkcert: https://github.com/FiloSottile/mkcert
- Uvicorn: https://www.uvicorn.org/

## 문의 및 도움
- 레포지토리 이슈 트래커를 사용해 주세요.


