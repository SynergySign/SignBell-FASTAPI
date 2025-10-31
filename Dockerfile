# ai-repo/fastapi/Dockerfile
FROM python:3.10.11

WORKDIR /app

# 시스템 패키지 설치 (Janus 필요 라이브러리 포함)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 환경 변수
ENV PYTHONUNBUFFERED=1
ENV USE_HTTPS=true

EXPOSE 8000 8188 8089

# FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
