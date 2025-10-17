"""간단한 REST 스모크 테스트 스크립트.

실행 전:
  1) uvicorn main:app --reload (다른 터미널에서 서버 실행)
  2) python smoke_test.py

aiortc 설치 여부와 무관하게 동작 가능한 기본 REST 엔드포인트만 검증.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx

BASE = os.getenv("SMOKE_BASE_URL", "http://127.0.0.1:8000")
TOKEN = os.getenv("INTERNAL_API_TOKEN")


def pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def auth_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    return headers


def check(path: str, method: str = "GET"):
    url = BASE + path
    if method == "GET":
        r = httpx.get(url, timeout=5)
    else:
        r = httpx.post(url, timeout=10)
    print(f"[{method}] {path} -> {r.status_code}")
    ctype = r.headers.get("content-type", "")
    try:
        if "application/json" in ctype:
            print(pretty(r.json()))
        else:
            # HTML 등은 길이만 출력
            print(f"<non-json len={len(r.text)}> 첫 120자: {r.text[:120]!r}")
    except Exception:
        print(r.text[:200])
    print("-" * 60)


def main():
    print("=== SignSense REST Smoke Test ===")
    print(f"BASE={BASE} TOKEN={'SET' if TOKEN else 'NONE'}")

    # 기본 엔드포인트들
    for path, method in [
        ("/", "GET"),
        ("/health", "GET"),
        ("/model/status", "GET"),
        ("/config", "GET"),
        ("/client", "GET"),  # HTML 테스트 페이지
        ("/simulate/predict", "POST"),
        ("/api/diagnostics/status", "GET"),
        ("/api/diagnostics/echo", "POST"),
    ]:
        try:
            # diagnostics/echo은 POST이므로 body 포함
            if path == "/api/diagnostics/echo":
                r = httpx.post(BASE + path, json={"msg": "hello"}, timeout=5)
                print(f"[POST] {path} -> {r.status_code}")
                try:
                    print(pretty(r.json()))
                except Exception:
                    print(r.text[:200])
                print("-" * 60)
                continue

            check(path, method)
        except Exception as e:  # noqa
            print(f"[ERR] {path}: {e}")

    # === Internal endpoints (require Bearer token) ===
    if not TOKEN:
        print("Skipping /api/internal/* tests because INTERNAL_API_TOKEN is not set in environment.")
    else:
        try:
            print("[POST] /api/internal/save-quiz")
            # payload matches schemas.SaveQuizRequest: session_id, predicted, score, timings/landmarks_info optional
            payload_quiz = {
                "session_id": "testsession",
                "predicted": "테스트",
                "score": 0.5,
                "timings": {"frame_count": 0},
            }
            r = httpx.post(BASE + "/api/internal/save-quiz", json=payload_quiz, headers=auth_headers(), timeout=10)
            print(f"/api/internal/save-quiz -> {r.status_code}")
            try:
                print(pretty(r.json()))
            except Exception:
                print(r.text[:200])
        except Exception as e:  # noqa
            print(f"[ERR] /api/internal/save-quiz: {e}")
        print("-" * 60)

        try:
            print("[POST] /api/internal/save-learning")
            # payload matches schemas.SaveLearningRequest: session_id, word, metadata optional
            payload_learning = {
                "session_id": "testsession",
                "word": "안녕",
                "metadata": {"source": "smoke_test"}
            }
            r = httpx.post(BASE + "/api/internal/save-learning", json=payload_learning, headers=auth_headers(), timeout=10)
            print(f"/api/internal/save-learning -> {r.status_code}")
            try:
                print(pretty(r.json()))
            except Exception:
                print(r.text[:200])
        except Exception as e:  # noqa
            print(f"[ERR] /api/internal/save-learning: {e}")
        print("-" * 60)

    print("완료. WebSocket /ws 및 WebRTC 흐름은 브라우저 /client 에서 수동 확인 필요.")


if __name__ == "__main__":
    main()
