"""간단한 REST 스모크 테스트 스크립트.

실행 전:
  1) uvicorn main:app --reload (다른 터미널에서 서버 실행)
  2) python smoke_test.py

aiortc 설치 여부와 무관하게 동작 가능한 기본 REST 엔드포인트만 검증.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

BASE = "http://127.0.0.1:8000"


def pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


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
    for path, method in [
        ("/", "GET"),
        ("/health", "GET"),
        ("/model/status", "GET"),
        ("/config", "GET"),
        ("/client", "GET"),  # HTML 테스트 페이지
        ("/simulate/predict", "POST"),
    ]:
        try:
            check(path, method)
        except Exception as e:  # noqa
            print(f"[ERR] {path}: {e}")

    print("완료. WebSocket /ws 및 WebRTC 흐름은 브라우저 /client 에서 수동 확인 필요.")


if __name__ == "__main__":
    main()
