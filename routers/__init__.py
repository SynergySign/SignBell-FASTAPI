"""
모듈: routers 패키지 초기화
설명:
- API 라우터 모듈들을 집계하는 패키지 초기화 파일입니다.
- 하위 라우터(`internal`, `diagnostics`)를 import 하여 앱에서 사용할 수 있도록 노출합니다.

since: 2025.10.17
author: 백승현
"""

from . import internal, diagnostics

__all__ = ["internal", "diagnostics"]

