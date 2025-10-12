"""
Dummy S3/DB saver module
- 실제 S3/DB 연동 없이 로컬 더미 동작으로 저장을 흉내냄
- 비동기 함수 `save_quiz`와 `save_learning`을 제공
"""
import os
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)


async def save_quiz(frames: List[bytes], inference_result: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    """퀴즈 추론 결과와 프레임을 로컬에 더미로 저장합니다."""
    try:
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')
        fname = f"quiz_{session_id or 'anon'}_{ts}.json"
        path = os.path.join(DATA_DIR, fname)

        # 프레임 자체는 바이너리라서 저장하지 않고 메타데이터만 저장합니다.
        payload = {
            'session_id': session_id,
            'timestamp': ts,
            'inference_result': inference_result,
            'frames_count': len(frames) if frames is not None else 0,
        }

        # I/O를 흉내내기 위해 짧게 대기
        await asyncio.sleep(0.01)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return {'ok': True, 'path': path}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


async def save_learning(frames: List[bytes], meta: Dict[str, Any]) -> Dict[str, Any]:
    """개인 학습 데이터 저장 더미 구현"""
    try:
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')
        fname = f"learning_{meta.get('session_id','anon')}_{ts}.json"
        path = os.path.join(DATA_DIR, fname)

        payload = {
            'meta': meta,
            'frames_count': len(frames) if frames is not None else 0,
        }

        await asyncio.sleep(0.01)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return {'ok': True, 'path': path}
    except Exception as e:
        return {'ok': False, 'error': str(e)}

