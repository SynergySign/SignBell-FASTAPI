"""
모듈: schemas.py
설명:
- API 요청/응답 스키마(Pydantic 모델)를 정의합니다.
- 내부 엔드포인트에서 사용하는 저장 요청(payload)과 서버 상태 응답 모델들을 포함합니다.

모델:
- SaveLearningRequest: 학습 데이터 저장 요청 페이로드
- SaveQuizRequest: 퀴즈(추론) 결과 저장 요청 페이로드
- InferenceResultModel: 추론 결과 응답 모델
- HealthResponse: 헬스 체크 응답 모델
- ModelStatusResponse: 모델 상태 응답 모델

since: 2025.10.17
author: 백승현
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class SaveLearningRequest(BaseModel):
    """학습용 데이터 저장 요청.

    필드:
    - session_id: 수집된 세션 ID
    - word: 학습할 단어(레이블)
    - metadata: 추가 메타데이터(선택)
    """
    session_id: str = Field(..., description="수집된 세션 ID")
    word: str = Field(..., description="학습할 단어(레이블)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터(선택)")


class SaveQuizRequest(BaseModel):
    """퀴즈(추론) 결과 저장 요청.

    필드:
    - session_id: 수집된 세션 ID
    - predicted: 추론된 레이블
    - score: 추론 점수(0.0-1.0)
    - timings: 수집/추론 타이밍 정보(선택)
    - landmarks_info: 랜드마크/시퀀스 관련 메타정보(선택)
    """
    session_id: str = Field(..., description="수집된 세션 ID")
    predicted: str = Field(..., description="추론된 레이블")
    score: float = Field(..., description="추론 점수(0.0-1.0)")
    timings: Optional[Dict[str, Any]] = Field(None, description="수집/추론 타이밍 정보")
    landmarks_info: Optional[Dict[str, Any]] = Field(None, description="랜드마크/시퀀스 관련 메타정보")


class InferenceResultModel(BaseModel):
    """추론 결과를 표현하는 응답 모델."""
    predicted: str
    score: float
    inference_start: float
    inference_end: float
    inference_ms: int
    frames_used: int
    landmarks: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """헬스 체크 응답 모델."""
    ok: bool


class ModelStatusResponse(BaseModel):
    """모델 로드 상태 응답 모델."""
    predictor_loaded: bool
    model_path: str
    device: str
