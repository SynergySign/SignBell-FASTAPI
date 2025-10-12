from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class SaveLearningRequest(BaseModel):
    session_id: str = Field(..., description="수집된 세션 ID")
    word: str = Field(..., description="학습할 단어(레이블)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터(선택)")


class SaveQuizRequest(BaseModel):
    session_id: str = Field(..., description="수집된 세션 ID")
    predicted: str = Field(..., description="추론된 레이블")
    score: float = Field(..., description="추론 점수(0.0-1.0)")
    timings: Optional[Dict[str, Any]] = Field(None, description="수집/추론 타이밍 정보")
    landmarks_info: Optional[Dict[str, Any]] = Field(None, description="랜드마크/시퀀스 관련 메타정보")


class InferenceResultModel(BaseModel):
    predicted: str
    score: float
    inference_start: float
    inference_end: float
    inference_ms: int
    frames_used: int
    landmarks: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    ok: bool


class ModelStatusResponse(BaseModel):
    predictor_loaded: bool
    model_path: str
    device: str
