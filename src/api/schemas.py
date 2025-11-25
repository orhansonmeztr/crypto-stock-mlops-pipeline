"""
Pydantic schemas for the FastAPI application.
Separated from main.py for modularity and reusability.
"""

from typing import Any

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    asset_name: str
    features: list[dict[str, float]]


class PredictionResponse(BaseModel):
    asset_name: str
    prediction: float


class BatchPredictionItem(BaseModel):
    asset_name: str
    features: list[dict[str, float]]


class BatchPredictionRequest(BaseModel):
    requests: list[BatchPredictionItem]


class BatchPredictionResult(BaseModel):
    asset_name: str
    prediction: float | None  # None if failed
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    predictions: list[BatchPredictionResult]
    metadata: dict[str, Any]


class ModelInfoItem(BaseModel):
    asset_name: str
    model_loaded: bool
    registry_name: str | None = None


class ModelInfoResponse(BaseModel):
    models: list[ModelInfoItem]
    total_loaded: int


class MetricsResponse(BaseModel):
    status: str
    metrics: dict[str, Any] | None = None
    message: str | None = None
