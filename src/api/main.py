import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.model_manager import ModelManager
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    MetricsResponse,
    ModelInfoItem,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.utils.config_utils import get_model_registry_name, sanitize_artifact_name

# SETUP
project_root = Path(__file__).resolve().parents[2]
load_dotenv()
config_path = project_root / "configs" / "training_config.yaml"

# LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# HELPERS
def load_assets_from_config() -> list[str]:
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        stocks = config.get("assets", {}).get("stocks", [])
        cryptos = config.get("assets", {}).get("cryptos", [])
        assets = stocks + cryptos
        logger.info(f"Loaded assets from config: {assets}")
        return assets
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return []


# LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Initialize ModelManager, load assets, and load models.
    Shutdown: Clear model cache.
    """
    logger.info("Starting API Lifespan...")

    # Setup MLflow
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Initialize Model Manager
    model_manager = ModelManager()

    # Load Assets
    assets = load_assets_from_config()
    if not assets:
        logger.warning("No assets found in config! API will start but serve no models.")

    model_manager.set_assets(assets)

    # Load Models
    model_manager.load_models()

    # Store in app state
    app.state.model_manager = model_manager
    yield

    logger.info("Shutting down API. Clearing model cache.")
    model_manager.clear()


# SECURITY
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY", "secret-default-key")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key")


# RATE LIMITING
limiter = Limiter(key_func=get_remote_address)

# FASTAPI APP
app = FastAPI(
    title="Crypto-Stock Live Inference API",
    description="Serves predictions using MLflow Champion models. Requires historical context for LSTM.",
    version="3.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ENDPOINTS
@app.get("/health")
def health_check(request: Request):
    model_manager = request.app.state.model_manager
    return {
        "status": "healthy",
        "loaded_models": model_manager.list_loaded_models(),
        "config_assets": model_manager.assets,
    }


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")
def predict(payload: PredictionRequest, request: Request, api_key: str = Depends(get_api_key)):
    asset = payload.asset_name
    model_manager = request.app.state.model_manager

    model = model_manager.get_model(asset)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model for '{asset}' not available.")

    try:
        input_df = pd.DataFrame(payload.features)
        if input_df.empty:
            raise HTTPException(status_code=400, detail="Features list cannot be empty.")

        # HybridModel wrapper handles the logic, unwrap to bypass strict MLflow signature validation
        prediction = model.unwrap_python_model().predict(context=None, model_input=input_df)

        # Handle numpy types
        if isinstance(prediction, (list, pd.Series, np.ndarray)):
            result_value = float(prediction[0]) if len(prediction) >= 1 else 0.0
        else:
            result_value = float(prediction)

        return {"asset_name": asset, "prediction": result_value}

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status codes (e.g. 400 for empty features)
        raise

    except ValueError as ve:
        logger.warning(f"Validation error for {asset}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve)) from ve

    except Exception as e:
        logger.error(f"Prediction error for {asset}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") from e


@app.post("/batch-predict", response_model=BatchPredictionResponse)
@limiter.limit("10/minute")
def batch_predict(
    payload: BatchPredictionRequest, request: Request, api_key: str = Depends(get_api_key)
):
    """
    Batch prediction endpoint for multiple assets.
    Processes all requests and returns results with error handling per asset.
    """
    start_time = time.time()

    model_manager = request.app.state.model_manager
    results = []
    successful = 0
    failed = 0

    for item in payload.requests:
        asset = item.asset_name
        try:
            model = model_manager.get_model(asset)
            if not model:
                results.append(
                    {
                        "asset_name": asset,
                        "prediction": None,
                        "error": f"Model for '{asset}' not available",
                    }
                )
                failed += 1
                continue

            input_df = pd.DataFrame(item.features)
            if input_df.empty:
                results.append(
                    {
                        "asset_name": asset,
                        "prediction": None,
                        "error": "Features list cannot be empty",
                    }
                )
                failed += 1
                continue

            # Make prediction, unwrap to bypass strict MLflow signature validation allowing lstm_pred
            prediction = model.unwrap_python_model().predict(context=None, model_input=input_df)

            # Handle numpy types
            if isinstance(prediction, (list, pd.Series, np.ndarray)):
                result_value = float(prediction[0]) if len(prediction) >= 1 else 0.0
            else:
                result_value = float(prediction)

            results.append({"asset_name": asset, "prediction": result_value, "error": None})
            successful += 1

        except ValueError as ve:
            logger.warning(f"Validation error for {asset}: {ve}")
            results.append(
                {"asset_name": asset, "prediction": None, "error": f"Validation error: {str(ve)}"}
            )
            failed += 1

        except Exception as e:
            logger.error(f"Prediction error for {asset}: {e}")
            results.append(
                {"asset_name": asset, "prediction": None, "error": f"Prediction failed: {str(e)}"}
            )
            failed += 1

    latency_ms = (time.time() - start_time) * 1000

    return {
        "predictions": results,
        "metadata": {
            "total_requests": len(payload.requests),
            "successful": successful,
            "failed": failed,
            "latency_ms": round(latency_ms, 2),
        },
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request):
    """Returns metadata about all configured models and their load status."""
    model_manager = request.app.state.model_manager
    models = []

    for asset in model_manager.assets:
        safe_name = sanitize_artifact_name(asset)
        base_model_name = f"hybrid_lstm_xgboost_{safe_name}"
        registry_name = get_model_registry_name(base_model_name)

        models.append(
            ModelInfoItem(
                asset_name=asset,
                model_loaded=asset in model_manager.models,
                registry_name=registry_name,
            )
        )

    return ModelInfoResponse(
        models=models,
        total_loaded=len(model_manager.models),
    )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics(request: Request):
    """Returns latest training run metrics from MLflow."""
    try:
        experiment_name = os.getenv("DATABRICKS_EXPERIMENT_PATH")
        if not experiment_name:
            return MetricsResponse(
                status="unavailable", message="DATABRICKS_EXPERIMENT_PATH not configured."
            )

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return MetricsResponse(
                status="unavailable", message=f"Experiment '{experiment_name}' not found."
            )

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs.empty:
            return MetricsResponse(status="unavailable", message="No finished runs found.")

        run = runs.iloc[0]
        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
        metrics = {}
        for col in metric_cols:
            value = run[col]
            if pd.notna(value):
                metrics[col.replace("metrics.", "")] = float(value)

        return MetricsResponse(
            status="available",
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Failed to fetch metrics: {e}")
        return MetricsResponse(
            status="error",
            message=str(e),
        )
