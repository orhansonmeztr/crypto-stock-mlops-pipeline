import logging
from typing import Any

import mlflow
import mlflow.pyfunc

from src.utils.config_utils import get_model_registry_name, sanitize_artifact_name

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.models: dict[str, Any] = {}
        self.assets: list[str] = []

    def set_assets(self, assets: list[str]):
        self.assets = assets

    def _get_model_uri(self, asset_name: str) -> str:
        safe_name = sanitize_artifact_name(asset_name)
        base_model_name = f"hybrid_lstm_xgboost_{safe_name}"
        registry_name = get_model_registry_name(base_model_name)
        return f"models:/{registry_name}@Champion"

    def load_models(self):
        """Loads all models defined in self.assets."""
        for asset in self.assets:
            try:
                model_uri = self._get_model_uri(asset)
                logger.info(f"Downloading & Loading model for {asset} from {model_uri}...")

                model = mlflow.pyfunc.load_model(model_uri)
                self.models[asset] = model
                logger.info(f"Loaded {asset} model successfully.")
            except Exception as e:
                logger.error(f"Failed to load model for {asset} from {model_uri}")
                logger.error(f"Details: {e}")

    def get_model(self, asset_name: str) -> Any:
        return self.models.get(asset_name)

    def list_loaded_models(self) -> list[str]:
        return list(self.models.keys())

    def clear(self):
        self.models.clear()
