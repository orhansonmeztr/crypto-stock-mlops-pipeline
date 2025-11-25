"""
Checks the Champion model alias for all configured assets in MLflow Model Registry.
Uses config file and shared utility functions for consistency.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.utils.config_utils import (
    get_model_registry_name,
    load_config,
    sanitize_artifact_name,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

project_root = Path(__file__).resolve().parents[2]


def check_champion(asset_name: str) -> None:
    """Checks and logs the Champion alias for a given asset's model."""
    client = MlflowClient()

    safe_name = sanitize_artifact_name(asset_name)
    base_model_name = f"hybrid_lstm_xgboost_{safe_name}"
    model_name = get_model_registry_name(base_model_name)

    try:
        champion_version = client.get_model_version_by_alias(model_name, "Champion")
        logging.info(
            f"{asset_name:<12} -> Champion Version: v{champion_version.version} "
            f"(Run ID: {champion_version.run_id})"
        )
    except Exception:
        logging.warning(f"{asset_name:<12} -> No Champion alias found yet.")


def main():
    config_path = project_root / "configs" / "training_config.yaml"
    config = load_config(config_path)

    stocks = config.get("assets", {}).get("stocks", [])
    cryptos = config.get("assets", {}).get("cryptos", [])
    all_assets = stocks + cryptos

    logging.info("-" * 50)
    logging.info("Checking Champion Models in Registry...")
    logging.info("-" * 50)

    for asset in all_assets:
        check_champion(asset)

    logging.info("-" * 50)


if __name__ == "__main__":
    main()
