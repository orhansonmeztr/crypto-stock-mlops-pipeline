import contextlib
import logging
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.utils.config_utils import get_model_registry_name, load_config, sanitize_artifact_name

project_root = Path(__file__).resolve().parents[2]

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def promote_best_model(asset_name: str, metric_name: str = "rmse", lower_is_better: bool = True):
    """
    Compares the latest trained model with the current 'Champion'.
    Promotes the new model if it performs better OR if no Champion exists.
    """
    client = MlflowClient()

    # 1. Construct the Model Registry Name (using our unified utility)
    safe_asset_name = sanitize_artifact_name(asset_name)
    base_model_name = f"hybrid_lstm_xgboost_{safe_asset_name}"
    registry_name = get_model_registry_name(base_model_name)

    logging.info(f"Checking model: {registry_name}...")

    # 2. Get all versions of the model
    try:
        all_versions = client.search_model_versions(f"name='{registry_name}'")
    except Exception as e:
        logging.warning(f"Model '{registry_name}' not found or error accessing registry: {e}")
        return

    if not all_versions:
        logging.warning(f"No versions found for {registry_name}")
        return

    # Sort by version number (descending) to get the latest
    all_versions.sort(key=lambda x: int(x.version), reverse=True)
    newest_model = all_versions[0]

    # 3. Find current Champion
    current_champion = None
    with contextlib.suppress(Exception):
        current_champion = client.get_model_version_by_alias(registry_name, "Champion")

    # 4. Logic for Promotion

    # Case A: No Champion exists -> Promote immediately
    if current_champion is None:
        logging.info(
            f"No Champion found for {registry_name}. Promoting v{newest_model.version} to Champion."
        )
        client.set_registered_model_alias(registry_name, "Champion", newest_model.version)
        return

    # Case B: The newest version is already the Champion -> Do nothing
    if newest_model.version == current_champion.version:
        logging.info(f"Newest version (v{newest_model.version}) is already Champion.")
        return

    # Case C: Compare Metrics
    try:
        new_run = client.get_run(newest_model.run_id)
        prod_run = client.get_run(current_champion.run_id)

        new_metric = new_run.data.metrics.get(metric_name)
        prod_metric = prod_run.data.metrics.get(metric_name)

        if new_metric is None or prod_metric is None:
            logging.warning(
                f"Metrics ({metric_name}) not found for comparison. Skipping promotion."
            )
            return

        logging.info(
            f"   v{newest_model.version} (New): {new_metric:.4f} vs v{current_champion.version} (Champ): {prod_metric:.4f}"
        )

        is_better = (new_metric < prod_metric) if lower_is_better else (new_metric > prod_metric)

        if is_better:
            logging.info(f"New model is better! Promoting v{newest_model.version} to Champion.")
            client.set_registered_model_alias(registry_name, "Champion", newest_model.version)
        else:
            logging.info(
                f"New model is not better. Keeping v{current_champion.version} as Champion."
            )

    except Exception as e:
        logging.error(f"Error during metric comparison: {e}")


def main():
    # Setup MLflow Tracking URI
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Load assets from config
    config_path = project_root / "configs" / "training_config.yaml"
    config = load_config(config_path)

    # Merge stocks and cryptos into a single list
    stocks = config.get("assets", {}).get("stocks", [])
    cryptos = config.get("assets", {}).get("cryptos", [])
    all_assets = stocks + cryptos

    logging.info(f"Starting Model Promotion Pipeline for assets: {all_assets}")

    for asset in all_assets:
        promote_best_model(asset)

    logging.info("Model promotion finished.")


if __name__ == "__main__":
    main()
