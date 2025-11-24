import logging
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: Path) -> dict:
    """
    Loads the YAML configuration file.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def sanitize_artifact_name(name: str) -> str:
    """
    Replaces invalid characters in the artifact name with underscores.
    Allowed characters are alphanumeric, underscores, dashes, and spaces (though spaces are best avoided).
    MLflow restriction: ('/', ':', '.', '%', '"', "'")

    Args:
        name (str): The original name (e.g., 'Amazon.com').

    Returns:
        str: Sanitized name (e.g., 'Amazon_com').
    """
    for char in [".", ":", "/", "%", '"', "'", " "]:
        name = name.replace(char, "_")
    return name


def train_model_for_asset(df_asset: pd.DataFrame, asset_name: str, config: dict):
    """
    Trains an XGBoost model for a specific asset using parameters from config.

    Args:
        df_asset (pd.DataFrame): Dataframe containing features for a single asset.
        asset_name (str): Name of the asset.
        config (dict): Full configuration dictionary containing training and model params.
    """
    logging.info(f"Starting training for {asset_name}...")

    # Extract training parameters
    train_params = config["training"]
    model_params = config["model_params"]["xgboost"]

    # Ensure reproducibility
    random_state = train_params.get("random_state", 42)
    model_params["random_state"] = random_state

    # Split data into train and test (Time-series split based on ratio)
    test_ratio = train_params.get("test_size_ratio", 0.2)
    split_idx = int(len(df_asset) * (1 - test_ratio))

    train_df = df_asset.iloc[:split_idx]
    test_df = df_asset.iloc[split_idx:]

    # Define features and target
    # Exclude non-feature columns
    exclude_cols = ["target", "asset_name", "asset_type", "close"]
    feature_cols = [c for c in df_asset.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    # Start a nested MLflow run for this specific asset
    with mlflow.start_run(run_name=f"train_{asset_name}", nested=True):
        # Log parameters from config
        mlflow.log_params(model_params)
        mlflow.log_params(train_params)
        mlflow.log_param("asset_name", asset_name)
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("test_samples", len(test_df))

        # Initialize and train model
        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train)

        # Evaluate
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)

        # Calculate RMSE (using numpy due to sklearn version changes)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        logging.info(f"Model {asset_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Log model with SANITIZED name
        safe_model_name = f"model_{sanitize_artifact_name(asset_name)}"
        mlflow.xgboost.log_model(model, artifact_path=safe_model_name)


def main():
    """
    Main function to execute batch training for all assets.
    """
    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    features_path = project_root / "data" / "features" / "multi_asset_features.csv"
    config_path = project_root / "configs" / "training_config.yaml"

    # Load Config
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        return
    config = load_config(config_path)

    # Load Data
    if not features_path.exists():
        logging.error(f"Features file not found: {features_path}")
        return

    logging.info("Loading features data...")
    df = pd.read_csv(features_path, parse_dates=["timestamp"], index_col="timestamp")

    # Get list of unique assets
    asset_names = df["asset_name"].unique()
    logging.info(f"Found assets to train: {asset_names}")

    # Setup MLflow Tracking URI
    # Use environment variable or default to databricks if configured
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logging.info(f"MLflow Tracking URI: {mlflow_tracking_uri}")

    # Setup Experiment
    experiment_path = os.getenv("DATABRICKS_EXPERIMENT_PATH", "/Shared/crypto-stock-forecasting")
    mlflow.set_experiment(experiment_path)
    logging.info(f"MLflow Experiment: {experiment_path}")

    # Start Parent Run
    with mlflow.start_run(run_name="Batch_Training_All_Assets") as parent_run:
        logging.info(f"Parent Run ID: {parent_run.info.run_id}")
        mlflow.log_param("assets_count", len(asset_names))

        # Loop through each asset and train using the config
        for asset in asset_names:
            asset_df = df[df["asset_name"] == asset]
            train_model_for_asset(asset_df, asset, config)

    logging.info("Batch training completed.")


if __name__ == "__main__":
    main()
