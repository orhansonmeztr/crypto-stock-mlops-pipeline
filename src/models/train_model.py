import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import logging

import mlflow
import mlflow.tensorflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.models.lstm_model import create_lstm_model, prepare_lstm_data

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def sanitize_artifact_name(name: str) -> str:
    for char in [".", ":", "/", "%", '"', "'", " "]:
        name = name.replace(char, "_")
    return name


# --- GLOBAL LIST TO STORE FORECASTS ---
latest_forecasts = []


def train_hybrid_model_for_asset(df_asset: pd.DataFrame, asset_name: str, config: dict):
    logging.info(f"Starting HYBRID training for {asset_name}...")

    # --- 1. LSTM ---
    lstm_config = config["model_params"]["lstm"]
    look_back = lstm_config.get("look_back", 60)

    X_lstm, y_lstm, scaler = prepare_lstm_data(
        df_asset, look_back=look_back, target_col="close", feature_cols=["close"]
    )

    train_size = int(len(X_lstm) * 0.8)
    X_lstm_train, X_lstm_test = X_lstm[:train_size], X_lstm[train_size:]
    y_lstm_train, y_lstm_test = y_lstm[:train_size], y_lstm[train_size:]

    with mlflow.start_run(run_name=f"train_{asset_name}_lstm", nested=True):
        mlflow.log_params(lstm_config)

        model_lstm = create_lstm_model(
            input_shape=(X_lstm.shape[1], X_lstm.shape[2]),
            units=lstm_config.get("units", 50),
            dropout_rate=lstm_config.get("dropout", 0.2),
            learning_rate=lstm_config.get("learning_rate", 0.001),
        )

        model_lstm.fit(
            X_lstm_train,
            y_lstm_train,
            epochs=lstm_config.get("epochs", 10),
            batch_size=lstm_config.get("batch_size", 32),
            validation_data=(X_lstm_test, y_lstm_test),
            verbose=0,
        )

        safe_name = sanitize_artifact_name(asset_name)

        signature = infer_signature(X_lstm_train, model_lstm.predict(X_lstm_train, verbose=0))
        mlflow.tensorflow.log_model(
            model_lstm, artifact_path=f"model_lstm_{safe_name}", signature=signature
        )

        lstm_preds_scaled = model_lstm.predict(X_lstm, verbose=0)
        lstm_preds = scaler.inverse_transform(lstm_preds_scaled)

        logging.info(f"LSTM training finished for {asset_name}")

    # --- 2. Hybrid ---
    df_trimmed = df_asset.iloc[look_back:].copy()
    df_trimmed["lstm_pred"] = lstm_preds.flatten()

    train_params = config["training"]
    xgb_params = config["model_params"]["xgboost"]

    test_ratio = train_params.get("test_size_ratio", 0.2)
    split_idx = int(len(df_trimmed) * (1 - test_ratio))

    train_df = df_trimmed.iloc[:split_idx]
    test_df = df_trimmed.iloc[split_idx:]

    exclude_cols = ["target", "asset_name", "asset_type", "close"]
    feature_cols = [c for c in df_trimmed.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    with mlflow.start_run(run_name=f"train_{asset_name}_hybrid", nested=True):
        mlflow.log_params(xgb_params)
        mlflow.log_param("features", feature_cols)

        model_xgb = XGBRegressor(**xgb_params)
        model_xgb.fit(X_train, y_train)

        predictions = model_xgb.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        logging.info(f"Hybrid Model {asset_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        signature_xgb = infer_signature(X_train, predictions)
        mlflow.xgboost.log_model(
            model_xgb, artifact_path=f"model_hybrid_{safe_name}", signature=signature_xgb
        )

        # --- 3. Generate Future Forecast (Next Day) ---
        # Use the very last data point to predict the 'next' unknown day
        last_row = df_trimmed.iloc[[-1]]  # Last available data point
        X_future = last_row[feature_cols]

        forecast_price = model_xgb.predict(X_future)[0]
        current_price = last_row["close"].values[0]

        forecast_entry = {
            "asset_name": asset_name,
            "last_date": last_row.index[0],
            "current_price": float(current_price),
            "forecast_price": float(forecast_price),
            "predicted_change_pct": float((forecast_price - current_price) / current_price * 100),
        }
        latest_forecasts.append(forecast_entry)
        logging.info(
            f"Forecast for {asset_name}: {forecast_price:.2f} (Change: {forecast_entry['predicted_change_pct']:.2f}%)"
        )


def main():
    features_path = project_root / "data" / "features" / "multi_asset_features.csv"
    config_path = project_root / "configs" / "training_config.yaml"
    predictions_path = project_root / "data" / "predictions"
    predictions_path.mkdir(exist_ok=True)

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiment_name = os.getenv("DATABRICKS_EXPERIMENT_PATH", "/Shared/crypto-stock-forecasting")
    mlflow.set_experiment(experiment_name)

    config = load_config(config_path)

    logging.info("Loading features data...")
    df = pd.read_csv(features_path, parse_dates=["timestamp"], index_col="timestamp")
    asset_names = df["asset_name"].unique()

    with mlflow.start_run(run_name="Batch_Hybrid_Training"):
        for asset in asset_names:
            try:
                asset_df = df[df["asset_name"] == asset]
                if len(asset_df) > config["model_params"]["lstm"]["look_back"] + 10:
                    train_hybrid_model_for_asset(asset_df, asset, config)
                else:
                    logging.warning(f"Not enough data for {asset}, skipping.")
            except Exception as e:
                logging.error(f"Failed to train {asset}: {e}")

    # Save all forecasts to CSV
    if latest_forecasts:
        forecast_df = pd.DataFrame(latest_forecasts)
        output_file = predictions_path / "latest_forecasts.csv"
        forecast_df.to_csv(output_file, index=False)
        logging.info(f"Saved latest forecasts to {output_file}")
        print("\n--- LATEST FORECASTS ---")
        print(forecast_df)


if __name__ == "__main__":
    main()
