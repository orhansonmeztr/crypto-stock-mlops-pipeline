import os

# Suppress TensorFlow Logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import logging
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.tensorflow
import mlflow.xgboost
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from xgboost import XGBRegressor

from src.data.data_processor import prepare_training_data, prepare_xgboost_features

# Local Imports
from src.models.hybrid_model import HybridModel
from src.models.lstm_model import create_lstm_model
from src.training.evaluate import evaluate_model
from src.utils.config_utils import (
    get_model_registry_name,
    load_config,
    override_config_with_params,
    sanitize_artifact_name,
)

# TF Warning Suppression
tf.get_logger().setLevel(logging.ERROR)

# Setup
project_root = Path(__file__).resolve().parents[2]
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_lstm_component(lstm_data, lstm_config, safe_name):
    """Trains and logs the LSTM model component."""
    with mlflow.start_run(run_name=f"train_{safe_name}_lstm", nested=True):
        mlflow.log_params(lstm_config)

        model_lstm = create_lstm_model(
            input_shape=(lstm_data["X_train"].shape[1], lstm_data["X_train"].shape[2]),
            units=lstm_config.get("units", 50),
            dropout_rate=lstm_config.get("dropout", 0.2),
            learning_rate=lstm_config.get("learning_rate", 0.001),
        )

        model_lstm.fit(
            lstm_data["X_train"],
            lstm_data["y_train"],
            epochs=lstm_config.get("epochs", 10),
            batch_size=lstm_config.get("batch_size", 32),
            validation_data=(lstm_data["X_val"], lstm_data["y_val"]),
            verbose=0,
        )

        # Log LSTM Model
        signature = infer_signature(
            lstm_data["X_train"], model_lstm.predict(lstm_data["X_train"], verbose=0)
        )
        mlflow.tensorflow.log_model(
            model_lstm,
            name=f"model_lstm_{safe_name}",
            signature=signature,
            input_example=lstm_data["X_train"][:1],
        )

        return model_lstm


def train_xgboost_component(xgb_data, xgb_params, safe_name):
    """Trains the XGBoost component and returns model + metrics."""
    with mlflow.start_run(run_name=f"train_{safe_name}_hybrid", nested=True):
        mlflow.log_params(xgb_params)
        mlflow.log_param("features", xgb_data["feature_cols"])

        model_xgb = XGBRegressor(**xgb_params)
        model_xgb.fit(xgb_data["X_train"], xgb_data["y_train"])

        # Evaluate with full metrics
        predictions = model_xgb.predict(xgb_data["X_test"])
        metrics = evaluate_model(xgb_data["y_test"].values, predictions)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        return model_xgb


def log_hybrid_model(model_xgb, model_lstm, scaler, look_back, xgb_data, safe_name):
    """Logs the composite HybridModel to MLflow."""
    base_model_name = f"hybrid_lstm_xgboost_{safe_name}"
    registry_name = get_model_registry_name(base_model_name)

    logging.info(f"Logging model to MLflow: {registry_name}")

    hybrid_model_instance = HybridModel(
        xgb_model=model_xgb,
        lstm_model=model_lstm,
        scaler=scaler,
        look_back=look_back,
        feature_cols=["close"],
    )

    # Infer signature
    # Dummy input matching API structure: XGBoost features (no lstm_pred) + 'close'
    signature_input = xgb_data["X_train"].head(look_back).copy()
    if "lstm_pred" in signature_input.columns:
        signature_input = signature_input.drop(columns=["lstm_pred"])

    # We need a dummy 'close' for signature, fetch from context if possible or just use 0.0
    # Ideally should come from data, but as signature it just needs type.
    signature_input["close"] = 0.0

    signature_xgb = infer_signature(
        signature_input, hybrid_model_instance.predict(context=None, model_input=signature_input)
    )

    mlflow.pyfunc.log_model(
        name=f"model_hybrid_{safe_name}",
        python_model=hybrid_model_instance,
        signature=signature_xgb,
        input_example=signature_input,
        registered_model_name=registry_name,
        pip_requirements=["xgboost", "pandas", "numpy", "tensorflow"],
    )


def generate_forecast(df_asset, model_lstm, model_xgb, scaler, look_back, feature_cols, asset_name):
    """Generates a next-day forecast."""
    last_window = df_asset[["close"]].tail(look_back)
    last_window_scaled_array = scaler.transform(last_window).reshape(1, look_back, 1)

    next_lstm_pred_scaled = model_lstm.predict(last_window_scaled_array, verbose=0)
    # create dummy context DataFrame to appease the scaler inverse transform feature name warning as well if possible, or just ignore for 1 length
    next_lstm_pred = scaler.inverse_transform(
        pd.DataFrame(next_lstm_pred_scaled, columns=["close"])
    )[0][0]

    last_row = df_asset.iloc[[-1]].copy()
    last_row["lstm_pred"] = float(next_lstm_pred)

    # Ensure correct feature columns for XGBoost
    X_future = last_row[feature_cols]

    forecast_price = model_xgb.predict(X_future)[0]
    current_price = last_row["close"].values[0]

    return {
        "asset_name": asset_name,
        "last_date": str(last_row.index[0]),
        "current_price": float(current_price),
        "forecast_price": float(forecast_price),
        "predicted_change_pct": float((forecast_price - current_price) / current_price * 100),
    }


def train_hybrid_model_for_asset(df_asset: pd.DataFrame, asset_name: str, config: dict):
    logging.info(f"Starting HYBRID training for {asset_name}...")
    safe_name = sanitize_artifact_name(asset_name)

    # 1. Data Prep
    lstm_data, xgb_data_raw, scaler = prepare_training_data(df_asset, config)
    if not lstm_data:
        return None

    lstm_config = config["model_params"]["lstm"]
    look_back = lstm_config.get("look_back", 30)

    # 2. Train LSTM
    model_lstm = train_lstm_component(lstm_data, lstm_config, safe_name)
    logging.info(f"LSTM training finished for {asset_name}")

    # 3. Generate LSTM Features for XGBoost
    # Predict on Val and Test
    lstm_pred_val = scaler.inverse_transform(model_lstm.predict(lstm_data["X_val"], verbose=0))
    lstm_pred_test = scaler.inverse_transform(model_lstm.predict(lstm_data["X_test"], verbose=0))

    xgb_data = prepare_xgboost_features(
        xgb_data_raw["df_val"], xgb_data_raw["df_test"], lstm_pred_val, lstm_pred_test
    )

    # 4. Train XGBoost
    xgb_params = config["model_params"]["xgboost"]
    model_xgb = train_xgboost_component(xgb_data, xgb_params, safe_name)

    # 5. Log Hybrid Model
    log_hybrid_model(model_xgb, model_lstm, scaler, look_back, xgb_data, safe_name)

    # 6. Forecast
    forecast = generate_forecast(
        df_asset, model_lstm, model_xgb, scaler, look_back, xgb_data["feature_cols"], asset_name
    )

    logging.info(f"Forecast for {asset_name}: {forecast['forecast_price']:.2f}")
    return forecast


def main():
    features_path = project_root / "data" / "features" / "multi_asset_features.csv"
    config_path = project_root / "configs" / "training_config.yaml"
    best_params_path = project_root / "configs" / "best_params.json"
    predictions_path = project_root / "data" / "predictions"
    predictions_path.mkdir(exist_ok=True)

    # MLflow Setup
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiment_name = os.getenv("DATABRICKS_EXPERIMENT_PATH")
    mlflow.set_experiment(experiment_name)

    if not features_path.exists():
        logging.error("Features file not found. Run 'build_features.py' first.")
        return

    # Load Config & Params
    config = load_config(config_path)
    if best_params_path.exists():
        logging.info(f"Loading optimized hyperparameters from {best_params_path}")
        with open(best_params_path) as f:
            best_params = json.load(f)
        override_config_with_params(config, best_params)
    else:
        logging.info("No optimized params found. Using defaults.")

    df = pd.read_csv(features_path, parse_dates=["timestamp"], index_col="timestamp")
    assets = config.get("assets", {})
    target_assets = assets.get("stocks", []) + assets.get("cryptos", [])

    forecasts = []

    with mlflow.start_run(run_name="pipeline_training_run"):
        mlflow.log_params(config["training"])

        for asset in target_assets:
            df_asset = df[df["asset_name"] == asset].copy()
            if not df_asset.empty:
                forecast = train_hybrid_model_for_asset(df_asset, asset, config)
                if forecast:
                    forecasts.append(forecast)
            else:
                logging.warning(f"No data found for {asset}, skipping...")

        if forecasts:
            forecast_df = pd.DataFrame(forecasts)
            forecast_file = predictions_path / "latest_forecasts.csv"
            forecast_df.to_csv(forecast_file, index=False)
            mlflow.log_artifact(str(forecast_file), artifact_path="forecasts")
            logging.info("Saved latest forecasts to CSV and MLflow.")


if __name__ == "__main__":
    main()
