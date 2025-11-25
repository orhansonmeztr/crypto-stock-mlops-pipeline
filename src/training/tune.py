import os

# Suppress TensorFlow Logs
# These settings must be applied BEFORE importing tensorflow or modules that use it.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=All, 1=Filter INFO, 2=Filter WARNING, 3=Filter ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suppress oneDNN custom operations info


import json
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf

# Suppress TensorFlow python-level warnings
tf.get_logger().setLevel(logging.ERROR)
# Suppress the specific retracing warning if possible, but setting level to ERROR should cover it.

from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from src.models.lstm_model import create_lstm_model, prepare_lstm_data
from src.utils.config_utils import load_config

project_root = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CONFIG_PATH = project_root / "configs" / "training_config.yaml"
BEST_PARAMS_PATH = project_root / "configs" / "best_params.json"


def save_best_params(best_params):
    """
    Saves the best hyperparameters found by Optuna to a JSON file.
    This avoids overwriting the main YAML config and dirtying the git history.
    """
    logging.info(f"Saving best parameters to {BEST_PARAMS_PATH}...")

    serializable_params = {}
    for k, v in best_params.items():
        # Check for both standard python float and numpy floating types
        if isinstance(v, (float, np.floating)):
            serializable_params[k] = float(v)
        else:
            serializable_params[k] = int(v)

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(serializable_params, f, indent=4)

    logging.info("Successfully saved best parameters.")


def objective(trial, x_lstm, y_lstm, df_asset, look_back):
    """
    Objective function to be optimized by Optuna.
    Optimizes both LSTM architecture and XGBoost parameters simultaneously.
    """

    # 1. LSTM Parameters
    lstm_units = trial.suggest_int("lstm_units", 20, 100)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.5)
    lstm_learning_rate = trial.suggest_float("lstm_lr", 1e-4, 1e-2, log=True)
    lstm_epochs = 5  # Keep low for tuning speed

    # LSTM Training (Simple split for tuning)
    train_size = int(len(x_lstm) * 0.8)
    X_train, X_val = x_lstm[:train_size], x_lstm[train_size:]
    y_train, y_val = y_lstm[:train_size], y_lstm[train_size:]

    model_lstm = create_lstm_model(
        input_shape=(x_lstm.shape[1], x_lstm.shape[2]),
        units=lstm_units,
        dropout_rate=lstm_dropout,
        learning_rate=lstm_learning_rate,
    )

    model_lstm.fit(X_train, y_train, epochs=lstm_epochs, batch_size=32, verbose=0)
    lstm_preds_scaled = model_lstm.predict(x_lstm, verbose=0)

    # 2. Hybrid Data Preparation
    df_trimmed = df_asset.iloc[look_back:].copy()
    min_len = min(len(df_trimmed), len(lstm_preds_scaled))
    df_trimmed = df_trimmed.iloc[:min_len]
    lstm_features = lstm_preds_scaled[:min_len].flatten()
    df_trimmed["lstm_pred"] = lstm_features

    # 3. XGBoost Parameters
    xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
    xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 10)
    xgb_learning_rate = trial.suggest_float("xgb_lr", 0.01, 0.3)
    xgb_subsample = trial.suggest_float("xgb_subsample", 0.5, 1.0)

    exclude_cols = ["target", "asset_name", "asset_type", "close"]
    feature_cols = [c for c in df_trimmed.columns if c not in exclude_cols]

    split_idx = int(len(df_trimmed) * 0.8)
    train_df = df_trimmed.iloc[:split_idx]
    val_df = df_trimmed.iloc[split_idx:]

    model_xgb = XGBRegressor(
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        subsample=xgb_subsample,
        random_state=42,
    )

    model_xgb.fit(train_df[feature_cols], train_df["target"])
    preds = model_xgb.predict(val_df[feature_cols])

    rmse = np.sqrt(mean_squared_error(val_df["target"], preds))

    tf.keras.backend.clear_session()
    return rmse


def run_optimization(asset_name, n_trials=10):
    logging.info(f"Starting optimization for {asset_name}...")

    features_path = project_root / "data" / "features" / "multi_asset_features.csv"

    if not features_path.exists():
        logging.error("Features file not found. Please run build_features.py first.")
        return

    df = pd.read_csv(features_path, parse_dates=["timestamp"], index_col="timestamp")
    df_asset = df[df["asset_name"] == asset_name].copy()

    if df_asset.empty:
        logging.error(f"No data for {asset_name}")
        return

    config = load_config(CONFIG_PATH)
    look_back = config["model_params"]["lstm"]["look_back"]

    X_lstm, y_lstm, _ = prepare_lstm_data(
        df_asset, look_back=look_back, target_col="close", feature_cols=["close"]
    )

    study = optuna.create_study(direction="minimize")

    def func(trial):
        return objective(trial, X_lstm, y_lstm, df_asset, look_back)

    study.optimize(func, n_trials=n_trials)

    logging.info(f"Best params for {asset_name}: {study.best_params}")

    # Save to JSON instead of overwriting YAML
    save_best_params(study.best_params)

    return study.best_params


if __name__ == "__main__":
    # Optimize for BTC-USD as a representative asset
    run_optimization("BTC-USD", n_trials=10)
