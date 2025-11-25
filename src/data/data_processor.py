import logging

import pandas as pd

from src.models.lstm_model import prepare_lstm_data


def prepare_training_data(
    df: pd.DataFrame, config: dict
) -> tuple[dict | None, dict | None, object | None]:
    """
    Prepares data for training Hybrid model (LSTM + XGBoost).

    Returns:
        tuple: (lstm_data_dict, xgboost_data_dict, scaler)
            lstm_data_dict keys: X_train, y_train, X_val, y_val, X_test, y_test
            xgboost_data_dict keys: X_train, y_train, X_test, y_test
    """
    lstm_config = config["model_params"]["lstm"]
    look_back = lstm_config.get("look_back", 30)

    # 1. Prepare LSTM Data
    X_lstm, y_lstm, scaler = prepare_lstm_data(
        df, look_back=look_back, target_col="close", feature_cols=["close"]
    )

    n_samples = len(X_lstm)
    if n_samples < 5:
        logging.warning(f"Not enough data for LSTM split. Samples: {n_samples}")
        return None, None, None

    # 2. Split Data
    train_ratio = config["training"].get("train_ratio", 0.6)
    val_ratio = config["training"].get("validation_ratio", 0.2)

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    lstm_data = {
        "X_train": X_lstm[:train_end],
        "y_train": y_lstm[:train_end],
        "X_val": X_lstm[train_end:val_end],
        "y_val": y_lstm[train_end:val_end],
        "X_test": X_lstm[val_end:],
        "y_test": y_lstm[val_end:],
    }

    # Validation check
    if (
        len(lstm_data["X_train"]) == 0
        or len(lstm_data["X_val"]) == 0
        or len(lstm_data["X_test"]) == 0
    ):
        logging.warning("Data split resulted in empty sets.")
        return None, None, None

    # 3. Prepare DataFrames for XGBoost features
    # df_trimmed must align with X_lstm (starts after look_back)
    df_trimmed = df.iloc[look_back:].copy()
    df_trimmed = df_trimmed.iloc[:n_samples]

    xgb_data_raw = {
        "df_val": df_trimmed.iloc[train_end:val_end].copy(),
        "df_test": df_trimmed.iloc[val_end:].copy(),
    }

    return lstm_data, xgb_data_raw, scaler


def prepare_xgboost_features(df_val, df_test, lstm_pred_val, lstm_pred_test):
    """Adds LSTM predictions to DataFrame and selects features."""

    # Add LSTM predictions
    df_val["lstm_pred"] = lstm_pred_val.flatten()
    df_test["lstm_pred"] = lstm_pred_test.flatten()

    exclude_cols = ["target", "asset_name", "asset_type", "close"]
    feature_cols = [c for c in df_val.columns if c not in exclude_cols]

    xgb_data = {
        "X_train": df_val[feature_cols],  # Train XGBoost on Validation set
        "y_train": df_val["target"],
        "X_test": df_test[feature_cols],
        "y_test": df_test["target"],
        "feature_cols": feature_cols,
    }

    return xgb_data
