import logging
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def eval_metrics(actual, pred):
    """Calculates and returns regression evaluation metrics."""

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_xgboost_model():
    """
    Trains an XGBoost model with cross-validation and logs the entire process
    into a single MLflow run.
    """

    mlflow.set_registry_uri("databricks")

    databricks_experiment_path = os.getenv("DATABRICKS_EXPERIMENT_PATH")
    if not databricks_experiment_path:
        logging.error("DATABRICKS_EXPERIMENT_PATH environment variable not set.")
        return

    project_root = Path(__file__).resolve().parents[2]
    features_path = project_root / "data" / "features" / "amazon_daily_features.csv"

    logging.info("Loading feature-engineered data...")
    df = pd.read_csv(features_path, index_col="timestamp", parse_dates=True)

    X = df.drop(columns=["target"])
    y = df["target"]

    mlflow.set_experiment(databricks_experiment_path)

    params = {
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    with mlflow.start_run() as parent_run:
        logging.info(f"Starting single parent MLflow run: {parent_run.info.run_id}")
        mlflow.log_params(params)

        tscv = TimeSeriesSplit(n_splits=5)
        metrics_by_fold = {"rmse": [], "mae": [], "r2": []}

        logging.info("Starting time-series cross-validation...")
        for fold, (train_index, val_index) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            predictions = model.predict(X_val)
            rmse, mae, r2 = eval_metrics(y_val, predictions)

            mlflow.log_metric(f"fold_{fold + 1}_rmse", rmse)
            mlflow.log_metric(f"fold_{fold + 1}_mae", mae)
            mlflow.log_metric(f"fold_{fold + 1}_r2", r2)

            metrics_by_fold["rmse"].append(rmse)
            metrics_by_fold["mae"].append(mae)
            metrics_by_fold["r2"].append(r2)

            logging.info(f"Fold {fold + 1}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

        avg_metrics = {f"avg_{k}": np.mean(v) for k, v in metrics_by_fold.items()}
        mlflow.log_metrics(avg_metrics)
        logging.info(f"Average CV Metrics: {avg_metrics}")

        logging.info("Training and logging final model on all data...")
        final_model = xgb.XGBRegressor(**params)
        final_model.fit(X, y, eval_set=[(X.iloc[-100:], y.iloc[-100:])], verbose=False)

        signature = mlflow.models.infer_signature(X, final_model.predict(X))
        mlflow.xgboost.log_model(
            xgb_model=final_model, artifact_path="final-amazon-predictor", signature=signature
        )

        logging.info("MLflow parent run completed successfully.")
        logging.info(f"Check your MLflow UI for run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    train_xgboost_model()
