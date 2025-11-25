import logging
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Thresholds for model acceptance
# Note: Since Bitcoin price is 100k+, absolute error (MAE/RMSE) might appear high.
# You can revise these values according to the project.
THRESHOLDS = {
    "rmse": 20000.0,  # Max acceptable RMSE
    "mae": 15000.0,  # Max acceptable MAE
}


def get_latest_run_metrics(experiment_name: str) -> dict[str, float] | None:
    """
    Retrieves metrics from the latest successful run in the specified experiment.
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logging.warning(f"Experiment '{experiment_name}' not found.")
            return None

        # Search for the latest finished run
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=10,
        )

        if runs.empty:
            logging.warning("No finished runs found.")
            return None

        # Let's try to find the first run with meaningful metrics among the latest runs
        for _, run in runs.iterrows():
            # MLflow might sometimes return NaN metrics, let's check
            rmse = run.get("metrics.rmse")
            if rmse is not None and not pd.isna(rmse):
                metrics = {
                    "rmse": float(rmse),
                    "mae": float(run.get("metrics.mae", 0.0)),
                    "run_id": run.run_id,
                    "run_name": run.get("tags.mlflow.runName", "unknown"),
                }
                logging.info(f"Analyzing Run: {metrics['run_name']} (ID: {metrics['run_id']})")
                logging.info(f"Metrics - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}")
                return metrics

        logging.warning("No runs with valid RMSE metrics found within last 10 runs.")
        return None

    except Exception as e:
        logging.error(f"Failed to fetch metrics: {e}")
        return None


def check_performance_thresholds(metrics: dict[str, float]) -> bool:
    """
    Compares current metrics against defined thresholds.
    """
    is_healthy = True

    if metrics["rmse"] > THRESHOLDS["rmse"]:
        logging.error(f"RMSE {metrics['rmse']:.2f} exceeds threshold {THRESHOLDS['rmse']}")
        is_healthy = False

    if metrics["mae"] > THRESHOLDS["mae"]:
        logging.error(f"MAE {metrics['mae']:.2f} exceeds threshold {THRESHOLDS['mae']}")
        is_healthy = False

    if is_healthy:
        logging.info("Model performance is within acceptable limits.")
    else:
        logging.warning("Model performance validation FAILED.")

    return is_healthy


def main():
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiment_name = os.getenv("DATABRICKS_EXPERIMENT_PATH")

    logging.info("Starting Model Performance Monitor...")

    metrics = get_latest_run_metrics(experiment_name)

    if metrics:
        success = check_performance_thresholds(metrics)
        # To raise an error in the CI/CD pipeline:
        # if not success: exit(1)
    else:
        logging.error("Could not retrieve metrics for analysis.")


if __name__ == "__main__":
    main()
