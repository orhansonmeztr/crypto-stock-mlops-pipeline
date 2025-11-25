import json
import logging
import os
import smtplib
import sys
from email.mime.text import MIMEText
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset

from src.utils.config_utils import load_config

load_dotenv()

# SETUP PATHS
project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "configs" / "training_config.yaml"
reports_dir = project_root / "reports" / "monitoring"
reports_dir.mkdir(parents=True, exist_ok=True)

# LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def send_drift_alert_email(asset_name: str, drift_share: float) -> None:
    """
    Sends a drift alert email via SMTP.
    Requires SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO
    environment variables to be set.
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    alert_to = os.getenv("ALERT_EMAIL_TO")

    if not all([smtp_host, smtp_user, smtp_password, alert_to]):
        logger.warning(
            "Email alert not configured. Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO."
        )
        return

    subject = f"[MLOps Alert] Data Drift Detected — {asset_name}"
    body = (
        f"Data drift has been detected for asset: {asset_name}\n\n"
        f"Drift share: {drift_share:.2%}\n\n"
        f"A model retraining has been triggered automatically.\n"
        f"Please check the MLflow dashboard for details."
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = alert_to

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [alert_to], msg.as_string())
        logger.info(f"Drift alert email sent to {alert_to}")
    except Exception as e:
        logger.error(f"Failed to send drift alert email: {e}")


def load_data(asset_name: str = None):
    """
    Loads features data. If asset_name is provided, filters for that asset.
    Simulates production monitoring by splitting data into Reference (Old) and Current (New).
    """
    features_path = project_root / "data" / "features" / "multi_asset_features.csv"

    if not features_path.exists():
        logger.error(f"Features file not found at {features_path}")
        sys.exit(1)

    df = pd.read_csv(features_path, parse_dates=["timestamp"])

    if asset_name:
        df = df[df["asset_name"] == asset_name].copy()

    if df.empty:
        logger.warning(f"No data found for asset: {asset_name}")
        return None, None

    df = df.sort_values("timestamp")

    # Simulate Production Scenario:
    # Instead of comparing 4 years of history to 1 year of recent data (which guarantees
    # massive drift for non-stationary stock prices), we compare recent windows.
    # Reference Data: 60 days prior
    # Current Data: Last 30 days
    if len(df) > 90:
        df = df.tail(90)
        split_idx = 60
    else:
        split_idx = int(len(df) * 0.66)

    reference_data = df.iloc[:split_idx]
    current_data = df.iloc[split_idx:]

    # Drop non-feature columns for drift analysis cleanliness
    cols_to_drop = ["timestamp", "asset_name", "asset_type", "target", "lstm_pred"]

    # Financial price data (close, MAs, lags) is non-stationary and will almost always
    # trigger drift warnings over short windows. We focus on stationary features.
    non_stationary_cols = [
        "close",
        "ma_3",
        "ma_7",
        "ma_14",
        "bb_high",
        "bb_low",
        "lag_1",
        "lag_2",
        "lag_3",
        "macd",
        "macd_signal",
    ]
    cols_to_drop.extend(non_stationary_cols)

    # Note: 'lstm_pred' might not exist in raw features yet, so we ignore errors
    reference_data = reference_data.drop(columns=cols_to_drop, errors="ignore")
    current_data = current_data.drop(columns=cols_to_drop, errors="ignore")

    return reference_data, current_data


def run_drift_analysis_for_asset(asset_name: str, drift_share_threshold: float = 0.5):
    logger.info(f"--- Running Drift Analysis for {asset_name} ---")

    reference_data, current_data = load_data(asset_name)

    if reference_data is None or current_data is None:
        return

    # 1. Generate Report using Evidently
    report = Report(metrics=[DataDriftPreset(drift_share=drift_share_threshold)])
    my_eval = report.run(reference_data=reference_data, current_data=current_data)

    # 2. Save Reports Locally
    safe_name = asset_name.replace(".", "_").replace(" ", "_").lower()
    html_path = reports_dir / f"drift_report_{safe_name}.html"
    json_path = reports_dir / f"drift_metrics_{safe_name}.json"

    my_eval.save_html(str(html_path))
    logging.info(f"HTML Report saved to: {html_path}")
    my_eval.save_json(str(json_path))
    logging.info(f"JSON Report saved to: {json_path}")

    # 3. Log to MLflow
    # We use a specific experiment for monitoring
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    monitoring_experiment = os.getenv(
        "DATABRICKS_EXPERIMENT_PATH_MONITORING", "/Shared/crypto-stock-monitoring"
    )
    mlflow.set_experiment(monitoring_experiment)

    with mlflow.start_run(run_name=f"Monitor_{asset_name}"):
        mlflow.log_artifact(str(html_path))
        mlflow.log_artifact(str(json_path))

        # Parse JSON to log key metrics
        with open(json_path) as f:
            my_eval = json.load(f)

        drift_detected = False
        drift_share = 0.0

        try:
            # Iterate through metrics to find 'DriftedColumnsCount'
            for metric in my_eval.get("metrics", []):
                # Check if this is the summary metric
                # The metric_name usually looks like "DriftedColumnsCount(drift_share=0.5)"
                metric_config = metric.get("config", {})
                metric_type = metric_config.get("type", "")

                if "DriftedColumnsCount" in metric_type:
                    value_data = metric.get("value", {})
                    drift_share = value_data.get("share", 0.0)
                    drift_threshold = metric_config.get("drift_share", drift_share)

                    # Determine drift based on share vs threshold
                    if drift_share >= drift_threshold:
                        drift_detected = True

                    mlflow.log_metric("drift_share", drift_share)
                    mlflow.log_metric("drift_detected", drift_detected)

                    break

            if drift_detected:
                logger.warning(f"DRIFT DETECTED for {asset_name}! Share: {drift_share:.2%}")
                # Create a marker file to indicate drift detected for GitHub Actions
                with open("drift_detected.txt", "w") as f:
                    f.write("true")
                # Send email alert
                send_drift_alert_email(asset_name, drift_share)
            else:
                logger.info(f"No significant drift for {asset_name}. Share: {drift_share:.2%}")

        except KeyError as e:
            logger.warning(f"Could not parse evidently metrics json for MLflow logging: {e}")


def main():
    config = load_config(config_path)

    # Combine all assets
    stocks = config.get("assets", {}).get("stocks", [])
    cryptos = config.get("assets", {}).get("cryptos", [])
    all_assets = stocks + cryptos

    if not all_assets:
        logger.error("No assets found in config.")
        return

    for asset in all_assets:
        run_drift_analysis_for_asset(asset)


if __name__ == "__main__":
    main()
