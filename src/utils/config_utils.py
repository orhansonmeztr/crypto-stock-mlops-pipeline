import logging
import os
from pathlib import Path
from typing import Any

import yaml


def get_model_registry_name(model_name: str) -> str:
    """
    Formats the model name based on the environment.

    Logic:
    - If Databricks Unity Catalog env vars are present: Returns 'catalog.schema.model_name'
    - If running locally (Standard MLflow): Returns 'model_name'

    Args:
        model_name (str): Base name of the model (e.g., 'hybrid_lstm_xgboost_btc')

    Returns:
        str: Formatted model name suitable for the current MLflow registry.
    """
    catalog = os.getenv("DATABRICKS_CATALOG", "workspace")
    schema = os.getenv("DATABRICKS_SCHEMA", "default")

    # Sanitize the name: remove spaces, dots, and convert to lowercase
    safe_name = model_name.replace(".", "_").replace(" ", "_").lower()

    if catalog and schema:
        # Format for Databricks Unity Catalog
        full_name = f"{catalog}.{schema}.{safe_name}"
        logging.info(f"Using Unity Catalog format: {full_name}")
        return full_name
    else:
        # Format for Local MLflow
        logging.info(f"Using Local MLflow format: {safe_name}")
        return safe_name


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path (Path): Path to the YAML file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        raise e


def sanitize_artifact_name(name: str) -> str:
    """
    Sanitizes the asset name to be safe for file paths and MLflow registry names.
    Replaces spaces and special chars with underscores.

        name (str): Original asset name (e.g. 'BTC-USD')

    Returns:
        str: Sanitized name (e.g. 'BTC_USD')
    """
    for char in [".", ":", "/", "%", '"', "'", " ", "-"]:
        name = name.replace(char, "_")
    return name


def override_config_with_params(config: dict[str, Any], best_params: dict[str, Any]) -> None:
    """
    Updates the configuration dictionary with optimized hyperparameters.

    Args:
        config (dict): The base configuration dictionary.
        best_params (dict): Dictionary containing optimized parameters.
    """
    # Update LSTM params
    if "lstm_units" in best_params:
        config["model_params"]["lstm"]["units"] = int(best_params["lstm_units"])
    if "lstm_dropout" in best_params:
        config["model_params"]["lstm"]["dropout"] = float(best_params["lstm_dropout"])
    if "lstm_lr" in best_params:
        config["model_params"]["lstm"]["learning_rate"] = float(best_params["lstm_lr"])

    # Update XGBoost params
    if "xgb_n_estimators" in best_params:
        config["model_params"]["xgboost"]["n_estimators"] = int(best_params["xgb_n_estimators"])
    if "xgb_max_depth" in best_params:
        config["model_params"]["xgboost"]["max_depth"] = int(best_params["xgb_max_depth"])
    if "xgb_lr" in best_params:
        config["model_params"]["xgboost"]["learning_rate"] = float(best_params["xgb_lr"])
    if "xgb_subsample" in best_params:
        config["model_params"]["xgboost"]["subsample"] = float(best_params["xgb_subsample"])
