"""
Evaluation metrics module for model performance assessment.

Provides MAPE, MinMax RMSE, Directional Accuracy, and standard MAE/RMSE
as specified in the project plan (referencing arxiv:2506.22055).
"""

import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    Returns percentage value (e.g., 5.2 means 5.2%).
    Filters out zero values in y_true to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        logging.warning("MAPE: All true values are zero, returning inf.")
        return float("inf")

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return float(mape)


def calculate_minmax_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MinMax Normalized RMSE.

    Normalizes RMSE by the range of true values: RMSE / (max - min).
    Returns a value between 0 and 1 (lower is better).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = calculate_rmse(y_true, y_pred)
    value_range = y_true.max() - y_true.min()

    if value_range == 0:
        logging.warning("MinMax RMSE: All true values are identical, returning inf.")
        return float("inf")

    return float(rmse / value_range)


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy — percentage of correct direction predictions.

    Compares the direction of change (up/down) between consecutive predictions
    and actual values. Returns percentage (e.g., 65.0 means 65%).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        logging.warning("Directional Accuracy: Need at least 2 samples.")
        return 0.0

    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))

    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)

    return float(correct / total * 100)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculates all evaluation metrics and returns them as a dictionary.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dictionary with keys: rmse, mae, mape, minmax_rmse, directional_accuracy
    """
    metrics = {
        "rmse": calculate_rmse(y_true, y_pred),
        "mae": calculate_mae(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred),
        "minmax_rmse": calculate_minmax_rmse(y_true, y_pred),
        "directional_accuracy": calculate_directional_accuracy(y_true, y_pred),
    }

    logging.info(
        f"Evaluation — RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, "
        f"MAPE: {metrics['mape']:.2f}%, MinMax RMSE: {metrics['minmax_rmse']:.4f}, "
        f"Dir Accuracy: {metrics['directional_accuracy']:.1f}%"
    )

    return metrics
