"""
XGBoost model module — provides model creation and training functions.
Extracted from train.py for modularity and testability.
"""

import logging

import numpy as np
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_xgboost_model(params: dict) -> XGBRegressor:
    """
    Creates an XGBRegressor with the given parameters.

    Args:
        params: Dictionary of XGBoost hyperparameters.

    Returns:
        XGBRegressor instance (not yet fitted).
    """
    return XGBRegressor(**params)


def train_xgboost(model: XGBRegressor, x_train, y_train) -> XGBRegressor:
    """
    Fits the XGBoost model on training data.

    Args:
        model: Unfitted XGBRegressor.
        x_train: Training features.
        y_train: Training targets.

    Returns:
        Fitted XGBRegressor.
    """
    model.fit(x_train, y_train)
    logging.info(
        f"XGBoost model trained. Features: {x_train.shape[1]}, Samples: {x_train.shape[0]}"
    )
    return model


def predict_xgboost(model: XGBRegressor, x) -> np.ndarray:
    """
    Makes predictions using a fitted XGBoost model.

    Args:
        model: Fitted XGBRegressor.
        x: Input features.

    Returns:
        Numpy array of predictions.
    """
    return model.predict(x)
