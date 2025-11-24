import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path so we can import src modules
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.features.build_features import create_time_series_features
from src.models.lstm_model import prepare_lstm_data
from src.models.train_model import sanitize_artifact_name


# --- Test 1: Artifact Name Sanitization ---
def test_sanitize_artifact_name():
    """Test if invalid characters are replaced correctly."""
    assert sanitize_artifact_name("Amazon.com") == "Amazon_com"
    assert sanitize_artifact_name("Bitcoin/USD") == "Bitcoin_USD"
    assert sanitize_artifact_name("NormalName") == "NormalName"
    assert sanitize_artifact_name("Space Name") == "Space_Name"


# --- Test 2: LSTM Data Preparation ---
def test_prepare_lstm_data():
    """Test if data is correctly reshaped for LSTM input."""
    # Create mock data: 100 days of prices
    df = pd.DataFrame({"close": np.arange(100), "feature_2": np.arange(100) * 2})

    look_back = 10
    X, y, scaler = prepare_lstm_data(
        df, look_back=look_back, target_col="close", feature_cols=["close"]
    )

    # Check dimensions
    # Total samples = 100 - 10 = 90
    assert X.shape == (90, look_back, 1)  # [samples, time_steps, features]
    assert y.shape == (90,)  # [samples]

    # Check logic: y at index 0 should be price at index 10 (because look_back=10)
    # Note: Data is scaled between 0-1, so we can't check exact values easily without inverse transform
    # But we can check shapes and types
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


# --- Test 3: Feature Engineering ---
def test_create_time_series_features():
    """Test if technical indicators are added without errors."""
    # Create mock data enough for indicators (need > 50 rows for ma_50)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({"close": np.random.rand(100) * 100}, index=dates)

    df_feat = create_time_series_features(df, target_col="close")

    # Check if new columns exist
    expected_columns = ["rsi", "macd", "bb_width", "ma_50", "log_return"]
    for col in expected_columns:
        assert col in df_feat.columns, f"Missing column: {col}"

    # Check if NaN values are handled (first few rows will be NaN due to lags/rolling)
    # The function itself returns NaNs, dropping them is done in process_asset
    assert pd.isna(df_feat["ma_50"].iloc[0])  # First row MA should be NaN
