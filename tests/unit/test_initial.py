import numpy as np
import pandas as pd

from src.features.build_features import create_time_series_features
from src.models.lstm_model import prepare_lstm_data
from src.utils.config_utils import sanitize_artifact_name

# Mock config
MOCK_CONFIG = {
    "features": {
        "moving_averages": [50],
        "lags": [1, 2],
        "rsi_window": 14,
        "bollinger_window": 20,
        "bollinger_dev": 2,
        "volatility_window": 30,
    }
}


# Test 1: Artifact Name Sanitization
def test_sanitize_artifact_name():
    """Test if invalid characters are replaced correctly."""
    assert sanitize_artifact_name("AMZN") == "AMZN"
    assert sanitize_artifact_name("BTC-USD") == "BTC_USD"
    assert sanitize_artifact_name("NormalName") == "NormalName"
    assert sanitize_artifact_name("Space Name") == "Space_Name"


# Test 2: LSTM Data Preparation
def test_prepare_lstm_data():
    """Test if data is correctly reshaped for LSTM input."""
    df = pd.DataFrame({"close": np.arange(100), "feature_2": np.arange(100) * 2})

    look_back = 10
    X, y, scaler = prepare_lstm_data(
        df, look_back=look_back, target_col="close", feature_cols=["close"]
    )

    assert X.shape == (90, look_back, 1)
    assert y.shape == (90,)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


# Test 3: Feature Engineering
def test_create_time_series_features():
    """Test if technical indicators are added without errors."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({"close": np.random.rand(100) * 100}, index=dates)

    # Passing MOCK_CONFIG
    df_feat = create_time_series_features(df, MOCK_CONFIG, target_col="close")

    expected_columns = ["rsi", "macd", "bb_width", "ma_50", "log_return"]
    for col in expected_columns:
        assert col in df_feat.columns, f"Missing column: {col}"

    assert pd.isna(df_feat["ma_50"].iloc[0])
