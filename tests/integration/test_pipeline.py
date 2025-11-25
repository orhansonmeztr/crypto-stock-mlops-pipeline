import numpy as np
import pandas as pd

from src.data.preprocess import preprocess_crypto, preprocess_stocks
from src.features.build_features import create_time_series_features

# Mock config for testing features
MOCK_CONFIG = {
    "features": {
        "moving_averages": [7, 30],
        "lags": [1],
        "rsi_window": 14,
        "bollinger_window": 20,
        "bollinger_dev": 2,
        "volatility_window": 30,
    }
}


def test_crypto_preprocessing_pipeline():
    """
    Test raw crypto data cleaning -> Feature Engineering integration.
    """
    # 1. Create Dummy Raw Crypto Data
    raw_crypto = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2025-03-17", periods=50),
            "name": ["BTC-USD"] * 50,
            "symbol": ["BTC"] * 50,
            "price_usd": [
                100.0 + i for i in range(50)
            ],  # Already numeric as per new raw data insights
            "vol_24h": [1000.0 + i * 10 for i in range(50)],
            "total_vol": [5.5] * 50,
            "chg_24h": [1.2] * 50,
            "chg_7d": [-0.5] * 50,
            "market_cap": [1000000.0] * 50,
        }
    )

    # 2. Run Preprocessing
    cleaned_df = preprocess_crypto(raw_crypto)

    assert not cleaned_df.empty
    assert "price_usd" in cleaned_df.columns
    assert pd.api.types.is_numeric_dtype(cleaned_df["price_usd"])
    assert cleaned_df.shape[0] == 50

    # 3. Run Feature Engineering
    # Passing MOCK_CONFIG
    features_df = create_time_series_features(cleaned_df, MOCK_CONFIG, target_col="price_usd")

    # Check if indicators are created
    expected_features = ["rsi", "macd", "bb_high", "return", "log_return"]
    for feat in expected_features:
        assert feat in features_df.columns, f"Missing feature: {feat}"

    # Check Lag Features
    assert "lag_1" in features_df.columns

    # Verify that initial rows have NaNs due to rolling windows
    assert np.isnan(features_df["rsi"].iloc[0])


def test_stocks_preprocessing_pipeline():
    """
    Test raw stocks data cleaning integration.
    """
    raw_stocks = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2025-03-17", periods=30),
            "name": ["Apple"] * 30,
            "last": [150.0 + i for i in range(30)],
            "high": [155.0] * 30,
            "low": [145.0] * 30,
            "chg_pct": [0.5] * 30,
            "vol": [10000000.0] * 30,
        }
    )

    processed_stocks = preprocess_stocks(raw_stocks)

    assert not processed_stocks.empty
    assert "vol" in processed_stocks.columns
    assert processed_stocks["vol"].iloc[0] == 10_000_000.0
