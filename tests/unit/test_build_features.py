from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.features.build_features import main, process_asset

# Mock config
MOCK_CONFIG = {
    "assets": {"stocks": ["AAPL"], "cryptos": ["BTC-USD"]},
    "features": {
        "moving_averages": [5],
        "macd_slow": 26,
        "macd_fast": 12,
        "macd_signal": 9,
        "rsi_window": 14,
        "bollinger_window": 20,
        "bollinger_dev": 2,
        "volatility_window": 30,
        "lags": [1, 2],
    },
}


class TestBuildFeatures:
    def test_process_asset_stock(self):
        dates = pd.date_range("2023-01-01", periods=100)
        df = pd.DataFrame(
            {"timestamp": dates, "name": ["AAPL"] * 100, "last": np.random.rand(100) * 100 + 50}
        )

        df_feat = process_asset(df, "AAPL", "stock", MOCK_CONFIG)

        # After dropna because of indicator windows (e.g. volatility 30), length will decrease
        assert not df_feat.empty
        assert "target" in df_feat.columns
        assert "asset_name" in df_feat.columns
        assert df_feat["asset_name"].iloc[0] == "AAPL"
        assert "asset_type" in df_feat.columns
        assert df_feat["asset_type"].iloc[0] == "stock"
        assert "close" in df_feat.columns

    def test_process_asset_crypto(self):
        dates = pd.date_range("2023-01-01", periods=100)
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "name": ["BTC-USD"] * 100,
                "price_usd": np.random.rand(100) * 1000 + 50000,
            }
        )

        df_feat = process_asset(df, "BTC-USD", "crypto", MOCK_CONFIG)

        assert not df_feat.empty
        assert "target" in df_feat.columns
        assert df_feat["asset_name"].iloc[0] == "BTC-USD"
        assert df_feat["asset_type"].iloc[0] == "crypto"
        assert "close" in df_feat.columns

    def test_process_asset_not_found(self):
        df = pd.DataFrame({"name": ["GOOGL"], "last": [100]})
        df_feat = process_asset(df, "AAPL", "stock", MOCK_CONFIG)
        assert df_feat.empty

    def test_process_asset_all_nans(self):
        dates = pd.date_range("2023-01-01", periods=10)
        df = pd.DataFrame({"timestamp": dates, "name": ["AAPL"] * 10, "last": [np.nan] * 10})
        df_feat = process_asset(df, "AAPL", "stock", MOCK_CONFIG)
        assert df_feat.empty

    @patch("src.features.build_features.Path")
    @patch("src.features.build_features.pd.read_csv")
    @patch("src.features.build_features.pd.DataFrame.to_csv")
    @patch("src.features.build_features.load_config")
    def test_main(self, mock_load_config, mock_to_csv, mock_read_csv, mock_path):
        mock_load_config.return_value = MOCK_CONFIG

        # Mock paths
        mock_root = MagicMock()
        mock_path.return_value.resolve.return_value.parents = [None, None, mock_root]

        mock_processed = mock_root / "data" / "processed"
        mock_features = mock_root / "data" / "features"

        mock_stocks_file = mock_processed / "stocks.csv"
        mock_crypto_file = mock_processed / "cryptocurrency.csv"

        mock_stocks_file.exists.return_value = True
        mock_crypto_file.exists.return_value = True

        # Mock read_csv returns DataFrames long enough to generate features
        dates = pd.date_range("2023-01-01", periods=100)
        df_stocks = pd.DataFrame(
            {"timestamp": dates, "name": ["AAPL"] * 100, "last": np.random.rand(100) * 100 + 50}
        )
        df_crypto = pd.DataFrame(
            {
                "timestamp": dates,
                "name": ["BTC-USD"] * 100,
                "price_usd": np.random.rand(100) * 1000 + 50000,
            }
        )

        mock_read_csv.side_effect = [df_stocks, df_crypto]

        main()

        mock_to_csv.assert_called_once()
