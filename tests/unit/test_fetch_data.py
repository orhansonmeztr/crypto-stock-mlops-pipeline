from unittest.mock import patch

import pandas as pd

from src.data.fetch_data import fetch_asset_data, load_config, main


class TestFetchData:
    @patch("src.data.fetch_data.yf.download")
    def test_fetch_asset_data_stock(self, mock_download):
        df_mock = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "Close": [105.0, 102.0],
                "High": [106.0, 103.0],
                "Low": [99.0, 100.0],
                "Volume": [1000, 2000],
            },
            index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
        )
        df_mock.index.name = "Date"
        mock_download.return_value = df_mock

        df_result = fetch_asset_data("AAPL", is_crypto=False)

        assert not df_result.empty
        assert "name" in df_result.columns
        assert df_result["name"].iloc[0] == "AAPL"
        assert "last" in df_result.columns
        assert df_result["last"].iloc[0] == 105.0
        assert "chg_abs" in df_result.columns
        assert df_result["chg_abs"].iloc[0] == 5.0
        assert "vol" in df_result.columns

    @patch("src.data.fetch_data.yf.download")
    def test_fetch_asset_data_crypto(self, mock_download):
        df_mock = pd.DataFrame(
            {
                "Open": [50000.0, 51000.0],
                "Close": [52000.0, 50500.0],
                "High": [53000.0, 52000.0],
                "Low": [49000.0, 50000.0],
                "Volume": [1000, 2000],
            },
            index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
        )
        df_mock.index.name = "Date"
        mock_download.return_value = df_mock

        df_result = fetch_asset_data("BTC-USD", is_crypto=True)

        assert not df_result.empty
        assert "name" in df_result.columns
        assert df_result["name"].iloc[0] == "BTC-USD"
        assert "symbol" in df_result.columns
        assert df_result["symbol"].iloc[0] == "BTC"
        assert "price_usd" in df_result.columns
        assert "chg_7d" in df_result.columns
        assert "market_cap" in df_result.columns

    @patch("src.data.fetch_data.yf.download")
    def test_fetch_asset_data_empty(self, mock_download):
        mock_download.return_value = pd.DataFrame()
        df_result = fetch_asset_data("UNKNOWN", is_crypto=False)
        assert df_result.empty

    @patch("src.data.fetch_data.yf.download")
    def test_fetch_asset_data_multiindex(self, mock_download):
        # Create a df with MultiIndex columns
        cols = pd.MultiIndex.from_tuples([("Close", "AAPL"), ("Open", "AAPL")])
        df_mock = pd.DataFrame([[150.0, 149.0]], columns=cols)
        df_mock.index = pd.to_datetime(["2023-01-01"])
        df_mock.index.name = "Date"
        mock_download.return_value = df_mock

        df_result = fetch_asset_data("AAPL", is_crypto=False)
        assert not df_result.empty
        assert "last" in df_result.columns
        assert df_result["last"].iloc[0] == 150.0

    @patch("src.data.fetch_data.Path")
    @patch("src.data.fetch_data.load_config")
    @patch("src.data.fetch_data.pd.DataFrame.to_csv")
    @patch("src.data.fetch_data.fetch_asset_data")
    def test_main(self, mock_fetch, mock_to_csv, mock_load_config, mock_path):
        mock_load_config.return_value = {"assets": {"stocks": ["AAPL"], "cryptos": ["BTC-USD"]}}

        df_stocks = pd.DataFrame({"name": ["AAPL"], "last": [150.0]})
        df_crypto = pd.DataFrame({"name": ["BTC-USD"], "price_usd": [50000.0]})

        mock_fetch.side_effect = [df_stocks, df_crypto]

        main()

        assert mock_to_csv.call_count == 2

    @patch("src.data.fetch_data.Path")
    @patch("src.data.fetch_data.load_config")
    @patch("src.data.fetch_data.pd.DataFrame.to_csv")
    @patch("src.data.fetch_data.fetch_asset_data")
    def test_main_empty_assets(self, mock_fetch, mock_to_csv, mock_load_config, mock_path):
        mock_load_config.return_value = {"assets": {}}

        main()
        mock_fetch.assert_not_called()
        mock_to_csv.assert_not_called()

    def test_load_config(self, tmp_path):
        config_file = tmp_path / "dummy.yaml"
        config_file.write_text("assets:\n  stocks:\n    - AAPL")
        config = load_config(config_file)
        assert "assets" in config
        assert "stocks" in config["assets"]
        assert config["assets"]["stocks"][0] == "AAPL"
