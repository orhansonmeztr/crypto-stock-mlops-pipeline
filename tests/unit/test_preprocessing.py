"""Unit tests for data preprocessing functions."""

from unittest.mock import patch

import pandas as pd

from src.data.preprocess import (
    clean_column_names,
    clean_numeric_columns,
    main,
    preprocess_crypto,
    preprocess_stocks,
)


class TestCleanColumnNames:
    def test_removes_special_chars(self):
        df = pd.DataFrame({"Chg_%": [1], "Price (USD)": [2], "Vol.": [3]})
        result = clean_column_names(df)
        for col in result.columns:
            assert col.isidentifier() or col.replace("_", "").isalnum()

    def test_lowercases_columns(self):
        df = pd.DataFrame({"Price": [1], "VOLUME": [2]})
        result = clean_column_names(df)
        assert all(col == col.lower() for col in result.columns)


class TestCleanNumericColumns:
    def test_already_numeric(self):
        series = pd.Series([1.0, 2.0, 3.0])
        result = clean_numeric_columns(series)
        assert pd.api.types.is_numeric_dtype(result)
        assert result.iloc[0] == 1.0

    def test_millions_suffix(self):
        series = pd.Series(["1.5M", "2M"])
        result = clean_numeric_columns(series)
        assert result.iloc[0] == 1_500_000
        assert result.iloc[1] == 2_000_000

    def test_millions_suffix_invalid(self):
        series = pd.Series(["invalidM"])
        result = clean_numeric_columns(series)
        assert pd.isna(result.iloc[0])

    def test_billions_suffix(self):
        series = pd.Series(["1B"])
        result = clean_numeric_columns(series)
        assert result.iloc[0] == 1_000_000_000

    def test_billions_suffix_invalid(self):
        series = pd.Series(["invalidB"])
        result = clean_numeric_columns(series)
        assert pd.isna(result.iloc[0])

    def test_invalid_value(self):
        series = pd.Series(["invalid"])
        result = clean_numeric_columns(series)
        assert pd.isna(result.iloc[0])

    def test_dollar_sign_and_comma(self):
        series = pd.Series(["$1,234.56"])
        result = clean_numeric_columns(series)
        assert abs(result.iloc[0] - 1234.56) < 0.01

    def test_nan_handling(self):
        series = pd.Series(["nan", None, "100"])
        result = clean_numeric_columns(series)
        assert pd.isna(result.iloc[0])
        assert result.iloc[2] == 100.0


class TestPreprocessStocks:
    def test_basic_processing(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=10),
                "name": ["Apple"] * 10,
                "last": [150.0 + i for i in range(10)],
                "high": [155.0] * 10,
                "low": [145.0] * 10,
                "chg_%": [0.5] * 10,
                "vol": [1000000.0] * 10,
            }
        )
        result = preprocess_stocks(df)
        assert not result.empty
        assert "last" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])

    def test_drops_rows_without_price(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=3),
                "name": ["Apple", None, "Apple"],
                "last": [100.0, None, 102.0],
            }
        )
        result = preprocess_stocks(df)
        assert len(result) <= 2  # NaN rows dropped


class TestPreprocessCrypto:
    def test_basic_processing(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=10),
                "name": ["BTC-USD"] * 10,
                "symbol": ["BTC"] * 10,
                "price_usd": [50000.0 + i * 100 for i in range(10)],
                "vol_24h": [1000.0] * 10,
                "total_vol": [5.0] * 10,
                "chg_24h": [1.0] * 10,
                "chg_7d": [-0.5] * 10,
                "market_cap": [1000000.0] * 10,
            }
        )
        result = preprocess_crypto(df)
        assert not result.empty
        assert "price_usd" in result.columns
        assert pd.api.types.is_numeric_dtype(result["price_usd"])

    def test_empty_after_na_drop(self):
        df = pd.DataFrame(
            {
                "timestamp": ["2025-01-01"],
                "name": [None],
                "price_usd": [None],
            }
        )
        result = preprocess_crypto(df)
        assert result.empty


class TestMain:
    @patch("src.data.preprocess.pd.read_csv")
    @patch("src.data.preprocess.pd.DataFrame.to_csv")
    @patch("src.data.preprocess.Path.exists")
    def test_main(self, mock_exists, mock_to_csv, mock_read_csv):
        # file exists logic -> returns true for tests
        mock_exists.return_value = True

        # mock pd.read_csv
        df_stocks = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=2),
                "name": ["Apple"] * 2,
                "last": [150.0, 151.0],
            }
        )

        df_crypto = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=2),
                "name": ["BTC-USD"] * 2,
                "price_usd": [50.0, 51.0],
            }
        )

        mock_read_csv.side_effect = [df_stocks, df_crypto]

        main()

        assert mock_to_csv.call_count == 2

    @patch("src.data.preprocess.pd.read_csv")
    @patch("src.data.preprocess.pd.DataFrame.to_csv")
    @patch("src.data.preprocess.Path.exists")
    def test_main_file_not_exist(self, mock_exists, mock_to_csv, mock_read_csv):
        # file does not exist logic
        mock_exists.return_value = False

        main()

        assert mock_to_csv.call_count == 0

    @patch("src.data.preprocess.pd.read_csv")
    @patch("src.data.preprocess.pd.DataFrame.to_csv")
    @patch("src.data.preprocess.Path.exists")
    def test_main_processing_error(self, mock_exists, mock_to_csv, mock_read_csv):
        mock_exists.return_value = True

        # raise exception
        mock_read_csv.side_effect = Exception("Test read error")

        main()

        assert mock_to_csv.call_count == 0
