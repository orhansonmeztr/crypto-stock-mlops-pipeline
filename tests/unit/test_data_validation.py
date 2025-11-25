"""Unit tests for data validation functions."""

import pandas as pd

from src.data.data_validation import (
    CRYPTO_SCHEMA,
    STOCKS_SCHEMA,
    ValidationResult,
    validate_columns,
    validate_dataset,
    validate_no_duplicates,
    validate_no_empty,
    validate_null_ratio,
    validate_numeric_types,
    validate_positive_values,
)


class TestValidateNoEmpty:
    def test_passes_non_empty(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = ValidationResult("test")
        validate_no_empty(df, result)
        assert result.passed

    def test_fails_empty(self):
        df = pd.DataFrame()
        result = ValidationResult("test")
        validate_no_empty(df, result)
        assert not result.passed


class TestValidateColumns:
    def test_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = ValidationResult("test")
        validate_columns(df, ["a", "b"], result)
        assert result.passed

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        result = ValidationResult("test")
        validate_columns(df, ["a", "b", "c"], result)
        assert not result.passed


class TestValidateNullRatio:
    def test_under_threshold(self):
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0, 4.0, None]})
        result = ValidationResult("test")
        validate_null_ratio(df, ["col"], 0.3, result)
        assert result.passed  # 20% < 30%

    def test_over_threshold(self):
        df = pd.DataFrame({"col": [1.0, None, None, None, None]})
        result = ValidationResult("test")
        validate_null_ratio(df, ["col"], 0.1, result)
        assert not result.passed  # 80% > 10%


class TestValidateNumericTypes:
    def test_numeric_column(self):
        df = pd.DataFrame({"price": [1.0, 2.0, 3.0]})
        result = ValidationResult("test")
        validate_numeric_types(df, ["price"], result)
        assert result.passed

    def test_non_numeric_column(self):
        df = pd.DataFrame({"price": ["a", "b", "c"]})
        result = ValidationResult("test")
        validate_numeric_types(df, ["price"], result)
        assert not result.passed


class TestValidateNoDuplicates:
    def test_no_duplicates(self):
        df = pd.DataFrame({"timestamp": [1, 2, 3], "name": ["A", "A", "A"]})
        result = ValidationResult("test")
        validate_no_duplicates(df, ["timestamp", "name"], result)
        assert result.passed

    def test_has_duplicates(self):
        df = pd.DataFrame({"timestamp": [1, 1, 2], "name": ["A", "A", "B"]})
        result = ValidationResult("test")
        validate_no_duplicates(df, ["timestamp", "name"], result)
        assert not result.passed


class TestValidatePositiveValues:
    def test_all_positive(self):
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0]})
        result = ValidationResult("test")
        validate_positive_values(df, ["price"], result)
        assert result.passed

    def test_negative_values(self):
        df = pd.DataFrame({"price": [10.0, -5.0, 30.0]})
        result = ValidationResult("test")
        validate_positive_values(df, ["price"], result)
        assert not result.passed


class TestValidateDataset:
    def test_valid_stocks_data(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=5),
                "name": ["AAPL"] * 5,
                "last": [150.0, 151.0, 152.0, 153.0, 154.0],
                "high": [155.0] * 5,
                "low": [145.0] * 5,
                "vol": [1000000.0] * 5,
            }
        )
        result = validate_dataset(df, STOCKS_SCHEMA, "TestStocks")
        assert result.passed

    def test_valid_crypto_data(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=5),
                "name": ["BTC-USD"] * 5,
                "price_usd": [50000.0, 50100.0, 50200.0, 50300.0, 50400.0],
                "vol_24h": [1000.0] * 5,
                "market_cap": [900000000.0] * 5,
            }
        )
        result = validate_dataset(df, CRYPTO_SCHEMA, "TestCrypto")
        assert result.passed


from unittest.mock import MagicMock, patch


class TestValidationResult:
    def test_summary_all_passed(self):
        result = ValidationResult("test_ds")
        result.add_check("check1", True, "detail1")
        result.add_check("check2", True, "detail2")
        summary = result.summary()
        assert "PASSED" in summary
        assert result.passed

    def test_summary_with_failure(self):
        result = ValidationResult("test_ds")
        result.add_check("check1", True, "ok")
        result.add_check("check2", False, "bad")
        summary = result.summary()
        assert "FAILED" in summary
        assert not result.passed


class TestValidateNullRatioEdgeCases:
    def test_column_not_in_df(self):
        """When column doesn't exist in df, no check is added."""
        df = pd.DataFrame({"a": [1, 2]})
        result = ValidationResult("test")
        validate_null_ratio(df, ["nonexistent"], 0.1, result)
        assert result.passed  # no check added, so still passed
        assert len(result.checks) == 0


class TestValidateNumericEdgeCases:
    def test_column_not_in_df(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = ValidationResult("test")
        validate_numeric_types(df, ["nonexistent"], result)
        assert result.passed
        assert len(result.checks) == 0


class TestValidateNoDuplicatesEdgeCases:
    def test_no_matching_subset_columns(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = ValidationResult("test")
        validate_no_duplicates(df, ["nonexistent"], result)
        assert result.passed
        assert len(result.checks) == 0


class TestValidatePositiveEdgeCases:
    def test_column_not_in_df(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = ValidationResult("test")
        validate_positive_values(df, ["nonexistent"], result)
        assert result.passed
        assert len(result.checks) == 0

    def test_all_null_column(self):
        df = pd.DataFrame({"price": [None, None]})
        result = ValidationResult("test")
        validate_positive_values(df, ["price"], result)
        assert result.passed
        assert len(result.checks) == 0


class TestMain:
    @patch("src.data.data_validation.Path")
    @patch("src.data.data_validation.load_config")
    @patch("src.data.data_validation.pd.read_csv")
    @patch("src.data.data_validation.validate_dataset")
    @patch("src.data.data_validation.sys.exit")
    def test_main_all_pass(
        self, mock_exit, mock_validate, mock_read_csv, mock_load_config, mock_path
    ):
        mock_load_config.return_value = {"assets": {"stocks": ["AAPL"], "cryptos": ["BTC-USD"]}}

        # Build a mock filesystem tree that behaves like real Path objects
        mock_root = MagicMock()
        mock_path.return_value.resolve.return_value.parents.__getitem__ = lambda s, i: mock_root

        mock_processed = MagicMock()
        mock_config_path = MagicMock()
        mock_root.__truediv__ = lambda s, k: {
            "data": MagicMock(__truediv__=lambda s2, k2: mock_processed),
            "configs": MagicMock(__truediv__=lambda s2, k2: mock_config_path),
        }[k]

        mock_stocks_file = MagicMock()
        mock_stocks_file.exists.return_value = True
        mock_crypto_file = MagicMock()
        mock_crypto_file.exists.return_value = True
        mock_processed.__truediv__ = lambda s, k: {
            "stocks.csv": mock_stocks_file,
            "cryptocurrency.csv": mock_crypto_file,
        }[k]

        df_stocks = pd.DataFrame({"name": ["AAPL"]})
        df_crypto = pd.DataFrame({"name": ["BTC-USD"]})
        mock_read_csv.side_effect = [df_stocks, df_crypto, df_stocks, df_crypto]

        mock_result = MagicMock()
        mock_result.passed = True
        mock_validate.return_value = mock_result

        from src.data.data_validation import main

        main()

        mock_exit.assert_not_called()

    @patch("src.data.data_validation.Path")
    @patch("src.data.data_validation.load_config")
    @patch("src.data.data_validation.pd.read_csv")
    @patch("src.data.data_validation.validate_dataset")
    @patch("src.data.data_validation.sys.exit")
    def test_main_validation_fails(
        self, mock_exit, mock_validate, mock_read_csv, mock_load_config, mock_path
    ):
        mock_load_config.return_value = {"assets": {"stocks": [], "cryptos": []}}

        mock_root = MagicMock()
        mock_path.return_value.resolve.return_value.parents.__getitem__ = lambda s, i: mock_root

        mock_processed = MagicMock()
        mock_config_path = MagicMock()
        mock_root.__truediv__ = lambda s, k: {
            "data": MagicMock(__truediv__=lambda s2, k2: mock_processed),
            "configs": MagicMock(__truediv__=lambda s2, k2: mock_config_path),
        }[k]

        mock_stocks_file = MagicMock()
        mock_stocks_file.exists.return_value = True
        mock_crypto_file = MagicMock()
        mock_crypto_file.exists.return_value = False
        mock_processed.__truediv__ = lambda s, k: {
            "stocks.csv": mock_stocks_file,
            "cryptocurrency.csv": mock_crypto_file,
        }[k]

        df_stocks = pd.DataFrame({"name": ["AAPL"]})
        mock_read_csv.side_effect = [df_stocks, df_stocks]

        mock_result = MagicMock()
        mock_result.passed = False
        mock_validate.return_value = mock_result

        from src.data.data_validation import main

        main()

        mock_exit.assert_called_once_with(1)
