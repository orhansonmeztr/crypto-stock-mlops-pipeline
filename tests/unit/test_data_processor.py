from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.data.data_processor import prepare_training_data, prepare_xgboost_features


class TestDataProcessor:
    @patch("src.data.data_processor.prepare_lstm_data")
    def test_prepare_training_data(self, mock_prepare):
        # Setup mock return
        n_samples = 100
        look_back = 10
        X_lstm = np.random.rand(n_samples, look_back, 1)
        y_lstm = np.random.rand(n_samples)
        scaler = MagicMock()
        mock_prepare.return_value = (X_lstm, y_lstm, scaler)

        config = {
            "model_params": {"lstm": {"look_back": look_back}},
            "training": {"train_ratio": 0.6, "validation_ratio": 0.2},
        }

        df = pd.DataFrame(
            {
                "close": np.random.rand(n_samples + look_back),
                "feature1": np.random.rand(n_samples + look_back),
            }
        )

        lstm_data, xgb_data_raw, returned_scaler = prepare_training_data(df, config)

        assert lstm_data is not None
        assert xgb_data_raw is not None
        assert returned_scaler is scaler

        # Check splits
        train_end = int(n_samples * 0.6)
        val_end = int(n_samples * 0.8)

        assert len(lstm_data["X_train"]) == train_end
        assert len(lstm_data["X_val"]) == val_end - train_end
        assert len(lstm_data["X_test"]) == n_samples - val_end

        # Check xgb data
        assert len(xgb_data_raw["df_val"]) == val_end - train_end
        assert len(xgb_data_raw["df_test"]) == n_samples - val_end

    @patch("src.data.data_processor.prepare_lstm_data")
    def test_prepare_training_data_small_sample(self, mock_prepare):
        # Setup mock return
        n_samples = 4
        X_lstm = np.random.rand(n_samples, 10, 1)
        y_lstm = np.random.rand(n_samples)
        mock_prepare.return_value = (X_lstm, y_lstm, None)

        config = {
            "model_params": {"lstm": {"look_back": 10}},
            "training": {"train_ratio": 0.6, "validation_ratio": 0.2},
        }

        df = pd.DataFrame({"close": np.random.rand(14)})

        lstm_data, xgb_data_raw, returned_scaler = prepare_training_data(df, config)

        assert lstm_data is None
        assert xgb_data_raw is None

    @patch("src.data.data_processor.prepare_lstm_data")
    def test_prepare_training_data_empty_splits(self, mock_prepare):
        # Setup mock return
        n_samples = 6
        X_lstm = np.random.rand(n_samples, 10, 1)
        y_lstm = np.random.rand(n_samples)
        mock_prepare.return_value = (X_lstm, y_lstm, None)

        config = {
            "model_params": {"lstm": {"look_back": 10}},
            # Very small val ratio so split gets rounded to empty
            "training": {"train_ratio": 0.99, "validation_ratio": 0.001},
        }

        df = pd.DataFrame({"close": np.random.rand(16)})

        lstm_data, xgb_data_raw, returned_scaler = prepare_training_data(df, config)

        assert lstm_data is None
        assert xgb_data_raw is None

    def test_prepare_xgboost_features(self):
        df_val = pd.DataFrame(
            {
                "target": [10, 11],
                "asset_name": ["A", "A"],
                "asset_type": ["S", "S"],
                "close": [9, 10],
                "feature1": [1.1, 1.2],
            }
        )

        df_test = pd.DataFrame(
            {
                "target": [12],
                "asset_name": ["A"],
                "asset_type": ["S"],
                "close": [11],
                "feature1": [1.3],
            }
        )

        lstm_pred_val = np.array([9.5, 10.5])
        lstm_pred_test = np.array([11.5])

        xgb_data = prepare_xgboost_features(df_val, df_test, lstm_pred_val, lstm_pred_test)

        assert "X_train" in xgb_data
        assert "y_train" in xgb_data
        assert "X_test" in xgb_data
        assert "y_test" in xgb_data

        # Excludes should not be in X_train
        assert "target" not in xgb_data["X_train"].columns
        assert "close" not in xgb_data["X_train"].columns

        assert "lstm_pred" in xgb_data["X_train"].columns
        assert "lstm_pred" in xgb_data["X_test"].columns

        assert len(xgb_data["X_train"]) == 2
        assert len(xgb_data["X_test"]) == 1
