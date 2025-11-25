from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.models.hybrid_model import HybridModel


class TestHybridModel:
    def test_predict_case_1_batch(self):
        xgb_mock = MagicMock()
        xgb_mock.predict.return_value = np.array([10.0])

        lstm_mock = MagicMock()
        scaler_mock = MagicMock()

        model = HybridModel(
            xgb_model=xgb_mock, lstm_model=lstm_mock, scaler=scaler_mock, look_back=2
        )

        # In batch mode, "lstm_pred" is already present
        df_input = pd.DataFrame(
            {"feature1": [1, 2], "lstm_pred": [5.0, 6.0], "close": [10.0, 11.0]}
        )

        preds = model.predict(context=None, model_input=df_input)

        # XGBoost should have been called
        xgb_mock.predict.assert_called_once()
        args, kwargs = xgb_mock.predict.call_args
        called_df = args[0]

        assert "close" not in called_df.columns
        assert "lstm_pred" in called_df.columns
        assert preds == np.array([10.0])
        lstm_mock.predict.assert_not_called()

    def test_predict_case_2_live_inference(self):
        xgb_mock = MagicMock()
        xgb_mock.predict.return_value = np.array([42.0])

        lstm_mock = MagicMock()
        lstm_mock.predict.return_value = np.array([[0.5]])

        scaler_mock = MagicMock()
        scaler_mock.transform.return_value = np.array([[1.0], [2.0]])
        scaler_mock.inverse_transform.return_value = np.array([[100.0]])

        model = HybridModel(
            xgb_model=xgb_mock, lstm_model=lstm_mock, scaler=scaler_mock, look_back=2
        )

        # Live mode: we don't have lstm_pred but we do have enough rows
        df_input = pd.DataFrame({"feature1": [1, 2, 3], "close": [10.0, 11.0, 12.0]})

        preds = model.predict(context=None, model_input=df_input)

        # LSTM should have been called
        lstm_mock.predict.assert_called_once()

        # Scaler checks
        scaler_mock.transform.assert_called_once()
        scaler_mock.inverse_transform.assert_called_once()

        # XGBoost should have been called on the final modified row
        xgb_mock.predict.assert_called_once()
        args, kwargs = xgb_mock.predict.call_args
        called_df = args[0]

        assert "close" not in called_df.columns
        assert "lstm_pred" in called_df.columns
        assert called_df["lstm_pred"].iloc[0] == 100.0
        assert preds == np.array([42.0])

    def test_predict_case_2_not_enough_data(self):
        model = HybridModel(xgb_model=None, lstm_model=None, scaler=None, look_back=5)
        df_input = pd.DataFrame({"feature1": [1, 2], "close": [10.0, 11.0]})

        with pytest.raises(ValueError, match="must contain at least 5 rows"):
            model.predict(context=None, model_input=df_input)
