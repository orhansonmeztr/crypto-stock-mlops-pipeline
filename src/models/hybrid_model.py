from typing import Any

import pandas as pd
from mlflow.pyfunc import PythonModel


class HybridModel(PythonModel):
    """
    A custom MLflow PythonModel that wraps the Hybrid LSTM + XGBoost logic.
    It handles the internal generation of LSTM features during inference.
    """

    def __init__(
        self,
        xgb_model: Any,
        lstm_model: Any,
        scaler: Any,
        look_back: int = 60,
        feature_cols: list[str] | None = None,
    ):
        self.xgb_model = xgb_model
        self.lstm_model = lstm_model
        self.scaler = scaler
        self.look_back = look_back
        self.feature_cols = feature_cols if feature_cols else ["close"]

    def predict(self, context, model_input: pd.DataFrame) -> Any:
        """
        Performs inference using the hybrid architecture.
        """
        # Case 1: Batch Prediction / Training Mode
        if "lstm_pred" in model_input.columns:
            # If 'close' is present but not needed for XGBoost, drop it to be safe
            if "close" in model_input.columns:
                model_input = model_input.drop(columns=["close"])
            return self.xgb_model.predict(model_input)

        # Case 2: Live Inference Mode
        if len(model_input) < self.look_back:
            raise ValueError(
                f"Input data must contain at least {self.look_back} rows for LSTM context."
            )

        # Prepare data for LSTM
        input_slice = model_input[self.feature_cols].tail(self.look_back)
        input_scaled = self.scaler.transform(input_slice)
        lstm_input = input_scaled.reshape(1, self.look_back, len(self.feature_cols))

        # Predict with LSTM
        lstm_pred_scaled = self.lstm_model.predict(lstm_input, verbose=0)
        lstm_pred = self.scaler.inverse_transform(
            pd.DataFrame(lstm_pred_scaled, columns=self.feature_cols)
        )

        # Prepare input for XGBoost
        xgb_input = model_input.iloc[[-1]].copy()
        xgb_input["lstm_pred"] = float(lstm_pred[0][0])

        # CLEANUP: Drop 'close' column because XGBoost was trained without it
        if "close" in xgb_input.columns:
            xgb_input = xgb_input.drop(columns=["close"])

        # Final Prediction with XGBoost
        return self.xgb_model.predict(xgb_input)
