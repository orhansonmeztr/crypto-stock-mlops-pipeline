import numpy as np
from xgboost import XGBRegressor

from src.models.xgboost_model import create_xgboost_model, predict_xgboost, train_xgboost


class TestXGBoostModel:
    def test_create_xgboost_model(self):
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}
        model = create_xgboost_model(params)
        assert isinstance(model, XGBRegressor)
        assert model.n_estimators == 10
        assert model.max_depth == 3

    def test_train_xgboost(self):
        params = {"n_estimators": 5, "max_depth": 2}
        model = create_xgboost_model(params)
        X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y_train = np.array([1.0, 2.0, 3.0])

        trained_model = train_xgboost(model, X_train, y_train)
        assert trained_model is model
        # Check if the model is fitted by trying to predict
        preds = trained_model.predict(X_train)
        assert len(preds) == 3

    def test_predict_xgboost(self):
        params = {"n_estimators": 5, "max_depth": 2}
        model = create_xgboost_model(params)
        X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y_train = np.array([1.0, 2.0, 3.0])
        trained_model = train_xgboost(model, X_train, y_train)

        preds = predict_xgboost(trained_model, X_train)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 3
