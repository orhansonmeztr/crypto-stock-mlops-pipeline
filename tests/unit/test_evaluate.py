"""Unit tests for the evaluate module."""

import numpy as np

from src.training.evaluate import (
    calculate_directional_accuracy,
    calculate_mae,
    calculate_mape,
    calculate_minmax_rmse,
    calculate_rmse,
    evaluate_model,
)


class TestRMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert calculate_rmse(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        rmse = calculate_rmse(y_true, y_pred)
        expected = np.sqrt(1 / 3)  # sqrt(mean([0, 0, 1]))
        assert abs(rmse - expected) < 1e-6


class TestMAE:
    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0])
        assert calculate_mae(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 22.0, 27.0])
        mae = calculate_mae(y_true, y_pred)
        assert abs(mae - 2.0) < 1e-6


class TestMAPE:
    def test_perfect_prediction(self):
        y = np.array([100.0, 200.0])
        assert calculate_mape(y, y) == 0.0

    def test_known_percentage(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        mape = calculate_mape(y_true, y_pred)
        # (10/100 + 20/200) / 2 * 100 = (0.1 + 0.1) / 2 * 100 = 10%
        assert abs(mape - 10.0) < 1e-6

    def test_handles_zero_true_values(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 2.0])
        result = calculate_mape(y_true, y_pred)
        assert result == float("inf")


class TestMinMaxRMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert calculate_minmax_rmse(y, y) == 0.0

    def test_normalized_between_0_and_1(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([12.0, 22.0, 28.0, 42.0, 48.0])
        result = calculate_minmax_rmse(y_true, y_pred)
        assert 0.0 <= result <= 1.0

    def test_identical_values(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([6.0, 7.0, 8.0])
        result = calculate_minmax_rmse(y_true, y_pred)
        assert result == float("inf")


class TestDirectionalAccuracy:
    def test_all_correct(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.5, 3.5, 5.0])
        assert calculate_directional_accuracy(y_true, y_pred) == 100.0

    def test_all_wrong(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])
        assert calculate_directional_accuracy(y_true, y_pred) == 0.0

    def test_insufficient_data(self):
        y_true = np.array([1.0])
        y_pred = np.array([2.0])
        assert calculate_directional_accuracy(y_true, y_pred) == 0.0


class TestEvaluateModel:
    def test_returns_all_metrics(self):
        y_true = np.array([100.0, 105.0, 110.0, 108.0, 115.0])
        y_pred = np.array([102.0, 104.0, 112.0, 107.0, 116.0])
        result = evaluate_model(y_true, y_pred)

        assert "rmse" in result
        assert "mae" in result
        assert "mape" in result
        assert "minmax_rmse" in result
        assert "directional_accuracy" in result
        assert all(isinstance(v, float) for v in result.values())
