"""Integration tests for FastAPI endpoints using TestClient."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Creates a TestClient with mocked model manager."""
    # Mock the model loading — we don't have real models in test
    with patch("src.api.main.ModelManager") as MockManager:
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.assets = ["BTC-USD", "ETH-USD"]
        mock_instance.models = {"BTC-USD": MagicMock(), "ETH-USD": MagicMock()}
        mock_instance.list_loaded_models.return_value = ["BTC-USD", "ETH-USD"]
        mock_instance.get_model.return_value = None  # Default: no model
        mock_instance.clear.return_value = None
        MockManager.return_value = mock_instance

        # Override API key for testing
        with patch("src.api.main.API_KEY", "test-key"), TestClient(app) as c:
            # Inject mock into app state
            c.app.state.model_manager = mock_instance
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "loaded_models" in data

    def test_health_no_auth_required(self, client):
        """Health endpoint should work without API key."""
        response = client.get("/health")
        assert response.status_code == 200


class TestPredictEndpoint:
    def test_predict_requires_api_key(self, client):
        response = client.post(
            "/predict",
            json={"asset_name": "BTC-USD", "features": [{"close": 100.0}]},
        )
        assert response.status_code == 403

    def test_predict_invalid_api_key(self, client):
        response = client.post(
            "/predict",
            json={"asset_name": "BTC-USD", "features": [{"close": 100.0}]},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 403

    def test_predict_model_not_found(self, client):
        client.app.state.model_manager.get_model.return_value = None
        response = client.post(
            "/predict",
            json={"asset_name": "Dogecoin", "features": [{"close": 100.0}]},
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 404

    def test_predict_empty_features(self, client):
        mock_model = MagicMock()
        client.app.state.model_manager.get_model.return_value = mock_model

        response = client.post(
            "/predict",
            json={"asset_name": "BTC-USD", "features": []},
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 400


class TestModelInfoEndpoint:
    def test_model_info_returns_list(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total_loaded" in data
        assert data["total_loaded"] >= 0


class TestMetricsEndpoint:
    def test_metrics_without_config(self, client):
        with patch.dict(os.environ, {"DATABRICKS_EXPERIMENT_PATH": ""}, clear=False):
            response = client.get("/metrics")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ("unavailable", "error")


class TestBatchPredictEndpoint:
    def test_batch_requires_api_key(self, client):
        payload = {"requests": [{"asset_name": "BTC-USD", "features": [{"close": 100.0}]}]}
        response = client.post("/batch-predict", json=payload)
        assert response.status_code == 403

    def test_batch_returns_metadata(self, client):
        mock_model = MagicMock()
        mock_model.predict.return_value = [42000.0]
        client.app.state.model_manager.get_model.return_value = mock_model

        payload = {"requests": [{"asset_name": "BTC-USD", "features": [{"close": 100.0}]}]}
        response = client.post(
            "/batch-predict",
            json=payload,
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "metadata" in data
        assert data["metadata"]["total_requests"] == 1
