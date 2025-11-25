"""Unit tests for config utility functions."""

import os
from unittest.mock import patch

import pytest

from src.utils.config_utils import (
    get_model_registry_name,
    load_config,
    override_config_with_params,
    sanitize_artifact_name,
)


class TestSanitizeArtifactName:
    def test_dot_replacement(self):
        assert sanitize_artifact_name("AMZN") == "AMZN"

    def test_space_replacement(self):
        assert sanitize_artifact_name("S P 500") == "S_P_500"

    def test_slash_replacement(self):
        assert sanitize_artifact_name("BTC-USD") == "BTC_USD"

    def test_no_change_needed(self):
        assert sanitize_artifact_name("BTC") == "BTC"

    def test_multiple_special_chars(self):
        assert sanitize_artifact_name("A.B/C:D") == "A_B_C_D"


class TestGetModelRegistryName:
    @patch.dict(os.environ, {"DATABRICKS_CATALOG": "my_catalog", "DATABRICKS_SCHEMA": "my_schema"})
    def test_unity_catalog_format(self):
        result = get_model_registry_name("test_model")
        assert result == "my_catalog.my_schema.test_model"

    @patch.dict(os.environ, {"DATABRICKS_CATALOG": "", "DATABRICKS_SCHEMA": ""}, clear=False)
    def test_local_format(self):
        # When catalog/schema are empty strings, use local format
        result = get_model_registry_name("test_model")
        assert "test_model" in result

    def test_sanitizes_model_name(self):
        result = get_model_registry_name("My Model.v1")
        assert "." not in result.split(".")[-1]  # The model name part has no dots
        assert " " not in result


class TestLoadConfig:
    def test_loads_existing_config(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("key: value\nnested:\n  a: 1\n")
        result = load_config(config_file)
        assert result["key"] == "value"
        assert result["nested"]["a"] == 1

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


class TestOverrideConfigWithParams:
    def test_overrides_lstm_params(self):
        config = {
            "model_params": {
                "lstm": {"units": 50, "dropout": 0.2, "learning_rate": 0.001},
                "xgboost": {"n_estimators": 100},
            }
        }
        best_params = {"lstm_units": 80, "lstm_dropout": 0.3, "lstm_lr": 0.005}
        override_config_with_params(config, best_params)

        assert config["model_params"]["lstm"]["units"] == 80
        assert config["model_params"]["lstm"]["dropout"] == 0.3
        assert config["model_params"]["lstm"]["learning_rate"] == 0.005

    def test_overrides_xgboost_params(self):
        config = {
            "model_params": {
                "lstm": {"units": 50},
                "xgboost": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                },
            }
        }
        best_params = {
            "xgb_n_estimators": 200,
            "xgb_max_depth": 8,
            "xgb_lr": 0.1,
            "xgb_subsample": 0.9,
        }
        override_config_with_params(config, best_params)

        assert config["model_params"]["xgboost"]["n_estimators"] == 200
        assert config["model_params"]["xgboost"]["max_depth"] == 8
        assert config["model_params"]["xgboost"]["learning_rate"] == 0.1
        assert config["model_params"]["xgboost"]["subsample"] == 0.9

    def test_leaves_unspecified_params_untouched(self):
        config = {
            "model_params": {
                "lstm": {"units": 50, "epochs": 20},
                "xgboost": {"n_estimators": 100},
            }
        }
        best_params = {"lstm_units": 80}
        override_config_with_params(config, best_params)

        assert config["model_params"]["lstm"]["units"] == 80
        assert config["model_params"]["lstm"]["epochs"] == 20  # Unchanged
