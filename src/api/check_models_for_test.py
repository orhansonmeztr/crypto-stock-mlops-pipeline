# check_models.py
import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

client = MlflowClient()

print("=== workspace.default Models ===\n")
models = client.search_registered_models()

for model in models:
    if model.name.startswith("workspace.default."):
        print(f"Model: {model.name}")
        versions = client.search_model_versions(f"name='{model.name}'")
        for version in versions:
            # aliases is a list in Unity Catalog
            aliases = version.aliases if hasattr(version, "aliases") and version.aliases else []
            aliases_str = str(aliases)
            print(f"  Version {version.version} - Aliases: {aliases_str}")
        print()

# Also check what config_utils is generating
print("\n=== Expected Model Names ===")
from src.utils.config_utils import get_model_registry_name, sanitize_artifact_name

test_assets = ["AMZN", "BTC-USD", "ETH-USD", "IBM"]
for asset in test_assets:
    safe_name = sanitize_artifact_name(asset)
    base_name = f"hybrid_lstm_xgboost_{safe_name}"
    registry_name = get_model_registry_name(base_name)
    print(f"{asset} -> {registry_name}")
