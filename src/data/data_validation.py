"""
Custom data validation module for processed datasets.
Replaces Great Expectations with lightweight pandas-based checks.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from src.utils.config_utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Validation Rules

STOCKS_SCHEMA = {
    "required_columns": ["timestamp", "name", "last"],
    "numeric_columns": ["last", "high", "low", "vol"],
    "max_null_ratio": 0.1,  # Max 10% null allowed per column
}

CRYPTO_SCHEMA = {
    "required_columns": ["timestamp", "name", "price_usd"],
    "numeric_columns": ["price_usd", "vol_24h", "market_cap"],
    "max_null_ratio": 0.1,
}


class ValidationError(Exception):
    """Raised when data validation fails."""


class ValidationResult:
    """Collects validation check results."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.checks: list[dict] = []
        self.passed = True

    def add_check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            self.passed = False

    def summary(self) -> str:
        lines = [f"\n--- Validation: {self.dataset_name} ---"]
        for check in self.checks:
            status = "PASS" if check["passed"] else "FAIL"
            detail = f" - {check['detail']}" if check["detail"] else ""
            lines.append(f"  {status}: {check['name']}{detail}")
        lines.append(f"  Overall: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


def validate_columns(df: pd.DataFrame, required: list[str], result: ValidationResult) -> None:
    """Check that all required columns are present."""
    missing = [col for col in required if col not in df.columns]
    result.add_check(
        "Required columns present",
        len(missing) == 0,
        f"Missing: {missing}" if missing else f"All {len(required)} columns found",
    )


def validate_no_empty(df: pd.DataFrame, result: ValidationResult) -> None:
    """Check that the DataFrame is not empty."""
    result.add_check(
        "DataFrame not empty",
        len(df) > 0,
        f"Rows: {len(df)}",
    )


def validate_null_ratio(
    df: pd.DataFrame, columns: list[str], max_ratio: float, result: ValidationResult
) -> None:
    """Check that null ratio doesn't exceed threshold for specified columns."""
    for col in columns:
        if col not in df.columns:
            continue
        null_ratio = df[col].isna().mean()
        result.add_check(
            f"Null ratio for '{col}'",
            null_ratio <= max_ratio,
            f"{null_ratio:.2%} (max: {max_ratio:.0%})",
        )


def validate_numeric_types(df: pd.DataFrame, columns: list[str], result: ValidationResult) -> None:
    """Check that specified columns are numeric."""
    for col in columns:
        if col not in df.columns:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        result.add_check(
            f"Column '{col}' is numeric",
            is_numeric,
            f"dtype: {df[col].dtype}",
        )


def validate_no_duplicates(df: pd.DataFrame, subset: list[str], result: ValidationResult) -> None:
    """Check for duplicate rows on given subset."""
    existing_subset = [col for col in subset if col in df.columns]
    if not existing_subset:
        return
    n_dupes = df.duplicated(subset=existing_subset).sum()
    result.add_check(
        f"No duplicates on {existing_subset}",
        n_dupes == 0,
        f"Duplicates: {n_dupes}",
    )


def validate_positive_values(
    df: pd.DataFrame, columns: list[str], result: ValidationResult
) -> None:
    """Check that price/volume columns contain only positive values (where not null)."""
    for col in columns:
        if col not in df.columns:
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        n_negative = (non_null < 0).sum()
        result.add_check(
            f"Column '{col}' has no negative values",
            n_negative == 0,
            f"Negative count: {n_negative}",
        )


def validate_dataset(df: pd.DataFrame, schema: dict, name: str) -> ValidationResult:
    """Run all validation checks on a dataset."""
    result = ValidationResult(name)

    validate_no_empty(df, result)
    validate_columns(df, schema["required_columns"], result)
    validate_numeric_types(df, schema["numeric_columns"], result)
    validate_null_ratio(df, schema["numeric_columns"], schema["max_null_ratio"], result)
    validate_no_duplicates(df, ["timestamp", "name"], result)
    validate_positive_values(df, schema["numeric_columns"], result)

    return result


def main():
    project_root = Path(__file__).resolve().parents[2]
    processed_path = project_root / "data" / "processed"
    config_path = project_root / "configs" / "training_config.yaml"

    config = load_config(config_path)

    all_passed = True

    # Validate stocks
    stocks_file = processed_path / "stocks.csv"
    if stocks_file.exists():
        logger.info(f"Validating {stocks_file}...")
        df_stocks = pd.read_csv(stocks_file)
        result = validate_dataset(df_stocks, STOCKS_SCHEMA, "Stocks")
        logger.info(result.summary())
        if not result.passed:
            all_passed = False
    else:
        logger.warning(f"Stocks file not found: {stocks_file}")

    # Validate crypto
    crypto_file = processed_path / "cryptocurrency.csv"
    if crypto_file.exists():
        logger.info(f"Validating {crypto_file}...")
        df_crypto = pd.read_csv(crypto_file)
        result = validate_dataset(df_crypto, CRYPTO_SCHEMA, "Cryptocurrency")
        logger.info(result.summary())
        if not result.passed:
            all_passed = False
    else:
        logger.warning(f"Crypto file not found: {crypto_file}")

    # Validate that configured assets exist in data
    stocks = config.get("assets", {}).get("stocks", [])
    cryptos = config.get("assets", {}).get("cryptos", [])

    if stocks_file.exists():
        df_stocks = pd.read_csv(stocks_file)
        stock_names = df_stocks["name"].unique() if "name" in df_stocks.columns else []
        missing_stocks = [s for s in stocks if s not in stock_names]
        if missing_stocks:
            logger.warning(f"Configured stocks not found in data: {missing_stocks}")

    if crypto_file.exists():
        df_crypto = pd.read_csv(crypto_file)
        crypto_names = df_crypto["name"].unique() if "name" in df_crypto.columns else []
        missing_cryptos = [c for c in cryptos if c not in crypto_names]
        if missing_cryptos:
            logger.warning(f"Configured cryptos not found in data: {missing_cryptos}")

    if not all_passed:
        logger.error("Data validation FAILED!")
        sys.exit(1)
    else:
        logger.info("All data validations PASSED.")


if __name__ == "__main__":
    main()
