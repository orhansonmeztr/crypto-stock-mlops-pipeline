import logging
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names: lowercase and removes special chars."""
    # Keeps letters, numbers, and underscores. Removes everything else.
    df.columns = df.columns.str.lower().str.replace("[^A-Za-z0-9_]+", "", regex=True)
    return df


def clean_numeric_columns(series: pd.Series) -> pd.Series:
    """
    Cleans numeric columns that might be strings or already floats.
    Handles 'M' (millions), '$', ',' etc.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    cleaned = series.astype(str).str.replace(r'[$,"]', "", regex=True)

    def convert_value(val):
        if pd.isna(val) or val == "nan":
            return None

        val_str = str(val).strip().upper()

        if "M" in val_str:
            try:
                return float(val_str.replace("M", "")) * 1_000_000
            except ValueError:
                return None
        if "B" in val_str:
            try:
                return float(val_str.replace("B", "")) * 1_000_000_000
            except ValueError:
                return None
        try:
            return float(val_str.replace("%", ""))
        except (ValueError, TypeError):
            return None

    return cleaned.apply(convert_value)


def preprocess_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the stocks DataFrame."""
    logging.info("Preprocessing stocks data...")

    # 1. Normalize case first
    df.columns = df.columns.str.lower()

    # 2. Rename BEFORE cleaning to catch special chars like '%'
    # This prevents 'chg_%' from becoming 'chg_' and colliding with 'chg_' (abs)
    rename_map = {
        "chg_%": "chg_pct",
        "chg%": "chg_pct",
        "change %": "chg_pct",
        "change_pct": "chg_pct",
        "chg_": "chg_abs",
        "chg": "chg_abs",
        "vol_": "vol",
        "volume": "vol",
    }
    df = df.rename(columns=rename_map)

    # 3. Clean Names (removes special chars, keeps underscores)
    df = clean_column_names(df)

    logging.info(f"Columns found: {df.columns.tolist()}")

    # 4. Filter Columns
    cols_to_keep = ["timestamp", "name", "last", "high", "low", "chg_abs", "chg_pct", "vol"]
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    df_processed = df[existing_cols].copy()

    # 5. Convert Types
    df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"])

    numeric_cols = ["last", "high", "low", "chg_abs", "vol", "chg_pct"]
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = clean_numeric_columns(df_processed[col])

    # 6. Drop Invalid Rows
    df_processed.dropna(subset=["last", "name"], inplace=True)

    logging.info(f"Stocks data preprocessed. Shape: {df_processed.shape}")
    return df_processed


def preprocess_crypto(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the cryptocurrency DataFrame."""
    logging.info("Preprocessing cryptocurrency data...")

    # 1. Clean Names
    df = clean_column_names(df)

    # 2. Filter Columns
    cols_to_keep = [
        "timestamp",
        "name",
        "symbol",
        "price_usd",
        "vol_24h",
        "total_vol",
        "chg_24h",
        "chg_7d",
        "market_cap",
    ]
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    df_processed = df[existing_cols].copy()

    # 3. Convert Types
    df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"])

    numeric_cols = ["price_usd", "vol_24h", "total_vol", "chg_24h", "chg_7d", "market_cap"]
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = clean_numeric_columns(df_processed[col])

    # 4. Drop Invalid Rows
    df_processed.dropna(subset=["price_usd", "name"], inplace=True)

    logging.info(f"Crypto data preprocessed. Shape: {df_processed.shape}")
    return df_processed


def main():
    """Main function to orchestrate the data preprocessing."""
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "raw"
    processed_path = project_root / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    file_processor_map = {
        "stocks.csv": preprocess_stocks,
        "cryptocurrency.csv": preprocess_crypto,
    }

    for file_name, processor_func in file_processor_map.items():
        raw_file_path = raw_path / file_name
        processed_file_path = processed_path / file_name

        if raw_file_path.exists():
            logging.info(f"Processing file: {raw_file_path}")
            try:
                df = pd.read_csv(raw_file_path)
                processed_df = processor_func(df)
                logging.info(f"Saving processed file to: {processed_file_path}")
                processed_df.to_csv(processed_file_path, index=False)
            except Exception as e:
                logging.error(f"Failed to process {file_name}: {e}")
        else:
            logging.warning(f"File not found, skipping: {raw_file_path}")


if __name__ == "__main__":
    main()
