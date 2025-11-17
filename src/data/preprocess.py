import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
#     """Performs basic cleaning operations on a DataFrame."""
#     # Make column names lowercase and replace special characters
#     df.columns = df.columns.str.lower().str.replace('[^A-Za-z0-9_]+', '', regex=True)

#     # Remove any completely empty rows or columns
#     df.dropna(how='all', axis=0, inplace=True)
#     df.dropna(how='all', axis=1, inplace=True)

#     return df


def clean_stock_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Specifically cleans and standardizes column names for the stocks dataframe
    to avoid name collisions.
    """
    rename_map = {"chg_": "chg_abs", "chg_%": "chg_pct", "vol_": "vol"}
    df.rename(columns=rename_map, inplace=True)
    df.columns = df.columns.str.lower().str.replace("[^A-Za-z0-9_]+", "", regex=True)
    return df


def clean_crypto_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans column names for the crypto dataframe."""
    df.columns = df.columns.str.lower().str.replace("[^A-Za-z0-9_]+", "", regex=True)
    return df


def clean_monetary_columns(series: pd.Series) -> pd.Series:
    """Cleans a series of monetary values."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"Input must be a pandas Series, but got {type(series)}")

    cleaned = series.astype(str).str.replace(r'[$,"]', "", regex=True)

    def convert_value(val):
        val = val.strip().upper()
        if "M" in val:
            return float(val.replace("M", "")) * 1_000_000
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    return cleaned.apply(convert_value)


def clean_percentage_columns(series: pd.Series) -> pd.Series:
    """Cleans a series of percentage values."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"Input must be a pandas Series, but got {type(series)}")
    cleaned_series = series.astype(str).str.replace(r"[,\+%\$]", "", regex=True)
    return pd.to_numeric(cleaned_series, errors="coerce")


def preprocess_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the stocks DataFrame."""
    logging.info("Preprocessing stocks data...")
    df_processed = clean_stock_column_names(df.copy())

    cols_to_keep = ["timestamp", "name", "last", "high", "low", "chg_abs", "chg_pct", "vol"]
    existing_cols = [col for col in cols_to_keep if col in df_processed.columns]
    df_processed = df_processed[existing_cols].copy()

    df_processed.loc[:, "timestamp"] = pd.to_datetime(df_processed["timestamp"])

    for col in ["last", "high", "low", "chg_abs", "vol"]:
        if col in df_processed.columns:
            df_processed.loc[:, col] = clean_monetary_columns(df_processed[col])

    if "chg_pct" in df_processed.columns:
        df_processed.loc[:, "chg_pct"] = clean_percentage_columns(df_processed["chg_pct"])

    df_processed.dropna(subset=["last", "name"], inplace=True)
    logging.info(f"Stocks data preprocessed. Shape: {df_processed.shape}")
    return df_processed


def preprocess_crypto(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the cryptocurrency DataFrame."""
    logging.info("Preprocessing cryptocurrency data...")
    df_processed = clean_crypto_column_names(df.copy())

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
    existing_cols = [col for col in cols_to_keep if col in df_processed.columns]
    df_processed = df_processed[existing_cols].copy()

    df_processed.loc[:, "timestamp"] = pd.to_datetime(df_processed["timestamp"])

    for col in ["price_usd", "vol_24h", "market_cap"]:
        if col in df_processed.columns:
            df_processed.loc[:, col] = clean_monetary_columns(df_processed[col])

    for col in ["chg_24h", "chg_7d", "total_vol"]:
        if col in df_processed.columns:
            df_processed.loc[:, col] = clean_percentage_columns(df_processed[col])

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
        "stocks.csv": (preprocess_stocks, None),
        "cryptocurrency.csv": (preprocess_crypto, clean_crypto_column_names),
    }

    for file_name, (processor_func, cleaner_func) in file_processor_map.items():
        raw_file_path = raw_path / file_name
        processed_file_path = processed_path / file_name

        if raw_file_path.exists():
            logging.info(f"Processing file: {raw_file_path}")
            df = pd.read_csv(raw_file_path)
            df_cleaned = cleaner_func(df) if cleaner_func else df
            processed_df = processor_func(df_cleaned)
            logging.info(f"Saving processed file to: {processed_file_path}")
            processed_df.to_csv(processed_file_path, index=False)
        else:
            logging.warning(f"File not found, skipping: {raw_file_path}")


if __name__ == "__main__":
    main()
