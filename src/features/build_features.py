import logging
from pathlib import Path

import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: Path) -> dict:
    """
    Loads the YAML configuration file.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_time_series_features(df: pd.DataFrame, target_col: str = "close") -> pd.DataFrame:
    """
    Generates technical indicators and time-series features.

    Args:
        df (pd.DataFrame): Input dataframe with price data.
        target_col (str): Name of the column containing the price (e.g., 'last', 'price_usd').

    Returns:
        pd.DataFrame: Dataframe with added features.
    """
    df_feat = df.copy()

    # Rename target column to standard 'close' for consistency across assets
    df_feat = df_feat.rename(columns={target_col: "close"})

    # Return calculation
    df_feat["return"] = df_feat["close"].pct_change()

    # Moving Averages
    df_feat["ma_7"] = df_feat["close"].rolling(window=7).mean()
    df_feat["ma_30"] = df_feat["close"].rolling(window=30).mean()

    # Volatility
    df_feat["volatility_30"] = df_feat["close"].rolling(window=30).std()

    # Lag features (t-1, t-2, ...)
    for lag in [1, 2, 3, 7]:
        df_feat[f"lag_{lag}"] = df_feat["close"].shift(lag)

    return df_feat


def process_asset(df: pd.DataFrame, asset_name: str, asset_type: str) -> pd.DataFrame:
    """
    Processes a single asset: filters data, resamples, and generates features.

    Args:
        df (pd.DataFrame): The complete dataframe containing multiple assets.
        asset_name (str): Name of the asset to process (e.g., 'Bitcoin', 'Amazon.com').
        asset_type (str): Type of asset ('stock' or 'crypto').

    Returns:
        pd.DataFrame: Processed dataframe with features and metadata, or empty if not found.
    """
    # Filter data for the specific asset
    mask = df["name"] == asset_name
    df_asset = df[mask].copy()

    if df_asset.empty:
        logging.warning(f"No data found for asset: {asset_name}")
        return pd.DataFrame()

    # Determine price column based on asset type
    price_col = "last" if asset_type == "stock" else "price_usd"

    # Resample to daily frequency and handle missing values
    # Using 'last' (closing price) for resampling
    daily_df = df_asset.set_index("timestamp")[price_col].resample("D").last().ffill()

    # Generate features
    df_features = create_time_series_features(daily_df.to_frame(), target_col=price_col)

    # Create target variable (Next day's price)
    df_features["target"] = df_features["close"].shift(-1)

    # Add metadata
    df_features["asset_name"] = asset_name
    df_features["asset_type"] = asset_type

    # Drop rows with NaN values (due to lags and shifting)
    df_features.dropna(inplace=True)

    return df_features


def main():
    """
    Main execution function for feature engineering pipeline.
    """
    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    processed_data_path = project_root / "data" / "processed"
    features_data_path = project_root / "data" / "features"
    config_path = project_root / "configs" / "training_config.yaml"

    # Ensure output directory exists
    features_data_path.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(config_path)

    all_features_list = []

    # 1. Process Stocks
    stocks_file = processed_data_path / "stocks.csv"
    if stocks_file.exists():
        logging.info("Loading stocks data...")
        df_stocks = pd.read_csv(stocks_file, parse_dates=["timestamp"])

        for stock_name in config["assets"]["stocks"]:
            logging.info(f"Processing Stock: {stock_name}")
            df_feat = process_asset(df_stocks, stock_name, "stock")
            if not df_feat.empty:
                all_features_list.append(df_feat)
    else:
        logging.warning(f"Stocks file not found at {stocks_file}")

    # 2. Process Cryptocurrencies
    crypto_file = processed_data_path / "cryptocurrency.csv"
    if crypto_file.exists():
        logging.info("Loading cryptocurrency data...")
        df_crypto = pd.read_csv(crypto_file, parse_dates=["timestamp"])

        for crypto_name in config["assets"]["cryptos"]:
            logging.info(f"Processing Crypto: {crypto_name}")
            df_feat = process_asset(df_crypto, crypto_name, "crypto")
            if not df_feat.empty:
                all_features_list.append(df_feat)
    else:
        logging.warning(f"Cryptocurrency file not found at {crypto_file}")

    # 3. Combine and Save
    if all_features_list:
        final_df = pd.concat(all_features_list)
        output_file = features_data_path / "multi_asset_features.csv"
        final_df.to_csv(output_file)
        logging.info(f"Successfully saved multi-asset features to {output_file}")
        logging.info(f"Total shape: {final_df.shape}")
        logging.info(f"Assets included: {final_df['asset_name'].unique()}")
    else:
        logging.error("No features generated. Check data files and configuration.")


if __name__ == "__main__":
    main()
