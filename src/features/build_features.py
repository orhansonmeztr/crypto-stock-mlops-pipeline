import logging
from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from src.utils.config_utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_time_series_features(
    df: pd.DataFrame, config: dict, target_col: str = "close"
) -> pd.DataFrame:
    """
    Generates advanced technical indicators and time-series features based on config.
    """
    df_feat = df.copy()
    feat_config = config["features"]

    # Rename target column to standard 'close'
    df_feat = df_feat.rename(columns={target_col: "close"})

    # 1. Basic Features
    df_feat["return"] = df_feat["close"].pct_change()
    df_feat["log_return"] = np.log(df_feat["close"] / df_feat["close"].shift(1))

    # 2. Trend Indicators (Moving Averages)
    for window in feat_config["moving_averages"]:
        df_feat[f"ma_{window}"] = df_feat["close"].rolling(window=window).mean()

    # 3. MACD (UPDATED)
    # Read params from config, default to standard if missing
    window_slow = feat_config.get("macd_slow", 26)
    window_fast = feat_config.get("macd_fast", 12)
    window_sign = feat_config.get("macd_signal", 9)

    macd = MACD(
        close=df_feat["close"],
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
    )
    df_feat["macd"] = macd.macd()
    df_feat["macd_signal"] = macd.macd_signal()
    df_feat["macd_diff"] = macd.macd_diff()

    # 4. Momentum Indicators
    # RSI
    rsi_window = feat_config.get("rsi_window", 14)
    rsi = RSIIndicator(close=df_feat["close"], window=rsi_window)
    df_feat["rsi"] = rsi.rsi()

    # 5. Volatility Indicators
    # Bollinger Bands
    bb_window = feat_config.get("bollinger_window", 20)
    bb_dev = feat_config.get("bollinger_dev", 2)

    bb = BollingerBands(close=df_feat["close"], window=bb_window, window_dev=bb_dev)
    df_feat["bb_high"] = bb.bollinger_hband()
    df_feat["bb_low"] = bb.bollinger_lband()

    # Avoid division by zero
    df_feat["bb_width"] = (df_feat["bb_high"] - df_feat["bb_low"]) / df_feat["close"].replace(
        0, np.nan
    )

    denominator = df_feat["bb_high"] - df_feat["bb_low"]
    df_feat["bb_position"] = (df_feat["close"] - df_feat["bb_low"]) / denominator.replace(0, np.nan)

    # Volatility
    vol_window = feat_config.get("volatility_window", 30)
    df_feat[f"volatility_{vol_window}"] = df_feat["close"].rolling(window=vol_window).std()

    # 6. Time-Based Features (Lags)
    lags = feat_config.get("lags", [1, 2, 3, 7])
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat["close"].shift(lag)

    return df_feat


def process_asset(df: pd.DataFrame, asset_name: str, asset_type: str, config: dict) -> pd.DataFrame:
    mask = df["name"] == asset_name
    df_asset = df[mask].copy()

    if df_asset.empty:
        logging.warning(f"No data found for asset: {asset_name}")
        return pd.DataFrame()

    price_col = "last" if asset_type == "stock" else "price_usd"

    # Force numeric conversion
    if price_col in df_asset.columns:
        df_asset[price_col] = pd.to_numeric(df_asset[price_col], errors="coerce")

    df_asset.dropna(subset=[price_col], inplace=True)

    if df_asset.empty:
        logging.warning(f"Asset {asset_name} has no valid price data after numeric conversion.")
        return pd.DataFrame()

    # Resample to daily
    daily_df = df_asset.set_index("timestamp")[price_col].resample("D").last().ffill()

    # Generate features
    df_features = create_time_series_features(daily_df.to_frame(), config, target_col=price_col)

    # Create target variable (Next day's price)
    df_features["target"] = df_features["close"].shift(-1)

    # Add metadata
    df_features["asset_name"] = asset_name
    df_features["asset_type"] = asset_type

    # Drop rows with NaN values
    original_len = len(df_features)
    df_features.dropna(inplace=True)

    if df_features.empty:
        logging.warning(
            f"Asset {asset_name}: All rows dropped after feature engineering (Original: {original_len}). Check if history is long enough for indicators."
        )

    return df_features


def main():
    project_root = Path(__file__).resolve().parents[2]
    processed_data_path = project_root / "data" / "processed"
    features_data_path = project_root / "data" / "features"
    config_path = project_root / "configs" / "training_config.yaml"

    features_data_path.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)

    all_features_list = []

    # 1. Process Stocks
    stocks_file = processed_data_path / "stocks.csv"
    if stocks_file.exists():
        logging.info("Loading stocks data...")
        df_stocks = pd.read_csv(stocks_file, parse_dates=["timestamp"])
        for stock_name in config["assets"]["stocks"]:
            logging.info(f"Processing Stock: {stock_name}")
            df_feat = process_asset(df_stocks, stock_name, "stock", config)
            if not df_feat.empty:
                all_features_list.append(df_feat)

    # 2. Process Cryptocurrencies
    crypto_file = processed_data_path / "cryptocurrency.csv"
    if crypto_file.exists():
        logging.info("Loading cryptocurrency data...")
        df_crypto = pd.read_csv(crypto_file, parse_dates=["timestamp"])
        for crypto_name in config["assets"]["cryptos"]:
            logging.info(f"Processing Crypto: {crypto_name}")
            df_feat = process_asset(df_crypto, crypto_name, "crypto", config)
            if not df_feat.empty:
                all_features_list.append(df_feat)

    # 3. Combine and Save
    if all_features_list:
        final_df = pd.concat(all_features_list)
        output_file = features_data_path / "multi_asset_features.csv"
        final_df.to_csv(output_file)
        logging.info(f"Successfully saved enhanced features to {output_file}")
        logging.info(f"Total shape: {final_df.shape}")

        unique_assets = final_df["asset_name"].unique()
        logging.info(f"Assets in feature file: {unique_assets}")
    else:
        logging.error("No features generated.")


if __name__ == "__main__":
    main()
