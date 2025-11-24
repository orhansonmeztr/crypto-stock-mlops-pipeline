import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_time_series_features(df: pd.DataFrame, target_col: str = "close") -> pd.DataFrame:
    """
    Generates advanced technical indicators and time-series features.
    """
    df_feat = df.copy()

    # Rename target column to standard 'close'
    df_feat = df_feat.rename(columns={target_col: "close"})

    # --- 1. Basic Features ---
    # Daily Return (Percentage change)
    df_feat["return"] = df_feat["close"].pct_change()
    # Log Return (More suitable for financial modeling)
    df_feat["log_return"] = np.log(df_feat["close"] / df_feat["close"].shift(1))

    # --- 2. Trend Indicators ---
    # Moving Averages
    df_feat["ma_7"] = df_feat["close"].rolling(window=7).mean()
    df_feat["ma_30"] = df_feat["close"].rolling(window=30).mean()
    df_feat["ma_50"] = df_feat["close"].rolling(window=50).mean()  # Added longer trend

    # MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df_feat["close"])
    df_feat["macd"] = macd.macd()
    df_feat["macd_signal"] = macd.macd_signal()
    df_feat["macd_diff"] = macd.macd_diff()  # Histogram

    # --- 3. Momentum Indicators ---
    # RSI (Relative Strength Index) - 14 day standard
    rsi = RSIIndicator(close=df_feat["close"], window=14)
    df_feat["rsi"] = rsi.rsi()

    # --- 4. Volatility Indicators ---
    # Bollinger Bands
    bb = BollingerBands(close=df_feat["close"], window=20, window_dev=2)
    df_feat["bb_high"] = bb.bollinger_hband()
    df_feat["bb_low"] = bb.bollinger_lband()
    # Band Width (High volatility = High width)
    df_feat["bb_width"] = (df_feat["bb_high"] - df_feat["bb_low"]) / df_feat["close"]
    # Position within bands (0=Lower Band, 1=Upper Band)
    df_feat["bb_position"] = (df_feat["close"] - df_feat["bb_low"]) / (
        df_feat["bb_high"] - df_feat["bb_low"]
    )

    # Volatility (Rolling Standard Deviation)
    df_feat["volatility_30"] = df_feat["close"].rolling(window=30).std()

    # --- 5. Time-Based Features ---
    # Lag features (t-1, t-2, ...) - Critical for XGBoost
    lags = [1, 2, 3, 7, 14]
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat["close"].shift(lag)

    return df_feat


def process_asset(df: pd.DataFrame, asset_name: str, asset_type: str) -> pd.DataFrame:
    mask = df["name"] == asset_name
    df_asset = df[mask].copy()

    if df_asset.empty:
        logging.warning(f"No data found for asset: {asset_name}")
        return pd.DataFrame()

    price_col = "last" if asset_type == "stock" else "price_usd"

    # Resample to daily
    daily_df = df_asset.set_index("timestamp")[price_col].resample("D").last().ffill()

    # Generate features
    df_features = create_time_series_features(daily_df.to_frame(), target_col=price_col)

    # Create target variable (Next day's return is often better to predict than price,
    # but we stick to Price for this phase as per plan)
    df_features["target"] = df_features["close"].shift(-1)

    # Add metadata
    df_features["asset_name"] = asset_name
    df_features["asset_type"] = asset_type

    # Drop rows with NaN values (Indicators need initial window to calculate)
    # MACD/RSI/Moving Averages create NaNs at the start
    df_features.dropna(inplace=True)

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
            df_feat = process_asset(df_stocks, stock_name, "stock")
            if not df_feat.empty:
                all_features_list.append(df_feat)

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

    # 3. Combine and Save
    if all_features_list:
        final_df = pd.concat(all_features_list)
        output_file = features_data_path / "multi_asset_features.csv"
        final_df.to_csv(output_file)
        logging.info(f"Successfully saved enhanced features to {output_file}")
        logging.info(f"Total shape: {final_df.shape}")
        logging.info(f"Columns: {final_df.columns.tolist()}")
    else:
        logging.error("No features generated.")


if __name__ == "__main__":
    main()
