import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_time_series_features(df: pd.DataFrame, target_col: str, lags: int = 5) -> pd.DataFrame:
    """
    Creates time-series features like lags and time-based rolling window statistics.

    Args:
        df (pd.DataFrame): Input DataFrame, must have a DatetimeIndex.
        target_col (str): The column to create features for (e.g., 'last' price).
        lags (int): The number of past days to use for lag features.

    Returns:
        pd.DataFrame: DataFrame with new time-series features.
    """
    df_features = df.copy()

    # Lag features for previous days
    for i in range(1, lags + 1):
        df_features[f"{target_col}_lag_{i}"] = df_features[target_col].shift(i)

    rolling_windows = ["7D", "14D", "30D"]
    for window in rolling_windows:
        # We shift by 1 day to prevent data leakage from the current day
        shifted_series = df_features[target_col].shift(1)

        df_features[f"{target_col}_roll_mean_{window}"] = shifted_series.rolling(
            window=window
        ).mean()
        df_features[f"{target_col}_roll_std_{window}"] = shifted_series.rolling(window=window).std()

    df_features["day_of_week"] = df_features.index.dayofweek
    df_features["month"] = df_features.index.month
    df_features["year"] = df_features.index.year
    df_features["day_of_year"] = df_features.index.dayofyear

    df_features.dropna(inplace=True)
    return df_features


def main():
    """Main function to load, resample, build features, and save data."""
    project_root = Path(__file__).resolve().parents[2]
    processed_data_path = project_root / "data" / "processed"
    features_path = project_root / "data" / "features"
    features_path.mkdir(parents=True, exist_ok=True)

    stock_file = processed_data_path / "stocks.csv"
    if not stock_file.exists():
        logging.error(f"Processed data not found at {stock_file}.")
        return

    logging.info(f"Loading data from {stock_file}...")
    df = pd.read_csv(stock_file, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    df_single_asset = df[df["name"] == "Amazon.com"].copy()
    if df_single_asset.empty:
        logging.error("No data found for 'Amazon.com'.")
        return

    logging.info("Resampling data to daily frequency...")
    daily_df = df_single_asset["last"].resample("D").last()  # Use last price of the day
    daily_df = daily_df.ffill()  # Forward-fill weekends and holidays

    logging.info("Building features for 'Amazon.com'...")
    df_features = create_time_series_features(daily_df.to_frame(), target_col="last")

    # Define the prediction target: the price 1 day ahead
    df_features["target"] = df_features["last"].shift(-1)
    df_features.dropna(inplace=True)

    output_file = features_path / "amazon_daily_features.csv"
    logging.info(f"Saving feature-engineered data to {output_file}...")
    df_features.to_csv(output_file)

    logging.info(f"Feature engineering complete. Final shape: {df_features.shape}")


if __name__ == "__main__":
    main()
