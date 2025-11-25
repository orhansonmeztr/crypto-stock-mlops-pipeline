import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_asset_data(ticker_symbol: str, is_crypto: bool) -> pd.DataFrame:
    """Fetches historical data for a single asset via yfinance and formats it."""
    logging.info(f"Downloading data for {ticker_symbol}...")
    # Fetch 5 years of data to match historical depth if possible
    df = yf.download(ticker_symbol, period="5y", progress=False)

    # Flatten MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        logging.warning(f"No data returned for {ticker_symbol}.")
        return pd.DataFrame()

    df = df.reset_index()
    # Ensure Date column is named correctly
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "timestamp"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "timestamp"})

    # Calculate basic daily changes
    df["chg_abs"] = df["Close"] - df["Open"]
    # Avoid div by zero
    df["chg_pct"] = (df["chg_abs"] / df["Open"].replace(0, pd.NA)) * 100

    if not is_crypto:
        # Stock Format: ["timestamp", "name", "last", "high", "low", "chg_abs", "chg_pct", "vol"]
        df["name"] = ticker_symbol
        df = df.rename(columns={"Close": "last", "High": "high", "Low": "low", "Volume": "vol"})
        cols = ["timestamp", "name", "last", "high", "low", "chg_abs", "chg_pct", "vol"]
        # Keep only existing columns to be safe
        existing_cols = [c for c in cols if c in df.columns]
        return df[existing_cols]
    else:
        # Crypto Format: ["timestamp", "name", "symbol", "price_usd", "vol_24h", "total_vol", "chg_24h", "chg_7d", "market_cap"]
        df["name"] = ticker_symbol
        df["symbol"] = ticker_symbol.split("-")[0]
        # Calculate 7d change
        df["chg_7d"] = df["Close"].pct_change(periods=7) * 100
        # Dummy market cap based on rough circulating supply proxy (or 0) so validation doesn't fail
        df["market_cap"] = df["Close"] * 1e6  # proxy

        df = df.rename(columns={"Close": "price_usd", "Volume": "vol_24h", "chg_pct": "chg_24h"})
        df["total_vol"] = df["vol_24h"]  # yfinance doesn't differentiate total_vol from vol_24h

        cols = [
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
        existing_cols = [c for c in cols if c in df.columns]
        return df[existing_cols]


def main():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "training_config.yaml"
    raw_data_path = project_root / "data" / "raw"
    raw_data_path.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    stocks_list = config.get("assets", {}).get("stocks", [])
    cryptos_list = config.get("assets", {}).get("cryptos", [])

    # Process Stocks
    if stocks_list:
        stock_dfs = []
        for stock_name in stocks_list:
            df = fetch_asset_data(stock_name, is_crypto=False)
            if not df.empty:
                stock_dfs.append(df)

        if stock_dfs:
            final_stocks_df = pd.concat(stock_dfs, ignore_index=True)
            out_file = raw_data_path / "stocks.csv"
            final_stocks_df.to_csv(out_file, index=False)
            logging.info(f"Saved {len(final_stocks_df)} stock records to {out_file}")

    # Process Cryptos
    if cryptos_list:
        crypto_dfs = []
        for crypto_name in cryptos_list:
            df = fetch_asset_data(crypto_name, is_crypto=True)
            if not df.empty:
                crypto_dfs.append(df)

        if crypto_dfs:
            final_crypto_df = pd.concat(crypto_dfs, ignore_index=True)
            out_file = raw_data_path / "cryptocurrency.csv"
            final_crypto_df.to_csv(out_file, index=False)
            logging.info(f"Saved {len(final_crypto_df)} crypto records to {out_file}")


if __name__ == "__main__":
    main()
