import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def setup_kaggle_api():
    """
    Sets up the Kaggle API credentials from environment variables.
    The kaggle library automatically looks for KAGGLE_USERNAME and KAGGLE_KEY.
    """
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        logging.error("Kaggle credentials not found in environment variables.")
        logging.info(
            "Please set KAGGLE_USERNAME and KAGGLE_KEY in your .env file or system environment."
        )
        raise OSError("Kaggle API credentials are not configured.")

    try:
        import kaggle

        kaggle.api.authenticate()
        logging.info("Kaggle API authenticated successfully.")
        return kaggle.api
    except ImportError:
        logging.error("The 'kaggle' package is not installed.")
        raise
    except Exception as e:
        logging.error(f"Kaggle API authentication failed: {e}")
        raise


def download_dataset(api, dataset_id: str, download_path: Path):
    """
    Downloads and unzips a dataset from Kaggle to the specified path.

    Args:
        api: The authenticated Kaggle API client.
        dataset_id (str): The identifier of the Kaggle dataset (e.g., 'user/dataset-name').
        download_path (Path): The local directory to download the data into.
    """
    if not download_path.exists():
        logging.info(f"Creating download directory at: {download_path}")
        download_path.mkdir(parents=True, exist_ok=True)
    else:
        if any(download_path.iterdir()):
            logging.warning(
                f"Download directory '{download_path}' is not empty. Files might be overwritten."
            )

    logging.info(f"Starting download of dataset '{dataset_id}'...")
    try:
        api.dataset_download_files(
            dataset=dataset_id,
            path=str(download_path),
            unzip=True,
            quiet=False,
        )
        logging.info(
            f"Dataset '{dataset_id}' downloaded and unzipped to '{download_path}' successfully."
        )
    except Exception as e:
        logging.error(f"Failed to download dataset. Error: {e}")
        raise


def main():
    """
    Main function to orchestrate the data fetching process.
    """
    load_dotenv()

    # Dataset details
    dataset_id = "adrianjuliusaluoch/crypto-and-stock-market-data-for-financial-analysis"
    project_root = Path(__file__).resolve().parents[2]
    raw_data_path = project_root / "data" / "raw"

    try:
        api = setup_kaggle_api()
        download_dataset(api=api, dataset_id=dataset_id, download_path=raw_data_path)
    except Exception as e:
        logging.critical(f"Data fetching process failed: {e}")


if __name__ == "__main__":
    main()
