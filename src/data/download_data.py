"""
download_data.py
----------------
Data ingestion step for the customer churn MLOps pipeline.

Since the dataset has already been downloaded manually and placed in data/raw/,
this script validates that the expected files exist and have reasonable sizes,
then prints a summary so the DVC stage has a concrete command to run.

If you need to re-download the dataset from Kaggle in the future, set the
environment variables KAGGLE_USERNAME and KAGGLE_KEY (never hardcode them)
and uncomment the kaggle API section below.
"""

import sys
import os
import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
TRAIN_FILE = RAW_DIR / "customer_churn_train.csv"
TEST_FILE = RAW_DIR / "customer_churn_test.csv"

EXPECTED_TRAIN_MIN_BYTES = 20_000_000  # ~20 MB lower bound
EXPECTED_TEST_MIN_BYTES = 2_000_000    # ~2 MB lower bound


def validate_file(path: Path, min_bytes: int) -> None:
    """Raise if a file does not exist or is suspiciously small."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}\n"
            "Place customer_churn_dataset-training-master.csv and "
            "customer_churn_dataset-testing-master.csv in data/raw/ "
            "and rename them as above."
        )
    size = path.stat().st_size
    if size < min_bytes:
        raise ValueError(
            f"File {path.name} is only {size:,} bytes, "
            f"expected at least {min_bytes:,} bytes. "
            "The file may be truncated or corrupted."
        )
    print(f"  OK  {path.name}  ({size / 1_048_576:.1f} MB)")


def md5_first_mb(path: Path) -> str:
    """Return MD5 of the first 1 MB of a file (fast sanity check)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        h.update(fh.read(1_048_576))
    return h.hexdigest()


def main() -> None:
    print("=" * 60)
    print("Data ingestion validation")
    print("=" * 60)
    print(f"Raw data directory: {RAW_DIR}")
    print()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    validate_file(TRAIN_FILE, EXPECTED_TRAIN_MIN_BYTES)
    validate_file(TEST_FILE, EXPECTED_TEST_MIN_BYTES)

    print()
    print("Checksums (first 1 MB, for quick integrity check):")
    print(f"  train: {md5_first_mb(TRAIN_FILE)}")
    print(f"  test:  {md5_first_mb(TEST_FILE)}")

    print()
    print("Data validation passed. Files are ready for preprocessing.")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Kaggle re-download (commented out; uncomment if you need to re-fetch)
    # -----------------------------------------------------------------------
    # Requires: pip install kaggle
    # Environment variables:  KAGGLE_USERNAME  and  KAGGLE_KEY
    #
    # import subprocess
    # dataset_slug = "muhammadshahidazeem/customer-churn-dataset"
    # username = os.environ.get("KAGGLE_USERNAME")
    # key      = os.environ.get("KAGGLE_KEY")
    # if not username or not key:
    #     raise EnvironmentError(
    #         "Set KAGGLE_USERNAME and KAGGLE_KEY environment variables "
    #         "before running the download. Do NOT hardcode credentials."
    #     )
    # subprocess.run(
    #     ["kaggle", "datasets", "download", "-d", dataset_slug,
    #      "-p", str(RAW_DIR), "--unzip"],
    #     check=True,
    # )


if __name__ == "__main__":
    main()
