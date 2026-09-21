"""
Download EMBER 2018 v2 Features Dataset from Kaggle
=====================================================
Dataset: dhoogla/ember-2018-v2-features (Kaggle)
Format: Parquet (tabular vectorized EMBER 2018 v2 features)

Usage:
    python data/download_ember.py
"""

import os
import sys
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_ember_dataset(destination_dir: Path | str | None = None) -> Path:
    """
    Download the EMBER 2018 v2 features dataset from Kaggle via kagglehub.
    
    Args:
        destination_dir: Optional path to copy/link the parquet files into.
                         Defaults to `data/ember_2018_v2/`.

    Returns:
        Path to the directory containing the downloaded parquet files.
    """
    import kagglehub

    logger.info("Initiating download for 'dhoogla/ember-2018-v2-features' from Kaggle...")
    download_path = Path(kagglehub.dataset_download("dhoogla/ember-2018-v2-features"))
    logger.info(f"Dataset successfully downloaded/cached at: {download_path}")

    if destination_dir is None:
        destination_dir = Path(__file__).parent / "ember_2018_v2"
    else:
        destination_dir = Path(destination_dir)

    destination_dir.mkdir(parents=True, exist_ok=True)

    # List parquet files
    parquet_files = list(download_path.glob("*.parquet")) + list(download_path.glob("*/*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files:")
    for pf in parquet_files:
        logger.info(f"  - {pf.name} ({pf.stat().st_size / (1024 * 1024):.2f} MB)")
        target_file = destination_dir / pf.name
        if not target_file.exists():
            logger.info(f"Copying {pf.name} to {target_file}...")
            shutil.copy2(pf, target_file)

    logger.info(f"EMBER 2018 v2 dataset ready in: {destination_dir}")
    return destination_dir


if __name__ == "__main__":
    download_ember_dataset()
