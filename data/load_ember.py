"""
EMBER 2018 v2 Feature Loader & Preprocessor
============================================
Loads parquet feature files from `dhoogla/ember-2018-v2-features` on Kaggle,
filters unlabeled samples, processes labels (0: benign, 1: malware),
and formats features for model training and benchmarking.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBER_LOCAL_DIR = Path(__file__).parent / "ember_2018_v2"


def find_ember_files() -> List[Path]:
    """Find all parquet files in local directory or kagglehub cache."""
    # Check local directory first
    if EMBER_LOCAL_DIR.exists():
        files = list(EMBER_LOCAL_DIR.glob("*.parquet"))
        if files:
            return sorted(files)

    # Check kagglehub default cache
    try:
        import kagglehub
        cache_path = Path(kagglehub.dataset_download("dhoogla/ember-2018-v2-features"))
        files = list(cache_path.glob("*.parquet")) + list(cache_path.glob("*/*.parquet"))
        if files:
            return sorted(files)
    except Exception as e:
        logger.warning(f"Could not locate cached files via kagglehub: {e}")

    return []


def load_ember_parquet(file_path: Path | str, sample_size: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    """
    Load a single EMBER parquet file, filter unlabeled rows (label == -1),
    and optionally sample a subset for fast development.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    logger.info(f"Reading parquet from {path.name}...")
    df = pd.read_parquet(path)

    # Check label column
    label_col = None
    for candidate in ["label", "Label", "target", "Target", "Class", "class"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is not None and label_col != "label":
        df = df.rename(columns={label_col: "label"})

    if "label" in df.columns:
        # EMBER uses -1 for unlabeled data; retain 0 (benign) and 1 (malware)
        initial_len = len(df)
        df = df[df["label"].isin([0, 1])].copy()
        df["label"] = df["label"].astype(int)
        logger.info(f"Filtered out unlabeled rows: {initial_len:,} -> {len(df):,} rows")

    if sample_size is not None and sample_size < len(df):
        logger.info(f"Sampling {sample_size:,} rows (random_state={random_state})...")
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    logger.info(f"Loaded {df.shape[0]:,} rows with {df.shape[1]:,} columns from {path.name}")
    return df


def prepare_ember_dataset(
    n_samples: int = 50_000,
    output_csv_path: Optional[Path | str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load available EMBER parquet files, combine, sample, and save as CSV.
    """
    parquet_files = find_ember_files()
    if not parquet_files:
        raise RuntimeError("No EMBER parquet files found. Run `python data/download_ember.py` first.")

    dfs = []
    for pf in parquet_files:
        try:
            df_part = load_ember_parquet(pf, sample_size=n_samples, random_state=random_state)
            dfs.append(df_part)
        except Exception as e:
            logger.warning(f"Failed to read {pf.name}: {e}")

    if not dfs:
        raise RuntimeError("Could not load any data from parquet files.")

    combined = pd.concat(dfs, ignore_index=True)
    if "label" in combined.columns:
        combined = combined[combined["label"].isin([0, 1])]

    if n_samples is not None and len(combined) > n_samples:
        combined = combined.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

    logger.info(f"Combined EMBER dataset shape: {combined.shape}")
    if "label" in combined.columns:
        logger.info(f"Label distribution: Benign={sum(combined['label']==0):,}, Malware={sum(combined['label']==1):,}")

    if output_csv_path is not None:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        logger.info(f"Saved dataset to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    return combined


if __name__ == "__main__":
    files = find_ember_files()
    print("Discovered EMBER Parquet files:", [f.name for f in files])
