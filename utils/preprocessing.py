"""
Preprocessing utilities for malware feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import joblib
from pathlib import Path

# ─── Feature Groups (used for Interaction Constraints) ───────────────────────

FEATURE_GROUPS = {
    "PE Header": [
        "file_size", "num_sections", "entry_point_offset", "image_base",
        "has_debug", "has_signature", "dll_characteristics",
    ],
    "Section Analysis": [
        "avg_section_entropy", "max_section_entropy", "section_size_variance",
        "num_executable_sections", "num_writable_sections",
    ],
    "Import Features": [
        "num_imports", "num_suspicious_imports", "num_unique_dlls",
        "uses_crypto_api", "uses_network_api", "uses_process_api", "uses_registry_api",
    ],
    "String Features": [
        "num_urls", "num_ips", "num_registry_keys", "num_file_paths",
        "avg_string_length", "num_printable_strings",
    ],
    "Behavioral": [
        "polymorphic_score", "obfuscation_level", "code_mutation_rate",
        "api_call_frequency", "memory_allocation_pattern",
    ],
    "Network": [
        "packet_entropy", "connection_attempts", "c2_similarity_score",
    ],
}

ALL_FEATURES = [f for feats in FEATURE_GROUPS.values() for f in feats]

def get_interaction_constraints(feature_names: Optional[List[str]] = None) -> List[List[str]]:
    """
    Return interaction constraints as lists of feature names (strings).
    XGBoost only allows interactions within the same domain group.
    This prevents spurious cross-group interactions that hurt generalization
    on polymorphic variants.
    """
    constraints = []
    for group_name, group_features in FEATURE_GROUPS.items():
        if feature_names is not None:
            feats = [f for f in group_features if f in feature_names]
        else:
            feats = list(group_features)
        if feats:
            constraints.append(feats)
    return constraints


# ─── Preprocessing Pipeline ───────────────────────────────────────────────────

class MalwarePreprocessor:
    """End-to-end preprocessing for malware detection dataset."""

    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type
        self.scaler: Optional[RobustScaler | StandardScaler] = None
        self.feature_names: List[str] = ALL_FEATURES
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> "MalwarePreprocessor":
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        self.scaler.fit(X[self.feature_names])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self.is_fitted, "Call fit() first."
        return self.scaler.transform(X[self.feature_names])

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "MalwarePreprocessor":
        return joblib.load(path)


# ─── Data Loading & Splitting ─────────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV, validate columns."""
    df = pd.read_csv(csv_path)
    missing = set(ALL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features in dataset: {missing}")
    return df


def prepare_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / validation / test sets (stratified).
    Returns: (train_df, val_df, test_df)
    """
    X = df[ALL_FEATURES]
    y = df["label"]

    # First split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Then split train / val from remaining
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, stratify=y_trainval,
        random_state=random_state,
    )

    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    val_df = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    return train_df, val_df, test_df


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Compute XGBoost scale_pos_weight for class imbalance."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return n_neg / n_pos


# ─── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that may improve detection."""
    df = df.copy()

    # Entropy ratio: suspicious when very high OR very low
    df["entropy_ratio"] = df["max_section_entropy"] / (df["avg_section_entropy"] + 1e-6)

    # Combined evasion score
    df["evasion_score"] = (
        df["polymorphic_score"] * 0.4 +
        df["obfuscation_level"] * 0.3 +
        df["code_mutation_rate"] * 0.3
    )

    # Import risk: combination of suspicious imports + dangerous APIs
    df["import_risk"] = (
        df["num_suspicious_imports"] / (df["num_imports"] + 1) +
        0.25 * (df["uses_crypto_api"] + df["uses_network_api"] +
                df["uses_process_api"] + df["uses_registry_api"])
    )

    # Network aggression
    df["network_threat"] = df["c2_similarity_score"] * np.log1p(df["connection_attempts"])

    return df


def get_feature_names_extended() -> List[str]:
    """Return all features including engineered ones."""
    return ALL_FEATURES + [
        "entropy_ratio", "evasion_score", "import_risk", "network_threat"
    ]
