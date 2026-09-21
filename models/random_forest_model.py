"""
Random Forest Classifier for Malware Detection
"""

import numpy as np
import joblib
import time
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


class RandomForestMalwareDetector:
    """Random Forest classifier with MDI + permutation feature importance."""

    MODEL_NAME = "Random Forest"

    def __init__(self, feature_names: Optional[List[str]] = None):
        from utils.preprocessing import ALL_FEATURES
        self.feature_names = feature_names or ALL_FEATURES
        self.train_time: float = 0.0
        self.model: Optional[RandomForestClassifier] = None
        self._perm_importance: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RandomForestMalwareDetector":
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,          # grow fully
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced", # handle imbalance
            n_jobs=-1,
            random_state=42,
            oob_score=True,
        )
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self, importance_type: str = "mdi") -> Dict[str, float]:
        """
        Get feature importances.

        Args:
            importance_type: 'mdi' (Mean Decrease in Impurity, built-in)
                             or 'permutation' (requires calling compute_permutation_importance first)
        """
        if importance_type == "mdi":
            scores = self.model.feature_importances_
        elif importance_type == "permutation":
            scores = self._perm_importance if self._perm_importance is not None else self.model.feature_importances_
        else:
            scores = self.model.feature_importances_

        return {fname: float(score) for fname, score in zip(self.feature_names, scores)}

    def compute_permutation_importance(
        self, X_val: np.ndarray, y_val: np.ndarray, n_repeats: int = 5
    ) -> None:
        """Compute permutation importance on validation set (expensive but unbiased)."""
        result = permutation_importance(
            self.model, X_val, y_val,
            n_repeats=n_repeats, random_state=42, n_jobs=-1,
            scoring="roc_auc",
        )
        self._perm_importance = result.importances_mean

    @property
    def oob_score(self) -> float:
        return getattr(self.model, "oob_score_", 0.0)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestMalwareDetector":
        return joblib.load(path)
