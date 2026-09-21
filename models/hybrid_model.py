"""
Hybrid Model: Isolation Forest + Gradient Boosting Classifier
==============================================================
Two-stage detection pipeline:
    Stage 1: Isolation Forest for anomaly pre-screening
              → Identifies samples that deviate from normal (benign) distribution
              → Particularly effective for novel/zero-day polymorphic variants
    Stage 2: Gradient Boosting Classifier for final classification
              → Trained on Isolation Forest anomaly scores + original features
              → Refined decision boundary using supervised labels

Rationale for Polymorphic Malware:
    Polymorphic malware changes signature but retains behavioral anomalies.
    Isolation Forest excels at detecting ANY deviation from benign baseline,
    even for unseen variants. The GBC then provides precise classification
    using the IF anomaly signal as an additional meta-feature.
"""

import numpy as np
import joblib
import time
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class HybridMalwareDetector:
    """Two-stage Isolation Forest + Gradient Boosting malware detector."""

    MODEL_NAME = "Hybrid (IF+GB)"

    def __init__(self, feature_names: Optional[List[str]] = None):
        from utils.preprocessing import ALL_FEATURES
        self.feature_names = feature_names or ALL_FEATURES
        self.train_time: float = 0.0
        self.isolation_forest: Optional[IsolationForest] = None
        self.gb_classifier: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None

    def _get_if_scores(self, X: np.ndarray) -> np.ndarray:
        """Get Isolation Forest anomaly scores (negative → more anomalous)."""
        # decision_function returns negative scores for anomalies → flip for intuitive direction
        raw = self.isolation_forest.decision_function(X)
        # Normalize to [0, 1] where 1 = most anomalous
        normalized = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        return normalized.reshape(-1, 1)

    def _augment_features(self, X: np.ndarray) -> np.ndarray:
        """Add IF anomaly score as extra feature for Stage 2."""
        if_scores = self._get_if_scores(X)
        return np.hstack([X, if_scores])

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "HybridMalwareDetector":
        t0 = time.time()

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # ── Stage 1: Isolation Forest (fit on benign-majority training data) ──
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            max_samples=0.8,
            contamination=0.20,        # ~20% malware = contamination
            max_features=0.8,
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
        )
        self.isolation_forest.fit(X_scaled)

        # ── Stage 2: GBC on original features + IF anomaly score ──────────────
        X_augmented = self._augment_features(X_scaled)
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos

        self.gb_classifier = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
        )
        self.gb_classifier.fit(X_augmented, y_train)
        self.train_time = time.time() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_aug = self._augment_features(X_scaled)
        return self.gb_classifier.predict(X_aug)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_aug = self._augment_features(X_scaled)
        return self.gb_classifier.predict_proba(X_aug)

    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return raw Isolation Forest anomaly scores for X."""
        X_scaled = self.scaler.transform(X)
        return self._get_if_scores(X_scaled).flatten()

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """Feature importance from GBC (includes IF anomaly score as extra feature)."""
        scores = self.gb_classifier.feature_importances_
        feature_names_aug = self.feature_names + ["if_anomaly_score"]
        return {fname: float(score) for fname, score in zip(feature_names_aug, scores)}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "HybridMalwareDetector":
        return joblib.load(path)
