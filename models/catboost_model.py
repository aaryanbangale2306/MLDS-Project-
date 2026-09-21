"""
CatBoost Classifier for Malware Detection
"""

import numpy as np
import joblib
import time
from pathlib import Path
from typing import Dict, List, Optional


class CatBoostMalwareDetector:
    """CatBoost with ordered boosting for robust malware detection."""

    MODEL_NAME = "CatBoost"

    def __init__(self, feature_names: Optional[List[str]] = None):
        from utils.preprocessing import ALL_FEATURES
        self.feature_names = feature_names or ALL_FEATURES
        self.train_time: float = 0.0
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "CatBoostMalwareDetector":
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("Install catboost: pip install catboost>=1.2.5")

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        spw = n_neg / max(n_pos, 1)

        self.model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            border_count=64,
            bagging_temperature=0.5,
            random_strength=1.0,
            scale_pos_weight=spw,
            boosting_type="Ordered",   # prevents overfitting via ordered TS
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        t0 = time.time()
        eval_pool = None
        if X_val is not None:
            from catboost import Pool
            eval_pool = Pool(X_val, y_val)

        self.model.fit(
            X_train, y_train,
            eval_set=eval_pool,
            early_stopping_rounds=50,
        )
        self.train_time = time.time() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self, importance_type: str = "FeatureImportance") -> Dict[str, float]:
        """
        importance_type: 'FeatureImportance' (default, gain-based),
                         'ShapValues', 'Interaction'
        """
        scores = self.model.get_feature_importance(type=importance_type)
        return {fname: float(score) for fname, score in zip(self.feature_names, scores)}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "CatBoostMalwareDetector":
        return joblib.load(path)
