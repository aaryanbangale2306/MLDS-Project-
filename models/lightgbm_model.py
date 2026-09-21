"""
LightGBM Classifier for Malware Detection
"""

import numpy as np
import joblib
import time
from pathlib import Path
from typing import Dict, List, Optional
import lightgbm as lgb


class LightGBMMalwareDetector:
    """LightGBM with DART boosting and leaf-wise growth for malware detection."""

    MODEL_NAME = "LightGBM"

    def __init__(self, feature_names: Optional[List[str]] = None):
        from utils.preprocessing import ALL_FEATURES
        self.feature_names = feature_names or ALL_FEATURES
        self.train_time: float = 0.0
        self.model: Optional[lgb.LGBMClassifier] = None
        self._evals_result: Dict = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LightGBMMalwareDetector":
        # Compute class weight ratio
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        spw = n_neg / max(n_pos, 1)

        self.model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            num_leaves=63,
            max_depth=-1,               # no depth limit (leaf-wise)
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            scale_pos_weight=spw,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        t0 = time.time()
        eval_set = [(X_val, y_val)] if X_val is not None else None
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric="auc",
            callbacks=callbacks if eval_set else [lgb.log_evaluation(period=-1)],
        )
        self.train_time = time.time() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        scores = self.model.booster_.feature_importance(importance_type=importance_type)
        return {fname: float(score) for fname, score in zip(self.feature_names, scores)}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "LightGBMMalwareDetector":
        return joblib.load(path)
