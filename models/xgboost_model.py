"""
XGBoost Classifier with Feature Interaction Constraints
=========================================================
PRIMARY MODEL for polymorphic malware detection.

Key innovation: interaction_constraints restrict which features can interact
within XGBoost trees, preventing spurious cross-group correlations that hurt
generalization on polymorphic variants (which mutate one signature domain at
a time, not all simultaneously).

Cybersecurity Rationale:
    Polymorphic malware changes its byte pattern/signature but retains
    behavioral and structural invariants *within* each domain:
      - PE Header structure remains self-consistent
      - Section entropy patterns remain consistent across sections
      - Behavioral indicators correlate within themselves
      - Network traffic patterns are internally coherent
    
    Without constraints, XGBoost may learn spurious interactions between
    e.g. section entropy and network traffic that only hold for training
    samples but not for novel polymorphic variants.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from utils.preprocessing import get_interaction_constraints, ALL_FEATURES, compute_scale_pos_weight


class XGBoostMalwareDetector:
    """
    XGBoost classifier with feature interaction constraints for malware detection.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        with_constraints: bool = True,
        scale_pos_weight: float = 4.0,
    ):
        self.feature_names = feature_names or ALL_FEATURES
        self.with_constraints = with_constraints
        self.scale_pos_weight = scale_pos_weight
        self.model: Optional[xgb.XGBClassifier] = None
        self.train_time: float = 0.0

        # Default hyperparameters (will be overridden by Optuna results)
        self.hyperparams = hyperparams or {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "colsample_bylevel": 0.7,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "gamma": 0.1,
        }

    def _build_model(self, early_stopping_rounds: Optional[int] = None) -> xgb.XGBClassifier:
        params = dict(
            **self.hyperparams,
            scale_pos_weight=self.scale_pos_weight,
            tree_method="hist",
            eval_metric="auc",
            random_state=42,
        )
        if early_stopping_rounds is not None:
            params["early_stopping_rounds"] = early_stopping_rounds
        if self.with_constraints:
            params["interaction_constraints"] = get_interaction_constraints(self.feature_names)
        return xgb.XGBClassifier(**params)

    def _to_df(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.feature_names)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostMalwareDetector":
        X_train_df = self._to_df(X_train)
        eval_set = [(self._to_df(X_val), y_val)] if X_val is not None else None
        early_stopping_rounds = 50 if eval_set is not None else None
        self.model = self._build_model(early_stopping_rounds=early_stopping_rounds)
        t0 = time.time()
        self.model.fit(
            X_train_df, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.train_time = time.time() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._to_df(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._to_df(X))

    def get_evals_result(self) -> Dict:
        """Return training/validation AUC history for learning curves."""
        return self.model.evals_result() if hasattr(self.model, "evals_result") else {}

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        Get feature importances.

        Args:
            importance_type: 'gain', 'weight', or 'cover'

        Returns:
            Dict mapping feature name → importance score.
        """
        scores = self.model.get_booster().get_score(importance_type=importance_type)
        # Map f0, f1, ... back to feature names
        result = {}
        for key, val in scores.items():
            if key.startswith("f") and key[1:].isdigit():
                idx = int(key[1:])
                if idx < len(self.feature_names):
                    result[self.feature_names[idx]] = val
            else:
                result[key] = val
        # Fill missing features with 0
        for fname in self.feature_names:
            if fname not in result:
                result[fname] = 0.0
        return result

    def get_interaction_constraints_formatted(self) -> List[List[str]]:
        """Return constraints as feature name groups (for display)."""
        from utils.preprocessing import FEATURE_GROUPS
        return [
            {"group": group_name, "features": features}
            for group_name, features in FEATURE_GROUPS.items()
        ]

    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostMalwareDetector":
        return joblib.load(path)
