"""
SHAP Explainability Utilities
"""

import numpy as np
import pandas as pd
import shap
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings("ignore")


class MalwareSHAPExplainer:
    """
    Wraps SHAP TreeExplainer for malware detection models.
    Provides summary, waterfall, and dependence plot data.
    """

    def __init__(self, model, feature_names: List[str], model_name: str = "XGBoost"):
        self.model = model
        self.feature_names = feature_names
        self.model_name = model_name
        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.base_value: Optional[float] = None

    def fit(self, X_background: np.ndarray) -> "MalwareSHAPExplainer":
        """
        Fit SHAP explainer on background data (subset of training data).

        Args:
            X_background: Background samples for SHAP (100-500 rows recommended).
        """
        try:
            self.explainer = shap.TreeExplainer(
                self.model,
                data=X_background,
                feature_perturbation="interventional",
                model_output="probability",
            )
        except Exception:
            # Fallback for models that don't support interventional
            self.explainer = shap.TreeExplainer(self.model)
        return self

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values. Returns array of shape (n_samples, n_features)."""
        assert self.explainer is not None, "Call fit() first."
        shap_vals = self.explainer.shap_values(X)
        # For binary classifiers that return [class0, class1], take class1
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        self.shap_values = shap_vals
        if hasattr(self.explainer, "expected_value"):
            ev = self.explainer.expected_value
            self.base_value = float(ev[1]) if isinstance(ev, (list, np.ndarray)) else float(ev)
        return shap_vals

    def get_feature_importance(self, X: np.ndarray) -> pd.DataFrame:
        """
        Return mean absolute SHAP values per feature (global importance).

        Returns:
            DataFrame with columns: feature, importance, rank
        """
        shap_vals = self.compute_shap_values(X)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "shap_importance": mean_abs,
        }).sort_values("shap_importance", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    def get_waterfall_data(self, sample: np.ndarray, max_display: int = 15) -> Dict[str, Any]:
        """
        Return data for a SHAP waterfall plot for a single sample.

        Args:
            sample: Single sample, shape (n_features,) or (1, n_features)
            max_display: Number of top features to display

        Returns:
            Dict with keys: base_value, shap_values, feature_values, feature_names
        """
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        shap_vals = self.explainer.shap_values(sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = shap_vals[0]  # (n_features,)

        # Sort by absolute value
        order = np.argsort(np.abs(shap_vals))[::-1][:max_display]

        return {
            "base_value": self.base_value or 0.0,
            "shap_values": shap_vals[order].tolist(),
            "feature_values": sample[0][order].tolist(),
            "feature_names": [self.feature_names[i] for i in order],
        }

    def get_dependence_data(
        self,
        X: np.ndarray,
        feature: str,
        interaction_feature: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return data for a SHAP dependence plot.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature: Primary feature name
            interaction_feature: Optional interaction feature to color by

        Returns:
            Dict with feature values, shap values, interaction values
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        feat_idx = self.feature_names.index(feature)
        shap_col = self.shap_values[:, feat_idx]
        feat_vals = X[:, feat_idx]

        result = {
            "feature": feature,
            "feature_values": feat_vals.tolist(),
            "shap_values": shap_col.tolist(),
        }

        if interaction_feature and interaction_feature in self.feature_names:
            int_idx = self.feature_names.index(interaction_feature)
            result["interaction_feature"] = interaction_feature
            result["interaction_values"] = X[:, int_idx].tolist()

        return result

    def get_summary_data(self, X: np.ndarray, top_n: int = 20) -> Dict[str, Any]:
        """
        Return data needed for beeswarm / summary plot.

        Returns:
            Dict with shap_values (n×k), feature_values (n×k), feature_names (k)
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        # Select top_n features
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_n]

        return {
            "shap_values": self.shap_values[:, top_idx].tolist(),
            "feature_values": X[:, top_idx].tolist(),
            "feature_names": [self.feature_names[i] for i in top_idx],
            "feature_importance": mean_abs[top_idx].tolist(),
        }
