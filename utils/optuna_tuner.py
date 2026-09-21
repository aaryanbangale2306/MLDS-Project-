"""
Optuna Hyperparameter Tuning Utilities
"""

import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    interaction_constraints: list,
    scale_pos_weight: float,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    study_name: str = "xgboost_malware",
) -> Dict[str, Any]:
    """
    Tune XGBoost hyperparameters with Optuna on ROC-AUC objective.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data for early stopping metric.
        interaction_constraints: List of feature index groups.
        scale_pos_weight: Class balance weight.
        n_trials: Number of Optuna trials.
        timeout: Max seconds for tuning (optional).
        study_name: Optuna study identifier.

    Returns:
        Dict of best hyperparameters.
    """
    from sklearn.metrics import roc_auc_score
    from utils.preprocessing import ALL_FEATURES

    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns=ALL_FEATURES[:X_train.shape[1]])
    if isinstance(X_val, np.ndarray):
        X_val = pd.DataFrame(X_val, columns=ALL_FEATURES[:X_val.shape[1]])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": scale_pos_weight,
            "interaction_constraints": interaction_constraints,
            "tree_method": "hist",
            "eval_metric": "auc",
            "early_stopping_rounds": 30,
            "random_state": 42,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_proba)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    logger.info(f"Best AUC: {study.best_value:.4f} | Params: {study.best_params}")
    return study.best_params, study


def get_optuna_history(study: optuna.Study) -> dict:
    """Extract trial history for plotting."""
    trials = study.trials
    return {
        "trial_numbers": [t.number for t in trials if t.state == optuna.trial.TrialState.COMPLETE],
        "values": [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE],
        "best_values": list(np.maximum.accumulate(
            [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        )),
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(trials),
    }
