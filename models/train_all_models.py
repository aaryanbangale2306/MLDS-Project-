"""
Master Training Pipeline — Train All 5 Models
===============================================
Run this script once to generate and save all model artifacts:

    python models/train_all_models.py

Outputs:
    - models/saved/xgboost_model.pkl
    - models/saved/random_forest_model.pkl
    - models/saved/lightgbm_model.pkl
    - models/saved/catboost_model.pkl
    - models/saved/hybrid_model.pkl
    - models/saved/preprocessor.pkl
    - models/saved/metrics.pkl         (all evaluation metrics)
    - models/saved/roc_data.pkl        (ROC curve data for all models)
    - models/saved/pr_data.pkl         (PR curve data)
    - models/saved/shap_data.pkl       (SHAP summary data for XGBoost)
    - models/saved/optuna_history.pkl  (Optuna tuning results)
    - models/saved/training_config.json
"""

import sys
import os
import json
import time
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.generate_dataset import generate_dataset
from utils.preprocessing import (
    MalwarePreprocessor, prepare_splits, compute_scale_pos_weight, ALL_FEATURES
)
from utils.evaluation import (
    compute_all_metrics, benchmark_inference_time,
    get_roc_curve_data, get_pr_curve_data,
)
from utils.optuna_tuner import tune_xgboost, get_optuna_history
from utils.shap_explainer import MalwareSHAPExplainer

from models.xgboost_model import XGBoostMalwareDetector
from models.random_forest_model import RandomForestMalwareDetector
from models.lightgbm_model import LightGBMMalwareDetector
from models.catboost_model import CatBoostMalwareDetector
from models.hybrid_model import HybridMalwareDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SAVED_DIR = project_root / "models" / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = project_root / "data" / "synthetic_malware_data.csv"

OPTUNA_TRIALS = 25


def banner(text: str) -> None:
    width = 60
    logger.info("=" * width)
    logger.info(f"  {text}")
    logger.info("=" * width)


def load_or_generate_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        logger.info(f"Loading existing dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        logger.info("Generating synthetic dataset (50,000 samples)...")
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
        logger.info(f"Dataset saved to {DATA_PATH}")
    logger.info(f"Dataset shape: {df.shape} | Malware: {df['label'].mean():.2%}")
    return df


def train_and_evaluate(
    model_obj,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str,
) -> dict:
    """Generic train → evaluate → save loop."""
    logger.info(f"Training {model_name}...")
    model_obj.fit(X_train, y_train, X_val, y_val)
    logger.info(f"  Train time: {model_obj.train_time:.1f}s")

    y_pred = model_obj.predict(X_test)
    y_proba = model_obj.predict_proba(X_test)[:, 1]

    inf_time = benchmark_inference_time(model_obj, X_test[:500])
    model_obj.save(save_path)
    model_size = os.path.getsize(save_path) / 1e6  # MB

    metrics = compute_all_metrics(
        y_test, y_pred, y_proba,
        model_name=model_name,
        train_time=model_obj.train_time,
        inference_time_ms=inf_time,
        model_size_mb=model_size,
    )
    logger.info(
        f"  AUC={metrics['roc_auc']:.4f} | F1={metrics['f1_score']:.4f} "
        f"| Recall={metrics['recall']:.4f} | Size={model_size:.1f}MB"
    )
    return metrics, y_proba


def main():
    banner("XGBoost Malware Detection — Training Pipeline")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    banner("Step 1/7: Data Loading")
    df = load_or_generate_data()
    train_df, val_df, test_df = prepare_splits(df)

    preprocessor = MalwarePreprocessor(scaler_type="robust")
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    spw = compute_scale_pos_weight(pd.Series(y_train))
    logger.info(f"scale_pos_weight = {spw:.2f}")

    preprocessor.save(str(SAVED_DIR / "preprocessor.pkl"))

    # ── 2. Optuna Hyperparameter Tuning ───────────────────────────────────────
    banner("Step 2/7: Optuna Tuning for XGBoost")
    from utils.preprocessing import get_interaction_constraints
    constraints = get_interaction_constraints(ALL_FEATURES)

    try:
        best_params, study = tune_xgboost(
            X_train, y_train, X_val, y_val,
            interaction_constraints=constraints,
            scale_pos_weight=spw,
            n_trials=OPTUNA_TRIALS,
        )
        optuna_history = get_optuna_history(study)
        logger.info(f"Best AUC: {optuna_history['best_value']:.4f}")
    except Exception as e:
        logger.warning(f"Optuna tuning failed ({e}), using default params")
        best_params = {}
        optuna_history = {"trial_numbers": [], "values": [], "best_values": [],
                          "best_params": {}, "best_value": 0.0, "n_trials": 0}

    joblib.dump(optuna_history, SAVED_DIR / "optuna_history.pkl")

    # ── 3. Train All 5 Models ─────────────────────────────────────────────────
    banner("Step 3/7: Training All 5 Models")

    all_metrics = []
    roc_data = {}
    pr_data = {}
    model_y_probas = {}

    # ── Model 1: XGBoost (primary) ────────────────────────────────────────────
    xgb_detector = XGBoostMalwareDetector(
        feature_names=ALL_FEATURES,
        hyperparams=best_params if best_params else None,
        with_constraints=True,
        scale_pos_weight=spw,
    )
    metrics_xgb, y_proba_xgb = train_and_evaluate(
        xgb_detector, "XGBoost",
        X_train, y_train, X_val, y_val, X_test, y_test,
        str(SAVED_DIR / "xgboost_model.pkl"),
    )
    all_metrics.append(metrics_xgb)
    fpr, tpr, _ = get_roc_curve_data(y_test, y_proba_xgb)
    roc_data["XGBoost"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": metrics_xgb["roc_auc"]}
    prec, rec, _ = get_pr_curve_data(y_test, y_proba_xgb)
    pr_data["XGBoost"] = {"precision": prec.tolist(), "recall": rec.tolist(), "pr_auc": metrics_xgb["pr_auc"]}
    model_y_probas["XGBoost"] = y_proba_xgb

    # ── XGBoost without constraints (for ablation study) ──────────────────────
    xgb_no_constraints = XGBoostMalwareDetector(
        feature_names=ALL_FEATURES,
        hyperparams=best_params if best_params else None,
        with_constraints=False,
        scale_pos_weight=spw,
    )
    xgb_no_constraints.fit(X_train, y_train, X_val, y_val)
    y_pred_nc = xgb_no_constraints.predict(X_test)
    y_proba_nc = xgb_no_constraints.predict_proba(X_test)[:, 1]
    metrics_nc = compute_all_metrics(y_test, y_pred_nc, y_proba_nc, model_name="XGBoost (no constraints)")
    xgb_no_constraints.save(str(SAVED_DIR / "xgboost_no_constraints.pkl"))
    joblib.dump({"with_constraints": metrics_xgb, "without_constraints": metrics_nc},
                SAVED_DIR / "ablation_results.pkl")
    logger.info(f"  Ablation — With constraints AUC: {metrics_xgb['roc_auc']:.4f} | "
                f"Without: {metrics_nc['roc_auc']:.4f}")

    # ── Model 2: Random Forest ────────────────────────────────────────────────
    rf_detector = RandomForestMalwareDetector(feature_names=ALL_FEATURES)
    metrics_rf, y_proba_rf = train_and_evaluate(
        rf_detector, "Random Forest",
        X_train, y_train, X_val, y_val, X_test, y_test,
        str(SAVED_DIR / "random_forest_model.pkl"),
    )
    all_metrics.append(metrics_rf)
    fpr, tpr, _ = get_roc_curve_data(y_test, y_proba_rf)
    roc_data["Random Forest"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": metrics_rf["roc_auc"]}
    prec, rec, _ = get_pr_curve_data(y_test, y_proba_rf)
    pr_data["Random Forest"] = {"precision": prec.tolist(), "recall": rec.tolist(), "pr_auc": metrics_rf["pr_auc"]}

    # ── Model 3: LightGBM ─────────────────────────────────────────────────────
    lgb_detector = LightGBMMalwareDetector(feature_names=ALL_FEATURES)
    metrics_lgb, y_proba_lgb = train_and_evaluate(
        lgb_detector, "LightGBM",
        X_train, y_train, X_val, y_val, X_test, y_test,
        str(SAVED_DIR / "lightgbm_model.pkl"),
    )
    all_metrics.append(metrics_lgb)
    fpr, tpr, _ = get_roc_curve_data(y_test, y_proba_lgb)
    roc_data["LightGBM"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": metrics_lgb["roc_auc"]}
    prec, rec, _ = get_pr_curve_data(y_test, y_proba_lgb)
    pr_data["LightGBM"] = {"precision": prec.tolist(), "recall": rec.tolist(), "pr_auc": metrics_lgb["pr_auc"]}

    # ── Model 4: CatBoost ─────────────────────────────────────────────────────
    try:
        cb_detector = CatBoostMalwareDetector(feature_names=ALL_FEATURES)
        metrics_cb, y_proba_cb = train_and_evaluate(
            cb_detector, "CatBoost",
            X_train, y_train, X_val, y_val, X_test, y_test,
            str(SAVED_DIR / "catboost_model.pkl"),
        )
        all_metrics.append(metrics_cb)
        fpr, tpr, _ = get_roc_curve_data(y_test, y_proba_cb)
        roc_data["CatBoost"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": metrics_cb["roc_auc"]}
        prec, rec, _ = get_pr_curve_data(y_test, y_proba_cb)
        pr_data["CatBoost"] = {"precision": prec.tolist(), "recall": rec.tolist(), "pr_auc": metrics_cb["pr_auc"]}
    except ImportError as e:
        logger.warning(f"CatBoost not installed, skipping: {e}")
        all_metrics.append({"model_name": "CatBoost", "roc_auc": 0, "f1_score": 0,
                             "accuracy": 0, "precision": 0, "recall": 0,
                             "pr_auc": 0, "mcc": 0, "f2_score": 0,
                             "false_positive_rate": 1, "false_negative_rate": 1,
                             "detection_at_1pct_fpr": 0, "train_time_s": 0,
                             "inference_time_ms": 0, "model_size_mb": 0,
                             "tp": 0, "tn": 0, "fp": 0, "fn": 0})

    # ── Model 5: Hybrid IF+GB ─────────────────────────────────────────────────
    hybrid_detector = HybridMalwareDetector(feature_names=ALL_FEATURES)
    metrics_hyb, y_proba_hyb = train_and_evaluate(
        hybrid_detector, "Hybrid (IF+GB)",
        X_train, y_train, X_val, y_val, X_test, y_test,
        str(SAVED_DIR / "hybrid_model.pkl"),
    )
    all_metrics.append(metrics_hyb)
    fpr, tpr, _ = get_roc_curve_data(y_test, y_proba_hyb)
    roc_data["Hybrid (IF+GB)"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": metrics_hyb["roc_auc"]}
    prec, rec, _ = get_pr_curve_data(y_test, y_proba_hyb)
    pr_data["Hybrid (IF+GB)"] = {"precision": prec.tolist(), "recall": rec.tolist(), "pr_auc": metrics_hyb["pr_auc"]}

    # ── 4. SHAP Analysis for XGBoost ─────────────────────────────────────────
    banner("Step 4/7: SHAP Analysis")
    try:
        background_idx = np.random.default_rng(42).choice(len(X_train), size=300, replace=False)
        shap_explainer = MalwareSHAPExplainer(
            xgb_detector.model, feature_names=ALL_FEATURES, model_name="XGBoost"
        )
        shap_explainer.fit(X_train[background_idx])
        shap_sample_idx = np.random.default_rng(42).choice(len(X_test), size=500, replace=False)
        summary_data = shap_explainer.get_summary_data(X_test[shap_sample_idx])

        # Waterfall for one malware sample
        malware_idx = np.where(y_test == 1)[0][:1]
        if len(malware_idx) > 0:
            waterfall_data = shap_explainer.get_waterfall_data(X_test[malware_idx[0]])
        else:
            waterfall_data = {}

        # Top-3 features for dependence plots
        top3_feat_idx = np.argsort(summary_data["feature_importance"])[::-1][:3]
        dep_data = {}
        for idx in top3_feat_idx:
            fname = summary_data["feature_names"][idx]
            if fname in ALL_FEATURES:
                dep_data[fname] = shap_explainer.get_dependence_data(X_test[shap_sample_idx], fname)

        shap_data = {
            "summary": summary_data,
            "waterfall": waterfall_data,
            "dependence": dep_data,
            "feature_importance": shap_explainer.get_feature_importance(X_test[shap_sample_idx]),
        }
        joblib.dump(shap_data, SAVED_DIR / "shap_data.pkl")
        joblib.dump(shap_explainer, SAVED_DIR / "shap_explainer.pkl")
        logger.info("SHAP analysis complete.")
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")
        joblib.dump({}, SAVED_DIR / "shap_data.pkl")

    # ── 5. XGBoost Feature Importance ─────────────────────────────────────────
    banner("Step 5/7: Feature Importance")
    fi_data = {}
    for importance_type in ["gain", "weight", "cover"]:
        try:
            fi_data[importance_type] = xgb_detector.get_feature_importance(importance_type)
        except Exception as e:
            logger.warning(f"Feature importance ({importance_type}) failed: {e}")
    joblib.dump(fi_data, SAVED_DIR / "feature_importance.pkl")

    # ── 6. Save All Metrics ───────────────────────────────────────────────────
    banner("Step 6/7: Saving Metrics & Curves")
    metrics_df = pd.DataFrame(all_metrics).set_index("model_name")
    joblib.dump(all_metrics, SAVED_DIR / "metrics.pkl")
    joblib.dump(roc_data, SAVED_DIR / "roc_data.pkl")
    joblib.dump(pr_data, SAVED_DIR / "pr_data.pkl")
    metrics_df.to_csv(SAVED_DIR / "metrics_summary.csv")

    # Save test data (small sample for dashboard demos)
    test_sample = pd.DataFrame(X_test[:1000], columns=ALL_FEATURES)
    test_sample["label"] = y_test[:1000]
    test_sample.to_csv(SAVED_DIR / "test_sample.csv", index=False)
    joblib.dump({"X_test": X_test[:500], "y_test": y_test[:500]}, SAVED_DIR / "test_data.pkl")

    # Save training config
    config = {
        "n_samples": len(df),
        "n_features": len(ALL_FEATURES),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "malware_ratio": float(df["label"].mean()),
        "scale_pos_weight": float(spw),
        "optuna_trials": OPTUNA_TRIALS,
        "best_xgb_params": best_params,
        "feature_names": ALL_FEATURES,
    }
    with open(SAVED_DIR / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── 7. Final Summary ──────────────────────────────────────────────────────
    banner("Step 7/7: Training Complete!")
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    print(metrics_df[["accuracy", "f1_score", "roc_auc", "pr_auc", "recall", "mcc"]].to_string())
    print("=" * 70)

    best_model = metrics_df["roc_auc"].idxmax()
    print(f"\n[BEST] Best Model (ROC-AUC): {best_model} ({metrics_df.loc[best_model, 'roc_auc']:.4f})")
    print(f"[SAVED] All models saved in: {SAVED_DIR}")

    return all_metrics


if __name__ == "__main__":
    main()
