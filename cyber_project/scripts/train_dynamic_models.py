"""
Train both AdaBoost and GradientBoosting models for dynamic data.
Supports F0 (11 features) and F1 (17 features) datasets.

Protocol:
- Fit base models on train split
- Calibrate on val split (for threshold tuning and test evaluation)
- Evaluate on test split

Note: Calibrated models are saved by Stage 2 (calibrate_dynamic.py).
This script saves only base models (*_dynamic.pkl).
"""
import argparse
import json
import os
import warnings
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, classification_report

# Suppress only specific sklearn warnings
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_csv(path: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["total_activity"], errors="ignore")
    if label_col not in df.columns:
        raise ValueError(f"Missing '{label_col}' in {path}")
    return df


def fit_artifacts(df_train: pd.DataFrame, label_col: str) -> Tuple[SimpleImputer, list]:
    # Exclude label and sample_id from features
    feature_cols = [c for c in df_train.columns if c not in (label_col, "sample_id")]
    imputer = SimpleImputer(strategy="median")
    imputer.fit(df_train[feature_cols])
    return imputer, feature_cols


def transform(df: pd.DataFrame, feature_cols: list, imputer: SimpleImputer, label_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[feature_cols]
    y = df[label_col].astype(int).to_numpy()
    X_imp = imputer.transform(X)
    X_imp = pd.DataFrame(X_imp, columns=feature_cols)
    return X_imp, y


def get_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def tune_thresholds(y_val: np.ndarray, p_val: np.ndarray) -> Dict[str, float]:
    """
    Minimal tuning on validation:
    Choose thresholds that keep FN low while limiting FP and REVIEW volume.
    """
    best = None
    best_pair = {"alert": 0.80, "review": 0.55}

    alert_grid = np.arange(0.70, 0.96, 0.05)
    review_grid = np.arange(0.50, 0.86, 0.05)

    FN_COST = 5.0
    FP_COST = 1.0
    REVIEW_COST = 0.2

    for t_alert in alert_grid:
        for t_review in review_grid:
            if t_review >= t_alert:
                continue

            decision = np.where(p_val >= t_alert, "ALERT",
                       np.where(p_val >= t_review, "REVIEW", "PASS"))
            y_pred_soc = np.where(decision == "PASS", 0, 1)

            cm = confusion_matrix(y_val, y_pred_soc, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            n_review = int((decision == "REVIEW").sum())

            cost = FN_COST * fn + FP_COST * fp + REVIEW_COST * n_review

            if best is None or cost < best:
                best = cost
                best_pair = {"alert": float(t_alert), "review": float(t_review)}

    return best_pair


def train_one_model(
    model_type: str,  # "ada" or "gb"
    train_csv: str,
    val_csv: str,
    test_csv: str,
    label_col: str,
    out_dir: str
) -> Dict:
    """Train one model (AdaBoost or GradientBoosting) and return evaluation report."""
    df_tr = load_csv(train_csv, label_col)
    df_va = load_csv(val_csv, label_col)
    df_te = load_csv(test_csv, label_col)

    imputer, feature_cols = fit_artifacts(df_tr, label_col)

    X_tr, y_tr = transform(df_tr, feature_cols, imputer, label_col)
    X_va, y_va = transform(df_va, feature_cols, imputer, label_col)
    X_te, y_te = transform(df_te, feature_cols, imputer, label_col)

    # Create base model
    if model_type == "ada":
        base = AdaBoostClassifier(random_state=42, n_estimators=100)
    elif model_type == "gb":
        base = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    base.fit(X_tr, y_tr)

    # Calibrate on validation (using FrozenEstimator for sklearn 1.6+ compatibility)
    try:
        from sklearn.frozen import FrozenEstimator
        frozen_base = FrozenEstimator(base)
        calibrator = CalibratedClassifierCV(
            estimator=frozen_base,
            method="sigmoid",
        )
    except ImportError:
        # Fallback for older sklearn versions
        try:
            calibrator = CalibratedClassifierCV(
                estimator=base,
                method="sigmoid",
                cv="prefit",
            )
        except TypeError:
            calibrator = CalibratedClassifierCV(
                base_estimator=base,
                method="sigmoid",
                cv="prefit",
            )
    calibrator.fit(X_va, y_va)

    p_va = get_proba(calibrator, X_va)
    thresholds = tune_thresholds(y_va, p_va)

    p_te = get_proba(calibrator, X_te)

    # Test evaluation
    auc = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) == 2 else float("nan")

    # SOC mapping confusion: ALERT+REVIEW => 1, PASS => 0
    decision_te = np.where(p_te >= thresholds["alert"], "ALERT",
                  np.where(p_te >= thresholds["review"], "REVIEW", "PASS"))
    y_pred_soc = np.where(decision_te == "PASS", 0, 1)
    cm_soc = confusion_matrix(y_te, y_pred_soc, labels=[0, 1])

    # Classification report for class 1 (malware)
    report_str = classification_report(y_te, y_pred_soc, labels=[0, 1], output_dict=False, digits=4, zero_division=0)
    report_dict = classification_report(y_te, y_pred_soc, labels=[0, 1], output_dict=True, zero_division=0)

    tn, fp, fn, tp = cm_soc.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    result = {
        "model_type": model_type,
        "dataset_paths": {"train": train_csv, "val": val_csv, "test": test_csv},
        "n_train": int(len(df_tr)),
        "n_val": int(len(df_va)),
        "n_test": int(len(df_te)),
        "n_features": int(len(feature_cols)),
        "thresholds": thresholds,
        "test": {
            "roc_auc": float(auc),
            "accuracy": float(accuracy),
            "f1_soc_mapping": float(f1_score(y_te, y_pred_soc, zero_division=0)),
            "precision_class1": float(report_dict.get("1", {}).get("precision", 0.0)),
            "recall_class1": float(report_dict.get("1", {}).get("recall", 0.0)),
            "f1_class1": float(report_dict.get("1", {}).get("f1-score", 0.0)),
            "confusion_matrix": cm_soc.tolist(),
            "confusion_counts": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "triage_counts": {
                "ALERT": int((decision_te == "ALERT").sum()),
                "REVIEW": int((decision_te == "REVIEW").sum()),
                "PASS": int((decision_te == "PASS").sum()),
            },
            "classification_report": report_str,
            "note": "SOC mapping treats ALERT+REVIEW as predicted malware (1), PASS as benign (0).",
        },
    }

    # Save artifacts
    ensure_dir(out_dir)
    joblib.dump(base, os.path.join(out_dir, f"{model_type}_dynamic.pkl"))
    # Note: Calibrated models are saved by Stage 2 (calibrate_dynamic.py)
    # We keep calibrator here only for threshold tuning and test evaluation
    
    # Save imputer and feature_cols (overwrite if exists, but that's fine for consistent feature sets)
    joblib.dump(imputer, os.path.join(out_dir, "dynamic_imputer.pkl"))
    joblib.dump(feature_cols, os.path.join(out_dir, "dynamic_feature_cols.pkl"))

    return result


def main():
    ap = argparse.ArgumentParser(description="Train AdaBoost and GradientBoosting for dynamic data.")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--dynamic_train", default="data/dynamic_train.csv")
    ap.add_argument("--dynamic_val", default="data/dynamic_val.csv")
    ap.add_argument("--dynamic_test", default="data/dynamic_test.csv")
    args = ap.parse_args()

    results = {}
    
    # Train AdaBoost
    print("\n=== Training AdaBoost ===")
    ada_result = train_one_model(
        "ada",
        args.dynamic_train,
        args.dynamic_val,
        args.dynamic_test,
        args.label_col,
        args.out_dir
    )
    results["ada"] = ada_result
    
    # Train GradientBoosting
    print("\n=== Training GradientBoosting ===")
    gb_result = train_one_model(
        "gb",
        args.dynamic_train,
        args.dynamic_val,
        args.dynamic_test,
        args.label_col,
        args.out_dir
    )
    results["gb"] = gb_result

    # Save combined report
    ensure_dir("outputs/reports")
    with open("outputs/reports/dynamic_models_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== TRAINING COMPLETE ===")
    print("Saved: outputs/reports/dynamic_models_report.json")
    print(f"\nAdaBoost thresholds: {ada_result['thresholds']}")
    print(f"GradientBoosting thresholds: {gb_result['thresholds']}")
    print(f"\nAdaBoost test accuracy: {ada_result['test']['accuracy']:.4f}")
    print(f"GradientBoosting test accuracy: {gb_result['test']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
