#!/usr/bin/env python3
"""
Threshold-free evaluation (demo): ROC-AUC and PR-AUC for current models.
Uses existing train/val/test splits and models; no retraining.
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def load_test_and_predict(model_path: str, imputer_path: str, feature_cols_path: str,
                          test_csv: str, source: str) -> tuple:
    """Load model, imputer, features; load test CSV; return y_true, p_pred."""
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    feature_cols = joblib.load(feature_cols_path)

    df = pd.read_csv(test_csv)
    df = df.drop(columns=["total_activity"], errors="ignore")
    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' in {test_csv}")
    y = df["label"].astype(int).to_numpy()

    X = df[[c for c in feature_cols if c in df.columns]].copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols]
    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_cols)
    p = model.predict_proba(X_imp)[:, 1]
    return y, p


def main():
    ap = argparse.ArgumentParser(description="Threshold-free metrics (ROC-AUC, PR-AUC) for current models")
    ap.add_argument("--output_dir", default="outputs/ablation")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = []

    print("\n" + "=" * 70)
    print("THRESHOLD-FREE EVALUATION (ROC-AUC, PR-AUC)")
    print("=" * 70)

    # Static GB
    if os.path.exists("models/gb_static_calibrated.pkl"):
        y, p = load_test_and_predict(
            "models/gb_static_calibrated.pkl",
            "models/static_imputer.pkl",
            "models/static_feature_cols.pkl",
            "data/static_test.csv",
            "static",
        )
        roc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
        pr = average_precision_score(y, p)
        results.append({"model": "static_GB", "roc_auc": roc, "pr_auc": pr, "n": len(y)})
        print(f"\nStatic (GradientBoosting, calibrated):")
        print(f"  Test n={len(y)}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")

    # Dynamic AdaBoost
    if os.path.exists("models/ada_dynamic_calibrated.pkl"):
        y, p = load_test_and_predict(
            "models/ada_dynamic_calibrated.pkl",
            "models/dynamic_imputer.pkl",
            "models/dynamic_feature_cols.pkl",
            "data/dynamic_test.csv",
            "dynamic",
        )
        roc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
        pr = average_precision_score(y, p)
        results.append({"model": "dynamic_AdaBoost", "roc_auc": roc, "pr_auc": pr, "n": len(y)})
        print(f"\nDynamic (AdaBoost, calibrated):")
        print(f"  Test n={len(y)}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")

    # Dynamic GB
    if os.path.exists("models/gb_dynamic_calibrated.pkl"):
        y, p = load_test_and_predict(
            "models/gb_dynamic_calibrated.pkl",
            "models/dynamic_imputer.pkl",
            "models/dynamic_feature_cols.pkl",
            "data/dynamic_test.csv",
            "dynamic",
        )
        roc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
        pr = average_precision_score(y, p)
        results.append({"model": "dynamic_GB", "roc_auc": roc, "pr_auc": pr, "n": len(y)})
        print(f"\nDynamic (GradientBoosting, calibrated):")
        print(f"  Test n={len(y)}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")

    if not results:
        print("\nNo calibrated models found. Train models first.")
        return

    print("\n" + "-" * 70)
    print(f"{'Model':<22} {'ROC-AUC':<10} {'PR-AUC':<10} {'N':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<22} {r['roc_auc']:<10.4f} {r['pr_auc']:<10.4f} {r['n']:<8}")
    print("=" * 70)

    out_path = Path(args.output_dir) / "threshold_free_demo.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}\n")


if __name__ == "__main__":
    main()
