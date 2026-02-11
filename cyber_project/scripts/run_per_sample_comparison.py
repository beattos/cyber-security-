#!/usr/bin/env python3
"""
Generate per-sample static vs dynamic comparison using sample_id.
Uses static_clean_with_id.csv and dynamic_clean_with_id.csv and trained models.
Prints comparison table and confusion-style summary.
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd


def _decision(p: float, t_alert: float, t_review: float) -> str:
    if p >= t_alert:
        return "ALERT"
    if p >= t_review:
        return "REVIEW"
    return "PASS"


def main():
    ap = argparse.ArgumentParser(description="Per-sample static vs dynamic comparison by sample_id")
    ap.add_argument("--static_with_id", default="data/static_clean_with_id.csv")
    ap.add_argument("--dynamic_with_id", default="data/dynamic_clean_with_id.csv")
    ap.add_argument("--output_dir", default="outputs/ablation")
    ap.add_argument("--head", type=int, default=15, help="Number of rows to print in comparison table")
    args = ap.parse_args()

    model_dir = Path("models")
    static_model_path = model_dir / "gb_static_calibrated.pkl"
    if not static_model_path.exists():
        static_model_path = model_dir / "gb_static.pkl"
    dynamic_model_path = model_dir / "gb_dynamic_calibrated.pkl"
    if not dynamic_model_path.exists():
        dynamic_model_path = model_dir / "gb_dynamic.pkl"
    for p in (static_model_path, dynamic_model_path):
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}. Train models first.")
    for p in (args.static_with_id, args.dynamic_with_id):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Dataset not found: {p}. Run create_sample_id first or use _with_id CSVs.")

    static_df = pd.read_csv(args.static_with_id)
    dynamic_df = pd.read_csv(args.dynamic_with_id)
    dynamic_df = dynamic_df.drop(columns=["total_activity"], errors="ignore")
    if "sample_id" not in static_df.columns or "sample_id" not in dynamic_df.columns:
        raise ValueError("Datasets must contain sample_id. Use create_sample_id or _with_id CSVs.")

    static_cols = joblib.load(model_dir / "static_feature_cols.pkl")
    dynamic_cols = joblib.load(model_dir / "dynamic_feature_cols.pkl")
    static_imputer = joblib.load(model_dir / "static_imputer.pkl")
    dynamic_imputer = joblib.load(model_dir / "dynamic_imputer.pkl")

    static_feats = [c for c in static_cols if c in static_df.columns]
    dynamic_feats = [c for c in dynamic_cols if c in dynamic_df.columns]
    if set(static_feats) != set(static_cols) or set(dynamic_feats) != set(dynamic_cols):
        raise ValueError("Feature mismatch between model and _with_id CSVs.")

    X_static = static_df[static_cols].copy()
    X_dynamic = dynamic_df[dynamic_cols].copy()
    X_static = pd.DataFrame(static_imputer.transform(X_static), columns=static_cols, index=X_static.index)
    X_dynamic = pd.DataFrame(dynamic_imputer.transform(X_dynamic), columns=dynamic_cols, index=X_dynamic.index)

    static_model = joblib.load(static_model_path)
    dynamic_model = joblib.load(dynamic_model_path)

    static_pred = static_model.predict(X_static)
    dynamic_pred = dynamic_model.predict(X_dynamic)
    try:
        static_proba = static_model.predict_proba(X_static)[:, 1]
        dynamic_proba = dynamic_model.predict_proba(X_dynamic)[:, 1]
    except Exception:
        static_proba = None
        dynamic_proba = None

    thresholds_path = model_dir / "thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            th = json.load(f)
        t_s = th.get("static", {"alert": 0.80, "review": 0.55})
        t_d = th.get("dynamic", {"alert": 0.80, "review": 0.55})
    else:
        t_s = t_d = {"alert": 0.80, "review": 0.55}

    comparison = pd.DataFrame({
        "sample_id": static_df["sample_id"],
        "label": static_df["label"],
        "static_pred": static_pred,
        "dynamic_pred": dynamic_pred,
    })
    if static_proba is not None:
        comparison["static_proba"] = static_proba
        comparison["static_decision"] = [
            _decision(p, t_s["alert"], t_s["review"]) for p in static_proba
        ]
    if dynamic_proba is not None:
        comparison["dynamic_proba"] = dynamic_proba
        comparison["dynamic_decision"] = [
            _decision(p, t_d["alert"], t_d["review"]) for p in dynamic_proba
        ]

    comparison["static_correct"] = (comparison["label"] == comparison["static_pred"]).astype(int)
    comparison["dynamic_correct"] = (comparison["label"] == comparison["dynamic_pred"]).astype(int)
    comparison["agreement"] = (comparison["static_pred"] == comparison["dynamic_pred"]).astype(int)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = Path(args.output_dir) / "static_vs_dynamic_by_sample.csv"
    comparison.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("PER-SAMPLE STATIC vs DYNAMIC COMPARISON (by sample_id)")
    print("=" * 80)
    n = len(comparison)
    both_correct = ((comparison["static_correct"] == 1) & (comparison["dynamic_correct"] == 1)).sum()
    both_wrong = ((comparison["static_correct"] == 0) & (comparison["dynamic_correct"] == 0)).sum()
    static_ok_dyn_fail = ((comparison["static_correct"] == 1) & (comparison["dynamic_correct"] == 0)).sum()
    static_fail_dyn_ok = ((comparison["static_correct"] == 0) & (comparison["dynamic_correct"] == 1)).sum()
    agree = comparison["agreement"].sum()
    print(f"\nSamples: {n}")
    print(f"  Both correct:     {both_correct} ({100*both_correct/n:.1f}%)")
    print(f"  Both wrong:       {both_wrong} ({100*both_wrong/n:.1f}%)")
    print(f"  Static ok, Dynamic wrong: {static_ok_dyn_fail}")
    print(f"  Static wrong, Dynamic ok: {static_fail_dyn_ok}")
    print(f"  Prediction agreement (static_pred == dynamic_pred): {agree} ({100*agree/n:.1f}%)")
    print(f"\nSaved: {out_path}")
    print("\nComparison table (first {} rows):".format(args.head))
    print("-" * 80)
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 8)
    print(comparison.head(args.head).to_string())
    print("=" * 80)


if __name__ == "__main__":
    main()
