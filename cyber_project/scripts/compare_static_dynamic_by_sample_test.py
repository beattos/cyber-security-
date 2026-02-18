#!/usr/bin/env python3
"""
Generate per-sample static vs dynamic comparison using sample_id (TEST ONLY).
Uses data/static_test.csv and data/dynamic_test.csv and trained models.
Prints comparison table and confusion-style summary for test split only.
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
    ap = argparse.ArgumentParser(description="Per-sample static vs dynamic comparison by sample_id (TEST ONLY)")
    ap.add_argument("--static_test", default="data/static_test.csv")
    ap.add_argument("--dynamic_test", default="data/dynamic_test.csv")
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
    for p in (args.static_test, args.dynamic_test):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Dataset not found: {p}. Run make_splits.py first to create test splits.")

    # Load test datasets
    static_df = pd.read_csv(args.static_test)
    dynamic_df = pd.read_csv(args.dynamic_test)
    dynamic_df = dynamic_df.drop(columns=["total_activity"], errors="ignore")
    
    # Guard: ensure sample_id exists
    if "sample_id" not in static_df.columns:
        raise ValueError(
            f"Missing 'sample_id' column in {args.static_test}. "
            f"Stage 0 (make_splits.py) must create sample_id before splitting. "
            f"Re-run: python scripts/make_splits.py"
        )
    if "sample_id" not in dynamic_df.columns:
        raise ValueError(
            f"Missing 'sample_id' column in {args.dynamic_test}. "
            f"Stage 0 (make_splits.py) must create sample_id before splitting. "
            f"Re-run: python scripts/make_splits.py"
        )

    # Merge on sample_id (inner join - only samples present in both)
    merged = pd.merge(
        static_df,
        dynamic_df,
        on="sample_id",
        how="inner",
        suffixes=("_static", "_dynamic")
    )
    
    # Ensure label columns match (should be same for same sample_id)
    if "label_static" in merged.columns and "label_dynamic" in merged.columns:
        # Verify labels match (sanity check)
        label_mismatch = (merged["label_static"] != merged["label_dynamic"]).sum()
        if label_mismatch > 0:
            print(f"Warning: {label_mismatch} samples have mismatched labels between static and dynamic.")
        merged["label"] = merged["label_static"]  # Use static label as canonical
    elif "label" in merged.columns:
        # Single label column (already merged)
        pass
    else:
        raise ValueError("Could not find label column after merge. Expected 'label' or 'label_static'/'label_dynamic'.")
    
    n_samples = len(merged)
    print(f"\nMerged test datasets on sample_id: {n_samples} samples")
    if n_samples == 0:
        raise ValueError("No samples found after merging on sample_id. Check that test splits include sample_id.")

    # Load models and preprocessing artifacts
    static_cols = joblib.load(model_dir / "static_feature_cols.pkl")
    dynamic_cols = joblib.load(model_dir / "dynamic_feature_cols.pkl")
    static_imputer = joblib.load(model_dir / "static_imputer.pkl")
    dynamic_imputer = joblib.load(model_dir / "dynamic_imputer.pkl")

    # Extract features (handle suffix columns from merge)
    static_feat_cols = [c for c in static_cols if c in merged.columns]
    dynamic_feat_cols = [c for c in dynamic_cols if c in merged.columns]
    
    if set(static_feat_cols) != set(static_cols):
        missing = set(static_cols) - set(static_feat_cols)
        raise ValueError(f"Missing static features after merge: {missing}")
    if set(dynamic_feat_cols) != set(dynamic_cols):
        missing = set(dynamic_cols) - set(dynamic_feat_cols)
        raise ValueError(f"Missing dynamic features after merge: {missing}")

    X_static = merged[static_cols].copy()
    X_dynamic = merged[dynamic_cols].copy()
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

    # Load thresholds
    thresholds_path = model_dir / "thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            th = json.load(f)
        t_s = th.get("static", {"alert": 0.80, "review": 0.55})
        t_d = th.get("dynamic", {"alert": 0.80, "review": 0.55})
    else:
        t_s = t_d = {"alert": 0.80, "review": 0.55}

    # Build comparison DataFrame
    comparison = pd.DataFrame({
        "sample_id": merged["sample_id"],
        "label": merged["label"],
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

    # Save to test-specific output path
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = Path(args.output_dir) / "static_vs_dynamic_by_sample_test.csv"
    comparison.to_csv(out_path, index=False)

    # Print summary (mirrors Stage 5 but with TEST ONLY header)
    print("\n" + "=" * 80)
    print("PER-SAMPLE STATIC vs DYNAMIC COMPARISON (TEST ONLY, by sample_id)")
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
