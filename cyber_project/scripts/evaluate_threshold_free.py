"""
ABLATION STUDY SCRIPT - NOT PART OF MAIN PIPELINE

This script performs ablation studies comparing F0 vs F1 feature sets.
It regenerates splits and retrains models, which is acceptable for research
but not part of the main evaluation pipeline.

For standard evaluation, use evaluate_threshold_free_demo.py instead.

Threshold-free evaluation: ROC-AUC and PR-AUC for dynamic models.
Evaluates AdaBoost vs GradientBoosting on F0 vs F1 feature sets.
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score


def load_test_data(test_csv: str, feature_cols: list) -> tuple[pd.DataFrame, np.ndarray]:
    """Load and prepare test data."""
    df = pd.read_csv(test_csv)
    df = df.drop(columns=["total_activity"], errors="ignore")
    
    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {test_csv}")
    
    # Sanity check: ensure label is not in features
    if "label" in feature_cols:
        raise ValueError("Label column found in feature list - potential leakage!")
    
    # Prepare features - align to feature_cols order
    X = pd.DataFrame(index=df.index)
    for feat in feature_cols:
        if feat in df.columns:
            col = pd.to_numeric(df[feat], errors="coerce")
            col = col.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X[feat] = col
        else:
            print(f"Warning: feature '{feat}' not found in CSV, filling with 0.0")
            X[feat] = 0.0
    
    X = X[feature_cols]  # Ensure correct order
    y = df["label"].astype(int).to_numpy()
    
    return X, y


def evaluate_model(
    model_path: str,
    test_csv: str,
    feature_cols: list,
    model_name: str
) -> dict:
    """Evaluate a model and return threshold-free metrics."""
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")
    model = joblib.load(model_path)
    
    # Load imputer
    imputer_path = "models/dynamic_imputer.pkl"
    if not os.path.exists(imputer_path):
        raise FileNotFoundError(f"Missing {imputer_path}")
    imputer = joblib.load(imputer_path)
    
    # Load test data
    X_test, y_test = load_test_data(test_csv, feature_cols)
    
    # Transform features
    X_test_imp = imputer.transform(X_test)
    X_test_imp = pd.DataFrame(X_test_imp, columns=feature_cols)
    
    # Get probabilities
    p_test = model.predict_proba(X_test_imp)[:, 1]
    
    # Sanity checks
    prob_min = float(np.min(p_test))
    prob_max = float(np.max(p_test))
    prob_mean = float(np.mean(p_test))
    n_samples = len(y_test)
    n_positives = int(np.sum(y_test == 1))
    n_negatives = int(np.sum(y_test == 0))
    
    # Compute metrics
    roc_auc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) == 2 else float("nan")
    pr_auc = average_precision_score(y_test, p_test)
    
    return {
        "model_name": model_name,
        "test_csv": test_csv,
        "n_samples": n_samples,
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "n_features": len(feature_cols),
        "prob_stats": {
            "min": prob_min,
            "max": prob_max,
            "mean": prob_mean,
        },
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def main():
    ap = argparse.ArgumentParser(description="Threshold-free evaluation: ROC-AUC and PR-AUC")
    ap.add_argument("--f0_test", default="data/dynamic_test.csv", help="F0 test CSV")
    ap.add_argument("--f1_test", default="data/dynamic_test.csv", help="F1 test CSV")
    ap.add_argument("--output_dir", default="outputs/ablation", help="Output directory")
    args = ap.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print("="*80)
    print("THRESHOLD-FREE EVALUATION: ROC-AUC and PR-AUC")
    print("="*80)
    print()
    
    # Evaluate F0 models
    print("Evaluating F0 models (11 features)...")
    print("-"*80)
    
    # Regenerate F0 splits to ensure we have F0 test data
    import subprocess
    subprocess.run(
        ["python", "scripts/make_splits.py", "--dynamic_csv", "data/dynamic_clean.csv"],
        capture_output=True
    )
    
    # Load F0 feature columns (should be 11)
    f0_feature_cols_path = "models/dynamic_feature_cols.pkl"
    if os.path.exists(f0_feature_cols_path):
        f0_feature_cols = joblib.load(f0_feature_cols_path)
        if len(f0_feature_cols) != 11:
            print(f"Warning: Expected 11 F0 features, found {len(f0_feature_cols)}")
            print("Regenerating F0 models...")
            # Train F0 models
            subprocess.run(
                ["python", "scripts/train_dynamic_models.py",
                 "--dynamic_train", "data/dynamic_train.csv",
                 "--dynamic_val", "data/dynamic_val.csv",
                 "--dynamic_test", "data/dynamic_test.csv"],
                capture_output=True
            )
            f0_feature_cols = joblib.load(f0_feature_cols_path)
    else:
        raise FileNotFoundError(f"Missing {f0_feature_cols_path}. Train F0 models first.")
    
    print(f"F0 feature columns: {len(f0_feature_cols)} features")
    
    # Evaluate F0 AdaBoost
    f0_ada_result = evaluate_model(
        "models/ada_dynamic_calibrated.pkl",
        "data/dynamic_test.csv",
        f0_feature_cols,
        "F0_AdaBoost"
    )
    results["F0_AdaBoost"] = f0_ada_result
    
    # Evaluate F0 GradientBoosting
    f0_gb_result = evaluate_model(
        "models/gb_dynamic_calibrated.pkl",
        "data/dynamic_test.csv",
        f0_feature_cols,
        "F0_GradientBoosting"
    )
    results["F0_GradientBoosting"] = f0_gb_result
    
    print()
    print("Evaluating F1 models (17 features)...")
    print("-"*80)
    
    # Regenerate F1 splits
    subprocess.run(
        ["python", "scripts/make_splits.py", "--dynamic_csv", "data/dynamic_clean_F1.csv"],
        capture_output=True
    )
    
    # Train F1 models
    print("Training F1 models...")
    subprocess.run(
        ["python", "scripts/train_dynamic_models.py",
         "--dynamic_train", "data/dynamic_train.csv",
         "--dynamic_val", "data/dynamic_val.csv",
         "--dynamic_test", "data/dynamic_test.csv"],
        capture_output=True
    )
    
    # Calibrate F1 models
    env = os.environ.copy()
    env["DYNAMIC_CSV"] = "data/dynamic_clean_F1.csv"
    subprocess.run(
        ["python", "scripts/calibrate_dynamic.py"],
        env=env,
        capture_output=True
    )
    
    # Load F1 feature columns (should be 17)
    f1_feature_cols = joblib.load(f0_feature_cols_path)  # Updated after F1 training
    if len(f1_feature_cols) != 17:
        raise ValueError(f"Expected 17 F1 features, found {len(f1_feature_cols)}")
    
    print(f"F1 feature columns: {len(f1_feature_cols)} features")
    
    # Evaluate F1 AdaBoost
    f1_ada_result = evaluate_model(
        "models/ada_dynamic_calibrated.pkl",
        "data/dynamic_test.csv",
        f1_feature_cols,
        "F1_AdaBoost"
    )
    results["F1_AdaBoost"] = f1_ada_result
    
    # Evaluate F1 GradientBoosting
    f1_gb_result = evaluate_model(
        "models/gb_dynamic_calibrated.pkl",
        "data/dynamic_test.csv",
        f1_feature_cols,
        "F1_GradientBoosting"
    )
    results["F1_GradientBoosting"] = f1_gb_result
    
    # Print results table
    print()
    print("="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(f"{'Model':<25} {'Feature Set':<12} {'ROC-AUC':<10} {'PR-AUC':<10} {'N Samples':<10}")
    print("-"*80)
    
    for key in ["F0_AdaBoost", "F0_GradientBoosting", "F1_AdaBoost", "F1_GradientBoosting"]:
        r = results[key]
        model_name = key.split("_")[1]
        feature_set = key.split("_")[0]
        print(f"{model_name:<25} {feature_set:<12} {r['roc_auc']:<10.4f} {r['pr_auc']:<10.4f} {r['n_samples']:<10}")
    
    print()
    print("="*80)
    print("SANITY CHECKS")
    print("="*80)
    for key, r in results.items():
        print(f"\n{key}:")
        print(f"  Samples: {r['n_samples']} (pos={r['n_positives']}, neg={r['n_negatives']})")
        print(f"  Features: {r['n_features']}")
        print(f"  Probabilities: min={r['prob_stats']['min']:.4f}, "
              f"mean={r['prob_stats']['mean']:.4f}, max={r['prob_stats']['max']:.4f}")
    
    # Save results
    json_path = os.path.join(args.output_dir, "threshold_free_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    txt_path = os.path.join(args.output_dir, "threshold_free_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("THRESHOLD-FREE EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write("Metrics: ROC-AUC and PR-AUC (Average Precision)\n")
        f.write("Test set only (no threshold tuning)\n\n")
        f.write("RESULTS TABLE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<25} {'Feature Set':<12} {'ROC-AUC':<10} {'PR-AUC':<10} {'N Samples':<10}\n")
        f.write("-"*80 + "\n")
        
        for key in ["F0_AdaBoost", "F0_GradientBoosting", "F1_AdaBoost", "F1_GradientBoosting"]:
            r = results[key]
            model_name = key.split("_")[1]
            feature_set = key.split("_")[0]
            f.write(f"{model_name:<25} {feature_set:<12} {r['roc_auc']:<10.4f} "
                   f"{r['pr_auc']:<10.4f} {r['n_samples']:<10}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for key, r in results.items():
            f.write(f"{key}:\n")
            f.write(f"  Test CSV: {r['test_csv']}\n")
            f.write(f"  Samples: {r['n_samples']} (pos={r['n_positives']}, neg={r['n_negatives']})\n")
            f.write(f"  Features: {r['n_features']}\n")
            f.write(f"  ROC-AUC: {r['roc_auc']:.6f}\n")
            f.write(f"  PR-AUC: {r['pr_auc']:.6f}\n")
            f.write(f"  Probability stats: min={r['prob_stats']['min']:.4f}, "
                   f"mean={r['prob_stats']['mean']:.4f}, max={r['prob_stats']['max']:.4f}\n")
            f.write("\n")
    
    print()
    print(f"Results saved to:")
    print(f"  {json_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()
