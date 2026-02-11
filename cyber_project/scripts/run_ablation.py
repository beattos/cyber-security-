"""
Run controlled ablation experiments: F0 vs F1, AdaBoost vs GradientBoosting.
Generates splits, trains, calibrates, and evaluates all four combinations.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def run_command(cmd: list, description: str):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed")
        print(result.stderr)
        sys.exit(1)
    # Print stdout and stderr (warnings go to stderr but are not errors)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        # Filter out sklearn FutureWarnings which are not errors
        stderr_lines = [l for l in result.stderr.split('\n') 
                       if 'FutureWarning' not in l and l.strip()]
        if stderr_lines:
            print("Warnings:", '\n'.join(stderr_lines))
    return result


def evaluate_model(
    model_path: str,
    test_csv: str,
    feature_cols: list,
    thresholds: dict,
    model_name: str
) -> dict:
    """Evaluate a calibrated model on test set."""
    import numpy as np
    from sklearn.impute import SimpleImputer
    
    # Load model and imputer
    model = joblib.load(model_path)
    imputer_path = "models/dynamic_imputer.pkl"
    if not os.path.exists(imputer_path):
        raise FileNotFoundError(f"Missing {imputer_path}")
    imputer = joblib.load(imputer_path)
    
    # Load test data
    df_test = pd.read_csv(test_csv)
    df_test = df_test.drop(columns=["total_activity"], errors="ignore")
    
    if "label" not in df_test.columns:
        raise ValueError(f"Missing 'label' in {test_csv}")
    
    # Prepare features
    X_test = df_test[feature_cols]
    y_test = df_test["label"].astype(int).to_numpy()
    
    # Transform
    X_test_imp = imputer.transform(X_test)
    X_test_imp = pd.DataFrame(X_test_imp, columns=feature_cols)
    
    # Predict
    p_test = model.predict_proba(X_test_imp)[:, 1]
    
    # Apply thresholds
    t_alert = thresholds.get("alert", 0.80)
    t_review = thresholds.get("review", 0.55)
    
    decision = np.where(p_test >= t_alert, "ALERT",
              np.where(p_test >= t_review, "REVIEW", "PASS"))
    y_pred_soc = np.where(decision == "PASS", 0, 1)
    
    # Metrics
    cm = confusion_matrix(y_test, y_pred_soc, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    report_dict = classification_report(y_test, y_pred_soc, labels=[0, 1], output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred_soc, labels=[0, 1], output_dict=False, digits=4, zero_division=0)
    
    return {
        "model_name": model_name,
        "n_test": len(df_test),
        "n_features": len(feature_cols),
        "thresholds": thresholds,
        "accuracy": float(accuracy),
        "precision_class1": float(report_dict.get("1", {}).get("precision", 0.0)),
        "recall_class1": float(report_dict.get("1", {}).get("recall", 0.0)),
        "f1_class1": float(report_dict.get("1", {}).get("f1-score", 0.0)),
        "confusion_matrix": cm.tolist(),
        "confusion_counts": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "triage_counts": {
            "ALERT": int((decision == "ALERT").sum()),
            "REVIEW": int((decision == "REVIEW").sum()),
            "PASS": int((decision == "PASS").sum()),
        },
        "classification_report": report_str,
    }


def run_experiment(
    feature_set: str,  # "F0" or "F1"
    csv_path: str,
    output_dir: str
):
    """Run complete experiment for one feature set."""
    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT: {feature_set}")
    print(f"{'#'*80}")
    
    exp_dir = os.path.join(output_dir, feature_set.lower())
    ensure_dir(exp_dir)
    
    # Step 1: Generate splits
    print(f"\n[Step 1] Generating splits for {feature_set}")
    run_command(
        ["python", "scripts/make_splits.py", "--dynamic_csv", csv_path],
        f"Generate splits for {feature_set}"
    )
    
    # Step 2: Train models
    print(f"\n[Step 2] Training models for {feature_set}")
    run_command(
        ["python", "scripts/train_dynamic_models.py",
         "--dynamic_train", "data/dynamic_train.csv",
         "--dynamic_val", "data/dynamic_val.csv",
         "--dynamic_test", "data/dynamic_test.csv"],
        f"Train AdaBoost and GradientBoosting for {feature_set}"
    )
    
    # Step 3: Calibrate models
    print(f"\n[Step 3] Calibrating models for {feature_set}")
    env = os.environ.copy()
    env["DYNAMIC_CSV"] = csv_path
    result = subprocess.run(
        ["python", "scripts/calibrate_dynamic.py"],
        env=env,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"ERROR: Calibration failed for {feature_set}")
        print(result.stderr)
        sys.exit(1)
    if result.stdout:
        print(result.stdout)
    # Filter warnings from stderr
    if result.stderr:
        stderr_lines = [l for l in result.stderr.split('\n') 
                       if 'FutureWarning' not in l and l.strip()]
        if stderr_lines:
            print("Calibration warnings:", '\n'.join(stderr_lines))
    
    # Step 4: Load feature columns and thresholds
    feature_cols = joblib.load("models/dynamic_feature_cols.pkl")
    print(f"\n[Step 4] Loaded feature columns: {len(feature_cols)} features")
    print(f"First 5: {feature_cols[:5]}")
    
    # Load thresholds from training report
    thresholds_path = "outputs/reports/dynamic_models_report.json"
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            training_report = json.load(f)
        ada_thresholds = training_report.get("ada", {}).get("thresholds", {"alert": 0.80, "review": 0.55})
        gb_thresholds = training_report.get("gb", {}).get("thresholds", {"alert": 0.80, "review": 0.55})
    else:
        ada_thresholds = {"alert": 0.80, "review": 0.55}
        gb_thresholds = {"alert": 0.80, "review": 0.55}
    
    # Step 5: Evaluate both models
    print(f"\n[Step 5] Evaluating models for {feature_set}")
    
    ada_results = evaluate_model(
        "models/ada_dynamic_calibrated.pkl",
        "data/dynamic_test.csv",
        feature_cols,
        ada_thresholds,
        f"{feature_set}_AdaBoost"
    )
    
    gb_results = evaluate_model(
        "models/gb_dynamic_calibrated.pkl",
        "data/dynamic_test.csv",
        feature_cols,
        gb_thresholds,
        f"{feature_set}_GradientBoosting"
    )
    
    # Save results
    results = {
        "feature_set": feature_set,
        "csv_path": csv_path,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "ada": ada_results,
        "gb": gb_results,
    }
    
    results_path = os.path.join(exp_dir, f"{feature_set.lower()}_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Also save text report
    report_path = os.path.join(exp_dir, f"{feature_set.lower()}_eval.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{feature_set} EXPERIMENT RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Feature set: {feature_set}\n")
        f.write(f"CSV path: {csv_path}\n")
        f.write(f"Number of features: {len(feature_cols)}\n")
        f.write(f"Feature columns: {feature_cols}\n\n")
        
        f.write("="*80 + "\n")
        f.write("ADABOOST RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Accuracy: {ada_results['accuracy']:.4f}\n")
        f.write(f"Precision (class 1): {ada_results['precision_class1']:.4f}\n")
        f.write(f"Recall (class 1): {ada_results['recall_class1']:.4f}\n")
        f.write(f"F1 (class 1): {ada_results['f1_class1']:.4f}\n")
        f.write(f"Confusion Matrix: TN={ada_results['confusion_counts']['TN']}, "
                f"FP={ada_results['confusion_counts']['FP']}, "
                f"FN={ada_results['confusion_counts']['FN']}, "
                f"TP={ada_results['confusion_counts']['TP']}\n")
        f.write(f"Thresholds: {ada_results['thresholds']}\n")
        f.write(f"\nClassification Report:\n{ada_results['classification_report']}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("GRADIENTBOOSTING RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Accuracy: {gb_results['accuracy']:.4f}\n")
        f.write(f"Precision (class 1): {gb_results['precision_class1']:.4f}\n")
        f.write(f"Recall (class 1): {gb_results['recall_class1']:.4f}\n")
        f.write(f"F1 (class 1): {gb_results['f1_class1']:.4f}\n")
        f.write(f"Confusion Matrix: TN={gb_results['confusion_counts']['TN']}, "
                f"FP={gb_results['confusion_counts']['FP']}, "
                f"FN={gb_results['confusion_counts']['FN']}, "
                f"TP={gb_results['confusion_counts']['TP']}\n")
        f.write(f"Thresholds: {gb_results['thresholds']}\n")
        f.write(f"\nClassification Report:\n{gb_results['classification_report']}\n")
    
    print(f"\nSaved results to: {results_path}")
    print(f"Saved text report to: {report_path}")
    
    return results


def main():
    ap = argparse.ArgumentParser(description="Run ablation experiments: F0 vs F1, AdaBoost vs GradientBoosting")
    ap.add_argument("--f0_csv", default="data/dynamic_clean.csv", help="F0 dataset (11 features)")
    ap.add_argument("--f1_csv", default="data/dynamic_clean_F1.csv", help="F1 dataset (17 features)")
    ap.add_argument("--output_dir", default="outputs/ablation", help="Output directory for results")
    args = ap.parse_args()
    
    ensure_dir(args.output_dir)
    
    all_results = {}
    
    # Run F0 experiment
    f0_results = run_experiment("F0", args.f0_csv, args.output_dir)
    all_results["F0"] = f0_results
    
    # Run F1 experiment
    f1_results = run_experiment("F1", args.f1_csv, args.output_dir)
    all_results["F1"] = f1_results
    
    # Generate summary report
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ABLATION EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("RESULTS TABLE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<20} {'Feature Set':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
        f.write("-"*80 + "\n")
        
        for feature_set in ["F0", "F1"]:
            results = all_results[feature_set]
            ada = results["ada"]
            gb = results["gb"]
            
            f.write(f"{'AdaBoost':<20} {feature_set:<12} {ada['accuracy']:<10.4f} "
                   f"{ada['precision_class1']:<10.4f} {ada['recall_class1']:<10.4f} "
                   f"{ada['f1_class1']:<10.4f}\n")
            f.write(f"{'GradientBoosting':<20} {feature_set:<12} {gb['accuracy']:<10.4f} "
                   f"{gb['precision_class1']:<10.4f} {gb['recall_class1']:<10.4f} "
                   f"{gb['f1_class1']:<10.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("CONFUSION MATRICES\n")
        f.write("="*80 + "\n\n")
        
        for feature_set in ["F0", "F1"]:
            results = all_results[feature_set]
            ada = results["ada"]
            gb = results["gb"]
            
            f.write(f"{feature_set} - AdaBoost:\n")
            f.write(f"  TN={ada['confusion_counts']['TN']}, FP={ada['confusion_counts']['FP']}, "
                   f"FN={ada['confusion_counts']['FN']}, TP={ada['confusion_counts']['TP']}\n")
            
            f.write(f"{feature_set} - GradientBoosting:\n")
            f.write(f"  TN={gb['confusion_counts']['TN']}, FP={gb['confusion_counts']['FP']}, "
                   f"FN={gb['confusion_counts']['FN']}, TP={gb['confusion_counts']['TP']}\n\n")
    
    print(f"\n{'#'*80}")
    print("# EXPERIMENTS COMPLETE")
    print(f"{'#'*80}")
    print(f"\nSummary saved to: {summary_path}")
    print("\nResults:")
    print(f"  F0: outputs/ablation/f0/f0_results.json")
    print(f"  F1: outputs/ablation/f1/f1_results.json")


if __name__ == "__main__":
    main()
