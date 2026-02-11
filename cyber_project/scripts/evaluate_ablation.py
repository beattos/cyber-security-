"""
Evaluate trained models and generate ablation report.
This script assumes models are already trained and calibrated.
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(
    model_path: str,
    test_csv: str,
    feature_cols: list,
    thresholds: dict,
    model_name: str
) -> dict:
    """Evaluate a calibrated model on test set."""
    # Load model and imputer
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")
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
    
    # Prepare features - ensure correct order
    X_test = df_test[feature_cols].copy()
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


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained models for ablation study")
    ap.add_argument("--test_csv", required=True, help="Test CSV path")
    ap.add_argument("--feature_set", required=True, help="Feature set name (F0 or F1)")
    ap.add_argument("--output_dir", default="outputs/ablation", help="Output directory")
    args = ap.parse_args()
    
    # Load feature columns
    feature_cols_path = "models/dynamic_feature_cols.pkl"
    if not os.path.exists(feature_cols_path):
        raise FileNotFoundError(f"Missing {feature_cols_path}. Train models first.")
    feature_cols = joblib.load(feature_cols_path)
    
    print(f"Loaded {len(feature_cols)} feature columns: {feature_cols[:5]}...")
    
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
    
    # Evaluate both models
    ada_results = evaluate_model(
        "models/ada_dynamic_calibrated.pkl",
        args.test_csv,
        feature_cols,
        ada_thresholds,
        f"{args.feature_set}_AdaBoost"
    )
    
    gb_results = evaluate_model(
        "models/gb_dynamic_calibrated.pkl",
        args.test_csv,
        feature_cols,
        gb_thresholds,
        f"{args.feature_set}_GradientBoosting"
    )
    
    # Save results
    results = {
        "feature_set": args.feature_set,
        "test_csv": args.test_csv,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "ada": ada_results,
        "gb": gb_results,
    }
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    exp_dir = os.path.join(args.output_dir, args.feature_set.lower())
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    
    results_path = os.path.join(exp_dir, f"{args.feature_set.lower()}_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Text report
    report_path = os.path.join(exp_dir, f"{args.feature_set.lower()}_eval.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{args.feature_set} EXPERIMENT RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Feature set: {args.feature_set}\n")
        f.write(f"Test CSV: {args.test_csv}\n")
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
    print(f"\nAdaBoost: Accuracy={ada_results['accuracy']:.4f}, F1={ada_results['f1_class1']:.4f}")
    print(f"GradientBoosting: Accuracy={gb_results['accuracy']:.4f}, F1={gb_results['f1_class1']:.4f}")


if __name__ == "__main__":
    main()
