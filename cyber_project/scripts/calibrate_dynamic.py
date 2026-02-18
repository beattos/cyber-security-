"""
Canonical calibration step for dynamic models.

Loads base models from Stage 1 (train_dynamic_models.py) and calibrates them
on validation data. This is the single source of truth for calibrated models.

Protocol:
- Train on train split (Stage 1)
- Calibrate on val split (Stage 2) <- This script
- Evaluate on test split (Stage 4)
"""
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix

# Suppress only specific sklearn warnings
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass


def load_feature_cols(pkl_path: str) -> list[str]:
    """Load feature column list from saved pkl file (matches training)."""
    return joblib.load(pkl_path)


def load_dynamic_dataset(csv_path: str, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and align columns to match training feature order."""
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column for calibration.")

    # Drop total_activity if present (EDA-only column)
    df = df.drop(columns=["total_activity"], errors="ignore")

    labels = pd.to_numeric(df["label"], errors="coerce")
    valid_mask = labels.notna()
    if not valid_mask.all():
        dropped = (~valid_mask).sum()
        print(f"Dropping {dropped} rows with non-numeric labels.")
    df = df.loc[valid_mask].reset_index(drop=True)
    labels = labels.loc[valid_mask].astype(int).reset_index(drop=True)

    # Align features to match training: use feature_cols order, fill missing with 0
    X = pd.DataFrame(index=df.index)
    for feat in feature_cols:
        if feat in df.columns:
            col = pd.to_numeric(df[feat], errors="coerce")
            col = col.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X[feat] = col
        else:
            # Missing feature (shouldn't happen if splits match, but handle gracefully)
            print(f"Warning: feature '{feat}' not found in CSV, filling with 0.0")
            X[feat] = 0.0

    # Ensure exact column order matches training
    X = X[feature_cols]
    y = labels
    return X, y


def analyze_probabilities(p: np.ndarray, y: np.ndarray, model_name: str) -> dict:
    """Analyze probability distribution for diagnostics."""
    p_rounded = np.round(p, 3)  # Round to 3 decimals for value_counts
    
    stats = {
        "model": model_name,
        "n_samples": int(len(p)),
        "n_positives": int(np.sum(y == 1)),
        "n_negatives": int(np.sum(y == 0)),
        "prob_stats": {
            "min": float(np.min(p)),
            "mean": float(np.mean(p)),
            "max": float(np.max(p)),
            "median": float(np.median(p)),
            "std": float(np.std(p)),
        },
        "percentiles": {
            "p25": float(np.percentile(p, 25)),
            "p50": float(np.percentile(p, 50)),
            "p75": float(np.percentile(p, 75)),
            "p90": float(np.percentile(p, 90)),
            "p95": float(np.percentile(p, 95)),
        },
        "below_thresholds": {
            "below_0.5": float(np.mean(p < 0.5) * 100),
            "below_0.7": float(np.mean(p < 0.7) * 100),
        },
        "top_10_probabilities": {}
    }
    
    # Top 10 most common probabilities (rounded)
    unique, counts = np.unique(p_rounded, return_counts=True)
    top_indices = np.argsort(counts)[-10:][::-1]
    for idx in top_indices:
        stats["top_10_probabilities"][f"{unique[idx]:.3f}"] = int(counts[idx])
    
    return stats


def select_thresholds_fpr(y_val: np.ndarray, p_val: np.ndarray, target_fpr: float = 0.05) -> dict:
    """
    Select thresholds to achieve target FPR on validation set.
    
    Strategy:
    - Find alert threshold that achieves target FPR (e.g., 5% false positives)
    - Set review threshold to be 0.15 below alert threshold (or minimum 0.5)
    """
    # Sort probabilities for threshold search
    sorted_indices = np.argsort(p_val)
    sorted_probs = p_val[sorted_indices]
    sorted_labels = y_val[sorted_indices]
    
    n_negatives = np.sum(y_val == 0)
    if n_negatives == 0:
        return {"alert": 0.80, "review": 0.55}  # Fallback
    
    target_fp = int(n_negatives * target_fpr)
    
    # Find threshold that gives approximately target FPR
    # We want: FP / n_negatives ≈ target_fpr
    # So: count negatives with p >= threshold ≈ target_fp
    # Since negatives should have low p, we want threshold high enough that few negatives exceed it
    
    best_threshold = 0.80
    best_fpr = 1.0
    
    # Search from high to low thresholds
    for threshold in np.arange(0.95, 0.50, -0.01):
        y_pred = (p_val >= threshold).astype(int)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / n_negatives if n_negatives > 0 else 0.0
            
            # Find threshold closest to target FPR
            if abs(fpr - target_fpr) < abs(best_fpr - target_fpr):
                best_threshold = threshold
                best_fpr = fpr
    
    # Set alert threshold
    t_alert = best_threshold
    
    # Set review threshold: 0.15 below alert, but minimum 0.5
    t_review = max(0.5, t_alert - 0.15)
    
    # Verify thresholds
    decision = np.where(p_val >= t_alert, "ALERT",
               np.where(p_val >= t_review, "REVIEW", "PASS"))
    y_pred_soc = np.where(decision == "PASS", 0, 1)
    cm = confusion_matrix(y_val, y_pred_soc, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    actual_fpr = fp / n_negatives if n_negatives > 0 else 0.0
    
    return {
        "alert": float(t_alert),
        "review": float(t_review),
        "fpr_achieved": float(actual_fpr),
        "target_fpr": target_fpr,
    }


def calibrate_model(model_path: str, X_calib: pd.DataFrame, y_calib: pd.Series, out_path: str):
    base_model = joblib.load(model_path)
    if not hasattr(base_model, "predict_proba"):
        raise TypeError(f"Model at {model_path} does not support predict_proba for calibration.")

    # Use FrozenEstimator for sklearn 1.6+ compatibility (replaces deprecated cv='prefit')
    try:
        from sklearn.frozen import FrozenEstimator
        frozen_model = FrozenEstimator(base_model)
        calibrator = CalibratedClassifierCV(
            estimator=frozen_model,
            method="sigmoid",
        )
    except ImportError:
        # Fallback for older sklearn versions
        try:
            calibrator = CalibratedClassifierCV(
                estimator=base_model,
                method="sigmoid",
                cv="prefit",
            )
        except TypeError:
            calibrator = CalibratedClassifierCV(
                base_estimator=base_model,
                method="sigmoid",
                cv="prefit",
            )
    calibrator.fit(X_calib, y_calib)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, out_path)
    print(f"Saved calibrated model to {out_path}")
    
    return calibrator


def main():
    # GUARDRAIL: Prevent data leakage by ensuring we never use test data or full clean CSV for calibration
    train_csv = "data/dynamic_train.csv"
    val_csv = "data/dynamic_val.csv"
    
    # Check that split files exist (created by Stage 0)
    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"Missing {train_csv}. Run Stage 0 (make_splits.py) first to create train/val/test splits."
        )
    if not os.path.exists(val_csv):
        raise FileNotFoundError(
            f"Missing {val_csv}. Run Stage 0 (make_splits.py) first to create train/val/test splits."
        )
    
    # GUARDRAIL: Explicitly prevent loading test data or full clean CSV
    forbidden_paths = ["data/dynamic_test.csv", "data/dynamic_clean.csv"]
    env_csv = os.getenv("DYNAMIC_CSV")
    if env_csv and env_csv in forbidden_paths:
        raise ValueError(
            f"DATA LEAKAGE PREVENTION: Cannot use {env_csv} for calibration. "
            f"Calibration must use only train/val data. Use train_csv={train_csv} and val_csv={val_csv}."
        )
    
    # Load feature columns from saved pkl (matches training)
    feature_cols_path = "models/dynamic_feature_cols.pkl"
    if not os.path.exists(feature_cols_path):
        raise FileNotFoundError(
            f"Missing {feature_cols_path}. Train models first or run export_pipeline_artifacts.py"
        )
    
    feature_cols = load_feature_cols(feature_cols_path)
    
    # Load train and val datasets (NEVER test)
    print(f"Loading calibration data from train/val splits (excluding test set)...")
    X_train, y_train = load_dynamic_dataset(train_csv, feature_cols)
    X_val, y_val = load_dynamic_dataset(val_csv, feature_cols)
    
    # Use validation set for calibration (as done in train_dynamic_models.py line 129)
    # This matches the training pipeline behavior
    X_calib = X_val
    y_calib = y_val
    
    print(f"Loaded calibration dataset: rows={len(X_calib)}, features={len(feature_cols)}")
    print(f"  - Train CSV: {train_csv} ({len(X_train)} rows, not used for calibration)")
    print(f"  - Val CSV: {val_csv} ({len(X_calib)} rows, used for calibration)")
    print(f"  - Test CSV: EXCLUDED (data leakage prevention)")
    print(f"Feature columns: {feature_cols[:5]}... ({len(feature_cols)} total)")
    print("Calibration label distribution:", y_calib.value_counts(normalize=True).to_dict())

    # Load imputer for feature transformation (required for consistent preprocessing)
    imputer_path = "models/dynamic_imputer.pkl"
    if not os.path.exists(imputer_path):
        raise FileNotFoundError(f"Missing {imputer_path}. Train models first.")
    imputer = joblib.load(imputer_path)
    
    # Transform validation features using imputer (matches training preprocessing)
    X_calib_imp = imputer.transform(X_calib)
    X_calib_imp = pd.DataFrame(X_calib_imp, columns=feature_cols)
    
    # Calibrate models and get probabilities for threshold selection
    print("\n=== Calibrating Models ===")
    ada_calibrator = calibrate_model(
        model_path="models/ada_dynamic.pkl",
        X_calib=X_calib_imp,
        y_calib=y_calib,
        out_path="models/ada_dynamic_calibrated.pkl",
    )
    gb_calibrator = calibrate_model(
        model_path="models/gb_dynamic.pkl",
        X_calib=X_calib_imp,
        y_calib=y_calib,
        out_path="models/gb_dynamic_calibrated.pkl",
    )
    
    # Get probabilities on validation set for diagnostics and threshold selection
    print("\n=== Analyzing Probabilities ===")
    p_ada = ada_calibrator.predict_proba(X_calib_imp)[:, 1]
    p_gb = gb_calibrator.predict_proba(X_calib_imp)[:, 1]
    
    # Use GB model probabilities for threshold selection (primary model)
    ada_stats = analyze_probabilities(p_ada, y_calib, "ada_dynamic")
    gb_stats = analyze_probabilities(p_gb, y_calib, "gb_dynamic")
    
    print(f"\nAdaBoost probability stats:")
    print(f"  Min: {ada_stats['prob_stats']['min']:.4f}, Mean: {ada_stats['prob_stats']['mean']:.4f}, Max: {ada_stats['prob_stats']['max']:.4f}")
    print(f"  Below 0.5: {ada_stats['below_thresholds']['below_0.5']:.1f}%, Below 0.7: {ada_stats['below_thresholds']['below_0.7']:.1f}%")
    
    print(f"\nGradientBoosting probability stats:")
    print(f"  Min: {gb_stats['prob_stats']['min']:.4f}, Mean: {gb_stats['prob_stats']['mean']:.4f}, Max: {gb_stats['prob_stats']['max']:.4f}")
    print(f"  Below 0.5: {gb_stats['below_thresholds']['below_0.5']:.1f}%, Below 0.7: {gb_stats['below_thresholds']['below_0.7']:.1f}%")
    
    print(f"\nTop 10 most common probabilities (GB):")
    for prob_val, count in list(gb_stats['top_10_probabilities'].items())[:10]:
        print(f"  {prob_val}: {count} samples")
    
    # Select thresholds based on FPR cap (using GB model as primary)
    print("\n=== Selecting Thresholds (FPR-capped) ===")
    thresholds_gb = select_thresholds_fpr(y_calib, p_gb, target_fpr=0.05)
    print(f"Selected thresholds (GB model): alert={thresholds_gb['alert']:.3f}, review={thresholds_gb['review']:.3f}")
    print(f"  Target FPR cap: ≤{thresholds_gb['target_fpr']:.1%}, Achieved validation FPR: {thresholds_gb['fpr_achieved']:.1%}")
    
    # Save probability summary
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    proba_summary = {
        "ada": ada_stats,
        "gb": gb_stats,
        "thresholds_selected": thresholds_gb,
    }
    with open("outputs/reports/dynamic_proba_summary.json", "w", encoding="utf-8") as f:
        json.dump(proba_summary, f, indent=2)
    print(f"\nSaved probability summary to outputs/reports/dynamic_proba_summary.json")
    
    # Update thresholds.json (preserve static thresholds, update dynamic)
    thresholds_path = "models/thresholds.json"
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
    else:
        thresholds = {"static": {"alert": 0.80, "review": 0.55}, "dynamic": {}}
    
    # Update dynamic thresholds only
    thresholds["dynamic"] = {
        "alert": thresholds_gb["alert"],
        "review": thresholds_gb["review"],
    }
    
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Updated thresholds.json: dynamic thresholds set to alert={thresholds_gb['alert']:.3f}, review={thresholds_gb['review']:.3f}")


if __name__ == "__main__":
    main()
