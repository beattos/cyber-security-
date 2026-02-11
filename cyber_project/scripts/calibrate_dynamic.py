import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# Suppress cv="prefit" deprecation warning (sklearn 1.6+)
warnings.filterwarnings("ignore", category=UserWarning, message=".*cv='prefit'.*")


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


def calibrate_model(model_path: str, X_calib: pd.DataFrame, y_calib: pd.Series, out_path: str):
    base_model = joblib.load(model_path)
    if not hasattr(base_model, "predict_proba"):
        raise TypeError(f"Model at {model_path} does not support predict_proba for calibration.")

    # sklearn API changed: older versions used base_estimator=..., newer uses estimator=...
    # cv="prefit" is deprecated in 1.6+ but still works; warning is suppressed at module level
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


def main():
    # Respect DYNAMIC_CSV env var (default to F0 for backward compatibility)
    csv_path = os.getenv("DYNAMIC_CSV", "data/dynamic_clean.csv")
    
    # Load feature columns from saved pkl (matches training)
    feature_cols_path = "models/dynamic_feature_cols.pkl"
    if not os.path.exists(feature_cols_path):
        raise FileNotFoundError(
            f"Missing {feature_cols_path}. Train models first or run export_pipeline_artifacts.py"
        )
    
    feature_cols = load_feature_cols(feature_cols_path)
    X, y = load_dynamic_dataset(csv_path, feature_cols)

    print(f"Loaded dynamic dataset for calibration: rows={len(X)}, features={len(feature_cols)}")
    print(f"CSV path: {csv_path}")
    print(f"Feature columns: {feature_cols[:5]}... ({len(feature_cols)} total)")
    print("Label distribution:", y.value_counts(normalize=True).to_dict())

    # Hold out a calibration split to avoid re-using the full dataset
    _, X_calib, _, y_calib = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    calibrate_model(
        model_path="models/ada_dynamic.pkl",
        X_calib=X_calib,
        y_calib=y_calib,
        out_path="models/ada_dynamic_calibrated.pkl",
    )
    calibrate_model(
        model_path="models/gb_dynamic.pkl",
        X_calib=X_calib,
        y_calib=y_calib,
        out_path="models/gb_dynamic_calibrated.pkl",
    )


if __name__ == "__main__":
    main()
