import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# Suppress cv="prefit" deprecation warning (sklearn 1.6+)
warnings.filterwarnings("ignore", category=UserWarning, message=".*cv='prefit'.*")


def load_feature_order(path: str) -> list[str]:
    with open(path, "r") as f:
        return json.load(f)


def load_dynamic_dataset(csv_path: str, feature_order: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column for calibration.")

    labels = pd.to_numeric(df["label"], errors="coerce")
    valid_mask = labels.notna()
    if not valid_mask.all():
        dropped = (~valid_mask).sum()
        print(f"Dropping {dropped} rows with non-numeric labels.")
    df = df.loc[valid_mask].reset_index(drop=True)
    labels = labels.loc[valid_mask].astype(int).reset_index(drop=True)

    features = {}
    for feat in feature_order:
        col = pd.to_numeric(df.get(feat, 0.0), errors="coerce")
        col = col.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        features[feat] = col

    X = pd.DataFrame(features, columns=feature_order)
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
    feature_order = load_feature_order("artifacts/dynamic_feature_order.json")
    X, y = load_dynamic_dataset("data/dynamic_clean.csv", feature_order)

    print(f"Loaded dynamic dataset for calibration: rows={len(X)}, features={len(feature_order)}")
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
