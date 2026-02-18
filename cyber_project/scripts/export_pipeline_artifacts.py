"""
UTILITY SCRIPT - NOT PART OF MAIN PIPELINE

This script exports preprocessing artifacts (imputers, feature columns) for
compatibility with older workflows. It fits imputers on the full dataset
(dynamic_clean.csv) which includes test data.

WARNING: The main pipeline (run_full_demo.sh) uses imputers fitted during
training on train split only. This script is for legacy support only.

For production use, rely on artifacts saved by train_dynamic_models.py.
"""
import os
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

def main():
    static_train_csv = "data/static_clean.csv"
    dynamic_train_csv = "data/dynamic_clean.csv"
    label_col = "label"

    if not os.path.exists(static_train_csv):
        raise FileNotFoundError(f"Missing {static_train_csv}. Update the path in this script.")
    if not os.path.exists(dynamic_train_csv):
        raise FileNotFoundError(f"Missing {dynamic_train_csv}. Update the path in this script.")

    os.makedirs("models", exist_ok=True)

    # ---- STATIC ----
    static_train = pd.read_csv(static_train_csv)
    if label_col not in static_train.columns:
        raise ValueError(f"{static_train_csv} has no '{label_col}' column. Found columns: {list(static_train.columns)[:10]}...")

    # Exclude label and sample_id from features
    Xs = static_train.drop(columns=[label_col, "sample_id"], errors="ignore")
    static_feature_cols = list(Xs.columns)

    static_imputer = SimpleImputer(strategy="median")
    static_imputer.fit(Xs)

    joblib.dump(static_feature_cols, "models/static_feature_cols.pkl")
    joblib.dump(static_imputer, "models/static_imputer.pkl")

    # ---- DYNAMIC ----
    dynamic_train = pd.read_csv(dynamic_train_csv)

    # keep consistent with your notebook: remove EDA-only column if present
    dynamic_train = dynamic_train.drop(columns=["total_activity"], errors="ignore")

    if label_col not in dynamic_train.columns:
        raise ValueError(f"{dynamic_train_csv} has no '{label_col}' column. Found columns: {list(dynamic_train.columns)[:10]}...")

    # Exclude label and sample_id from features
    Xd = dynamic_train.drop(columns=[label_col, "sample_id"], errors="ignore")
    dynamic_feature_cols = list(Xd.columns)

    dynamic_imputer = SimpleImputer(strategy="median")
    dynamic_imputer.fit(Xd)

    joblib.dump(dynamic_feature_cols, "models/dynamic_feature_cols.pkl")
    joblib.dump(dynamic_imputer, "models/dynamic_imputer.pkl")

    print("Exported artifacts:")
    print(" - models/static_feature_cols.pkl")
    print(" - models/static_imputer.pkl")
    print(" - models/dynamic_feature_cols.pkl")
    print(" - models/dynamic_imputer.pkl")
    print(f"Static features: {len(static_feature_cols)} | Dynamic features: {len(dynamic_feature_cols)}")

if __name__ == "__main__":
    main()
