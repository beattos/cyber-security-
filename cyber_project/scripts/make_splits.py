import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_shared_sample_id(static_df: pd.DataFrame, dynamic_df: pd.DataFrame, label_col: str = "label") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create shared sample_id for static and dynamic datasets.
    
    Strategy:
    1. If sample_id already exists in both datasets, use existing values
    2. If a common identifier column exists (e.g., sha256, filename), use that
    3. Otherwise, create sample_id from row index (validates 1:1 alignment)
    """
    # Check for existing sample_id
    if "sample_id" in static_df.columns and "sample_id" in dynamic_df.columns:
        print("sample_id already exists in both datasets, using existing values.")
        return static_df, dynamic_df
    
    # Check for common identifier columns
    identifier_candidates = [
        'sha256', 'md5', 'sha1', 'sha512',
        'filename', 'file_name', 'path', 'file_path',
        'sample_name', 'id', 'uuid', 'hash', 'file_hash'
    ]
    
    static_cols = set(static_df.columns)
    dynamic_cols = set(dynamic_df.columns)
    
    join_key = None
    for candidate in identifier_candidates:
        if candidate in static_cols and candidate in dynamic_cols:
            # Verify uniqueness
            if (static_df[candidate].nunique() == len(static_df) and 
                dynamic_df[candidate].nunique() == len(dynamic_df)):
                join_key = candidate
                print(f"Using existing identifier column '{candidate}' as sample_id")
                break
    
    if join_key:
        # Use existing identifier as sample_id
        static_df = static_df.copy()
        dynamic_df = dynamic_df.copy()
        static_df["sample_id"] = static_df[join_key]
        dynamic_df["sample_id"] = dynamic_df[join_key]
    else:
        # Fallback: Create sample_id from row index (requires explicit validation)
        # Step 1: Validate row counts match
        if len(static_df) != len(dynamic_df):
            raise ValueError(
                f"Static and Dynamic datasets have different row counts. "
                f"Cannot create aligned sample_id. "
                f"Static: {len(static_df)} rows, Dynamic: {len(dynamic_df)} rows."
            )
        
        # Step 2: Validate label alignment (≥99.9% agreement required)
        if label_col not in static_df.columns:
            raise ValueError(f"Missing '{label_col}' column in static dataset. Cannot validate alignment.")
        if label_col not in dynamic_df.columns:
            raise ValueError(f"Missing '{label_col}' column in dynamic dataset. Cannot validate alignment.")
        
        # Calculate label agreement
        static_labels = static_df[label_col].reset_index(drop=True).values
        dynamic_labels = dynamic_df[label_col].reset_index(drop=True).values
        agreement = (static_labels == dynamic_labels).mean()
        
        # Print diagnostics
        print(f"[sample_id fallback] Row count: {len(static_df)}")
        print(f"[sample_id fallback] Label agreement: {agreement*100:.4f}%")
        
        # Step 3: Fail fast if alignment is unsafe
        if agreement < 0.999:
            mismatches = (static_labels != dynamic_labels).sum()
            raise ValueError(
                f"Static and Dynamic datasets are not row-aligned. "
                f"Index-based sample_id creation is unsafe. "
                f"Label agreement: {agreement*100:.4f}% ({mismatches} mismatches out of {len(static_df)} rows). "
                f"Required: ≥99.9%. "
                f"Use a stable identifier (e.g., sha256 or filename) or fix dataset alignment."
            )
        
        # Step 4: Validation passed - create sample_id
        print("[sample_id fallback] Alignment validated. Creating sample_id from row index.")
        static_df = static_df.copy()
        dynamic_df = dynamic_df.copy()
        static_df.insert(0, "sample_id", range(len(static_df)))
        dynamic_df.insert(0, "sample_id", range(len(dynamic_df)))
        print(f"Created sample_id from row index (0 to {len(static_df)-1})")
    
    return static_df, dynamic_df


def split_df(df: pd.DataFrame, label_col: str, seed: int,
             train_ratio: float, val_ratio: float, test_ratio: float):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0")

    y = df[label_col].astype(int)

    df_train, df_temp = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=y
    )

    # split temp into val and test
    y_temp = df_temp[label_col].astype(int)
    val_share_of_temp = val_ratio / (val_ratio + test_ratio)

    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - val_share_of_temp),
        random_state=seed,
        stratify=y_temp
    )

    return df_train, df_val, df_test


def main():
    ap = argparse.ArgumentParser(description="Create 70/15/15 splits from clean CSVs with shared sample_id.")
    ap.add_argument("--static_csv", default="data/static_clean.csv")
    ap.add_argument("--dynamic_csv", default="data/dynamic_clean.csv")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)

    # Load both datasets
    print("Loading datasets...")
    static_df = pd.read_csv(args.static_csv)
    dynamic_df = pd.read_csv(args.dynamic_csv)
    
    if args.label_col not in static_df.columns:
        raise ValueError(f"static: missing '{args.label_col}' in {args.static_csv}")
    if args.label_col not in dynamic_df.columns:
        raise ValueError(f"dynamic: missing '{args.label_col}' in {args.dynamic_csv}")

    # Drop EDA-only column
    dynamic_df = dynamic_df.drop(columns=["total_activity"], errors="ignore")
    
    # Create shared sample_id BEFORE splitting
    print("\nCreating shared sample_id...")
    static_df, dynamic_df = create_shared_sample_id(static_df, dynamic_df, label_col=args.label_col)
    
    # Verify sample_id was created
    if "sample_id" not in static_df.columns or "sample_id" not in dynamic_df.columns:
        raise ValueError("Failed to create sample_id. Check dataset alignment.")
    
    print(f"sample_id created: {static_df['sample_id'].nunique()} unique values in static, "
          f"{dynamic_df['sample_id'].nunique()} unique values in dynamic")

    # Split both datasets (sample_id will be preserved)
    for name, df in [("static", static_df), ("dynamic", dynamic_df)]:
        df_train, df_val, df_test = split_df(
            df, label_col=args.label_col, seed=args.seed,
            train_ratio=args.train, val_ratio=args.val, test_ratio=args.test
        )

        # Verify sample_id is preserved in splits
        for split_name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
            if "sample_id" not in df_split.columns:
                raise ValueError(f"{name}_{split_name}: sample_id missing after split!")

        df_train.to_csv(f"data/{name}_train.csv", index=False)
        df_val.to_csv(f"data/{name}_val.csv", index=False)
        df_test.to_csv(f"data/{name}_test.csv", index=False)

        print(f"\n{name.upper()} split sizes:")
        print(f"  train: {len(df_train)} -> data/{name}_train.csv (sample_id: {df_train['sample_id'].nunique()} unique)")
        print(f"  val  : {len(df_val)} -> data/{name}_val.csv (sample_id: {df_val['sample_id'].nunique()} unique)")
        print(f"  test : {len(df_test)} -> data/{name}_test.csv (sample_id: {df_test['sample_id'].nunique()} unique)")


if __name__ == "__main__":
    main()
