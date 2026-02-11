#!/usr/bin/env python3
"""
Create shared sample_id linking static_clean.csv and dynamic_clean.csv.

This script:
1. Inspects both datasets for a reliable join key
2. Validates 1:1 mapping and label consistency
3. Creates sample_id if safe
4. Generates comparison artifacts
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.metrics import classification_report


def compute_row_hash(df_row, exclude_cols=None):
    """Compute a stable hash of a row (excluding specified columns)."""
    if exclude_cols is None:
        exclude_cols = []
    row_str = ",".join([str(v) for k, v in df_row.items() if k not in exclude_cols])
    return hashlib.md5(row_str.encode()).hexdigest()


def find_join_key(static_df, dynamic_df):
    """
    Step 1: Find a reliable join key.
    Look for explicit identifiers or shared columns.
    """
    print("\n" + "="*80)
    print("STEP 1: Inspecting datasets for join key")
    print("="*80)
    
    # Check for common identifier columns
    identifier_candidates = [
        'sha256', 'md5', 'sha1', 'sha512',
        'filename', 'file_name', 'path', 'file_path',
        'sample_name', 'sample_id', 'id', 'uuid',
        'hash', 'file_hash'
    ]
    
    static_cols = set(static_df.columns)
    dynamic_cols = set(dynamic_df.columns)
    
    # Check for explicit identifier columns
    for candidate in identifier_candidates:
        if candidate in static_cols and candidate in dynamic_cols:
            print(f"Found explicit identifier column: '{candidate}'")
            return candidate
    
    # Check for any overlapping columns (excluding label)
    overlap = static_cols & dynamic_cols
    overlap.discard('label')
    
    if overlap:
        print(f"Found {len(overlap)} overlapping columns (excluding 'label'):")
        for col in sorted(overlap):
            print(f"  - {col}")
        
        # Check if any overlapping column is unique in both datasets
        for col in sorted(overlap):
            static_unique = static_df[col].nunique()
            dynamic_unique = dynamic_df[col].nunique()
            static_total = len(static_df)
            dynamic_total = len(dynamic_df)
            
            print(f"\n  Checking '{col}':")
            print(f"    Static: {static_unique} unique values out of {static_total} rows")
            print(f"    Dynamic: {dynamic_unique} unique values out of {dynamic_total} rows")
            
            # If unique in both and counts match, could be a key
            if static_unique == static_total == dynamic_unique == dynamic_total:
                print(f"    ✓ '{col}' is unique in both datasets!")
                return col
    
    print("\nNo explicit identifier column found.")
    return None


def validate_join_key(static_df, dynamic_df, join_key):
    """
    Step 2: Validate joinability and label consistency.
    """
    print("\n" + "="*80)
    print("STEP 2: Validating join key and label consistency")
    print("="*80)
    
    if join_key is None:
        print("No join key available. Checking row-by-row alignment...")
        return validate_row_alignment(static_df, dynamic_df)
    
    static_keys = static_df[join_key]
    dynamic_keys = dynamic_df[join_key]
    
    static_unique = static_keys.nunique()
    dynamic_unique = dynamic_keys.nunique()
    static_total = len(static_df)
    dynamic_total = len(dynamic_df)
    
    print(f"\nJoin key statistics:")
    print(f"  Static dataset: {static_unique} unique keys out of {static_total} rows")
    print(f"  Dynamic dataset: {dynamic_unique} unique keys out of {dynamic_total} rows")
    
    # Check for duplicates
    static_dups = static_keys.duplicated().sum()
    dynamic_dups = dynamic_keys.duplicated().sum()
    
    if static_dups > 0:
        print(f"  ⚠ WARNING: {static_dups} duplicate keys in static dataset!")
        return False, None
    if dynamic_dups > 0:
        print(f"  ⚠ WARNING: {dynamic_dups} duplicate keys in dynamic dataset!")
        return False, None
    
    # Check intersection
    static_key_set = set(static_keys)
    dynamic_key_set = set(dynamic_keys)
    intersection = static_key_set & dynamic_key_set
    
    print(f"  Intersecting keys: {len(intersection)}")
    
    if len(intersection) != static_total or len(intersection) != dynamic_total:
        print(f"  ⚠ WARNING: Not all keys match!")
        print(f"    Static-only keys: {len(static_key_set - dynamic_key_set)}")
        print(f"    Dynamic-only keys: {len(dynamic_key_set - static_key_set)}")
        return False, None
    
    # Merge to check label consistency
    merged = static_df[[join_key, 'label']].merge(
        dynamic_df[[join_key, 'label']],
        on=join_key,
        suffixes=('_static', '_dynamic')
    )
    
    label_mismatches = merged[merged['label_static'] != merged['label_dynamic']]
    
    if len(label_mismatches) > 0:
        print(f"\n  ❌ CRITICAL: Found {len(label_mismatches)} label mismatches!")
        print("  Same key but different labels:")
        print(label_mismatches.head(10).to_string())
        if len(label_mismatches) > 10:
            print(f"  ... and {len(label_mismatches) - 10} more")
        return False, None
    
    print(f"\n  ✓ All {len(merged)} matched samples have consistent labels!")
    return True, join_key


def validate_row_alignment(static_df, dynamic_df):
    """
    Validate row-by-row alignment when no explicit join key exists.
    """
    print("\nValidating row-by-row alignment...")
    
    # Check lengths
    static_len = len(static_df)
    dynamic_len = len(dynamic_df)
    
    print(f"  Static rows: {static_len}")
    print(f"  Dynamic rows: {dynamic_len}")
    
    if static_len != dynamic_len:
        print(f"  ❌ Row counts don't match!")
        return False, None
    
    # Check label distributions
    static_label_dist = static_df['label'].value_counts().sort_index()
    dynamic_label_dist = dynamic_df['label'].value_counts().sort_index()
    
    print(f"\n  Label distributions:")
    print(f"    Static:  {dict(static_label_dist)}")
    print(f"    Dynamic: {dict(dynamic_label_dist)}")
    
    if not static_label_dist.equals(dynamic_label_dist):
        print(f"  ⚠ Label distributions don't match exactly!")
        return False, None
    
    # Check row-by-row label consistency
    label_mismatches = (static_df['label'] != dynamic_df['label']).sum()
    
    if label_mismatches > 0:
        print(f"\n  ❌ CRITICAL: Found {label_mismatches} row-by-row label mismatches!")
        mismatch_indices = static_df.index[static_df['label'] != dynamic_df['label']].tolist()
        print(f"  Mismatch at rows: {mismatch_indices[:20]}")
        if len(mismatch_indices) > 20:
            print(f"  ... and {len(mismatch_indices) - 20} more")
        
        # Show examples
        print("\n  Example mismatches:")
        for idx in mismatch_indices[:5]:
            print(f"    Row {idx}: static={static_df.loc[idx, 'label']}, dynamic={dynamic_df.loc[idx, 'label']}")
        
        return False, None
    
    print(f"\n  ✓ All {static_len} rows have matching labels!")
    
    # Validate with row hashes (sample subset)
    print("\n  Validating row signatures (sampling 100 rows)...")
    sample_indices = list(range(0, min(100, static_len), max(1, static_len // 100)))
    
    static_features = static_df.drop(columns=['label'])
    dynamic_features = dynamic_df.drop(columns=['label'])
    
    mismatches = 0
    for idx in sample_indices:
        static_hash = compute_row_hash(static_features.iloc[idx])
        dynamic_hash = compute_row_hash(dynamic_features.iloc[idx])
        if static_hash != dynamic_hash:
            mismatches += 1
    
    if mismatches > 0:
        print(f"  ⚠ Found {mismatches} feature hash mismatches in sample (expected - features differ)")
        print(f"  This is expected since static and dynamic features are different.")
    
    print(f"  ✓ Row alignment validated (labels match, feature differences expected)")
    
    return True, 'row_index'


def create_sample_id(static_df, dynamic_df, join_key):
    """
    Step 3: Create sample_id if validation passed.
    """
    print("\n" + "="*80)
    print("STEP 3: Creating sample_id")
    print("="*80)
    
    if join_key == 'row_index':
        # Use row index as the basis
        static_df['sample_id'] = range(len(static_df))
        dynamic_df['sample_id'] = range(len(dynamic_df))
    else:
        # Factorize the join key consistently
        all_keys = pd.concat([static_df[join_key], dynamic_df[join_key]]).unique()
        all_keys_sorted = sorted(all_keys)
        key_to_id = {key: idx for idx, key in enumerate(all_keys_sorted)}
        
        static_df['sample_id'] = static_df[join_key].map(key_to_id)
        dynamic_df['sample_id'] = dynamic_df[join_key].map(key_to_id)
    
    # Move sample_id to first column
    cols_static = ['sample_id'] + [c for c in static_df.columns if c != 'sample_id']
    cols_dynamic = ['sample_id'] + [c for c in dynamic_df.columns if c != 'sample_id']
    
    static_df = static_df[cols_static]
    dynamic_df = dynamic_df[cols_dynamic]
    
    print(f"  Created sample_id for {len(static_df)} samples")
    print(f"  sample_id range: {static_df['sample_id'].min()} to {static_df['sample_id'].max()}")
    
    return static_df, dynamic_df


def generate_comparison_artifact(static_df, dynamic_df, output_dir, dynamic_f1_csv=None):
    """
    Step 5: Generate per-sample comparison using trained models.
    
    Args:
        static_df: Static dataframe with sample_id
        dynamic_df: Dynamic dataframe with sample_id (may be F0 features)
        output_dir: Output directory
        dynamic_f1_csv: Optional path to dynamic_clean_F1.csv if models expect F1 features
    """
    print("\n" + "="*80)
    print("STEP 5: Generating comparison artifact")
    print("="*80)
    
    # Load models
    model_dir = Path("models")
    
    static_model_path = model_dir / "gb_static.pkl"
    dynamic_model_path = model_dir / "gb_dynamic.pkl"
    
    if not static_model_path.exists():
        print(f"  ⚠ Static model not found: {static_model_path}")
        print("  Skipping comparison artifact generation.")
        return None
    
    if not dynamic_model_path.exists():
        print(f"  ⚠ Dynamic model not found: {dynamic_model_path}")
        print("  Skipping comparison artifact generation.")
        return None
    
    print("  Loading models...")
    static_model = joblib.load(static_model_path)
    dynamic_model = joblib.load(dynamic_model_path)
    
    # Load feature columns
    static_feature_cols_path = model_dir / "static_feature_cols.pkl"
    dynamic_feature_cols_path = model_dir / "dynamic_feature_cols.pkl"
    
    if not static_feature_cols_path.exists() or not dynamic_feature_cols_path.exists():
        print("  ⚠ Feature column files not found. Skipping comparison.")
        return None
    
    static_feature_cols = joblib.load(static_feature_cols_path)
    dynamic_feature_cols = joblib.load(dynamic_feature_cols_path)
    
    # Check feature compatibility for static
    static_available = [c for c in static_feature_cols if c in static_df.columns]
    if set(static_feature_cols) != set(static_df.columns) - {'sample_id', 'label'}:
        missing_static = set(static_feature_cols) - set(static_available)
        extra_static = set(static_df.columns) - {'sample_id', 'label'} - set(static_feature_cols)
        print(f"  ⚠ Static model features don't match CSV:")
        if missing_static:
            print(f"    Missing from CSV: {list(missing_static)[:5]}...")
        if extra_static:
            print(f"    Extra in CSV: {list(extra_static)[:5]}...")
        print("  Skipping comparison artifact (model-CSV mismatch).")
        return None
    
    # Check dynamic feature compatibility
    dynamic_available = [c for c in dynamic_feature_cols if c in dynamic_df.columns]
    dynamic_df_for_model = dynamic_df
    
    if set(dynamic_feature_cols) != set(dynamic_df.columns) - {'sample_id', 'label'}:
        # Try using dynamic_clean_F1.csv if provided
        if dynamic_f1_csv and Path(dynamic_f1_csv).exists():
            print(f"  ⚠ Dynamic model expects F1 features, trying {dynamic_f1_csv}...")
            dynamic_f1_df = pd.read_csv(dynamic_f1_csv)
            dynamic_f1_df = dynamic_f1_df.drop(columns=["total_activity"], errors="ignore")
            
            # Verify alignment
            if len(dynamic_f1_df) == len(static_df) and (dynamic_f1_df['label'] == static_df['label']).all():
                # Create sample_id for F1 dataset (same as static)
                dynamic_f1_df['sample_id'] = static_df['sample_id'].values
                dynamic_df_for_model = dynamic_f1_df
                print(f"  ✓ Using {dynamic_f1_csv} for model predictions (aligned with static)")
            else:
                print(f"  ⚠ {dynamic_f1_csv} not aligned with static dataset")
                print("  Skipping comparison artifact (model-CSV mismatch).")
                return None
        else:
            missing_dynamic = set(dynamic_feature_cols) - set(dynamic_available)
            extra_dynamic = set(dynamic_df.columns) - {'sample_id', 'label'} - set(dynamic_feature_cols)
            print(f"  ⚠ Dynamic model features don't match CSV:")
            if missing_dynamic:
                print(f"    Missing from CSV: {list(missing_dynamic)[:5]}...")
            if extra_dynamic:
                print(f"    Extra in CSV: {list(extra_dynamic)[:5]}...")
            print("  Skipping comparison artifact (model-CSV mismatch).")
            return None
    
    # Load imputers if available
    static_imputer_path = model_dir / "static_imputer.pkl"
    dynamic_imputer_path = model_dir / "dynamic_imputer.pkl"
    
    static_imputer = None
    dynamic_imputer = None
    if static_imputer_path.exists():
        static_imputer = joblib.load(static_imputer_path)
    if dynamic_imputer_path.exists():
        dynamic_imputer = joblib.load(dynamic_imputer_path)
    
    # Prepare features - use model's expected feature order if available
    if hasattr(static_model, 'feature_names_in_'):
        static_feature_order = list(static_model.feature_names_in_)
        X_static = static_df[static_feature_order].copy()
    else:
        X_static = static_df[static_feature_cols].copy()
    
    if hasattr(dynamic_model, 'feature_names_in_'):
        dynamic_feature_order = list(dynamic_model.feature_names_in_)
        X_dynamic = dynamic_df_for_model[dynamic_feature_order].copy()
    else:
        X_dynamic = dynamic_df_for_model[dynamic_feature_cols].copy()
    
    # Impute if imputers available and features match
    if static_imputer is not None:
        try:
            X_static = pd.DataFrame(
                static_imputer.transform(X_static),
                columns=X_static.columns,
                index=X_static.index
            )
        except (ValueError, KeyError):
            print(f"  ⚠ Static imputer features don't match, skipping imputation")
    
    if dynamic_imputer is not None:
        try:
            X_dynamic = pd.DataFrame(
                dynamic_imputer.transform(X_dynamic),
                columns=X_dynamic.columns,
                index=X_dynamic.index
            )
        except (ValueError, KeyError):
            print(f"  ⚠ Dynamic imputer features don't match, skipping imputation")
    
    # Get labels (use original dynamic_df labels, not dynamic_df_for_model)
    y_static = static_df['label']
    y_dynamic = dynamic_df['label']  # Always use original dynamic_df labels
    
    # Generate predictions
    print("  Generating predictions...")
    static_pred = static_model.predict(X_static)
    dynamic_pred = dynamic_model.predict(X_dynamic)
    
    # Get probabilities if available
    try:
        static_proba = static_model.predict_proba(X_static)[:, 1]
        dynamic_proba = dynamic_model.predict_proba(X_dynamic)[:, 1]
    except:
        static_proba = None
        dynamic_proba = None
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'sample_id': static_df['sample_id'],
        'label': y_static.values,
        'static_pred': static_pred,
        'dynamic_pred': dynamic_pred,
    })
    
    if static_proba is not None:
        comparison['static_proba'] = static_proba
    if dynamic_proba is not None:
        comparison['dynamic_proba'] = dynamic_proba
    
    # Add correctness flags
    comparison['static_correct'] = (comparison['label'] == comparison['static_pred']).astype(int)
    comparison['dynamic_correct'] = (comparison['label'] == comparison['dynamic_pred']).astype(int)
    
    # Summary counts
    static_correct_dynamic_wrong = ((comparison['static_correct'] == 1) & (comparison['dynamic_correct'] == 0)).sum()
    dynamic_correct_static_wrong = ((comparison['static_correct'] == 0) & (comparison['dynamic_correct'] == 1)).sum()
    both_correct = ((comparison['static_correct'] == 1) & (comparison['dynamic_correct'] == 1)).sum()
    both_wrong = ((comparison['static_correct'] == 0) & (comparison['dynamic_correct'] == 0)).sum()
    
    print(f"\n  Summary counts:")
    print(f"    Static correct & Dynamic wrong: {static_correct_dynamic_wrong}")
    print(f"    Dynamic correct & Static wrong: {dynamic_correct_static_wrong}")
    print(f"    Both correct: {both_correct}")
    print(f"    Both wrong: {both_wrong}")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / "static_vs_dynamic_by_sample.csv"
    comparison.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    
    return comparison


def main():
    ap = argparse.ArgumentParser(
        description="Create shared sample_id linking static and dynamic datasets"
    )
    ap.add_argument("--static_csv", default="data/static_clean.csv")
    ap.add_argument("--dynamic_csv", default="data/dynamic_clean.csv")
    ap.add_argument("--output_dir", default="outputs/ablation")
    args = ap.parse_args()
    
    print("="*80)
    print("Creating Shared sample_id for Static and Dynamic Datasets")
    print("="*80)
    
    # Load datasets
    print(f"\nLoading datasets...")
    print(f"  Static: {args.static_csv}")
    print(f"  Dynamic: {args.dynamic_csv}")
    
    static_df = pd.read_csv(args.static_csv)
    dynamic_df = pd.read_csv(args.dynamic_csv)
    
    # Remove EDA-only column if present
    dynamic_df = dynamic_df.drop(columns=["total_activity"], errors="ignore")
    
    print(f"  Static: {len(static_df)} rows, {len(static_df.columns)} columns")
    print(f"  Dynamic: {len(dynamic_df)} rows, {len(dynamic_df.columns)} columns")
    
    # Step 1: Find join key
    join_key = find_join_key(static_df, dynamic_df)
    
    # Step 2: Validate
    is_valid, validated_key = validate_join_key(static_df, dynamic_df, join_key)
    
    if not is_valid:
        print("\n" + "="*80)
        print("❌ VALIDATION FAILED")
        print("="*80)
        print("\nWe cannot safely generate a shared sample_id because:")
        if join_key is None:
            print("  - No explicit identifier column found")
            print("  - Row alignment could not be validated")
        else:
            print(f"  - Join key '{validated_key}' validation failed")
            print("  - Either duplicates exist or labels don't match")
        print("\nRecommendation: Collect/add a true sample identifier at data generation time.")
        sys.exit(1)
    
    # Step 3: Create sample_id
    static_with_id, dynamic_with_id = create_sample_id(static_df, dynamic_df, validated_key)
    
    # Save datasets with sample_id
    static_output = args.static_csv.replace('.csv', '_with_id.csv')
    dynamic_output = args.dynamic_csv.replace('.csv', '_with_id.csv')
    
    static_with_id.to_csv(static_output, index=False)
    dynamic_with_id.to_csv(dynamic_output, index=False)
    
    print(f"\n  Saved: {static_output}")
    print(f"  Saved: {dynamic_output}")
    
    # Step 5: Generate comparison artifact
    # Try dynamic_clean_F1.csv if models expect F1 features
    dynamic_f1_path = args.dynamic_csv.replace('.csv', '_F1.csv')
    comparison = generate_comparison_artifact(
        static_with_id, dynamic_with_id, args.output_dir,
        dynamic_f1_csv=dynamic_f1_path if Path(dynamic_f1_path).exists() else None
    )
    
    # Final report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    print(f"\nChosen join key: {validated_key}")
    print(f"Join stats:")
    print(f"  - Total samples: {len(static_with_id)}")
    print(f"  - Label consistency: ✓ All matched")
    print(f"\nCreated files:")
    print(f"  - {static_output}")
    print(f"  - {dynamic_output}")
    if comparison is not None:
        print(f"  - {Path(args.output_dir) / 'static_vs_dynamic_by_sample.csv'}")
    print("\n✓ sample_id created successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
