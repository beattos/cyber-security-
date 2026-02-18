# Sample ID Fix Summary

## Problem

Stage 5b was failing because test splits (`data/static_test.csv`, `data/dynamic_test.csv`) did not include `sample_id` column needed for merging.

## Solution

Modified Stage 0 (`scripts/make_splits.py`) to create shared `sample_id` **before** splitting, ensuring all train/val/test splits include it.

## Changes Made

### 1. `scripts/make_splits.py` - Create sample_id before splitting

**Added:**
- `create_shared_sample_id()` function that:
  - Checks for existing identifier columns (sha256, filename, etc.)
  - If found and unique, uses as sample_id
  - Otherwise, creates sample_id from row index (assumes 1:1 alignment)
  - Validates datasets have same length if using row index

**Modified:**
- Main loop now loads both datasets first
- Creates shared sample_id before splitting
- Verifies sample_id is preserved in all splits
- Prints sample_id counts in split summary

**Key Logic:**
```python
# Create shared sample_id BEFORE splitting
static_df, dynamic_df = create_shared_sample_id(static_df, dynamic_df)

# Split both datasets (sample_id preserved)
for name, df in [("static", static_df), ("dynamic", dynamic_df)]:
    df_train, df_val, df_test = split_df(...)
    # sample_id automatically preserved in all splits
```

### 2. `scripts/train_dynamic_models.py` - Exclude sample_id from features

**Changed:**
```python
# Before:
feature_cols = [c for c in df_train.columns if c != label_col]

# After:
feature_cols = [c for c in df_train.columns if c not in (label_col, "sample_id")]
```

### 3. `scripts/train_models.py` - Exclude sample_id from features

**Changed:** Same as above - exclude `sample_id` from feature columns.

### 4. `scripts/compare_static_dynamic_by_sample_test.py` - Better error message

**Changed:** Error message now explains that Stage 0 must create sample_id:
```python
raise ValueError(
    f"Missing 'sample_id' column in {args.static_test}. "
    f"Stage 0 (make_splits.py) must create sample_id before splitting. "
    f"Re-run: python scripts/make_splits.py"
)
```

### 5. `scripts/export_pipeline_artifacts.py` - Exclude sample_id (utility script)

**Changed:** For consistency, also exclude `sample_id` from features in utility script.

## How It Works

1. **Stage 0 (`make_splits.py`):**
   - Loads `static_clean.csv` and `dynamic_clean.csv`
   - Creates shared `sample_id`:
     - If common identifier exists → use it
     - Otherwise → use row index (0..N-1)
   - Adds `sample_id` to both datasets
   - Splits both datasets (sample_id preserved)
   - Saves all splits with `sample_id` included

2. **Training (`train_*.py`):**
   - Loads train CSV (includes `sample_id`)
   - Extracts features: excludes `label` and `sample_id`
   - Trains models on features only
   - Saves feature columns (without `sample_id`)

3. **Stage 5b (`compare_static_dynamic_by_sample_test.py`):**
   - Loads `static_test.csv` and `dynamic_test.csv` (both have `sample_id`)
   - Merges on `sample_id` (inner join)
   - Compares predictions per sample

## Verification

After running `./run_full_demo.sh`:

```bash
# Verify Stage 0 creates sample_id
python scripts/make_splits.py
# Expected: "sample_id created: X unique values in static, X unique values in dynamic"

# Verify test splits have sample_id
head -1 data/static_test.csv | grep sample_id
head -1 data/dynamic_test.csv | grep sample_id
# Expected: sample_id column present

# Verify Stage 5b works
python scripts/compare_static_dynamic_by_sample_test.py
# Expected: Merges successfully, outputs CSV with 533 rows

# Verify training excludes sample_id
python scripts/train_dynamic_models.py --dynamic_train data/dynamic_train.csv ...
# Expected: feature_cols does not include sample_id
```

## Expected Output

**Stage 0:**
```
Creating shared sample_id...
Created sample_id from row index (0 to 3549)

STATIC split sizes:
  train: 2484 -> data/static_train.csv (sample_id: 2484 unique)
  val  : 533 -> data/static_val.csv (sample_id: 533 unique)
  test : 533 -> data/static_test.csv (sample_id: 533 unique)

DYNAMIC split sizes:
  train: 2484 -> data/dynamic_train.csv (sample_id: 2484 unique)
  val  : 533 -> data/dynamic_val.csv (sample_id: 533 unique)
  test : 533 -> data/dynamic_test.csv (sample_id: 533 unique)
```

**Stage 5b:**
```
Merged test datasets on sample_id: 533 samples

================================================================================
PER-SAMPLE STATIC vs DYNAMIC COMPARISON (TEST ONLY, by sample_id)
================================================================================

Samples: 533
  Both correct:     XXX (XX.X%)
  ...
Saved: outputs/ablation/static_vs_dynamic_by_sample_test.csv
```

## Files Modified

1. `scripts/make_splits.py` - Added sample_id creation before splitting
2. `scripts/train_dynamic_models.py` - Exclude sample_id from features
3. `scripts/train_models.py` - Exclude sample_id from features
4. `scripts/compare_static_dynamic_by_sample_test.py` - Better error message
5. `scripts/export_pipeline_artifacts.py` - Exclude sample_id (utility script)

## Safety

✅ **Backward Compatible:** If `sample_id` already exists, uses it  
✅ **No Data Leakage:** sample_id is metadata, not used in training  
✅ **Consistent:** All splits have same sample_id values  
✅ **Validated:** Checks dataset alignment before creating sample_id  
