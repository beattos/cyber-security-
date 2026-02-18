# Stage 5b Implementation Summary

## Overview

Added a TEST-ONLY version of Stage 5 that compares static vs dynamic predictions on the test split only.

## Files Created/Modified

### 1. New Script: `scripts/compare_static_dynamic_by_sample_test.py`

**Purpose:** Compare static vs dynamic predictions on test split only, merging on `sample_id`.

**Key Features:**
- Loads `data/static_test.csv` and `data/dynamic_test.csv`
- Merges on `sample_id` (inner join)
- Uses same logic/metrics as Stage 5:
  - Both correct
  - Both wrong
  - Static ok, Dynamic wrong
  - Static wrong, Dynamic ok
  - Prediction agreement (static_pred == dynamic_pred)
- Saves to `outputs/ablation/static_vs_dynamic_by_sample_test.csv`
- Prints summary with "TEST ONLY" in header

**Guards:**
- Checks for `sample_id` column in both test CSVs
- Validates merge produces >0 samples
- Verifies feature columns match models

### 2. Updated: `run_full_demo.sh`

**Change:** Added Stage 5b call immediately after Stage 5:

```bash
# --- Stage 5b: Per-sample static vs dynamic comparison (TEST ONLY, by sample_id) ---
echo ""
echo ">>> STAGE 5b: PER-SAMPLE STATIC vs DYNAMIC COMPARISON (TEST ONLY, by sample_id)"
echo "--------------------------------------------------------------------------------"
python scripts/compare_static_dynamic_by_sample_test.py \
  --static_test data/static_test.csv \
  --dynamic_test data/dynamic_test.csv \
  --output_dir outputs/ablation \
  --head 15
```

## Differences from Stage 5

| Aspect | Stage 5 | Stage 5b |
|--------|---------|----------|
| **Input** | `static_clean_with_id.csv`<br>`dynamic_clean_with_id.csv` | `static_test.csv`<br>`dynamic_test.csv` |
| **Dataset** | Full dataset | Test split only |
| **Merge** | Already has sample_id | Merges on sample_id (inner join) |
| **Output** | `static_vs_dynamic_by_sample.csv` | `static_vs_dynamic_by_sample_test.csv` |
| **Header** | "PER-SAMPLE STATIC vs DYNAMIC COMPARISON" | "PER-SAMPLE STATIC vs DYNAMIC COMPARISON (TEST ONLY)" |

## Expected Output

When running `./run_full_demo.sh`, Stage 5b should show:

```
>>> STAGE 5b: PER-SAMPLE STATIC vs DYNAMIC COMPARISON (TEST ONLY, by sample_id)
--------------------------------------------------------------------------------

Merged test datasets on sample_id: 533 samples

================================================================================
PER-SAMPLE STATIC vs DYNAMIC COMPARISON (TEST ONLY, by sample_id)
================================================================================

Samples: 533
  Both correct:     XXX (XX.X%)
  Both wrong:       XXX (XX.X%)
  Static ok, Dynamic wrong: XXX
  Static wrong, Dynamic ok: XXX
  Prediction agreement (static_pred == dynamic_pred): XXX (XX.X%)

Saved: outputs/ablation/static_vs_dynamic_by_sample_test.csv

Comparison table (first 15 rows):
...
```

## Validation

After running `./run_full_demo.sh`:

```bash
# Verify output file exists
ls -lh outputs/ablation/static_vs_dynamic_by_sample_test.csv

# Verify it contains test samples (should be ~533 rows)
wc -l outputs/ablation/static_vs_dynamic_by_sample_test.csv

# Verify sample_id column exists
head -1 outputs/ablation/static_vs_dynamic_by_sample_test.csv | grep sample_id
```

## Constraints Met

✅ **No changes to Stage 5:** Original Stage 5 behavior/output unchanged  
✅ **Test-only evaluation:** No training/calibration/threshold selection  
✅ **Same logic/metrics:** Reuses Stage 5 comparison logic  
✅ **Correct output path:** `outputs/ablation/static_vs_dynamic_by_sample_test.csv`  
✅ **TEST ONLY header:** Summary block clearly marked  
✅ **sample_id guard:** Raises clear error if missing  
✅ **Sample count:** Prints number of samples after merge  

## Implementation Details

**Merge Strategy:**
- Inner join on `sample_id` ensures only samples present in both test splits are compared
- Handles label columns: uses `label_static` if suffixes exist, otherwise `label`
- Validates label consistency (warns if mismatches found)

**Feature Handling:**
- Handles column suffixes from merge (`_static`, `_dynamic`)
- Validates all required features present after merge
- Uses same imputation/preprocessing as Stage 5

**Model Loading:**
- Prefers calibrated models (`*_calibrated.pkl`)
- Falls back to base models if calibrated not available
- Same threshold loading logic as Stage 5
