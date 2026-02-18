# Stage 2 Calibration Data Leakage Audit Report

**Date:** 2026-02-12  
**Auditor:** Code Audit Agent  
**Scope:** Stage 2 "CALIBRATION (dynamic models)" data leakage verification

---

## A) Findings (Paths + Code Snippets)

### 1. Location of Stage 2 Calibration Code

**File:** `scripts/calibrate_dynamic.py`  
**Entry Point:** `main()` function (lines 81-146)  
**Called From:** `run_full_demo.sh` line 48: `python scripts/calibrate_dynamic.py || true`

### 2. Data Leakage Issue Identified

**Problem Location:** `scripts/calibrate_dynamic.py` lines 84-108

**Before Fix (LEAKAGE):**
```python
# Line 84: Loads FULL dataset including test samples
csv_path = os.getenv("DYNAMIC_CSV", "data/dynamic_clean.csv")

# Line 94: Loads ALL 3550 rows (train + val + test)
X, y = load_dynamic_dataset(csv_path, feature_cols)

# Lines 102-108: Splits the FULL dataset, which includes test samples
_, X_calib, _, y_calib = train_test_split(
    X,  # Contains test data!
    y,  # Contains test labels!
    test_size=0.25,
    random_state=42,
    stratify=y,
)
```

**Evidence:**
- `data/dynamic_clean.csv` contains **3550 rows** (all data: train=2484 + val=533 + test=533)
- The script loads this full CSV and splits it again
- Test samples from `data/dynamic_test.csv` are included in `dynamic_clean.csv`
- When `train_test_split` runs on the full dataset, test samples can end up in `X_calib` = **DATA LEAKAGE**

**Split File Sizes (verified):**
- `data/dynamic_train.csv`: 2484 rows
- `data/dynamic_val.csv`: 533 rows  
- `data/dynamic_test.csv`: 533 rows
- `data/dynamic_clean.csv`: 3550 rows (sum of above)

### 3. Comparison with Training Pipeline

**Reference:** `scripts/train_dynamic_models.py` lines 116-129

The training pipeline correctly uses:
- Train set for model fitting (line 114)
- **Validation set ONLY for calibration** (line 129)
- Test set ONLY for evaluation (line 134)

```python
# train_dynamic_models.py line 129 - CORRECT approach
calibrator.fit(X_va, y_va)  # Uses validation set only
```

Stage 2 calibration was inconsistent with this approach.

---

## B) Leakage Verdict

**VERDICT: ✅ LEAKAGE CONFIRMED AND FIXED**

**Evidence:**
1. **Line 84:** `csv_path = os.getenv("DYNAMIC_CSV", "data/dynamic_clean.csv")` loads full dataset
2. **Line 94:** `X, y = load_dynamic_dataset(csv_path, feature_cols)` loads all 3550 rows including test samples
3. **Lines 102-108:** `train_test_split(X, y, ...)` splits data that includes test samples, allowing test data to leak into calibration set
4. **Mismatch:** Training uses `dynamic_train.csv` + `dynamic_val.csv` separately, but calibration used `dynamic_clean.csv` (full dataset)

**Impact:**
- Calibration may have been fitted on test samples, leading to overfitting and optimistic performance estimates
- Violates train/val/test separation established in Stage 0

---

## C) Patch (Diff-Style)

### Fixed Code: `scripts/calibrate_dynamic.py`

**Changes:**
1. Removed `train_test_split` import (no longer needed)
2. Replaced full CSV loading with train/val split loading
3. Added guardrails to prevent test data usage
4. Updated logging to show which CSVs are used

**Key Changes:**

```diff
- from sklearn.model_selection import train_test_split

def main():
-    # Respect DYNAMIC_CSV env var (default to F0 for backward compatibility)
-    csv_path = os.getenv("DYNAMIC_CSV", "data/dynamic_clean.csv")
+    # GUARDRAIL: Prevent data leakage by ensuring we never use test data or full clean CSV for calibration
+    train_csv = "data/dynamic_train.csv"
+    val_csv = "data/dynamic_val.csv"
+    
+    # Check that split files exist (created by Stage 0)
+    if not os.path.exists(train_csv):
+        raise FileNotFoundError(...)
+    if not os.path.exists(val_csv):
+        raise FileNotFoundError(...)
+    
+    # GUARDRAIL: Explicitly prevent loading test data or full clean CSV
+    forbidden_paths = ["data/dynamic_test.csv", "data/dynamic_clean.csv"]
+    env_csv = os.getenv("DYNAMIC_CSV")
+    if env_csv and env_csv in forbidden_paths:
+        raise ValueError("DATA LEAKAGE PREVENTION: Cannot use {env_csv} for calibration...")
    
-    feature_cols = load_feature_cols(feature_cols_path)
-    X, y = load_dynamic_dataset(csv_path, feature_cols)
+    feature_cols = load_feature_cols(feature_cols_path)
+    
+    # Load train and val datasets (NEVER test)
+    print(f"Loading calibration data from train/val splits (excluding test set)...")
+    X_train, y_train = load_dynamic_dataset(train_csv, feature_cols)
+    X_val, y_val = load_dynamic_dataset(val_csv, feature_cols)
+    
+    # Use validation set for calibration (as done in train_dynamic_models.py line 129)
+    # This matches the training pipeline behavior
+    X_calib = X_val
+    y_calib = y_val
    
-    print(f"Loaded dynamic dataset for calibration: rows={len(X)}, features={len(feature_cols)}")
-    print(f"CSV path: {csv_path}")
+    print(f"Loaded calibration dataset: rows={len(X_calib)}, features={len(feature_cols)}")
+    print(f"  - Train CSV: {train_csv} ({len(X_train)} rows, not used for calibration)")
+    print(f"  - Val CSV: {val_csv} ({len(X_calib)} rows, used for calibration)")
+    print(f"  - Test CSV: EXCLUDED (data leakage prevention)")
     print(f"Feature columns: {feature_cols[:5]}... ({len(feature_cols)} total)")
-    print("Label distribution:", y.value_counts(normalize=True).to_dict())
+    print("Calibration label distribution:", y_calib.value_counts(normalize=True).to_dict())
    
-    # Hold out a calibration split to avoid re-using the full dataset
-    _, X_calib, _, y_calib = train_test_split(
-        X,
-        y,
-        test_size=0.25,
-        random_state=42,
-        stratify=y,
-    )
```

**Result:**
- Calibration now uses **validation set only** (533 rows), matching `train_dynamic_models.py` behavior
- Test data is completely excluded
- Guardrails prevent accidental test data usage

---

## D) Guardrail Implementation

### Guardrails Added:

1. **File Existence Check** (lines 87-94):
   - Ensures `dynamic_train.csv` and `dynamic_val.csv` exist before proceeding
   - Fails fast if Stage 0 (splits) hasn't been run

2. **Forbidden Path Prevention** (lines 96-103):
   - Explicitly blocks `data/dynamic_test.csv` and `data/dynamic_clean.csv` from being used
   - Raises `ValueError` with clear error message if attempted via `DYNAMIC_CSV` env var

3. **Explicit CSV Usage** (lines 114-122):
   - Hardcodes `train_csv` and `val_csv` paths (no env var override for forbidden paths)
   - Uses validation set only (matches training pipeline)

4. **Clear Logging** (lines 124-129):
   - Logs which CSVs are loaded and their row counts
   - Explicitly states test CSV is excluded

### Assertion Example:

If code tries to load test data, it will fail with:
```
ValueError: DATA LEAKAGE PREVENTION: Cannot use data/dynamic_test.csv for calibration. 
Calibration must use only train/val data. Use train_csv=data/dynamic_train.csv and val_csv=data/dynamic_val.csv.
```

---

## E) How to Verify (Commands + Expected Key Log Lines)

### Verification Steps:

1. **Run Stage 2 calibration:**
```bash
cd /Users/beatos/HIT-ai-cybersecurity-labs/labs/githubPush/cyber-security-/cyber_project
python scripts/calibrate_dynamic.py
```

2. **Expected Output (AFTER FIX):**
```
Loading calibration data from train/val splits (excluding test set)...
Loaded calibration dataset: rows=533, features=11
  - Train CSV: data/dynamic_train.csv (2484 rows, not used for calibration)
  - Val CSV: data/dynamic_val.csv (533 rows, used for calibration)
  - Test CSV: EXCLUDED (data leakage prevention)
Feature columns: ['feature1', 'feature2', ...]... (11 total)
Calibration label distribution: {...}
Saved calibrated model to models/ada_dynamic_calibrated.pkl
Saved calibrated model to models/gb_dynamic_calibrated.pkl
```

**Key Verification Points:**
- ✅ Row count is **533** (val size), NOT 3550 (full dataset)
- ✅ Log explicitly states "Test CSV: EXCLUDED"
- ✅ Log shows train CSV loaded but not used for calibration
- ✅ Models saved to same paths as before (`ada_dynamic_calibrated.pkl`, `gb_dynamic_calibrated.pkl`)

3. **Test Guardrail (should fail):**
```bash
DYNAMIC_CSV=data/dynamic_test.csv python scripts/calibrate_dynamic.py
# Expected: ValueError with "DATA LEAKAGE PREVENTION" message
```

4. **Full Pipeline Test:**
```bash
./run_full_demo.sh
# Stage 2 should now show 533 rows instead of 3550
```

### Before vs After Comparison:

**BEFORE (LEAKAGE):**
```
Loaded dynamic dataset for calibration: rows=3550, features=11
CSV path: data/dynamic_clean.csv
```

**AFTER (FIXED):**
```
Loaded calibration dataset: rows=533, features=11
  - Train CSV: data/dynamic_train.csv (2484 rows, not used for calibration)
  - Val CSV: data/dynamic_val.csv (533 rows, used for calibration)
  - Test CSV: EXCLUDED (data leakage prevention)
```

---

## Summary

✅ **Issue Found:** Stage 2 calibration was loading `data/dynamic_clean.csv` (3550 rows including test data) and splitting it, causing test data leakage into calibration set.

✅ **Issue Fixed:** Calibration now uses only `data/dynamic_val.csv` (533 rows), matching the training pipeline behavior.

✅ **Guardrails Added:** Multiple checks prevent accidental test data usage.

✅ **Backward Compatible:** Models still saved to same paths; pipeline behavior unchanged except for data source.

---

**Files Modified:**
- `scripts/calibrate_dynamic.py` (lines 81-129)

**Files Unchanged:**
- `run_full_demo.sh` (no changes needed)
- Model output paths (same as before)
- All other pipeline stages (unaffected)
