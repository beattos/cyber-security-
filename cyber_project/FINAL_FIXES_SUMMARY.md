# Final Pipeline Hardening - Last-Mile Fixes

**Date:** 2026-02-12  
**Commit:** `fix(stream): remove duplicate banner + enforce thresholds.json`

---

## What Was Fixed

### 1. Removed Duplicate Stream Banner ✅

**File:** `pipeline/consumer.py`

**Issue:** Two "STREAM DEMO START" banners were printed:
- First banner (correct): `run_stream_demo.py` with thresholds from `thresholds.json`
- Second banner (misleading): `consumer.py` with hardcoded defaults (0.80/0.55)

**Fix:** Removed duplicate banner from `consumer.py`. Single banner now printed by `run_stream_demo.py` with correct thresholds.

**Change:**
```diff
- print("\n=== STREAM DEMO START ===")
- print(f"Static model : {static_model_path}")
- print(f"Dynamic model: {dynamic_model_path}")
- print(f"Thresholds   : ALERT>={t_alert:.2f}, REVIEW>={t_review:.2f}")
- print(f"Features     : static={len(static_cols)}, dynamic={len(dynamic_cols)}")
- print("=========================\n")
+ # Banner is printed by run_stream_demo.py with correct thresholds from thresholds.json
+ # This function uses thresholds_by_source if provided, else falls back to t_alert/t_review defaults
```

### 2. Precise FPR Wording in Stage 2 ✅

**File:** `scripts/calibrate_dynamic.py`

**Issue:** Log wording implied exact FPR match ("Target FPR: 5.0%"), but it's actually a cap (≤5%).

**Fix:** Updated wording to accurately reflect FPR cap and achieved value.

**Change:**
```diff
- print("\n=== Selecting Thresholds (FPR-based) ===")
- print(f"Selected thresholds (GB model): alert={thresholds_gb['alert']:.3f}, review={thresholds_gb['review']:.3f}")
- print(f"  Target FPR: {thresholds_gb['target_fpr']:.1%}, Achieved FPR: {thresholds_gb['fpr_achieved']:.1%}")
+ print("\n=== Selecting Thresholds (FPR-capped) ===")
+ print(f"Selected thresholds (GB model): alert={thresholds_gb['alert']:.3f}, review={thresholds_gb['review']:.3f}")
+ print(f"  Target FPR cap: ≤{thresholds_gb['target_fpr']:.1%}, Achieved validation FPR: {thresholds_gb['fpr_achieved']:.1%}")
```

### 3. Confirmed Single Source of Truth ✅

**Verified:** Stream demo uses `thresholds.json` as single source of truth:
- `run_stream_demo.py` loads via `get_thresholds()` → `thresholds.json`
- Passes to `consumer.py` via `thresholds_by_source` parameter
- `consumer.py` uses `thresholds_by_source` if available (always in normal flow)
- Fallback defaults only used if `thresholds.json` missing (edge case)

---

## Why It Matters

**Prevents Ambiguity:**
- Single banner eliminates confusion about which thresholds are actually used
- Accurate FPR wording prevents overclaiming threshold selection precision
- Clear documentation ensures reviewers understand threshold source

**Maintains Consistency:**
- All thresholds come from `thresholds.json` (single source of truth)
- No hardcoded overrides in normal execution path
- Fallback defaults clearly documented as edge case only

---

## Before vs After

### Before:

```
=== STREAM DEMO START ===
Static model : models/gb_static.pkl
Dynamic model: models/gb_dynamic_calibrated.pkl
Thresholds   : static(ALERT>=0.70, REVIEW>=0.50) | dynamic(ALERT>=0.95, REVIEW>=0.80)
=========================

=== STREAM DEMO START ===  # DUPLICATE
Static model : models/gb_static.pkl
Dynamic model: models/gb_dynamic_calibrated.pkl
Thresholds   : ALERT>=0.80, REVIEW>=0.55  # MISLEADING
Features     : static=XX, dynamic=XX
=========================
```

### After:

```
=== STREAM DEMO START ===
Static model : models/gb_static.pkl
Dynamic model: models/gb_dynamic_calibrated.pkl
Thresholds   : static(ALERT>=0.70, REVIEW>=0.50) | dynamic(ALERT>=0.95, REVIEW>=0.80)
=========================
```

**Stage 2 Log:**

**Before:**
```
=== Selecting Thresholds (FPR-based) ===
Selected thresholds (GB model): alert=0.950, review=0.800
  Target FPR: 5.0%, Achieved FPR: 0.0%
```

**After:**
```
=== Selecting Thresholds (FPR-capped) ===
Selected thresholds (GB model): alert=0.950, review=0.800
  Target FPR cap: ≤5.0%, Achieved validation FPR: 0.0%
```

---

## Confirmation Steps

```bash
# Step 1: Verify single banner
grep -R "STREAM DEMO START" -n .
# Expected: Only one occurrence in pipeline/run_stream_demo.py

# Step 2: Run pipeline
./run_full_demo.sh

# Step 3: Verify Stage 3 shows single banner with correct thresholds
# Expected output:
# === STREAM DEMO START ===
# Static model : models/gb_static.pkl
# Dynamic model: models/gb_dynamic_calibrated.pkl
# Thresholds   : static(ALERT>=0.70, REVIEW>=0.50) | dynamic(ALERT>=0.95, REVIEW>=0.80)
# =========================

# Step 4: Verify thresholds.json matches banner
cat models/thresholds.json
# Expected: dynamic thresholds match banner (e.g., alert=0.95, review=0.80)

# Step 5: Verify Stage 2 log wording
# Expected: "Target FPR cap: ≤5.0%" not "Target FPR: 5.0%"
```

---

## Files Changed

1. `pipeline/consumer.py` - Removed duplicate banner (5 lines removed)
2. `scripts/calibrate_dynamic.py` - Updated FPR wording (2 lines changed)
3. `MINOR_FIXES_SUMMARY.md` - Added final fixes section

**Total Changes:** ~7 lines modified, minimal and safe

---

## Summary

✅ **Duplicate banner removed:** Single source of truth for stream demo output  
✅ **FPR wording precise:** Accurately reflects cap (≤5%) not exact match  
✅ **Thresholds enforced:** `thresholds.json` is single source, no ambiguity  
✅ **Minimal changes:** Only 7 lines modified, no behavior changes  

**Confidence:** 100% (removes ambiguity, no functional changes)
