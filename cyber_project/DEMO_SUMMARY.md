# Full Pipeline Demo – What You See at Each Stage

This document describes the **end-to-end pipeline demo** and what appears at each stage when you run `./run_full_demo.sh` (or the equivalent commands below).

---

## How to Run

**Single command (full run):**
```bash
cd /path/to/cyber_project
./run_full_demo.sh
```

**Or run stages manually** (from project root):
```bash
# 0. Splits
python scripts/make_splits.py --static_csv data/static_clean.csv --dynamic_csv data/dynamic_clean.csv

# 1. Training
python scripts/train_models.py
python scripts/train_dynamic_models.py

# 2. Calibration (optional)
python scripts/calibrate_dynamic.py

# 3. Stream demo
python -m pipeline.run_stream_demo \
  --static_csv data/static_clean.csv --dynamic_csv data/dynamic_clean.csv \
  --static_model models/gb_static_calibrated.pkl --dynamic_model models/gb_dynamic_calibrated.pkl \
  --max_events 400

# 4a. Threshold-based evaluation
python scripts/evaluate_confusion_matrices.py

# 4b. Threshold-free evaluation
python scripts/evaluate_threshold_free_demo.py --output_dir outputs/ablation

# 5. Per-sample comparison (ensure _with_id CSVs exist first; create_sample_id if needed)
python scripts/run_per_sample_comparison.py --head 15
```

---

## Stage 0: Data / Splits

**Command:** `scripts/make_splits.py`

**What you see:**
- Console output showing **static** and **dynamic** split sizes (train / val / test).
- Files written: `data/static_train.csv`, `data/static_val.csv`, `data/static_test.csv`, and the same for `dynamic_*`.

**Purpose:** Load static and dynamic clean datasets and create 70/15/15 train/val/test splits used for training and evaluation.

---

## Stage 1: Training (AdaBoost + GradientBoosting)

**Commands:** `scripts/train_models.py`, then `scripts/train_dynamic_models.py`

**What you see:**
- **train_models.py:** Trains **GradientBoosting** for both static and dynamic data; prints saved paths, `models/thresholds.json`, and static/dynamic thresholds.
- **train_dynamic_models.py:** Trains **AdaBoost** and **GradientBoosting** for dynamic data; prints thresholds and test accuracy for both.

**Artifacts:**
- `models/gb_static.pkl`, `models/gb_static_calibrated.pkl`
- `models/gb_dynamic.pkl`, `models/gb_dynamic_calibrated.pkl`
- `models/ada_dynamic.pkl`, `models/ada_dynamic_calibrated.pkl`
- `models/static_imputer.pkl`, `models/static_feature_cols.pkl`
- `models/dynamic_imputer.pkl`, `models/dynamic_feature_cols.pkl`
- `models/thresholds.json`
- `outputs/reports/train_val_test_report.json`, `outputs/reports/dynamic_models_report.json`

---

## Stage 2: Calibration

**Command:** `scripts/calibrate_dynamic.py`

**What you see:**
- Message that the dynamic dataset was loaded for calibration (rows, features, label distribution).
- “Saved calibrated model to …” for Ada and GB dynamic models.

**Purpose:** Re-calibrate dynamic models on a held-out calibration set (optional; training already produces calibrated models).

---

## Stage 3: Stream Demo

**Command:** `python -m pipeline.run_stream_demo ...`

**What you see:**
- Header with static/dynamic model paths and thresholds.
- A line per event: `[PRODUCER→PIPELINE] event=… source=static|dynamic | [ROUTE] … | [INFERENCE] p_malware=… | [DECISION] ALERT|REVIEW|PASS | latency=…ms`
- **STREAM SUMMARY:** total events, ALERT/REVIEW/PASS counts per source, average latency, path to `outputs/stream_results.csv`.

**Purpose:** Simulate the full pipeline (producer → consumer) with real models and thresholds; produce the CSV used for threshold-based evaluation.

---

## Stage 4a: Threshold-Based Evaluation (Confusion Matrices)

**Command:** `scripts/evaluate_confusion_matrices.py`

**What you see:**
- For **static** and **dynamic**:
  - **A) As-run decision:** confusion matrix `[[TN, FP], [FN, TP]]`, TN/FP/FN/TP counts, and classification report (precision, recall, F1).
  - **B) Auto-policy from thresholds.json:** same format using thresholds from `models/thresholds.json`.
  - **C) Pure probability @0.50:** model-only binary predictions at 0.5.
- Triage distributions (as-run and auto-policy) for each source.

**Purpose:** Show threshold-based metrics and confusion matrices from the stream run and from re-applying thresholds.

---

## Stage 4b: Threshold-Free Evaluation (ROC-AUC, PR-AUC)

**Command:** `scripts/evaluate_threshold_free_demo.py`

**What you see:**
- For each model (static GB, dynamic AdaBoost, dynamic GB): test size, **ROC-AUC**, **PR-AUC**.
- A short results table: Model, ROC-AUC, PR-AUC, N.
- Path to `outputs/ablation/threshold_free_demo.json`.

**Purpose:** Report threshold-free metrics on the current test sets without retraining.

---

## Stage 5: Per-Sample Static vs Dynamic Comparison (sample_id)

**Commands:** Optionally `scripts/create_sample_id.py` if `*_with_id.csv` are missing; then `scripts/run_per_sample_comparison.py`.

**What you see:**
- **PER-SAMPLE STATIC vs DYNAMIC COMPARISON (by sample_id):**
  - Summary counts: both correct, both wrong, static ok / dynamic wrong, static wrong / dynamic ok, prediction agreement.
  - Path to `outputs/ablation/static_vs_dynamic_by_sample.csv`.
  - **Comparison table:** first N rows (e.g. 15) with `sample_id`, `label`, `static_pred`, `dynamic_pred`, probabilities/decisions if available, correctness and agreement.

**Purpose:** Use `sample_id` to compare static vs dynamic predictions per sample and show the comparison matrix.

---

## Stage 6: Final Summary

**What you see (from `run_full_demo.sh`):**
- “DEMO COMPLETE” with timestamp.
- List of main artifacts: `models/thresholds.json`, `outputs/stream_results.csv`, `outputs/ablation/static_vs_dynamic_by_sample.csv`, `outputs/ablation/threshold_free_demo.json`, and report JSONs.

---

## Summary Table

| Stage | Label | Main output |
|-------|--------|-------------|
| 0 | Data / Splits | train/val/test CSVs for static and dynamic |
| 1 | Training | GB static/dynamic, Ada/GB dynamic, thresholds, reports |
| 2 | Calibration | Re-calibrated dynamic models (optional) |
| 3 | Stream demo | Per-event pipeline log, `stream_results.csv` |
| 4a | Threshold-based | Confusion matrices and classification reports (as-run, auto-policy, @0.5) |
| 4b | Threshold-free | ROC-AUC, PR-AUC table and JSON |
| 5 | Per-sample comparison | Summary counts and comparison table by `sample_id` |
| 6 | Final | List of artifacts and paths |

---

## Video vs Script-Only

- **Video:** Record the terminal while running `./run_full_demo.sh`; the stages above are what the viewer sees in order.
- **Script-only:** Use the “Single command” or “run stages manually” commands above; `DEMO_SUMMARY.md` is the step-by-step explanation of what appears at each stage.
