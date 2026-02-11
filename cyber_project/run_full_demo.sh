#!/usr/bin/env bash
# Full end-to-end pipeline demo: data -> train -> calibrate -> stream -> evaluate -> comparison.
# Run from project root: ./run_full_demo.sh
# Or: bash run_full_demo.sh

set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTHONUNBUFFERED=1

echo ""
echo "================================================================================"
echo "  FULL PIPELINE DEMO - $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

# --- Stage 0: Data / Splits ---
echo ""
echo ">>> STAGE 0: DATA / SPLITS (load datasets, create train/val/test)"
echo "--------------------------------------------------------------------------------"
python scripts/make_splits.py \
  --static_csv data/static_clean.csv \
  --dynamic_csv data/dynamic_clean.csv \
  --label_col label

# --- Stage 1: Training (AdaBoost + GradientBoosting) ---
echo ""
echo ">>> STAGE 1: TRAINING (AdaBoost + GradientBoosting)"
echo "--------------------------------------------------------------------------------"
python scripts/train_models.py \
  --static_train data/static_train.csv \
  --static_val data/static_val.csv \
  --static_test data/static_test.csv \
  --dynamic_train data/dynamic_train.csv \
  --dynamic_val data/dynamic_val.csv \
  --dynamic_test data/dynamic_test.csv \
  --out_dir models

python scripts/train_dynamic_models.py \
  --dynamic_train data/dynamic_train.csv \
  --dynamic_val data/dynamic_val.csv \
  --dynamic_test data/dynamic_test.csv \
  --out_dir models

# --- Stage 2: Calibration (optional; training already calibrates) ---
echo ""
echo ">>> STAGE 2: CALIBRATION (dynamic models; training already calibrated)"
echo "--------------------------------------------------------------------------------"
python scripts/calibrate_dynamic.py || true

# --- Stage 3: Stream demo (producer -> consumer -> stream_results.csv) ---
echo ""
echo ">>> STAGE 3: STREAM DEMO (static + dynamic events -> pipeline -> stream_results.csv)"
echo "--------------------------------------------------------------------------------"
python -m pipeline.run_stream_demo \
  --static_csv data/static_clean.csv \
  --dynamic_csv data/dynamic_clean.csv \
  --static_model models/gb_static_calibrated.pkl \
  --dynamic_model models/gb_dynamic_calibrated.pkl \
  --max_events 400 \
  --sleep_ms 0 \
  --suppress_sklearn_pickle_warnings

# --- Stage 4: Evaluation (threshold-based + threshold-free) ---
echo ""
echo ">>> STAGE 4a: THRESHOLD-BASED EVALUATION (confusion matrices from stream_results)"
echo "--------------------------------------------------------------------------------"
python scripts/evaluate_confusion_matrices.py

echo ""
echo ">>> STAGE 4b: THRESHOLD-FREE EVALUATION (ROC-AUC, PR-AUC)"
echo "--------------------------------------------------------------------------------"
python scripts/evaluate_threshold_free_demo.py --output_dir outputs/ablation

# --- Stage 5: Per-sample static vs dynamic comparison (sample_id) ---
echo ""
echo ">>> STAGE 5: PER-SAMPLE STATIC vs DYNAMIC COMPARISON (by sample_id)"
echo "--------------------------------------------------------------------------------"
if [[ ! -f data/static_clean_with_id.csv || ! -f data/dynamic_clean_with_id.csv ]]; then
  echo "Creating sample_id and _with_id datasets..."
  python scripts/create_sample_id.py \
    --static_csv data/static_clean.csv \
    --dynamic_csv data/dynamic_clean.csv \
    --output_dir outputs/ablation
fi
python scripts/run_per_sample_comparison.py \
  --static_with_id data/static_clean_with_id.csv \
  --dynamic_with_id data/dynamic_clean_with_id.csv \
  --output_dir outputs/ablation \
  --head 15

# --- Stage 6: Final summary ---
echo ""
echo "================================================================================"
echo "  DEMO COMPLETE - $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
echo ""
echo "Artifacts:"
echo "  - models/thresholds.json"
echo "  - outputs/stream_results.csv"
echo "  - outputs/ablation/static_vs_dynamic_by_sample.csv"
echo "  - outputs/ablation/threshold_free_demo.json"
echo "  - outputs/reports/train_val_test_report.json"
echo "  - outputs/reports/dynamic_models_report.json"
echo ""
