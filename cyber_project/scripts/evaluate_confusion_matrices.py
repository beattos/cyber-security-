# scripts/evaluate_confusion_matrices.py
import json
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def load_thresholds(path="models/thresholds.json"):
    if not os.path.exists(path):
        return {
            "static": {"alert": 0.80, "review": 0.55},
            "dynamic": {"alert": 0.80, "review": 0.55},
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decision_from_thresholds(p, t_alert, t_review):
    if p >= t_alert:
        return "ALERT"
    if p >= t_review:
        return "REVIEW"
    return "PASS"


def binary_from_decision(decision):
    # SOC mapping: ALERT+REVIEW => malware(1), PASS => benign(0)
    return 0 if decision == "PASS" else 1


def print_cm_block(title, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{title}")
    print("Format: [[TN, FP], [FN, TP]]")
    print(cm)
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
    print(classification_report(y_true, y_pred, labels=[0, 1], digits=3, zero_division=0))


def main():
    results_path = "outputs/stream_results.csv"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing {results_path}. Run the stream demo first.")

    df = pd.read_csv(results_path)

    required = {"source", "label", "p_malware", "decision"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {results_path}: {missing}")

    df = df.dropna(subset=["label", "p_malware", "decision"]).copy()
    df["label"] = df["label"].astype(int)
    df["p_malware"] = pd.to_numeric(df["p_malware"], errors="coerce")
    df = df.dropna(subset=["p_malware"])

    thresholds = load_thresholds("models/thresholds.json")

    for src in ["static", "dynamic"]:
        sub = df[df["source"] == src].copy()
        if sub.empty:
            print(f"\nNo rows for source={src}")
            continue

        print("\n" + "=" * 60)
        print(f"{src.upper()} CONFUSION MATRICES (n={len(sub)})")
        print("=" * 60)

        y_true = sub["label"].to_numpy()

        # A) As-run decision (what pipeline logged)
        y_pred_asrun = sub["decision"].apply(binary_from_decision).to_numpy()
        print_cm_block(f"{src.upper()} | A) As-run decision (ALERT/REVIEW=1, PASS=0)", y_true, y_pred_asrun)

        # Extra: triage distribution
        triage_counts = sub["decision"].value_counts().to_dict()
        print(f"\n{src.upper()} triage distribution (as-run): {triage_counts}")


if __name__ == "__main__":
    main()
