# pipeline/consumer.py
import csv
import os
import time
import warnings
from queue import Queue
from typing import List

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:
    InconsistentVersionWarning = None


def _predict_p_malware(model, X_df: pd.DataFrame) -> float:
    """
    Returns p(malware) for a single-row DataFrame.
    Works best with predict_proba (esp. calibrated models), and falls back if needed.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_df)
        return float(proba[0, 1])

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X_df)[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    pred = int(model.predict(X_df)[0])
    return 1.0 if pred == 1 else 0.0


def _decide(p_malware: float, t_alert: float, t_review: float) -> str:
    if p_malware >= t_alert:
        return "ALERT"
    if p_malware >= t_review:
        return "REVIEW"
    return "PASS"


def consume(
    q: Queue,
    static_model_path: str,
    dynamic_model_path: str,
    out_csv: str = "outputs/stream_results.csv",
    t_alert: float = 0.80,
    t_review: float = 0.55,
    interactive_review: bool = False,
    print_every: int = 1,
    suppress_sklearn_pickle_warnings: bool = False,
) -> None:
    """
    Loads models + imputers + feature column lists, consumes events, prints per-event pipeline line,
    writes results CSV, and prints a summary.
    """
    # Optional: keep console clean (not required)
    if suppress_sklearn_pickle_warnings and InconsistentVersionWarning is not None:
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

    # Always suppress this noisy warning by doing proper DataFrame inference below,
    # but keep it in case another estimator emits it.
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names*",
        category=UserWarning,
    )

    # Load models
    static_model = joblib.load(static_model_path)
    dynamic_model = joblib.load(dynamic_model_path)

    # Load preprocessing artifacts
    static_imputer = joblib.load("models/static_imputer.pkl")
    dynamic_imputer = joblib.load("models/dynamic_imputer.pkl")
    static_cols = joblib.load("models/static_feature_cols.pkl")
    dynamic_cols = joblib.load("models/dynamic_feature_cols.pkl")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    print("\n=== STREAM DEMO START ===")
    print(f"Static model : {static_model_path}")
    print(f"Dynamic model: {dynamic_model_path}")
    print(f"Thresholds   : ALERT>={t_alert:.2f}, REVIEW>={t_review:.2f}")
    print(f"Features     : static={len(static_cols)}, dynamic={len(dynamic_cols)}")
    print("=========================\n")

    counters = {
        "static": {"ALERT": 0, "REVIEW": 0, "PASS": 0},
        "dynamic": {"ALERT": 0, "REVIEW": 0, "PASS": 0},
    }
    latencies_ms: List[float] = []

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["event_id", "source", "p_malware", "decision", "latency_ms", "label", "user_action"],
        )
        writer.writeheader()

        processed = 0

        while True:
            ev = q.get()
            if ev is None:
                break

            t0 = time.time()

            if ev.source == "static":
                # Build single-row DF with correct columns -> imputer -> DF again
                X_raw = pd.DataFrame([ev.features], columns=static_cols)
                X_imp = static_imputer.transform(X_raw)
                X_df = pd.DataFrame(X_imp, columns=static_cols)

                p = _predict_p_malware(static_model, X_df)
                routed = "static_model"
            else:
                X_raw = pd.DataFrame([ev.features], columns=dynamic_cols)
                X_imp = dynamic_imputer.transform(X_raw)
                X_df = pd.DataFrame(X_imp, columns=dynamic_cols)

                p = _predict_p_malware(dynamic_model, X_df)
                routed = "dynamic_model"

            decision = _decide(p, t_alert=t_alert, t_review=t_review)

            user_action = ""
            if interactive_review and decision == "REVIEW":
                print(
                    f"[USER ACTION] Event {ev.event_id} is REVIEW (p_malware={p:.3f}). "
                    f"Choose: [a]=alert [p]=pass [r]=keep review"
                )
                ans = input(">> ").strip().lower()
                if ans == "a":
                    decision = "ALERT"
                    user_action = "promoted_to_alert"
                elif ans == "p":
                    decision = "PASS"
                    user_action = "downgraded_to_pass"
                else:
                    user_action = "kept_review"

            t1 = time.time()
            latency = (t1 - t0) * 1000.0
            latencies_ms.append(latency)

            counters[ev.source][decision] += 1
            processed += 1

            if processed % print_every == 0:
                print(
                    f"[PRODUCER→PIPELINE] event={ev.event_id} source={ev.source} | "
                    f"[ROUTE] {routed} | [INFERENCE] p_malware={p:.3f} | "
                    f"[DECISION] {decision} | latency={latency:.1f}ms"
                )

            writer.writerow(
                {
                    "event_id": ev.event_id,
                    "source": ev.source,
                    "p_malware": f"{p:.6f}",
                    "decision": decision,
                    "latency_ms": f"{latency:.3f}",
                    "label": "" if ev.label is None else int(ev.label),
                    "user_action": user_action,
                }
            )

    def _avg(xs: List[float]) -> float:
        return sum(xs) / max(1, len(xs))

    print("\n=== STREAM SUMMARY ===")
    total = sum(sum(v.values()) for v in counters.values())
    print(f"Total events: {total}")
    for src in ["static", "dynamic"]:
        print(f"{src}: ALERT={counters[src]['ALERT']} REVIEW={counters[src]['REVIEW']} PASS={counters[src]['PASS']}")
    print(f"Avg latency: {_avg(latencies_ms):.1f}ms")
    print(f"Results saved to: {out_csv}")
