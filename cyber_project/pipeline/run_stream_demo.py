# pipeline/run_stream_demo.py
from __future__ import annotations

import argparse
import time
from queue import Queue
from threading import Thread

import joblib

from pipeline.consumer import consume
from pipeline.producer import produce_both
from pipeline.decision import get_thresholds


def main():
    ap = argparse.ArgumentParser(description="Streaming demo: producer -> consumer malware detection")
    ap.add_argument("--static_csv", required=True)
    ap.add_argument("--dynamic_csv", required=True)
    ap.add_argument("--static_model", required=True)
    ap.add_argument("--dynamic_model", required=True)
    ap.add_argument("--max_events", type=int, default=300)
    ap.add_argument("--sleep_ms", type=int, default=0)
    ap.add_argument("--interactive_review", action="store_true")
    ap.add_argument("--suppress_sklearn_pickle_warnings", action="store_true")
    ap.add_argument("--thresholds_json", default=None, help="Optional path to thresholds.json (defaults to models/thresholds.json)")
    args = ap.parse_args()

    thresholds = get_thresholds(args.thresholds_json)

    print("\n=== STREAM DEMO START ===")
    print(f"Static model : {args.static_model}")
    print(f"Dynamic model: {args.dynamic_model}")
    print(f"Thresholds   : static(ALERT>={thresholds['static'].alert:.2f}, REVIEW>={thresholds['static'].review:.2f}) | "
          f"dynamic(ALERT>={thresholds['dynamic'].alert:.2f}, REVIEW>={thresholds['dynamic'].review:.2f})")
    print("=========================\n")

    q = Queue()
    static_cols = joblib.load("models/static_feature_cols.pkl")
    dynamic_cols = joblib.load("models/dynamic_feature_cols.pkl")

    def run_producer():
        produce_both(
            q,
            static_csv=args.static_csv,
            dynamic_csv=args.dynamic_csv,
            static_feature_cols=static_cols,
            dynamic_feature_cols=dynamic_cols,
            max_events=args.max_events,
            sleep_ms=args.sleep_ms,
        )

    start = time.time()
    prod_thread = Thread(target=run_producer)
    prod_thread.start()
    consume(
        q,
        static_model_path=args.static_model,
        dynamic_model_path=args.dynamic_model,
        thresholds_by_source=thresholds,
        interactive_review=args.interactive_review,
        suppress_sklearn_pickle_warnings=args.suppress_sklearn_pickle_warnings,
    )
    prod_thread.join()
    elapsed = time.time() - start
    print(f"\nTotal stream time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
