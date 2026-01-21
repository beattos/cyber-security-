# pipeline/run_stream_demo.py
import argparse
from queue import Queue
from threading import Thread

import joblib

from pipeline.producer import produce_both
from pipeline.consumer import consume


def main():
    parser = argparse.ArgumentParser(description="Producer–Consumer streaming demo (static + dynamic).")
    parser.add_argument("--static_csv", required=True)
    parser.add_argument("--dynamic_csv", required=True)
    parser.add_argument("--static_model", required=True)
    parser.add_argument("--dynamic_model", required=True)
    parser.add_argument("--out_csv", default="outputs/stream_results.csv")
    parser.add_argument("--t_alert", type=float, default=0.80)
    parser.add_argument("--t_review", type=float, default=0.55)
    parser.add_argument("--interleave", choices=["alternate", "random"], default="alternate")
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--sleep_ms", type=int, default=0)
    parser.add_argument("--interactive_review", action="store_true")
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument(
        "--suppress_sklearn_pickle_warnings",
        action="store_true",
        help="Hide scikit-learn version mismatch warnings during demo output.",
    )

    args = parser.parse_args()

    # Load exact feature order from saved artifacts (no manual lists)
    static_cols = joblib.load("models/static_feature_cols.pkl")
    dynamic_cols = joblib.load("models/dynamic_feature_cols.pkl")

    q = Queue(maxsize=1000)

    t_prod = Thread(
        target=produce_both,
        kwargs=dict(
            q=q,
            static_csv=args.static_csv,
            dynamic_csv=args.dynamic_csv,
            static_feature_cols=static_cols,
            dynamic_feature_cols=dynamic_cols,
            label_col="label",
            interleave=args.interleave,
            seed=42,
            max_events=args.max_events,
            sleep_ms=args.sleep_ms,
        ),
        daemon=True,
    )

    t_cons = Thread(
        target=consume,
        kwargs=dict(
            q=q,
            static_model_path=args.static_model,
            dynamic_model_path=args.dynamic_model,
            out_csv=args.out_csv,
            t_alert=args.t_alert,
            t_review=args.t_review,
            interactive_review=args.interactive_review,
            print_every=args.print_every,
            suppress_sklearn_pickle_warnings=args.suppress_sklearn_pickle_warnings,
        ),
        daemon=True,
    )

    t_prod.start()
    t_cons.start()
    t_prod.join()
    t_cons.join()


if __name__ == "__main__":
    main()
