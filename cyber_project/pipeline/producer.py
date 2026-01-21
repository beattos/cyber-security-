# pipeline/producer.py
import random
import time
from dataclasses import dataclass
from queue import Queue
from typing import Dict, Any, List, Optional

import pandas as pd


@dataclass
class Event:
    event_id: int
    ts: float
    source: str  # "static" | "dynamic"
    features: List[float]
    label: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


def _df_to_events(
    df: pd.DataFrame,
    source: str,
    feature_cols: List[str],
    label_col: str = "label",
) -> List[Event]:
    # Enforce exact order of feature columns
    X = df[feature_cols]

    y = None
    if label_col in df.columns:
        y = df[label_col]

    events: List[Event] = []
    for idx in range(len(X)):
        feats = [float(v) for v in X.iloc[idx].tolist()]
        label = int(y.iloc[idx]) if y is not None else None
        events.append(
            Event(
                event_id=0,
                ts=time.time(),
                source=source,
                features=feats,
                label=label,
                meta={"row_index": int(idx)},
            )
        )
    return events


def produce_both(
    q: Queue,
    static_csv: str,
    dynamic_csv: str,
    static_feature_cols: List[str],
    dynamic_feature_cols: List[str],
    label_col: str = "label",
    interleave: str = "alternate",  # "alternate" | "random"
    seed: int = 42,
    max_events: Optional[int] = None,
    sleep_ms: int = 0,  # optional: slow down for demo
):
    """
    Reads both CSVs, builds events with correct feature order, and pushes into queue.
    Sends a sentinel None when finished.
    """
    static_df = pd.read_csv(static_csv)
    dynamic_df = pd.read_csv(dynamic_csv)

    # Keep consistent with your notebook: remove EDA-only column if present
    dynamic_df = dynamic_df.drop(columns=["total_activity"], errors="ignore")

    static_events = _df_to_events(static_df, "static", static_feature_cols, label_col=label_col)
    dynamic_events = _df_to_events(dynamic_df, "dynamic", dynamic_feature_cols, label_col=label_col)

    merged: List[Event] = []
    if interleave == "alternate":
        i = j = 0
        while i < len(static_events) or j < len(dynamic_events):
            if i < len(static_events):
                merged.append(static_events[i]); i += 1
            if j < len(dynamic_events):
                merged.append(dynamic_events[j]); j += 1
    elif interleave == "random":
        rng = random.Random(seed)
        merged = static_events + dynamic_events
        rng.shuffle(merged)
    else:
        raise ValueError("interleave must be 'alternate' or 'random'")

    if max_events is not None:
        merged = merged[:max_events]

    # Push to queue
    for eid, ev in enumerate(merged, start=1):
        ev.event_id = eid
        ev.ts = time.time()

        # Producer visibility (optional; consumer prints the canonical line)
        # print(f"[PRODUCER] sent event={ev.event_id} source={ev.source}")

        q.put(ev)
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    q.put(None)  # sentinel
