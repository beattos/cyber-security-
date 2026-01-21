import argparse
import json
import os
import time
from typing import Any, Dict

import pandas as pd

from src.streaming.kafka_io import build_producer


def row_to_event(row: pd.Series, idx: int, source_type: str, feature_cols: list[str]) -> Dict[str, Any]:
    features = {}
    for f in feature_cols:
        v = row.get(f, 0.0)
        try:
            features[f] = float(v)
        except Exception:
            features[f] = 0.0

    sample_id = f"{source_type.upper()}-STREAM-{idx:05d}"

    event = {
        "sample_id": sample_id,
        "source_type": source_type,
        "features": features,
        "metadata": {
            "row_idx": int(idx),
            "source": "kafka_producer_demo",
        },
    }
    return event


def main():
    parser = argparse.ArgumentParser(description="Kafka producer demo for malware-input topic.")
    parser.add_argument("--source-type", choices=["static", "dynamic"], required=True)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--sleep-ms", type=int, default=200)
    args = parser.parse_args()

    source_type = args.source_type
    rows = max(1, args.rows)
    sleep_ms = max(0, args.sleep_ms)

    csv_path = "data/static_clean.csv" if source_type == "static" else "data/dynamic_clean.csv"

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "label"]
    n = min(rows, len(df))

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    topic = os.getenv("KAFKA_INPUT_TOPIC", "malware-input")

    producer = build_producer(bootstrap_servers=bootstrap)

    print(f"Sending {n} {source_type} rows from {csv_path} to topic '{topic}' via {bootstrap}")

    try:
        for i in range(n):
            row = df.iloc[i]
            event = row_to_event(row, i, source_type, feature_cols)
            producer.send(topic, key=event["sample_id"], value=event)
            producer.flush()
            print(f"Sent sample_id={event['sample_id']}")
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)
    finally:
        producer.close()


if __name__ == "__main__":
    main()

