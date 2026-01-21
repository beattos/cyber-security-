import json
import pandas as pd

from src.agents.common.contract import Sample
from src.agents.inference.ada_static import AdaStaticAgent
from src.agents.inference.gb_static import GBStaticAgent
from src.agents.inference.ada_dynamic import AdaDynamicAgent
from src.agents.inference.gb_dynamic import GBDynamicAgent
from src.pipeline.orchestrator import Orchestrator


def load_order(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_row(csv_path: str, row_idx: int, feature_order: list[str], source_type: str) -> Sample:
    df = pd.read_csv(csv_path)

    if row_idx < 0 or row_idx >= len(df):
        raise IndexError(f"row_idx {row_idx} out of range (0..{len(df)-1})")

    row = df.iloc[row_idx]

    label = None
    if "label" in df.columns:
        try:
            label = int(row["label"])
        except Exception:
            label = None

    feats = {}
    for f in feature_order:
        v = row.get(f, 0.0)
        try:
            feats[f] = float(v)
        except Exception:
            feats[f] = 0.0

    return Sample(
        sample_id=f"{source_type.upper()}-{row_idx:05d}",
        source_type=source_type,
        features=feats,
        metadata={"csv": csv_path, "row_idx": row_idx},
        label=label,
    )


def main():
    # Choose which demo to run:
    SOURCE_TYPE = "dynamic"   # change to "dynamic" to test dynamic
    ROW_IDX = 0

    if SOURCE_TYPE == "static":
        order = load_order("artifacts/static_feature_order.json")
        csv_path = "data/static_clean.csv"
        agents = [
            AdaStaticAgent("ada_static", "models/ada_static.pkl", order),
            GBStaticAgent("gb_static", "models/gb_static.pkl", order),
        ]
    elif SOURCE_TYPE == "dynamic":
        order = load_order("artifacts/dynamic_feature_order.json")
        # You need this file locally (next step if not yet):
        csv_path = "data/dynamic_clean.csv"
        agents = [
            AdaDynamicAgent("ada_dynamic", "models/ada_dynamic.pkl", order),
            GBDynamicAgent("gb_dynamic", "models/gb_dynamic.pkl", order),
        ]
    else:
        raise ValueError("SOURCE_TYPE must be 'static' or 'dynamic'")

    print(f"Loaded {SOURCE_TYPE} features: {len(order)}")

    orch = Orchestrator(agents)
    sample = load_row(csv_path, row_idx=ROW_IDX, feature_order=order, source_type=SOURCE_TYPE)

    print(f"Running inference on source={SOURCE_TYPE} row_idx={ROW_IDX}, label={sample.label}")
    decision = orch.run(sample)
    print(decision.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
