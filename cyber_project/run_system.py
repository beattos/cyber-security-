import json
import pandas as pd

from src.agents.common.contract import Sample
from src.agents.common.utils import load_policy_config
from src.agents.inference.ada_static import AdaStaticAgent
from src.agents.inference.gb_static import GBStaticAgent
from src.agents.inference.ada_dynamic import AdaDynamicAgent
from src.agents.inference.gb_dynamic import GBDynamicAgent
from src.agents.enforcing.enforce import ConfidenceEnforcingAgent
from src.pipeline.orchestrator import Orchestrator


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_orchestrator():
    static_order = load_json("artifacts/static_feature_order.json")
    dynamic_order = load_json("artifacts/dynamic_feature_order.json")
    policy = load_policy_config()

    enforcing_agent = ConfidenceEnforcingAgent(
        static_t_high=policy["static_t_high"],
        static_t_low=policy["static_t_low"],
        dynamic_t_high=policy["dynamic_t_high"],
        dynamic_t_low=policy["dynamic_t_low"],
        disagreement_penalty=policy["disagreement_penalty"],
        impute_penalty_mid=policy["impute_penalty_mid"],
        impute_penalty_high=policy["impute_penalty_high"],
    )

    agents = [
        AdaStaticAgent("ada_static", "models/ada_static.pkl", static_order, mal_threshold=policy["static_mal_threshold"]),
        GBStaticAgent("gb_static", "models/gb_static.pkl", static_order, mal_threshold=policy["static_mal_threshold"]),
        AdaDynamicAgent("ada_dynamic", "models/ada_dynamic_calibrated.pkl", dynamic_order, mal_threshold=policy["dynamic_mal_threshold"]),
        GBDynamicAgent("gb_dynamic", "models/gb_dynamic_calibrated.pkl", dynamic_order, mal_threshold=policy["dynamic_mal_threshold"]),
    ]
    return Orchestrator(agents, enforcing_agent=enforcing_agent), static_order, dynamic_order


def sample_from_row(df: pd.DataFrame, idx: int, source_type: str, feature_order: list[str]) -> Sample:
    row = df.iloc[idx]

    label = None
    if "label" in df.columns:
        try:
            label = int(row["label"])
        except Exception:
            label = None

    feats = {f: float(row.get(f, 0.0)) for f in feature_order}

    return Sample(
        sample_id=f"{source_type.upper()}-{idx:05d}",
        source_type=source_type,
        features=feats,
        metadata={"idx": idx},
        label=label,
    )


def main():
    orch, static_order, dynamic_order = build_orchestrator()

    # Choose a demo run:
    source_type = "static"   # change to "dynamic"
    idx = 0

    if source_type == "static":
        df = pd.read_csv("data/static_clean.csv")
        sample = sample_from_row(df, idx, "static", static_order)
    else:
        df = pd.read_csv("data/dynamic_clean.csv")
        sample = sample_from_row(df, idx, "dynamic", dynamic_order)

    print(f"Unified system loaded. Running {source_type} idx={idx} label={sample.label}")
    decision = orch.run(sample)
    print(decision.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
