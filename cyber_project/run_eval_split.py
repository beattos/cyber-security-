import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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


def sample_from_row(row: pd.Series, idx: int, source_type: str, feature_order: list[str]) -> Sample:
    label = None
    if "label" in row.index:
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
        sample_id=f"{source_type.upper()}-TEST-{idx:05d}",
        source_type=source_type,
        features=feats,
        metadata={"split": "test", "idx": idx},
        label=label,
    )


def eval_dataset(csv_path: str, source_type: str, feature_order: list[str], orch: Orchestrator, test_size=0.15, seed=42):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"{csv_path} has no 'label' column; cannot do evaluation split.")

    y = df["label"].astype(int)
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    test_df = df.loc[test_idx].reset_index(drop=True)

    rows = []
    y_true, y_pred = [], []

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        sample = sample_from_row(row, i, source_type, feature_order)
        decision = orch.run(sample)

        per = decision.judgement.per_agent
        max_imputed_ratio = max(float(o.explain_stub.get("imputed_ratio", 0.0) or 0.0) for o in per)
        chosen = next(o for o in per if o.agent_name == decision.judgement.chosen_agent)
        chosen_imputed_ratio = float(chosen.explain_stub.get("imputed_ratio", 0.0) or 0.0)

        rows.append({
            "sample_id": sample.sample_id,
            "source_type": source_type,
            "label": sample.label,
            "final_decision": decision.judgement.decision,
            "action": decision.action,
            "effective_confidence": decision.effective_confidence,
            "penalty": decision.penalty,
            "chosen_agent": decision.judgement.chosen_agent,
            "combined_confidence": decision.judgement.combined_confidence,
            "agreement": decision.judgement.explain.get("agreement"),
            "votes_malicious": decision.judgement.explain.get("votes", {}).get("malicious"),
            "votes_benign": decision.judgement.explain.get("votes", {}).get("benign"),
            "max_imputed_ratio": max_imputed_ratio,
            "chosen_imputed_ratio": chosen_imputed_ratio,
        })

        y_true.append(sample.label)
        y_pred.append(decision.judgement.decision)

    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    return pd.DataFrame(rows), report


def main():
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    orch, static_order, dynamic_order = build_orchestrator()

    static_out, static_rep = eval_dataset("data/static_clean.csv", "static", static_order, orch)
    static_out.to_csv("outputs/reports/static_test_report.csv", index=False)
    print("Wrote outputs/reports/static_test_report.csv")
    print("Static TEST classification report:\n", static_rep)

    dynamic_out, dynamic_rep = eval_dataset("data/dynamic_clean.csv", "dynamic", dynamic_order, orch)
    dynamic_out.to_csv("outputs/reports/dynamic_test_report.csv", index=False)
    print("Wrote outputs/reports/dynamic_test_report.csv")
    print("Dynamic TEST classification report:\n", dynamic_rep)


if __name__ == "__main__":
    main()
