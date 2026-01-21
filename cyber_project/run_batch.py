import json
from pathlib import Path

import pandas as pd
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


def sample_from_row(df: pd.DataFrame, idx: int, source_type: str, feature_order: list[str]) -> Sample:
    row = df.iloc[idx]

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
        sample_id=f"{source_type.upper()}-{idx:05d}",
        source_type=source_type,
        features=feats,
        metadata={"idx": idx},
        label=label,
    )


def run_batch(csv_path: str, source_type: str, feature_order: list[str], orch: Orchestrator, limit: int | None = None):
    df = pd.read_csv(csv_path)
    n = len(df) if limit is None else min(limit, len(df))

    rows = []
    y_true = []
    y_pred = []

    for i in range(n):
        sample = sample_from_row(df, i, source_type, feature_order)
        decision = orch.run(sample)

        per = decision.judgement.per_agent

        max_imputed_ratio = max(float(o.explain_stub.get("imputed_ratio", 0.0) or 0.0) for o in per)
        max_imputed_features = max(int(o.explain_stub.get("imputed_features", 0) or 0) for o in per)

        chosen = next(o for o in per if o.agent_name == decision.judgement.chosen_agent)
        chosen_imputed_ratio = float(chosen.explain_stub.get("imputed_ratio", 0.0) or 0.0)
        chosen_imputed_features = int(chosen.explain_stub.get("imputed_features", 0) or 0)

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
            "max_imputed_features": max_imputed_features,
            "chosen_imputed_ratio": chosen_imputed_ratio,
            "chosen_imputed_features": chosen_imputed_features,
        })

        if sample.label is not None:
            y_true.append(sample.label)
            y_pred.append(decision.judgement.decision)

    out_df = pd.DataFrame(rows)

    metrics_text = None
    if y_true:
        metrics_text = classification_report(y_true, y_pred, digits=4, zero_division=0)


    return out_df, metrics_text


def main():
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    orch, static_order, dynamic_order = build_orchestrator()

    # You can set a small limit for quick tests (e.g., 50), then remove for full run
    LIMIT = 200

    static_df, static_metrics = run_batch(
        csv_path="data/static_clean.csv",
        source_type="static",
        feature_order=static_order,
        orch=orch,
        limit=LIMIT,
    )
    static_out = "outputs/reports/static_report.csv"
    static_df.to_csv(static_out, index=False)
    print(f"Wrote: {static_out} rows={len(static_df)}")
    if static_metrics:
        print("Static classification report:")
        print(static_metrics)

    dynamic_df, dynamic_metrics = run_batch(
        csv_path="data/dynamic_clean.csv",
        source_type="dynamic",
        feature_order=dynamic_order,
        orch=orch,
        limit=LIMIT,
    )
    dynamic_out = "outputs/reports/dynamic_report.csv"
    dynamic_df.to_csv(dynamic_out, index=False)
    print(f"Wrote: {dynamic_out} rows={len(dynamic_df)}")
    if dynamic_metrics:
        print("Dynamic classification report:")
        print(dynamic_metrics)


if __name__ == "__main__":
    main()
