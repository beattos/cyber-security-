from __future__ import annotations
from src.agents.common.contract import Judgement, EnforcementDecision

class ConfidenceEnforcingAgent:
    """
    SOC-style enforcement:
    - Different thresholds per source type (static vs dynamic)
    - Penalize disagreement
    - Penalize low data quality (many imputed features)
    """

    def __init__(
        self,
        # thresholds per source
        static_t_high: float = 0.75,
        static_t_low: float = 0.45,
        dynamic_t_high: float = 0.80,
        dynamic_t_low: float = 0.55,
        # penalties
        disagreement_penalty: float = 0.15,
        impute_penalty_mid: float = 0.05,   # if imputed_ratio > 0.05
        impute_penalty_high: float = 0.15,  # if imputed_ratio > 0.15
    ):
        self.static_t_high = static_t_high
        self.static_t_low = static_t_low
        self.dynamic_t_high = dynamic_t_high
        self.dynamic_t_low = dynamic_t_low

        self.disagreement_penalty = disagreement_penalty
        self.impute_penalty_mid = impute_penalty_mid
        self.impute_penalty_high = impute_penalty_high

    def enforce(self, judgement: Judgement) -> EnforcementDecision:
        agreement = bool(judgement.explain.get("agreement", False))

        # Decide which thresholds to use (based on the first agent's source_type)
        st = judgement.per_agent[0].source_type if judgement.per_agent else "static"
        if st == "dynamic":
            t_high, t_low = self.dynamic_t_high, self.dynamic_t_low
        else:
            t_high, t_low = self.static_t_high, self.static_t_low

        penalty = 0.0

        # Disagreement penalty
        if not agreement:
            penalty += self.disagreement_penalty

        # Data-quality penalty (max imputed ratio across agents)
        max_imputed_ratio = 0.0
        for o in judgement.per_agent:
            r = float(o.explain_stub.get("imputed_ratio", 0.0) or 0.0)
            max_imputed_ratio = max(max_imputed_ratio, r)

        if max_imputed_ratio > 0.15:
            penalty += self.impute_penalty_high
        elif max_imputed_ratio > 0.05:
            penalty += self.impute_penalty_mid

        effective = max(0.0, min(1.0, judgement.combined_confidence - penalty))

        if effective >= t_high:
            action = "ALLOW"
            reason = "High effective confidence"
        elif effective >= t_low:
            action = "REVIEW"
            reason = "Medium confidence or disagreement/data-quality risk"
        else:
            action = "NO_ACTION"
            reason = "Low effective confidence"

        return EnforcementDecision(
            sample_id=judgement.sample_id,
            action=action,
            effective_confidence=float(effective),
            penalty=float(penalty),
            reason=reason,
            judgement=judgement,
        )
