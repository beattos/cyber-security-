from __future__ import annotations
from typing import List
from src.agents.common.contract import InferenceOutput, Judgement

class ParadigmComparisonAgent:
    def judge(self, outputs: List[InferenceOutput]) -> Judgement:
        if not outputs:
            raise ValueError("No inference outputs provided")

        sample_id = outputs[0].sample_id

        best = max(outputs, key=lambda o: o.confidence)
        combined_conf = sum(o.confidence for o in outputs) / len(outputs)

        agreement = len({o.y_pred for o in outputs}) == 1
        votes = {
            "malicious": sum(o.y_pred == 1 for o in outputs),
            "benign": sum(o.y_pred == 0 for o in outputs),
        }

        rationale = (
            f"Selected {best.agent_name} due to highest confidence ({best.confidence:.3f}). "
            f"Avg confidence={combined_conf:.3f}. Agreement={agreement} votes={votes}."
        )

        return Judgement(
            sample_id=sample_id,
            decision=best.y_pred,
            chosen_agent=best.agent_name,
            rationale=rationale,
            per_agent=outputs,
            combined_confidence=float(combined_conf),
            explain={"agreement": agreement, "votes": votes},
        )
