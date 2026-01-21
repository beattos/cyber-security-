from __future__ import annotations
from typing import List

from src.agents.common.contract import Sample, EnforcementDecision
from src.agents.judging.judge import ParadigmComparisonAgent
from src.agents.enforcing.enforce import ConfidenceEnforcingAgent

class Orchestrator:
    def __init__(self, inference_agents: List, judge_agent=None, enforcing_agent=None):
        self.inference_agents = inference_agents
        self.judge_agent = judge_agent or ParadigmComparisonAgent()
        self.enforcing_agent = enforcing_agent or ConfidenceEnforcingAgent()

    def run(self, sample: Sample) -> EnforcementDecision:
        outputs = []
        skipped = []

        for agent in self.inference_agents:
            try:
                outputs.append(agent.predict(sample))
            except ValueError as e:
                skipped.append(f"{getattr(agent, 'agent_name', agent.__class__.__name__)}: {e}")

        if not outputs:
            raise ValueError(
                "No inference outputs produced. This means no agent matched the sample source_type.\n"
                f"sample.source_type={sample.source_type}\n"
                "Skipped agents:\n- " + "\n- ".join(skipped)
            )

        judgement = self.judge_agent.judge(outputs)
        return self.enforcing_agent.enforce(judgement)
