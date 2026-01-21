from src.agents.inference.base import BaseSklearnInferenceAgent

class GBDynamicAgent(BaseSklearnInferenceAgent):
    def source_type(self) -> str:
        return "dynamic"
