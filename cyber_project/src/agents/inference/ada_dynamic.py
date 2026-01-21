from src.agents.inference.base import BaseSklearnInferenceAgent

class AdaDynamicAgent(BaseSklearnInferenceAgent):
    def source_type(self) -> str:
        return "dynamic"
