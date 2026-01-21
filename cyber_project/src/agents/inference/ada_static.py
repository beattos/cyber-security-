from src.agents.inference.base import BaseSklearnInferenceAgent

class AdaStaticAgent(BaseSklearnInferenceAgent):
    def source_type(self) -> str:
        return "static"
