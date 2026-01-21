from src.agents.inference.base import BaseSklearnInferenceAgent

class GBStaticAgent(BaseSklearnInferenceAgent):
    def source_type(self) -> str:
        return "static"
