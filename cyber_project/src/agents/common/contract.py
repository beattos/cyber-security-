from __future__ import annotations
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

from pydantic import BaseModel, ConfigDict

class _Base(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


SourceType = Literal["static", "dynamic"]

class Sample(_Base):
    sample_id: str
    source_type: SourceType
    features: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    label: Optional[int] = None

class InferenceOutput(_Base):
    sample_id: str
    agent_name: str
    source_type: SourceType
    y_pred: int
    proba_malicious: float
    confidence: float
    explain_stub: Dict[str, Any] = Field(default_factory=dict)
    model_version: str = "v1"
    latency_ms: float = 0.0

class Judgement(_Base):
    sample_id: str
    decision: int
    chosen_agent: str
    rationale: str
    per_agent: List[InferenceOutput]
    combined_confidence: float
    explain: Dict[str, Any] = Field(default_factory=dict)

class EnforcementDecision(_Base):
    sample_id: str
    action: Literal["ALLOW", "REVIEW", "NO_ACTION"]
    effective_confidence: float
    penalty: float
    reason: str
    judgement: Judgement
