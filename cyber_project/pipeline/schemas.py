from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import time

@dataclass
class Event:
    event_id: int
    ts: float
    source: str  # "static" | "dynamic"
    features: List[float]
    label: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None

def now_ts() -> float:
    return time.time()
