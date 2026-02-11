# pipeline/decision.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Thresholds:
    alert: float = 0.80
    review: float = 0.55

    def decide(self, p_malware: float) -> str:
        if p_malware >= self.alert:
            return "ALERT"
        if p_malware >= self.review:
            return "REVIEW"
        return "PASS"


def load_thresholds(thresholds_path: str) -> Dict[str, Thresholds]:
    """
    Loads thresholds.json in the format:
    {
      "static": {"alert": 0.7, "review": 0.5},
      "dynamic": {"alert": 0.7, "review": 0.5}
    }
    Returns a dict with keys 'static' and 'dynamic'.
    """
    with open(thresholds_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[str, Thresholds] = {}
    for key in ("static", "dynamic"):
        cfg = raw.get(key, {})
        out[key] = Thresholds(
            alert=float(cfg.get("alert", 0.80)),
            review=float(cfg.get("review", 0.55)),
        )
    return out


def get_thresholds(thresholds_path: Optional[str] = None) -> Dict[str, Thresholds]:
    """
    If thresholds_path is provided and exists -> load it.
    Else, if models/thresholds.json exists -> load it.
    Else -> defaults.
    """
    if thresholds_path and os.path.exists(thresholds_path):
        return load_thresholds(thresholds_path)

    default_path = os.path.join("models", "thresholds.json")
    if os.path.exists(default_path):
        return load_thresholds(default_path)

    return {"static": Thresholds(), "dynamic": Thresholds()}
