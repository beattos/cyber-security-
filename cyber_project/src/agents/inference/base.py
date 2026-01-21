from __future__ import annotations
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List

import math
import pandas as pd
from typing import Dict
import numpy as np
import joblib

from src.agents.common.contract import Sample, InferenceOutput

class BaseSklearnInferenceAgent(ABC):
    def __init__(self, agent_name: str, model_path: str, feature_order: List[str],mal_threshold: float = 0.5):
        self.agent_name = agent_name
        self.model = joblib.load(model_path)
        self.feature_order = feature_order
        self.mal_threshold = mal_threshold

        # sanity
        if not hasattr(self.model, "predict_proba"):
            raise TypeError(f"Model at {model_path} does not support predict_proba")

    @abstractmethod
    def source_type(self) -> str:
        ...

    def _safe_float(self, v) -> float:
        try:
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                return 0.0
            return x
        except Exception:
            return 0.0

    def _vectorize(self, features: Dict[str, float]):
        imputed = 0
        row = {}
        for f in self.feature_order:
            raw = features.get(f, 0.0)
            val = self._safe_float(raw)
            # if raw was NaN/inf/non-numeric, _safe_float returns 0.0; we count those cases
            try:
                bad = (raw is None) or (isinstance(raw, float) and (math.isnan(raw) or math.isinf(raw)))
            except Exception:
                bad = True

            # also treat strings / non-castable as imputed
            if bad:
                imputed += 1

            row[f] = val

        df = pd.DataFrame([row], columns=self.feature_order)
        return df, imputed
    
    def predict(self, sample: Sample) -> InferenceOutput:
        st = str(sample.source_type).strip().lower()
        at = str(self.source_type()).strip().lower()
        if st != at:
            raise ValueError(f"{self.agent_name} expects {at} but got {st}")

        t0 = time.time()
        X, imputed = self._vectorize(sample.features)


        proba = float(self.model.predict_proba(X)[0][1])  # class 1 = malicious
        proba_malicious = float(self.model.predict_proba(X)[0][1])
        y_pred = 1 if proba_malicious >= self.mal_threshold else 0
        confidence = abs(proba - 0.5) * 2.0  # [0..1]
        
        if os.getenv("DEBUG_INFER") == "1" and sample.sample_id.endswith("00000"):
            print(f"[DEBUG] {self.agent_name} thr={self.mal_threshold} proba={proba_malicious:.4f} y_pred={y_pred}")


        explain_stub = {
            "imputed_features": int(imputed),
            "imputed_ratio": float(imputed / max(1, len(self.feature_order))),
        }

        if hasattr(self.model, "feature_importances_"):
            imps = self.model.feature_importances_
            top_idx = np.argsort(imps)[::-1][:5]
            explain_stub["top_features"] = [
                {"feature": self.feature_order[i], "importance": float(imps[i])}
                for i in top_idx
            ]

        latency_ms = (time.time() - t0) * 1000.0

        return InferenceOutput(
            sample_id=sample.sample_id,
            agent_name=self.agent_name,
            source_type=sample.source_type,
            y_pred=y_pred,
            proba_malicious=proba,
            confidence=float(confidence),
            explain_stub=explain_stub,
            latency_ms=float(latency_ms),
        )
