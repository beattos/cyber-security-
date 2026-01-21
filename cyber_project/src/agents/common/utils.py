from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import pandas as pd

def load_feature_orders(
    static_path: str = "artifacts/static_feature_order.json",
    dynamic_path: str = "artifacts/dynamic_feature_order.json",
) -> Tuple[List[str], List[str]]:
    with open(static_path, "r") as f:
        static_order = json.load(f)
    with open(dynamic_path, "r") as f:
        dynamic_order = json.load(f)

    if len(static_order) != 202:
        raise ValueError(f"Expected 202 static features, got {len(static_order)}")
    if len(dynamic_order) != 11:
        raise ValueError(f"Expected 11 dynamic features, got {len(dynamic_order)}")

    return static_order, dynamic_order


def assert_feature_alignment(df: pd.DataFrame, expected_order: List[str], label_col: str = "label") -> None:
    cols = df.drop(columns=[label_col], errors="ignore").columns.tolist()
    expected = list(expected_order)

    set_cols = set(cols)
    set_expected = set(expected)

    missing = sorted(list(set_expected - set_cols))
    extra = sorted(list(set_cols - set_expected))

    same_order = (cols == expected)

    if missing or extra or (not same_order):
        msg = ["Feature alignment failed:"]
        if missing:
            msg.append(f"- Missing ({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        if extra:
            msg.append(f"- Extra ({len(extra)}): {extra[:20]}{' ...' if len(extra) > 20 else ''}")
        if not same_order:
            min_len = min(len(cols), len(expected))
            mismatch_idx = next((i for i in range(min_len) if cols[i] != expected[i]), None)
            msg.append(
                f"- Order mismatch: first mismatch at index {mismatch_idx}, "
                f"got '{cols[mismatch_idx]}' expected '{expected[mismatch_idx]}'"
            )
            raise ValueError("\n".join(msg))


def load_policy_config(config_path: str = "config/policy.json") -> dict:
    """Load policy configuration from JSON file.
    
    Raises FileNotFoundError with clear message if config is missing.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Policy configuration file not found: {config_path}\n"
            f"Please create {config_path} with required threshold and penalty values.\n"
            f"Required keys: static_mal_threshold, dynamic_mal_threshold, "
            f"static_t_high, static_t_low, dynamic_t_high, dynamic_t_low, "
            f"disagreement_penalty, impute_penalty_mid, impute_penalty_high"
        )
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    required_keys = [
        "static_mal_threshold", "dynamic_mal_threshold",
        "static_t_high", "static_t_low",
        "dynamic_t_high", "dynamic_t_low",
        "disagreement_penalty", "impute_penalty_mid", "impute_penalty_high"
    ]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(
            f"Policy configuration missing required keys: {missing}\n"
            f"Required keys: {required_keys}"
        )
    
    return config
