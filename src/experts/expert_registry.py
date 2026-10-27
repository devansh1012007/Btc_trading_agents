"""
Expert Registry (§12): centralized construction of all three experts from
settings.yaml, so the pipeline, the shared replay helper, and every
training script build them identically.
"""
from typing import Dict

from ..labeling.hold_policy import alignment_settings
from .base_expert import BaseExpert
from .breakout_expert import BreakoutExpert
from .mean_reversion_expert import MeanReversionExpert
from .trend_expert import TrendExpert


def build_experts(config: dict, data_granularity: str = "1m") -> Dict[str, BaseExpert]:
    # Raises HoldPolicyError if the minimum-trade-duration policy cannot be
    # satisfied: an expert is never built on a config that makes every one
    # of its trades illegal.
    hold_policy, require_aligned = alignment_settings(config)
    experts_cfg = config.get("experts", {})
    models_cfg = config.get("models", {})
    experts_dir = models_cfg.get("experts_dir", "data/models/experts/")

    trend_cfg = experts_cfg.get("trend", {})
    mr_cfg = experts_cfg.get("mean_reversion", {})
    bo_cfg = experts_cfg.get("breakout", {})

    return {
        "trend_expert": TrendExpert(
            config=trend_cfg,
            model_path=trend_cfg.get("model_path", f"{experts_dir}trend_expert.pkl"),
            data_granularity=data_granularity,
            hold_policy=hold_policy, require_aligned_models=require_aligned,
        ),
        "mean_reversion_expert": MeanReversionExpert(
            config=mr_cfg,
            model_path=mr_cfg.get("model_path", f"{experts_dir}mean_reversion_expert.pkl"),
            data_granularity=data_granularity,
            hold_policy=hold_policy, require_aligned_models=require_aligned,
        ),
        "breakout_expert": BreakoutExpert(
            config=bo_cfg,
            model_path=bo_cfg.get("model_path", f"{experts_dir}breakout_expert.pkl"),
            data_granularity=data_granularity,
            hold_policy=hold_policy, require_aligned_models=require_aligned,
        ),
    }
