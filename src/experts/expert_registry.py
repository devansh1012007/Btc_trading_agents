"""
Expert Registry (§12): centralized construction of all three experts from
settings.yaml, so the pipeline, the shared replay helper, and every
training script build them identically.
"""
from typing import Dict

from .base_expert import BaseExpert
from .trend_expert import TrendExpert
from .mean_reversion_expert import MeanReversionExpert
from .breakout_expert import BreakoutExpert


def build_experts(config: dict, data_granularity: str = "1m") -> Dict[str, BaseExpert]:
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
        ),
        "mean_reversion_expert": MeanReversionExpert(
            config=mr_cfg,
            model_path=mr_cfg.get("model_path", f"{experts_dir}mean_reversion_expert.pkl"),
            data_granularity=data_granularity,
        ),
        "breakout_expert": BreakoutExpert(
            config=bo_cfg,
            model_path=bo_cfg.get("model_path", f"{experts_dir}breakout_expert.pkl"),
            data_granularity=data_granularity,
        ),
    }
