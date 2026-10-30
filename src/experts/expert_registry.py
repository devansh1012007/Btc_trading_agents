"""
Expert Registry (§12 + Phase 9 Per-Regime Expert Instances): centralized
construction of experts from settings.yaml, supporting both the default 3
experts (`trend_expert`, `mean_reversion_expert`, `breakout_expert`) and
arbitrary `(template x regime)` instances in `experts.instances`.
"""
from typing import Dict

from ..labeling.hold_policy import alignment_settings
from .base_expert import BaseExpert
from .breakout_expert import BreakoutExpert
from .mean_reversion_expert import MeanReversionExpert
from .trend_expert import TrendExpert

TEMPLATE_CLASSES = {
    "trend": TrendExpert,
    "trend_expert": TrendExpert,
    "mean_reversion": MeanReversionExpert,
    "mean_reversion_expert": MeanReversionExpert,
    "breakout": BreakoutExpert,
    "breakout_expert": BreakoutExpert,
}


def build_experts(config: dict, data_granularity: str = "1m") -> Dict[str, BaseExpert]:
    hold_policy, require_aligned = alignment_settings(config)
    experts_cfg = config.get("experts", {}) or {}
    models_cfg = config.get("models", {}) or {}
    experts_dir = models_cfg.get("experts_dir", "data/models/experts/")

    instances_cfg = experts_cfg.get("instances")
    if isinstance(instances_cfg, list) and instances_cfg:
        out: Dict[str, BaseExpert] = {}
        for item in instances_cfg:
            if not isinstance(item, dict):
                continue
            tmpl_key = str(item.get("template", "trend"))
            cls = TEMPLATE_CLASSES.get(tmpl_key)
            if cls is None:
                raise ValueError(f"Unknown expert template {tmpl_key!r} in experts.instances")
            base_key = tmpl_key.replace("_expert", "")
            base_cfg = dict(experts_cfg.get(base_key, {}) or {})
            base_cfg.update(item.get("params") or {})
            if "regime_scope" in item:
                scope = item["regime_scope"]
                base_cfg["regime_requirement"] = [scope] if isinstance(scope, str) else list(scope)
            inst_id = str(item.get("id") or f"{base_key}_expert@{item.get('regime', 'all')}")
            enabled = bool(item.get("enabled", True))
            base_cfg["enabled"] = enabled
            m_path = item.get("model_path", base_cfg.get("model_path", f"{experts_dir}{inst_id}.pkl"))
            exp = cls(
                config=base_cfg,
                model_path=m_path,
                data_granularity=data_granularity,
                hold_policy=hold_policy,
                require_aligned_models=require_aligned,
            )
            exp.instance_id = inst_id
            exp.expert_name = inst_id
            exp.enabled = enabled
            out[inst_id] = exp
        return out

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
