"""
Typed configuration Switchboard and cross-section validator (Phase 1, T1.2 / §4.3, §4.6).

Provides:
- Typed configuration models for `pipeline`, `regime`, `experts`, `exits`,
  `sizing`, `execution`, `router`, `market`, `positions`, `cooldowns`,
  `halts`, `governor`, `costs`, `data`, `monitoring`, and `handoff`.
- `Switchboard.from_config(cfg)` with safe defaults so absent sections
  reproduce legacy behavior bit-for-bit.
- `Switchboard.is_legacy()` returning True when all new switches are at
  their legacy/default-off values.
- `validate_switchboard(cfg, base_cfg=None, overlay_cfg=None)` raising
  `ConfigConsistencyError` naming the exact key path on unknown keys,
  invalid ranges, unmet stage dependencies, or conflicting layer overrides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .consistency_validator import ConfigConsistencyError

VALID_PIPELINE_ORDERS = ("legacy", "staged")
VALID_REGIME_BACKENDS = ("gaussian_hmm", "gmm", "kmeans", "ms_ar", "garch_mixture")
VALID_REGIME_K_MODES = ("fixed", "auto")
VALID_SERVING_SOURCES = ("dynamic_lgbm", "legacy_multiclass4", "backend_filter", "rules", "auto", "hmm_filter", "lgbm_classifier", "rule_based")
VALID_ZONE_MODES = ("every_bar", "every_n", "event")
VALID_EXIT_MODES = ("model", "expert_plan", "hard")
VALID_EXIT_PAIRINGS = ("template", "instance")
VALID_EXECUTION_EXIT_MODELS = ("barrier_level", "delayed_market")
VALID_STOP_SOURCES = ("invalidation_distance", "exit_plan")
VALID_MARKET_PROFILES = ("crypto_spot", "crypto_perp", "equity", "option")
VALID_DERIVATIVE_MODES = ("none", "atm")
VALID_SELECTION_METRICS = ("net_return", "expectancy", "sharpe", "sortino", "max_drawdown", "cvar95", "composite")
VALID_SELECTION_SCOPES = ("global", "per_regime")
VALID_COOLDOWN_SCOPES = ("instance", "expert", "regime")
VALID_HALT_RESUMES = ("auto", "manual")
VALID_HALT_CLOCKS = ("bar", "wall")
VALID_GOVERNOR_MODES = ("off", "rules", "ml")
VALID_GOVERNOR_ACTIONS = ("suspend_instances", "scale_size", "lower_concurrency", "trigger_halt")
VALID_SLIPPAGE_MODELS = ("fixed", "volatility_scaled", "depth_based")
VALID_SPREAD_SOURCES = ("auto", "synthetic", "quote")
VALID_HANDOFF_MODES = ("off", "shadow", "advisory", "execute")
VALID_HANDOFF_SINKS = ("webhook", "file", "callable", "stdout")
VALID_HANDOFF_ON_ERROR = ("fail_closed", "internal")
VALID_DATA_SOURCES = ("bars", "ticks")
VALID_EXEC_RESOLUTIONS = ("bars", "ticks")

ALLOWED_STAGE_KEYS = {
    "regime_dynamic": {"enabled"},
    "zone_selection": {"enabled", "mode", "n", "events", "band_pct"},
    "zone_survival": {"enabled"},
    "barrier_sizing_model": {"enabled"},
    "exit_models": {"enabled"},
    "routers": {"enabled"},
    "opportunity_confirm": {"enabled"},
    "risk_sizing_model": {"enabled"},
}

ALLOWED_SECTION_KEYS: Dict[str, Set[str]] = {
    "pipeline": {"order", "stages"},
    "regime": {
        "n_states", "confidence_floor", "persistence_window_bars",
        "persistence_transition_threshold", "panic_z_threshold",
        "observation_features", "backend", "k", "min_regime_support_pct",
        "min_mean_dwell_bars", "serving_chain",
    },
    "experts": {
        "trend", "mean_reversion", "breakout", "instances", "selection",
        "min_samples", "sparse_policy", "plan_grid",
    },
    "exits": {"default_mode", "per_expert", "pairing", "hard", "model"},
    "sizing": {"barrier_model", "risk_model", "stop_source"},
    "positions": {
        "max_concurrent", "max_concurrent_per_expert", "max_gross_exposure_pct",
        "one_entry_per_bar", "hard_cap",
    },
    "cooldowns": {"enabled", "default", "per_instance", "wrong_if"},
    "halts": {"enabled", "rules", "resume", "clock", "flatten_on_halt"},
    "governor": {"mode", "may", "ml", "min_active_instances", "min_dwell_bars"},
    "costs": {
        "fees", "slippage", "spread", "delay", "adverse_selection_pct_per_100ms",
        "impact", "rejection", "partial_fill_policy", "carry", "per_market",
    },
    "handoff": {"mode", "sink", "include", "simulated_latency_seconds", "transport", "endpoint_url", "timeout_ms", "on_error", "secret_env_var", "verify_response_signature", "queue_dir", "command", "secret_key", "jsonl_path"},
    "market": {"min_spread_pct", "max_spread_pct", "min_depth_usdt", "profile", "derivatives"},
}


def _reject_unknown_keys(mapping: Mapping[str, Any], allowed: Set[str], path: str) -> None:
    for k in mapping:
        if k not in allowed:
            raise ConfigConsistencyError(
                f"Unknown config key '{path}.{k}' (allowed: {sorted(allowed)})"
            )


@dataclass(frozen=True)
class StageSwitchConfig:
    regime_dynamic: bool = False
    zone_selection: bool = False
    zone_selection_mode: str = "every_bar"
    zone_selection_n: int = 10
    zone_selection_events: Tuple[str, ...] = ("pivot", "regime_change", "zone_break", "band_entry")
    zone_survival: bool = False
    barrier_sizing_model: bool = False
    exit_models: bool = False
    routers: bool = False
    opportunity_confirm: bool = False
    risk_sizing_model: bool = False


@dataclass(frozen=True)
class RegimeKConfig:
    mode: str = "fixed"
    min_k: int = 2
    max_k: Optional[int] = None
    hard_cap: int = 12
    lock_between_retrains: bool = True


@dataclass(frozen=True)
class Switchboard:
    """Resolved, validated typed configuration switchboard."""
    legacy_mode: bool
    pipeline_order: str
    stages: StageSwitchConfig
    regime_backend: str
    regime_k: RegimeKConfig
    regime_min_support_pct: float
    regime_min_dwell_bars: int
    regime_serving_chain: Tuple[str, ...]
    expert_instances: Tuple[Dict[str, Any], ...]
    expert_selection: Dict[str, Any]
    exits_default_mode: str
    exits_per_expert: Dict[str, str]
    exits_pairing: str
    exits_hard: Dict[str, Any]
    exits_model: Dict[str, Any]
    sizing_stop_source: str
    barrier_model_enabled: bool
    barrier_model_bounds: Dict[str, Tuple[float, float]]
    risk_model_enabled: bool
    risk_model_min_multiplier: float
    entry_delay_seconds: float
    exit_delay_seconds: float
    execution_exit_model: str
    router_min_expert_confidence: float
    market_profile: str
    derivatives_mode: str
    derivatives_atm: Dict[str, Any]
    max_concurrent: int
    max_concurrent_per_expert: int
    max_gross_exposure_pct: float
    one_entry_per_bar: bool
    positions_hard_cap: int
    cooldowns_enabled: bool
    cooldowns_default: Dict[str, Any]
    halts_enabled: bool
    halts_rules: Tuple[Dict[str, Any], ...]
    halts_resume: str
    halts_clock: str
    governor_mode: str
    governor_may: Tuple[str, ...]
    data_source: str
    execution_resolution: str
    handoff_mode: str
    raw_config: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], validate: bool = True) -> "Switchboard":
        if validate:
            validate_switchboard(config)

        legacy_override = bool(config.get("legacy_mode", False))
        pipe_cfg = (config.get("pipeline", {}) or {}) if not legacy_override else {}
        stages_cfg = pipe_cfg.get("stages", {}) or {}

        def _stg_bool(name: str) -> bool:
            if legacy_override:
                return False
            blk = stages_cfg.get(name, {}) or {}
            return bool(blk.get("enabled", False)) if isinstance(blk, Mapping) else bool(blk)

        zs_blk = (stages_cfg.get("zone_selection", {}) or {}) if not legacy_override else {}
        stages = StageSwitchConfig(
            regime_dynamic=_stg_bool("regime_dynamic"),
            zone_selection=_stg_bool("zone_selection"),
            zone_selection_mode=str(zs_blk.get("mode", "every_bar")) if _stg_bool("zone_selection") else "every_bar",
            zone_selection_n=int(zs_blk.get("n", 10)),
            zone_selection_events=tuple(zs_blk.get("events", ("pivot", "regime_change", "zone_break", "band_entry"))),
            zone_survival=_stg_bool("zone_survival"),
            barrier_sizing_model=_stg_bool("barrier_sizing_model"),
            exit_models=_stg_bool("exit_models"),
            routers=_stg_bool("routers"),
            opportunity_confirm=_stg_bool("opportunity_confirm"),
            risk_sizing_model=_stg_bool("risk_sizing_model"),
        )

        regime_cfg = (config.get("regime", {}) or {}) if not legacy_override else {}
        k_cfg = regime_cfg.get("k", {}) or {}
        regime_k = RegimeKConfig(
            mode=str(k_cfg.get("mode", "fixed" if not stages.regime_dynamic else "auto")),
            min_k=int(k_cfg.get("min", 2)),
            max_k=int(k_cfg["max"]) if k_cfg.get("max") is not None else None,
            hard_cap=int(k_cfg.get("hard_cap", 12)),
            lock_between_retrains=bool(k_cfg.get("lock_between_retrains", True)),
        )

        experts_cfg = (config.get("experts", {}) or {}) if not legacy_override else {}
        instances_raw = experts_cfg.get("instances") or ()
        selection_raw = dict(experts_cfg.get("selection") or {
            "top_n": None, "metric": "net_return", "scope": "global", "min_trades": 30
        })

        exits_cfg = (config.get("exits", {}) or {}) if not legacy_override else {}
        default_exit_mode = str(exits_cfg.get("default_mode", "hard" if not stages.exit_models else "model"))
        per_expert_exits = {str(k): str(v) for k, v in (exits_cfg.get("per_expert") or {}).items()}
        hard_exits = dict(exits_cfg.get("hard") or {
            "sl_pct": 0.5, "tp_pct": 1.0,
            "trailing": {"activation_pct": 0.4, "distance_pct": 0.25},
            "max_loss_pct": 1.0,
        })
        model_exits = dict(exits_cfg.get("model") or {
            "profit_threshold": 0.65, "loss_threshold": 0.70,
        })

        sizing_cfg = (config.get("sizing", {}) or {}) if not legacy_override else {}
        bm_cfg = sizing_cfg.get("barrier_model", {}) or {}
        bm_bounds_raw = bm_cfg.get("bounds", {}) or {}
        bm_bounds = {
            "k_sl": tuple(float(x) for x in bm_bounds_raw.get("k_sl", (0.3, 3.0))),
            "k_tp": tuple(float(x) for x in bm_bounds_raw.get("k_tp", (0.3, 5.0))),
        }
        rm_cfg = sizing_cfg.get("risk_model", {}) or {}

        exec_cfg = config.get("execution", {}) or {}
        costs_cfg = (config.get("costs", {}) or {}) if not legacy_override else {}
        delay_cfg = costs_cfg.get("delay", {}) or {}

        if legacy_override:
            entry_delay = 0.0
            exit_delay = 0.0
            exec_exit_model = "barrier_level"
        else:
            entry_delay = float(
                delay_cfg.get(
                    "entry_seconds",
                    exec_cfg.get("entry_delay_seconds", float(exec_cfg.get("simulated_latency_ms", 0)) / 1000.0),
                )
            )
            exit_delay = float(delay_cfg.get("exit_seconds", exec_cfg.get("exit_delay_seconds", 0.0)))
            exec_exit_model = str(exec_cfg.get("exit_model", "barrier_level"))

        router_cfg = config.get("router", {}) or {}
        market_cfg = (config.get("market", {}) or {}) if not legacy_override else {}
        deriv_cfg = market_cfg.get("derivatives", {}) or {}

        pos_cfg = (config.get("positions", {}) or {}) if not legacy_override else {}
        risk_breakers = ((config.get("risk", {}) or {}).get("circuit_breakers", {}) or {})
        max_conc = int(pos_cfg.get("max_concurrent", risk_breakers.get("max_concurrent_positions", 1)))

        cd_cfg = (config.get("cooldowns", {}) or {}) if not legacy_override else {}
        cd_enabled = bool(cd_cfg.get("enabled", bool(cd_cfg.get("default") or cd_cfg.get("per_instance"))))

        halts_cfg = (config.get("halts", {}) or {}) if not legacy_override else {}
        halts_enabled = bool(halts_cfg.get("enabled", False))

        gov_cfg = (config.get("governor", {}) or {}) if not legacy_override else {}
        data_cfg = (config.get("data", {}) or {}) if not legacy_override else {}
        handoff_cfg = (config.get("handoff", {}) or {}) if not legacy_override else {}

        return cls(
            legacy_mode=legacy_override,
            pipeline_order="legacy" if legacy_override else str(pipe_cfg.get("order", "legacy")),
            stages=stages,
            regime_backend=str(regime_cfg.get("backend", "gaussian_hmm")),
            regime_k=regime_k,
            regime_min_support_pct=float(regime_cfg.get("min_regime_support_pct", 3.0)),
            regime_min_dwell_bars=int(regime_cfg.get("min_mean_dwell_bars", 5)),
            regime_serving_chain=tuple(
                regime_cfg.get("serving_chain", ("dynamic_lgbm", "legacy_multiclass4", "backend_filter", "rules"))
            ),
            expert_instances=tuple(dict(x) for x in instances_raw),
            expert_selection=selection_raw,
            exits_default_mode=default_exit_mode,
            exits_per_expert=per_expert_exits,
            exits_pairing=str(exits_cfg.get("pairing", "template")),
            exits_hard=hard_exits,
            exits_model=model_exits,
            sizing_stop_source=str(sizing_cfg.get("stop_source", "invalidation_distance")),
            barrier_model_enabled=bool(bm_cfg.get("enabled", False)) or stages.barrier_sizing_model,
            barrier_model_bounds=bm_bounds,  # type: ignore[arg-type]
            risk_model_enabled=bool(rm_cfg.get("enabled", False)) or stages.risk_sizing_model,
            risk_model_min_multiplier=float(rm_cfg.get("min_multiplier", 0.25)),
            entry_delay_seconds=entry_delay,
            exit_delay_seconds=exit_delay,
            execution_exit_model=exec_exit_model,
            router_min_expert_confidence=float(router_cfg.get("min_expert_confidence", 0.70)),
            market_profile=str(market_cfg.get("profile", "crypto_spot")),
            derivatives_mode=str(deriv_cfg.get("mode", "none")),
            derivatives_atm=dict(deriv_cfg.get("atm") or {"min_dte": 1, "roll_dte": 0}),
            max_concurrent=max_conc,
            max_concurrent_per_expert=int(pos_cfg.get("max_concurrent_per_expert", 1)),
            max_gross_exposure_pct=float(pos_cfg.get("max_gross_exposure_pct", 100.0)),
            one_entry_per_bar=bool(pos_cfg.get("one_entry_per_bar", True)),
            positions_hard_cap=int(pos_cfg.get("hard_cap", 20)),
            cooldowns_enabled=cd_enabled,
            cooldowns_default=dict(cd_cfg.get("default") or {"after_losses": 2, "duration": "30m", "scope": "instance"}),
            halts_enabled=halts_enabled,
            halts_rules=tuple(dict(r) for r in (halts_cfg.get("rules") or ())),
            halts_resume=str(halts_cfg.get("resume", "auto")),
            halts_clock=str(halts_cfg.get("clock", "bar")),
            governor_mode=str(gov_cfg.get("mode", "off")),
            governor_may=tuple(gov_cfg.get("may") or VALID_GOVERNOR_ACTIONS),
            data_source=str(data_cfg.get("source", "bars")),
            execution_resolution=str(data_cfg.get("execution_resolution", "bars")),
            handoff_mode=str(handoff_cfg.get("mode", "off")),
            raw_config=dict(config),
        )

    def is_legacy(self) -> bool:
        """True when all new switches are at their legacy defaults (today's behavior)."""
        if self.legacy_mode:
            return True
        st = self.stages
        any_stage_on = (
            st.regime_dynamic or st.zone_selection or st.zone_survival
            or st.barrier_sizing_model or st.exit_models or st.routers
            or st.opportunity_confirm or st.risk_sizing_model
        )
        return (
            not any_stage_on
            and self.pipeline_order == "legacy"
            and self.regime_k.mode == "fixed"
            and len(self.expert_instances) == 0
            and self.exits_default_mode == "hard"
            and not self.exits_per_expert
            and self.sizing_stop_source == "invalidation_distance"
            and not self.barrier_model_enabled
            and not self.risk_model_enabled
            and self.entry_delay_seconds == 0.0
            and self.exit_delay_seconds == 0.0
            and self.execution_exit_model == "barrier_level"
            and self.market_profile == "crypto_spot"
            and self.derivatives_mode == "none"
            and self.max_concurrent == 1
            and not self.cooldowns_enabled
            and not self.halts_enabled
            and self.governor_mode == "off"
            and self.data_source == "bars"
            and self.handoff_mode == "off"
        )


def validate_switchboard(
    config: Mapping[str, Any],
    *,
    base_config: Optional[Mapping[str, Any]] = None,
    overlay_config: Optional[Mapping[str, Any]] = None,
) -> None:
    """Validate all new configuration sections and cross-section dependencies.

    Raises :class:`ConfigConsistencyError` naming the offending key path.
    """
    # 1. Unknown key checks in new / extended sections
    for sec_name, allowed_keys in ALLOWED_SECTION_KEYS.items():
        sec = config.get(sec_name)
        if isinstance(sec, Mapping):
            _reject_unknown_keys(sec, allowed_keys, sec_name)

    # 2. Pipeline order & stages
    pipe_cfg = config.get("pipeline")
    if isinstance(pipe_cfg, Mapping):
        order = pipe_cfg.get("order", "legacy")
        if order not in VALID_PIPELINE_ORDERS:
            raise ConfigConsistencyError(
                f"pipeline.order={order!r} is invalid; expected one of {VALID_PIPELINE_ORDERS}"
            )
        stages_cfg = pipe_cfg.get("stages")
        if isinstance(stages_cfg, Mapping):
            _reject_unknown_keys(stages_cfg, set(ALLOWED_STAGE_KEYS), "pipeline.stages")
            for stg_name, stg_val in stages_cfg.items():
                if isinstance(stg_val, Mapping):
                    _reject_unknown_keys(stg_val, ALLOWED_STAGE_KEYS[stg_name], f"pipeline.stages.{stg_name}")
            zs = stages_cfg.get("zone_selection")
            if isinstance(zs, Mapping):
                zmode = zs.get("mode", "every_bar")
                if zmode not in VALID_ZONE_MODES:
                    raise ConfigConsistencyError(
                        f"pipeline.stages.zone_selection.mode={zmode!r} is invalid; expected {VALID_ZONE_MODES}"
                    )
                zn = int(zs.get("n", 10))
                if zn < 1:
                    raise ConfigConsistencyError(
                        f"pipeline.stages.zone_selection.n={zn} must be >= 1"
                    )

        # Dependency closure checks
        if order == "staged" and isinstance(stages_cfg, Mapping):
            oc = stages_cfg.get("opportunity_confirm")
            if isinstance(oc, Mapping) and oc.get("enabled") is False:
                raise ConfigConsistencyError(
                    "pipeline.order='staged' requires pipeline.stages.opportunity_confirm.enabled to not be explicitly false"
                )

    # 3. Regime section
    regime_cfg = config.get("regime")
    if isinstance(regime_cfg, Mapping):
        backend = regime_cfg.get("backend", "gaussian_hmm")
        if backend not in VALID_REGIME_BACKENDS:
            raise ConfigConsistencyError(
                f"regime.backend={backend!r} is invalid; expected one of {VALID_REGIME_BACKENDS}"
            )
        k_cfg = regime_cfg.get("k")
        if isinstance(k_cfg, Mapping):
            _reject_unknown_keys(k_cfg, {"mode", "min", "max", "hard_cap", "lock_between_retrains"}, "regime.k")
            k_mode = k_cfg.get("mode", "auto")
            if k_mode not in VALID_REGIME_K_MODES:
                raise ConfigConsistencyError(f"regime.k.mode={k_mode!r} must be one of {VALID_REGIME_K_MODES}")
            min_k = int(k_cfg.get("min", 2))
            hard_cap = int(k_cfg.get("hard_cap", 12))
            max_k = k_cfg.get("max")
            if min_k < 2:
                raise ConfigConsistencyError(f"regime.k.min={min_k} must be >= 2")
            if hard_cap < min_k:
                raise ConfigConsistencyError(f"regime.k.hard_cap={hard_cap} must be >= regime.k.min={min_k}")
            if max_k is not None:
                if int(max_k) < min_k:
                    raise ConfigConsistencyError(f"regime.k.max={max_k} must be >= regime.k.min={min_k}")
                if int(max_k) > hard_cap:
                    raise ConfigConsistencyError(f"regime.k.max={max_k} exceeds regime.k.hard_cap={hard_cap}")

        support_pct = float(regime_cfg.get("min_regime_support_pct", 3.0))
        if not (0.0 < support_pct <= 50.0):
            raise ConfigConsistencyError(
                f"regime.min_regime_support_pct={support_pct} must be in (0, 50]"
            )
        dwell = int(regime_cfg.get("min_mean_dwell_bars", 5))
        if dwell < 1:
            raise ConfigConsistencyError(f"regime.min_mean_dwell_bars={dwell} must be >= 1")

        chain = regime_cfg.get("serving_chain")
        if chain is not None:
            if not isinstance(chain, Sequence) or isinstance(chain, str) or len(chain) == 0:
                raise ConfigConsistencyError("regime.serving_chain must be a non-empty list")
            for src in chain:
                if src not in VALID_SERVING_SOURCES:
                    raise ConfigConsistencyError(
                        f"regime.serving_chain entry {src!r} is invalid; expected {VALID_SERVING_SOURCES}"
                    )

    # 4. Experts section
    experts_cfg = config.get("experts")
    enabled_instance_count = 3
    if isinstance(experts_cfg, Mapping):
        instances = experts_cfg.get("instances")
        if instances is not None:
            if not isinstance(instances, Sequence):
                raise ConfigConsistencyError("experts.instances must be a list")
            seen_ids: Set[str] = set()
            enabled_instance_count = 0
            for idx, inst in enumerate(instances):
                if not isinstance(inst, Mapping):
                    raise ConfigConsistencyError(f"experts.instances[{idx}] must be a mapping")
                _reject_unknown_keys(inst, {"id", "template", "regime", "enabled", "params"}, f"experts.instances[{idx}]")
                inst_id = inst.get("id")
                tmpl = inst.get("template")
                if not inst_id:
                    raise ConfigConsistencyError(f"experts.instances[{idx}].id is required")
                if inst_id in seen_ids:
                    raise ConfigConsistencyError(f"Duplicate experts.instances id {inst_id!r}")
                seen_ids.add(str(inst_id))
                if tmpl not in ("trend", "mean_reversion", "breakout", "trend_expert", "mean_reversion_expert", "breakout_expert"):
                    raise ConfigConsistencyError(
                        f"experts.instances[{idx}].template={tmpl!r} is invalid"
                    )
                if bool(inst.get("enabled", True)):
                    enabled_instance_count += 1

        sel = experts_cfg.get("selection")
        if isinstance(sel, Mapping):
            _reject_unknown_keys(
                sel, {"top_n", "metric", "scope", "min_trades", "profit_weight", "loss_weight"},
                "experts.selection",
            )
            metric = sel.get("metric", "net_return")
            if metric not in VALID_SELECTION_METRICS:
                raise ConfigConsistencyError(f"experts.selection.metric={metric!r} must be one of {VALID_SELECTION_METRICS}")
            scope = sel.get("scope", "global")
            if scope not in VALID_SELECTION_SCOPES:
                raise ConfigConsistencyError(f"experts.selection.scope={scope!r} must be one of {VALID_SELECTION_SCOPES}")
            top_n = sel.get("top_n")
            if top_n is not None and int(top_n) < 1:
                raise ConfigConsistencyError(f"experts.selection.top_n={top_n} must be >= 1")
            min_tr = int(sel.get("min_trades", 30))
            if min_tr < 1:
                raise ConfigConsistencyError(f"experts.selection.min_trades={min_tr} must be >= 1")

    # Stage dependency closure for routers & exit_models
    if isinstance(pipe_cfg, Mapping) and isinstance(pipe_cfg.get("stages"), Mapping):
        st_map = pipe_cfg["stages"]
        routers_on = isinstance(st_map.get("routers"), Mapping) and bool(st_map["routers"].get("enabled", False))
        exit_models_on = isinstance(st_map.get("exit_models"), Mapping) and bool(st_map["exit_models"].get("enabled", False))
        if routers_on and enabled_instance_count < 2:
            raise ConfigConsistencyError(
                f"pipeline.stages.routers.enabled=true requires at least 2 enabled expert instances (got {enabled_instance_count})"
            )
        if exit_models_on and enabled_instance_count < 1:
            raise ConfigConsistencyError(
                "pipeline.stages.exit_models.enabled=true requires at least 1 enabled expert instance"
            )

    # 5. Exits section
    exits_cfg = config.get("exits")
    if isinstance(exits_cfg, Mapping):
        dm = exits_cfg.get("default_mode", "hard")
        if dm not in VALID_EXIT_MODES:
            raise ConfigConsistencyError(f"exits.default_mode={dm!r} must be one of {VALID_EXIT_MODES}")
        pairing = exits_cfg.get("pairing", "template")
        if pairing not in VALID_EXIT_PAIRINGS:
            raise ConfigConsistencyError(f"exits.pairing={pairing!r} must be one of {VALID_EXIT_PAIRINGS}")
        pe = exits_cfg.get("per_expert")
        if isinstance(pe, Mapping):
            for exp_k, exp_mode in pe.items():
                if exp_mode not in VALID_EXIT_MODES:
                    raise ConfigConsistencyError(
                        f"exits.per_expert.{exp_k}={exp_mode!r} must be one of {VALID_EXIT_MODES}"
                    )
        hard_blk = exits_cfg.get("hard")
        if isinstance(hard_blk, Mapping):
            _reject_unknown_keys(hard_blk, {"sl_pct", "tp_pct", "trailing", "max_loss_pct"}, "exits.hard")
            for k_pct in ("sl_pct", "tp_pct", "max_loss_pct"):
                if k_pct in hard_blk and float(hard_blk[k_pct]) <= 0:
                    raise ConfigConsistencyError(f"exits.hard.{k_pct} must be > 0")
            tr_blk = hard_blk.get("trailing")
            if isinstance(tr_blk, Mapping):
                _reject_unknown_keys(tr_blk, {"activation_pct", "distance_pct"}, "exits.hard.trailing")
                if float(tr_blk.get("activation_pct", 0.4)) <= 0 or float(tr_blk.get("distance_pct", 0.25)) <= 0:
                    raise ConfigConsistencyError("exits.hard.trailing activation_pct and distance_pct must be > 0")
        mod_blk = exits_cfg.get("model")
        if isinstance(mod_blk, Mapping):
            _reject_unknown_keys(mod_blk, {"profit_threshold", "loss_threshold", "margin_pct", "max_exit_rate"}, "exits.model")
            for tk in ("profit_threshold", "loss_threshold"):
                if tk in mod_blk and not (0.0 < float(mod_blk[tk]) < 1.0):
                    raise ConfigConsistencyError(f"exits.model.{tk} must be in (0, 1)")

    # 6. Sizing section
    sizing_cfg = config.get("sizing")
    if isinstance(sizing_cfg, Mapping):
        stop_src = sizing_cfg.get("stop_source", "invalidation_distance")
        if stop_src not in VALID_STOP_SOURCES:
            raise ConfigConsistencyError(f"sizing.stop_source={stop_src!r} must be one of {VALID_STOP_SOURCES}")
        bm = sizing_cfg.get("barrier_model")
        if isinstance(bm, Mapping):
            _reject_unknown_keys(bm, {"enabled", "bounds", "quantiles"}, "sizing.barrier_model")
            bnds = bm.get("bounds")
            if isinstance(bnds, Mapping):
                _reject_unknown_keys(bnds, {"k_sl", "k_tp"}, "sizing.barrier_model.bounds")
                for bk in ("k_sl", "k_tp"):
                    if bk in bnds:
                        lo, hi = bnds[bk]
                        if float(lo) <= 0 or float(hi) < float(lo):
                            raise ConfigConsistencyError(f"sizing.barrier_model.bounds.{bk}=[{lo}, {hi}] is invalid")
        rm = sizing_cfg.get("risk_model")
        if isinstance(rm, Mapping):
            _reject_unknown_keys(rm, {"enabled", "min_multiplier", "mode"}, "sizing.risk_model")
            mm = float(rm.get("min_multiplier", 0.25))
            if not (0.0 < mm <= 1.0):
                raise ConfigConsistencyError(
                    f"sizing.risk_model.min_multiplier={mm} must be in (0, 1.0] (ADR-0004: ML may only reduce)"
                )

    # 7. Costs & Execution alias conflict checks
    costs_cfg = config.get("costs")
    exec_cfg = config.get("execution", {}) or {}
    labeling_cfg = config.get("labeling", {}) or {}
    if isinstance(costs_cfg, Mapping):
        from ..execution.cost_model import has_legacy_cost_conflict
        if has_legacy_cost_conflict(config):
            # If user explicitly supplied both in a raw config without migration alias flag
            pass
        fees_blk = costs_cfg.get("fees")
        if isinstance(fees_blk, Mapping):
            _reject_unknown_keys(fees_blk, {"taker_pct", "maker_pct", "entry_liquidity", "exit_liquidity"}, "costs.fees")
            if float(fees_blk.get("taker_pct", 0.04)) < 0 or float(fees_blk.get("maker_pct", 0.02)) < 0:
                raise ConfigConsistencyError("costs.fees percentages must be >= 0")
        slip_blk = costs_cfg.get("slippage")
        if isinstance(slip_blk, Mapping):
            _reject_unknown_keys(
                slip_blk, {"model", "fixed_pct", "vol_scaled", "depth_based", "max_slippage_pct"},
                "costs.slippage",
            )
            smod = slip_blk.get("model", "fixed")
            if smod not in VALID_SLIPPAGE_MODELS:
                raise ConfigConsistencyError(f"costs.slippage.model={smod!r} must be one of {VALID_SLIPPAGE_MODELS}")
        spr_blk = costs_cfg.get("spread")
        if isinstance(spr_blk, Mapping):
            _reject_unknown_keys(spr_blk, {"source", "synthetic"}, "costs.spread")
            if spr_blk.get("source", "auto") not in VALID_SPREAD_SOURCES:
                raise ConfigConsistencyError(f"costs.spread.source must be one of {VALID_SPREAD_SOURCES}")
        dly_blk = costs_cfg.get("delay")
        if isinstance(dly_blk, Mapping):
            _reject_unknown_keys(dly_blk, {"entry_seconds", "exit_seconds", "jitter_seconds"}, "costs.delay")
            if float(dly_blk.get("entry_seconds", 0)) < 0 or float(dly_blk.get("exit_seconds", 0)) < 0:
                raise ConfigConsistencyError("costs.delay seconds must be >= 0")

    if "entry_delay_seconds" in exec_cfg and "simulated_latency_ms" in exec_cfg:
        raise ConfigConsistencyError(
            "Both 'execution.entry_delay_seconds' and 'execution.simulated_latency_ms' are set; use one"
        )
    if "exit_model" in exec_cfg and exec_cfg["exit_model"] not in VALID_EXECUTION_EXIT_MODELS:
        raise ConfigConsistencyError(
            f"execution.exit_model={exec_cfg['exit_model']!r} must be one of {VALID_EXECUTION_EXIT_MODELS}"
        )

    # 8. Positions, Cooldowns, Halts, Governor
    pos_cfg = config.get("positions")
    if isinstance(pos_cfg, Mapping):
        mc = int(pos_cfg.get("max_concurrent", 1))
        hc = int(pos_cfg.get("hard_cap", 20))
        mce = int(pos_cfg.get("max_concurrent_per_expert", 1))
        mge = float(pos_cfg.get("max_gross_exposure_pct", 100.0))
        if mc < 1:
            raise ConfigConsistencyError(f"positions.max_concurrent={mc} must be >= 1")
        if mc > hc:
            raise ConfigConsistencyError(f"positions.max_concurrent={mc} exceeds positions.hard_cap={hc}")
        if mce < 1:
            raise ConfigConsistencyError(f"positions.max_concurrent_per_expert={mce} must be >= 1")
        if mge <= 0:
            raise ConfigConsistencyError(f"positions.max_gross_exposure_pct={mge} must be > 0")

    cd_cfg = config.get("cooldowns")
    if isinstance(cd_cfg, Mapping):
        dflt = cd_cfg.get("default")
        if isinstance(dflt, Mapping):
            _reject_unknown_keys(dflt, {"after_losses", "duration", "scope", "escalation"}, "cooldowns.default")
            if int(dflt.get("after_losses", 2)) < 1:
                raise ConfigConsistencyError("cooldowns.default.after_losses must be >= 1")
            if dflt.get("scope", "instance") not in VALID_COOLDOWN_SCOPES:
                raise ConfigConsistencyError(f"cooldowns.default.scope must be one of {VALID_COOLDOWN_SCOPES}")

    halts_cfg = config.get("halts")
    if isinstance(halts_cfg, Mapping):
        res = halts_cfg.get("resume", "auto")
        if res not in VALID_HALT_RESUMES:
            raise ConfigConsistencyError(f"halts.resume={res!r} must be one of {VALID_HALT_RESUMES}")
        clk = halts_cfg.get("clock", "bar")
        if clk not in VALID_HALT_CLOCKS:
            raise ConfigConsistencyError(f"halts.clock={clk!r} must be one of {VALID_HALT_CLOCKS}")
        if exec_cfg.get("mode") == "live" and clk != "wall":
            raise ConfigConsistencyError("halts.clock must be 'wall' when execution.mode=='live'")

    gov_cfg = config.get("governor")
    if isinstance(gov_cfg, Mapping):
        gmode = gov_cfg.get("mode", "off")
        if gmode not in VALID_GOVERNOR_MODES:
            raise ConfigConsistencyError(f"governor.mode={gmode!r} must be one of {VALID_GOVERNOR_MODES}")
        may = gov_cfg.get("may", VALID_GOVERNOR_ACTIONS)
        if not isinstance(may, Sequence) or isinstance(may, str):
            raise ConfigConsistencyError("governor.may must be a list of tightening actions")
        for act in may:
            if act not in VALID_GOVERNOR_ACTIONS:
                raise ConfigConsistencyError(
                    f"governor.may action {act!r} is forbidden (only tightening actions {VALID_GOVERNOR_ACTIONS} are allowed per ADR-0004)"
                )

    # 9. Handoff & Market & Data
    handoff_cfg = config.get("handoff")
    if isinstance(handoff_cfg, Mapping):
        hmode = handoff_cfg.get("mode", "off")
        if hmode not in VALID_HANDOFF_MODES:
            raise ConfigConsistencyError(f"handoff.mode={hmode!r} must be one of {VALID_HANDOFF_MODES}")
        if hmode == "execute" and exec_cfg.get("mode") == "live":
            raise ConfigConsistencyError(
                "handoff.mode='execute' is blocked when execution.mode='live' (C18 safety gate)"
            )
        sink = handoff_cfg.get("sink")
        if isinstance(sink, Mapping):
            _reject_unknown_keys(
                sink, {"type", "url_env", "url", "auth", "secret_env", "timeout_seconds", "on_error", "path", "callable", "max_failures"},
                "handoff.sink",
            )
            if sink.get("type", "webhook") not in VALID_HANDOFF_SINKS:
                raise ConfigConsistencyError(f"handoff.sink.type must be one of {VALID_HANDOFF_SINKS}")
            if sink.get("on_error", "fail_closed") not in VALID_HANDOFF_ON_ERROR:
                raise ConfigConsistencyError(f"handoff.sink.on_error must be one of {VALID_HANDOFF_ON_ERROR}")

    market_cfg = config.get("market")
    if isinstance(market_cfg, Mapping):
        prof = market_cfg.get("profile", "crypto_spot")
        if prof not in VALID_MARKET_PROFILES:
            raise ConfigConsistencyError(f"market.profile={prof!r} must be one of {VALID_MARKET_PROFILES}")
        deriv = market_cfg.get("derivatives")
        if isinstance(deriv, Mapping):
            _reject_unknown_keys(deriv, {"mode", "atm"}, "market.derivatives")
            if deriv.get("mode", "none") not in VALID_DERIVATIVE_MODES:
                raise ConfigConsistencyError(f"market.derivatives.mode must be one of {VALID_DERIVATIVE_MODES}")

    # 10. Cross-layer conflict detection (base vs overlay)
    if base_config is not None and overlay_config is not None:
        _validate_layer_conflicts(base_config, overlay_config)


def _validate_layer_conflicts(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> None:
    """Detect unsafe cross-layer contradictions between base and overlay configs."""
    base_order = ((base.get("pipeline") or {}).get("order"))
    over_order = ((overlay.get("pipeline") or {}).get("order"))
    if base_order and over_order and base_order != over_order:
        raise ConfigConsistencyError(
            f"Conflicting pipeline.order between base ({base_order!r}) and overlay ({over_order!r})"
        )
    over_exec = overlay.get("execution") or {}
    if over_exec.get("mode") == "live":
        halts_clk = ((overlay.get("halts") or base.get("halts") or {}).get("clock", "bar"))
        if halts_clk != "wall" and ((overlay.get("halts") or base.get("halts") or {}).get("enabled", False)):
            raise ConfigConsistencyError(
                "Overlay sets execution.mode='live' while halts.clock='bar'; live mode requires halts.clock='wall'"
            )
