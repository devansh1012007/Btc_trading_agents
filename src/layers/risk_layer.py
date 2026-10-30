"""
Risk Layer (Phase 4, §5): the strongest gate in the system. Sits between
the Router and the Execution Layer -- only ever called for a NEW entry
candidate (RouterDecision.selected_expert != "no_trade"). Exits are
managed by the triple-barrier TP/SL/timeout mechanics already running in
the Execution Layer once a position is open, so "a daily-loss halt stops
new entries but not existing positions' exit management" (§5.3's table)
falls out of this architecture directly rather than needing special-cased
code: this layer is never in the exit path at all.

Five independently tripped, independently reset circuit breakers (§5.3),
persisted in SQLite (data/registry.db) so a process restart doesn't
forget today's daily loss limit was already hit -- plus a manual kill
switch checked first, before anything else, with zero dependencies.
"""
import json
import logging
import os
import pickle
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..labeling.hold_policy import HoldPolicy, model_alignment_problem
from ..registry.db import DEFAULT_DB_PATH, MEMORY_PATH, connect, open_connection
from ..state.contracts import (
    FEATURE_SCHEMA_VERSION,
    ExpertAction,
    ExpertOutputs,
    FeatureVector,
    MarketSnapshot,
    RegimeState,
    RiskDecision,
    RouterDecision,
    ZoneMap,
)
from ..utils.clock import Clock, WallClock
from ..utils.config_helpers import parse_regime_list

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "invalidation_distance", "router_selection_confidence", "regime_confidence",
    "volatility_medium", "spread_pct", "liquidity_depth_usdt", "expected_return", "expected_risk",
]


def build_features(
    expert_output, router_decision: RouterDecision, regime: RegimeState,
    fv: FeatureVector, market: MarketSnapshot,
) -> Dict[str, float]:
    depth_usdt = (market.bid_volume + market.ask_volume) * market.close
    return {
        "invalidation_distance": expert_output.invalidation_distance,
        "router_selection_confidence": router_decision.selection_confidence,
        "regime_confidence": regime.regime_confidence,
        "volatility_medium": fv.volatility_medium,
        "spread_pct": market.spread_pct,
        "liquidity_depth_usdt": depth_usdt,
        "expected_return": expert_output.expected_return,
        "expected_risk": expert_output.expected_risk,
    }


def compute_base_size(equity: float, expert_output, cfg: dict, exit_plan=None) -> float:
    """
    Standard risk-based sizing (§5.1 + Phase 4 Stop-Source Unification):
    Supports `cfg.get("stop_source", "invalidation_distance")`:
    * `"invalidation_distance"` (default): uses `expert_output.invalidation_distance`.
    * `"exit_plan"`: uses `exit_plan.effective_stop_loss_pct` (falling back to
      `expert_output.exit_plan.effective_stop_loss_pct`, then `invalidation_distance`).
    """
    risk_amount = equity * (cfg.get("risk_pct_per_trade", 0.5) / 100)
    stop_source = str(cfg.get("stop_source", "invalidation_distance"))
    stop_distance_pct: float
    if stop_source == "exit_plan":
        plan = exit_plan if exit_plan is not None else getattr(expert_output, "exit_plan", None)
        if plan is not None:
            stop_distance_pct = float(plan.effective_stop_loss_pct)
        else:
            stop_distance_pct = float(expert_output.invalidation_distance)
    else:
        stop_distance_pct = float(expert_output.invalidation_distance)
    if stop_distance_pct <= 0:
        return 0.0
    return risk_amount / (stop_distance_pct / 100)


def liquidity_size_multiplier(depth_usdt: float, cfg: dict) -> float:
    """Smooth 0-1 reduction, not a hard reject (Opportunity Filter already
    hard-rejects the extreme cases) -- linear ramp between thin_threshold
    and full_size_threshold."""
    curve = cfg.get("liquidity_size_reduction_curve", {})
    thin = curve.get("thin_threshold_usdt", 20000)
    full = curve.get("full_size_threshold_usdt", 50000)
    if full <= thin:
        return 1.0
    if depth_usdt <= thin:
        return 0.0
    if depth_usdt >= full:
        return 1.0
    return (depth_usdt - thin) / (full - thin)


class RiskStateStore:
    """
    Persistent (SQLite-backed) circuit-breaker state -- survives process
    restarts by design (§5.3, §14 test_circuit_breaker_persists_across_restart).
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        persistent: bool = True,
        clock: Optional[Clock] = None,
    ):
        """``persistent=False`` keeps the state in a private in-memory
        database that dies with the store. ``clock`` defaults to ``WallClock``
        and can be ``BarClock`` in simulated backtests/replays."""
        self.db_path = db_path if persistent else MEMORY_PATH
        self._private = open_connection(MEMORY_PATH) if not persistent else None
        self.clock: Clock = clock or WallClock()

    def advance_clock(self, timestamp: float) -> None:
        self.clock.advance(timestamp)

    @contextmanager
    def _connect(self):
        """One pooled connection per store (WAL), or the private in-memory one."""
        if self._private is not None:
            try:
                yield self._private
                self._private.commit()
            except BaseException:
                self._private.rollback()
                raise
        else:
            with connect(self.db_path, wal=True) as conn:
                yield conn

    def _get(self, key: str, default=None):
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM risk_state WHERE key=?", (key,)).fetchone()
            return json.loads(row["value"]) if row else default

    def _set(self, key: str, value) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO risk_state (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, json.dumps(value), self.clock.now_iso()),
            )

    # ---- execution errors ----
    def record_execution_result(self, success: bool) -> int:
        count = 0 if success else self._get("consecutive_execution_errors", 0) + 1
        self._set("consecutive_execution_errors", count)
        return count

    def consecutive_execution_errors(self) -> int:
        return self._get("consecutive_execution_errors", 0)

    # ---- trade outcomes: consecutive losses + daily loss ----
    def record_trade_result(self, pnl_pct: float, pnl_usdt: float = 0.0) -> Dict:
        losses = self._get("consecutive_losses", 0)
        losses = losses + 1 if pnl_pct <= 0 else 0
        self._set("consecutive_losses", losses)

        today = self.clock.today_iso()
        stored_date = self._get("daily_loss_date")
        cumulative = self._get("daily_loss_pct", 0.0) if stored_date == today else 0.0
        if pnl_pct < 0:
            cumulative += abs(pnl_pct)
        self._set("daily_loss_date", today)
        self._set("daily_loss_pct", cumulative)

        return {"consecutive_losses": losses, "daily_loss_pct": cumulative}

    def consecutive_losses(self) -> int:
        return self._get("consecutive_losses", 0)

    def daily_loss_pct(self) -> float:
        today = self.clock.today_iso()
        stored_date = self._get("daily_loss_date")
        return self._get("daily_loss_pct", 0.0) if stored_date == today else 0.0

    # ---- equity / drawdown ----
    def update_equity(self, equity: float) -> float:
        peak = max(self._get("equity_peak", equity), equity)
        self._set("equity_peak", peak)
        self._set("current_equity", equity)
        return (peak - equity) / peak * 100 if peak > 0 else 0.0

    def current_drawdown_pct(self) -> float:
        equity = self._get("current_equity")
        peak = self._get("equity_peak")
        if equity is None or peak is None or peak <= 0:
            return 0.0
        return (peak - equity) / peak * 100

    # ---- data staleness (heartbeat) ----
    def is_stale(self, threshold_seconds: int) -> bool:
        last = self._get("last_bar_processed_at")
        if last is None:
            return False  # no data yet at all isn't "stale", it's "not started"
        last_dt = datetime.fromisoformat(last)
        return (self.clock.now_utc() - last_dt).total_seconds() > threshold_seconds

    def heartbeat(self) -> None:
        self._set("last_bar_processed_at", self.clock.now_iso())

    # ---- breaker trip/reset (independent of the counters above) ----
    def trip(self, breaker_type: str, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO circuit_breaker_halts (breaker_type, tripped_at, reason, active, reset_at) "
                "VALUES (?, ?, ?, 1, NULL) "
                "ON CONFLICT(breaker_type) DO UPDATE SET tripped_at=excluded.tripped_at, "
                "reason=excluded.reason, active=1, reset_at=NULL",
                (breaker_type, self.clock.now_iso(), reason),
            )

    def reset(self, breaker_type: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE circuit_breaker_halts SET active=0, reset_at=? WHERE breaker_type=?",
                (self.clock.now_iso(), breaker_type),
            )
        if breaker_type == "consecutive_execution_errors":
            self._set("consecutive_execution_errors", 0)
        if breaker_type == "consecutive_losses":
            self._set("consecutive_losses", 0)

    def is_tripped(self, breaker_type: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT active FROM circuit_breaker_halts WHERE breaker_type=?", (breaker_type,)
            ).fetchone()
            return bool(row["active"]) if row else False

    def active_halts(self) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM circuit_breaker_halts WHERE active=1").fetchall()
            return [dict(r) for r in rows]


class RiskLayer:
    def __init__(
        self,
        config: dict,
        store: Optional[RiskStateStore] = None,
        model_path: Optional[str] = None,
        data_granularity: str = "1m",
        kill_switch_path: str = "data/control/EMERGENCY_STOP",
        hold_policy: Optional[HoldPolicy] = None,
        require_aligned_models: bool = False,
    ):
        self.hold_policy = hold_policy
        self.require_aligned_models = require_aligned_models
        self.config = config
        self.sizing_cfg = config.get("sizing", {})
        self.hard_rules = config.get("hard_rules", {})
        self.breaker_cfg = config.get("circuit_breakers", {})
        ml_cfg = config.get("ml_model", {})
        self.blend_weight = ml_cfg.get("blend_weight", 0.0)
        self.data_granularity = data_granularity
        self.kill_switch_path = kill_switch_path

        self.store = store or RiskStateStore()
        self._reject_regimes = parse_regime_list(self.hard_rules.get("reject_on_regime", []))

        self._model = None
        self._model_feature_names = None
        if model_path:
            self._load_model(model_path)

        # Own switching-context tracking (§5.2's min_bars_between_switches
        # is a HARD rule here, independent of Router's soft-feature
        # tracking of the same concept -- deliberately duplicated, not
        # reused, so a bug in one doesn't silently disable the other).
        self._current_expert: Optional[str] = None
        self._bars_since_switch = 0

        # Trade rate limiting (Phase 5, §6.3): enforce max_trades_per_hour
        # and min_bars_between_orders from config.
        self._trade_timestamps: list = []  # timestamps of last N trades
        self._min_bars_between_orders = config.get("min_bars_between_orders", self.breaker_cfg.get("min_bars_between_orders", 2))
        self._bars_since_last_order = self._min_bars_between_orders  # start at threshold so first eval isn't blocked
        self._max_trades_per_hour = self.breaker_cfg.get("max_trades_per_hour", 10)

    def _load_model(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.info(f"No risk model at {p}; using pure hard-rules/liquidity-curve sizing")
            return
        try:
            with open(p, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load risk model at {p}: {e}")
            return

        trained_granularity = payload.get("granularity")
        if trained_granularity is not None and trained_granularity != self.data_granularity:
            logger.warning(
                f"Risk model trained on granularity={trained_granularity}, running on "
                f"{self.data_granularity}. Falling back to rule-based sizing only."
            )
            return

        trained_feature_version = payload.get("feature_version")
        if trained_feature_version is not None and trained_feature_version != FEATURE_SCHEMA_VERSION:
            logger.warning(
                f"Risk model trained on feature_schema={trained_feature_version}, "
                f"current schema={FEATURE_SCHEMA_VERSION}. Falling back to rule-based sizing only."
            )
            return

        problem = model_alignment_problem(
            payload, self.hold_policy, require_aligned=self.require_aligned_models,
        )
        if problem is not None:
            logger.warning(
                f"Risk model at {p} is not aligned with the running harness ({problem}). "
                f"Falling back to rule-based sizing only. Retrain with scripts/train_risk_model.py."
            )
            return

        self._model = payload["model"]
        self._model_feature_names = payload["feature_names"]
        logger.info(f"Loaded risk model from {p}")

    def _kill_switch_active(self) -> bool:
        return os.path.exists(self.kill_switch_path)

    def _update_switch_state(self, expert_name: Optional[str]) -> None:
        if expert_name != self._current_expert:
            self._current_expert = expert_name
            self._bars_since_switch = 0
        else:
            self._bars_since_switch += 1

    def evaluate(
        self,
        router_decision: RouterDecision,
        expert_outputs: ExpertOutputs,
        regime_state: RegimeState,
        zone_map: Optional[ZoneMap],
        fv: FeatureVector,
        market: MarketSnapshot,
        equity: float,
    ) -> RiskDecision:
        ts = router_decision.timestamp
        symbol = router_decision.symbol

        def reject(reason: str, risk_score: float = 1.0) -> RiskDecision:
            return RiskDecision(
                timestamp=ts, symbol=symbol, approved=False, risk_score=risk_score,
                base_size=0.0, adjusted_size=0.0, size_reduction_pct=100.0, rejection_reason=reason,
            )

        # 1. Manual kill switch -- checked first, no dependencies, can't be
        # broken by a bug anywhere else in this method.
        if self._kill_switch_active():
            return reject("manual_kill_switch_active")

        # 2. Persisted hard-halt breakers (independent of each other).
        for breaker_type in ("consecutive_execution_errors", "max_drawdown", "consecutive_losses_hard"):
            if self.store.is_tripped(breaker_type):
                return reject(f"circuit_breaker_active:{breaker_type}")

        # 3. Daily loss limit -- halts NEW entries only. This method is
        # only ever called for new-entry candidates (exits run through the
        # Execution Layer's own TP/SL/timeout mechanics, never back
        # through here), so "existing positions keep their SL/TP
        # management" falls out of the architecture rather than needing a
        # special case.
        if self.store.daily_loss_pct() >= self.breaker_cfg.get("daily_loss_limit_pct", 2.0):
            return reject("circuit_breaker_active:daily_loss_limit")

        # 4. Data staleness.
        if self.store.is_stale(self.breaker_cfg.get("data_staleness_halt_seconds", 120)):
            return reject("circuit_breaker_active:data_staleness")

        # 5. Nothing to evaluate.
        if router_decision.selected_expert == "no_trade":
            return reject("no_trade_selected", risk_score=0.0)

        expert_output = expert_outputs.get_by_name(router_decision.selected_expert)
        if expert_output is None or expert_output.action == ExpertAction.NO_TRADE:
            return reject("selected_expert_has_no_candidate", risk_score=0.0)

        # 5b. Trade rate limiting (Phase 5, §6.3): enforce max_trades_per_hour
        # and min_bars_between_orders from config.
        # Clean up old timestamps (older than 1 hour)
        current_ts = ts
        cutoff_hour_ago = current_ts - 3600
        self._trade_timestamps = [t for t in self._trade_timestamps if t > cutoff_hour_ago]
        if len(self._trade_timestamps) >= self._max_trades_per_hour:
            return reject(f"max_trades_per_hour({self._max_trades_per_hour}) reached")

        # Min bars between orders
        if self._bars_since_last_order < self._min_bars_between_orders:
            return reject(f"min_bars_between_orders({self._min_bars_between_orders}) not met")

        # 6. Hard rules: invalidation distance bounds.
        dist = expert_output.invalidation_distance
        if dist < self.hard_rules.get("min_invalidation_distance_pct", 0.0):
            return reject(f"invalidation_distance_too_tight({dist:.4f})")
        if dist > self.hard_rules.get("max_invalidation_distance_pct", float("inf")):
            return reject(f"invalidation_distance_too_wide({dist:.4f})")

        # 7. Hard rule: switching penalty (independent of Router's own
        # soft tracking of the same concept -- see __init__ note).
        min_bars_between_switches = self.hard_rules.get("min_bars_between_switches", 0)
        would_switch = router_decision.selected_expert != self._current_expert
        if would_switch and self._current_expert is not None and self._bars_since_switch < min_bars_between_switches:
            self._update_switch_state(self._current_expert)  # rejected: don't count as a switch
            return reject(f"switched_too_recently(bars_since_switch={self._bars_since_switch})")

        # 8. Regime reject -- redundant with the Opportunity Filter on
        # purpose (defense in depth: two independent code paths have to
        # both be wrong for a panic-regime trade to get through).
        if regime_state.primary_regime in self._reject_regimes:
            self._update_switch_state(self._current_expert)
            return reject(f"regime_blocked({regime_state.primary_regime.value})")

        # --- Approved: compute sizing ---
        base_size = compute_base_size(equity, expert_output, self.sizing_cfg, exit_plan=getattr(expert_output, 'exit_plan', None))
        max_position = equity * (self.sizing_cfg.get("max_position_pct_of_equity", 100.0) / 100)
        base_size = min(base_size, max_position)

        depth_usdt = (market.bid_volume + market.ask_volume) * market.close
        liq_mult = liquidity_size_multiplier(depth_usdt, self.hard_rules)

        # Soft consecutive-loss pause: halve size on top of everything else.
        soft_pause_mult = 1.0
        if self.store.consecutive_losses() >= self.breaker_cfg.get("consecutive_losses_soft_pause", 3):
            soft_pause_mult = 0.5

        adjusted_size = base_size * liq_mult * soft_pause_mult
        size_reduction_pct = (1 - (adjusted_size / base_size)) * 100 if base_size > 0 else 0.0

        ml_risk_component = None
        if self._model is not None:
            try:
                features = build_features(expert_output, router_decision, regime_state, fv, market)
                row = np.array([[features[name] for name in self._model_feature_names]])
                ml_risk_component = float(self._model.predict_proba(row)[0][1])  # P(stop-out)
            except Exception as e:
                logger.warning(f"Risk model inference failed, using rule-based risk_score only: {e}")

        rule_based_risk = 1.0 - liq_mult  # 0 = ample liquidity, 1 = thin
        if ml_risk_component is not None:
            w = self.blend_weight
            risk_score = (1 - w) * rule_based_risk + w * ml_risk_component
            apply_to_size = bool(
                (self.config.get("ml_model") or {}).get("apply_to_size", False)
                or (self.sizing_cfg.get("ml_risk_sizing") or {}).get("enabled", False)
            )
            if apply_to_size:
                from ..sizing.risk_model import risk_score_to_size_multiplier
                min_mult = float((self.sizing_cfg.get("ml_risk_sizing") or {}).get("min_multiplier", 0.20))
                ml_mult = risk_score_to_size_multiplier(ml_risk_component, min_multiplier=min_mult)
                adjusted_size = adjusted_size * ml_mult
                size_reduction_pct = (1 - (adjusted_size / base_size)) * 100 if base_size > 0 else 0.0
        else:
            risk_score = rule_based_risk

        self._update_switch_state(router_decision.selected_expert)

        return RiskDecision(
            timestamp=ts, symbol=symbol, approved=True, risk_score=float(np.clip(risk_score, 0.0, 1.0)),
            base_size=base_size, adjusted_size=adjusted_size, size_reduction_pct=float(size_reduction_pct),
            rejection_reason=None,
        )

    def record_trade(self, timestamp: int) -> None:
        """Record a successful trade for rate limiting."""
        self._trade_timestamps.append(timestamp)
        self._bars_since_last_order = 0

    def advance_bar(self) -> None:
        """Called each bar to update bar counters."""
        self._bars_since_last_order += 1
