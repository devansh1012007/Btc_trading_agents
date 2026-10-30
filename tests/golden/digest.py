"""
Canonical JSON digest generator for golden regression tests (Phase 0, T0.2).

Produces a deterministic, canonical JSON digest of:
- trade list (entry/exit ts, prices, reason, bars/seconds held, pnl)
- equity curve
- per-bar terminal_stage sequence
- trade_duration report

Used to prove that default-off configuration switches leave existing
pipeline/lifecycle behavior bit-for-bit identical across all phases.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from src.evaluation.trade_duration import duration_report
from src.execution.position_lifecycle import (
    OpenPosition,
    build_execution_engine,
    process_bar,
)
from src.labeling.hold_policy import HoldPolicy, alignment_settings, parse_granularity_seconds
from src.layers.market_data import MarketDataLayer
from src.layers.risk_layer import RiskLayer, RiskStateStore
from src.pipeline import TradingPipeline
from src.utils.config_helpers import load_layered_config
from src.utils.synthetic import SYNTHETIC_START, synthetic_ohlcv

GOLDEN_DIR = Path(__file__).resolve().parent

SCENARIOS = {
    "paper_sim_1m_900": {"granularity": "1m", "n_bars": 900, "seed": 42},
    "paper_sim_10s_900": {"granularity": "10s", "n_bars": 900, "seed": 42},
}


def _relax_config(base_cfg: Mapping[str, Any], granularity: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(base_cfg))
    cfg.setdefault("system", {})["data_granularity"] = granularity
    cfg.setdefault("execution", {})["mode"] = "paper_sim"
    hr = cfg.setdefault("opportunity_filter", {}).setdefault("hard_rules", {})
    hr["max_spread_pct"] = 0.5
    hr["min_depth_usdt"] = 0.0
    hr["edge_vs_fees_threshold"] = -10.0
    return cfg


def _round_float(val: Any) -> Any:
    if isinstance(val, float):
        return round(val, 10)
    if isinstance(val, dict):
        return {str(k): _round_float(v) for k, v in sorted(val.items())}
    if isinstance(val, (list, tuple)):
        return [_round_float(v) for v in val]
    return val


def compute_digest(
    granularity: str = "1m",
    n_bars: int = 900,
    seed: int = 42,
    config_overrides: Optional[Mapping[str, Any]] = None,
    initial_equity: float = 10_000.0,
) -> Dict[str, Any]:
    """Run a seeded synthetic session through the full stack and return its canonical digest."""
    base_cfg = load_layered_config("paper")
    cfg = _relax_config(base_cfg, granularity)
    if config_overrides:
        from src.utils.config_helpers import _deep_merge
        cfg = _deep_merge(cfg, dict(config_overrides))

    symbol = cfg.get("system", {}).get("symbol", "BTC/USDT")
    bar_sec = int(parse_granularity_seconds(granularity))
    df_ohlcv = synthetic_ohlcv(n_bars, seed=seed, start=SYNTHETIC_START, bar_seconds=bar_sec)

    tmp_dir = tempfile.mkdtemp(prefix="golden_digest_", dir="/tmp")
    try:
        pipeline = TradingPipeline(symbol=symbol, config=cfg, data_dir=f"{tmp_dir}/data")
        hold_policy, require_aligned = alignment_settings(cfg)
        risk_layer = RiskLayer(
            config=cfg.get("risk", {}),
            hold_policy=hold_policy,
            require_aligned_models=require_aligned,
            store=RiskStateStore(persistent=False),
            model_path=cfg.get("risk", {}).get("ml_model", {}).get("model_path"),
            data_granularity=granularity,
            kill_switch_path=f"{tmp_dir}/NO_KILL_SWITCH",
        )
        execution_engine = build_execution_engine(cfg, pipeline.connector, symbol)
        fee_pct = float(cfg.get("labeling", {}).get("fee_pct", 0.04))

        snapshots = MarketDataLayer.batch_create_from_dataframe(df_ohlcv, symbol)
        equity = float(initial_equity)
        open_pos: Optional[OpenPosition] = None
        trades: List[Dict[str, Any]] = []
        equity_curve: List[float] = []
        terminal_stages: List[str] = []
        bar_timestamps: List[int] = []

        def on_open(position: OpenPosition) -> str:
            return f"golden-{position.entry_timestamp}"

        def on_close(position: OpenPosition, exit_timestamp: int, exit_info: dict) -> None:
            trades.append({
                "trade_id": position.trade_id,
                "entry_timestamp": int(position.entry_timestamp),
                "exit_timestamp": int(exit_timestamp),
                "entry_price": round(float(position.entry_price), 8),
                "exit_price": round(float(exit_info["exit_price"]), 8),
                "direction": int(position.direction),
                "size_usdt": round(float(position.size_usdt), 8),
                "take_profit_pct": round(float(position.take_profit_pct), 8),
                "stop_loss_pct": round(float(position.stop_loss_pct), 8),
                "exit_reason": str(exit_info["exit_reason"]),
                "bars_held": int(exit_info["bars_held"]),
                "hold_seconds": round(float(exit_info["hold_seconds"]), 6),
                "min_hold_bars": int(position.min_hold_bars),
                "pnl_pct": round(float(exit_info["realized_pnl_pct"]), 8),
                "pnl_usdt": round(float(exit_info["realized_pnl_usdt"]), 8),
                "expert": str(position.expert),
                "regime": str(position.regime),
            })

        for snapshot in snapshots:
            state = pipeline.run_live_bar(snapshot)
            if state is None:
                continue
            outcome = process_bar(
                state,
                risk_layer=risk_layer,
                execution_engine=execution_engine,
                position=open_pos,
                equity=equity,
                fee_pct=fee_pct,
                config=cfg,
                on_trade_open=on_open,
                on_trade_close=on_close,
            )
            open_pos, equity = outcome.position, outcome.equity
            terminal_stages.append(outcome.terminal_stage)
            equity_curve.append(round(float(equity), 8))
            bar_timestamps.append(int(outcome.state.timestamp))

        dur_rep = _round_float(duration_report(trades, HoldPolicy.from_config(cfg)))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    payload = {
        "granularity": granularity,
        "n_bars": n_bars,
        "seed": seed,
        "processed_bars": len(bar_timestamps),
        "first_timestamp": bar_timestamps[0] if bar_timestamps else None,
        "last_timestamp": bar_timestamps[-1] if bar_timestamps else None,
        "final_equity": round(float(equity), 8),
        "trades": trades,
        "equity_curve": equity_curve,
        "terminal_stages": terminal_stages,
        "trade_duration": dur_rep,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def golden_path(scenario_name: str) -> Path:
    return GOLDEN_DIR / f"{scenario_name}.json"


def load_golden_digest(scenario_name: str) -> Dict[str, Any]:
    path = golden_path(scenario_name)
    return json.loads(path.read_text(encoding="utf-8"))


def diff_digests(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> str:
    """Return a human-readable diff between two digests (empty string if identical)."""
    if expected == actual:
        return ""
    lines: List[str] = []
    for key in ("granularity", "n_bars", "processed_bars", "final_equity", "sha256"):
        if expected.get(key) != actual.get(key):
            lines.append(f"  {key}: expected={expected.get(key)!r} != actual={actual.get(key)!r}")
    exp_trades = expected.get("trades", [])
    act_trades = actual.get("trades", [])
    if len(exp_trades) != len(act_trades):
        lines.append(f"  trades count: expected={len(exp_trades)} != actual={len(act_trades)}")
    for i, (et, at) in enumerate(zip(exp_trades, act_trades)):
        if et != at:
            lines.append(f"  trade[{i}] differs:\n    expected={et}\n    actual  ={at}")
            break
    exp_stages = expected.get("terminal_stages", [])
    act_stages = actual.get("terminal_stages", [])
    if exp_stages != act_stages:
        for i, (es, as_) in enumerate(zip(exp_stages, act_stages)):
            if es != as_:
                lines.append(f"  terminal_stages[{i}]: expected={es!r} != actual={as_!r}")
                break
    if expected.get("trade_duration") != actual.get("trade_duration"):
        lines.append(f"  trade_duration differs: expected={expected.get('trade_duration')} != actual={actual.get('trade_duration')}")
    return "\n".join(lines) if lines else "Digests differ in equity_curve or metadata"


def generate_golden_files() -> Dict[str, Path]:
    written: Dict[str, Path] = {}
    for name, params in SCENARIOS.items():
        digest = compute_digest(**params)
        path = golden_path(name)
        path.write_text(json.dumps(digest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written[name] = path
    return written


if __name__ == "__main__":
    out = generate_golden_files()
    for k, p in out.items():
        print(f"Wrote {k} -> {p}")
