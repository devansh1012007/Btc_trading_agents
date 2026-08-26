# Monitoring & Feedback

## Feature Drift Monitor (`src/monitoring/drift_monitor.py`)

### Population Stability Index (PSI)

Standard convention: under 0.1 = stable, 0.1–0.25 = moderate shift, over 0.25 = significant.

```python
def population_stability_index(reference, current, n_bins=10):
    # Quantile-binned: edges from reference distribution's percentiles
    edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
```

**Key design**: Edges are derived from the reference distribution's quantiles, so each reference bin has roughly equal weight. This prevents empty bins and division-by-zero issues.

### `check_feature_drift()`

Compares reference vs. current feature distributions; returns `{feature: {"psi": float, "alert": bool}}` for every feature present in both dicts.

> **Status**: implemented and unit-tested, but nothing schedules it — wiring a daily job (the config's `monitoring.drift.check_schedule` is aspirational) is deferred work.

### `check_regime_distribution()`

Production-health check for unusual regime patterns:

```python
def check_regime_distribution(recent_regimes, panic_frequency_alert_threshold=0.15,
                               low_confidence_alert_threshold=0.35, recent_confidences=None):
    panic_frequency = sum(1 for r in recent_regimes if r == "panic_spike") / len(recent_regimes)
    
    result = {
        "panic_frequency": panic_frequency,
        "panic_alert": panic_frequency > panic_frequency_alert_threshold,
        "avg_confidence": float(np.mean(recent_confidences)) if recent_confidences else None,
        "low_confidence_alert": avg_confidence < low_confidence_alert_threshold,
    }
    return result
```

Alerts on:
- High PANIC_SPIKE frequency (> 15% of recent regimes)
- Sustained low regime confidence (< 35%, only when confidences are supplied)

## Performance Monitor (`src/monitoring/performance_monitor.py`)

Compares live vs. validation expectancy per component (from trade-journal rows). Detects when live performance deviates significantly from walk-forward expectations.

```python
def compute_expectancy_gap(validation_expectancy_pct, live_trades_pnl_pct,
                           gap_alert_threshold_pct=30.0) -> dict:
    """live_trades_pnl_pct: realized pnl% of the most recent rolling window.
    gap = (validation - live) / |validation| * 100; alert when gap >= threshold
    (or unconditionally negative expectancy when validation == 0)."""

def expectancy_gap_by_component(trades, validation_expectancies,
                                rolling_window_trades=50,
                                gap_alert_threshold_pct=30.0) -> dict:
    """Groups journal rows by expert_used and applies the check per component."""
```

> **Status**: implemented and unit-tested, but nothing in the repo calls
> these functions yet — no scheduler or loop consumes them. Same for
> `drift_monitor.check_feature_drift()`.

## Trade Journal (`src/feedback/trade_journal.py`)

SQLite-backed (same DB as circuit breakers and model registry). Human-readable artifact — what you'd actually scroll through to sanity-check the bot.

### Functions

| Function | Purpose |
|---|---|
| `record_trade_entry(...)` | Insert open trade, return `trade_id` |
| `record_trade_exit(trade_id, ...)` | Update trade with exit info |
| `get_recent_trades(limit=50)` | Last N trades, ordered by entry_timestamp |
| `get_open_trades()` | Trades where `exit_timestamp IS NULL` |
| `summarize()` | Today's trade count, PnL, win rate |

### Schema

```sql
CREATE TABLE trade_journal (
    trade_id TEXT PRIMARY KEY,
    entry_timestamp INTEGER, entry_price REAL,
    exit_timestamp INTEGER, exit_price REAL, exit_reason TEXT,
    expert_used TEXT, regime_at_entry TEXT, direction INTEGER,
    size REAL, realized_pnl_pct REAL, realized_pnl_usdt REAL,
    fees_paid REAL, slippage_realized REAL,
    router_confidence REAL, risk_score_at_entry REAL, mode TEXT
);
```

## Zone Event Log (`src/feedback/zone_event_log.py`)

Flushes `structure_engine.lifecycle_events` to Parquet. Append via read-concat-write pattern; returns the total row count after the write. An empty event list is a no-op that still returns the existing count.

```python
def flush_zone_events(events, path="data/logs/zone_events.parquet") -> int: ...
def load_zone_events(path=...) -> pd.DataFrame  # empty typed frame if absent
```

### Event Types

`created`, `rejected`, `broke`, `merged`, `decayed` (from the lifecycle log — plain touches are not separate events). Each row carries: `zone_id`, `event_type`, `timestamp`, `bar_index`, `price_at_event`, `zone_strength_at_event`, `related_zone_id` (merges), and `regime_at_event`.

## System State Archive (`src/feedback/system_state_archive.py`)

Complete `SystemState` archive. Partitioned by (symbol, UTC date) — one Parquet file per partition. Every field of every dataclass is flattened (verified by `test_full_systemstate_archive_completeness`).

```python
def write_system_state_archive(states, base_dir="data/logs/system_state"):
    by_partition = defaultdict(list)
    for state in states:
        dt = datetime.fromtimestamp(state.timestamp, tz=timezone.utc)
        partition_key = (state.symbol.replace("/", "_"), dt.strftime("%Y-%m-%d"))
        by_partition[partition_key].append(system_state_to_dict(state))
    
    for (symbol, date_str), records in by_partition.items():
        path = Path(base_dir) / symbol / f"{date_str}.parquet"
        # Append via read-concat-write
        ...
```

### Partitioning

Adding a second trading pair later is a new partition, not a schema change. Path: `data/logs/system_state/BTC_USDT/2026-01-01.parquet`.

### `realized_pnl` on Every Bar

`SystemState.realized_pnl` is populated on every bar — 0.0 on no-trade bars, not `None`. This is critical for future training (contextual bandit / PPO needs (state, action, reward) every bar).

## Decision Traces (`src/diagnostics/decision_trace.py`)

One record per bar — the full stage-by-stage journey through the pipeline. Written to `data/logs/decision_traces/` as partitioned JSONL. Each `StageResult` carries optional `duration_ms`, and each trace carries `total_duration_ms` / `stage_timings_ms` (populated by the pipeline's per-stage timing).

```bash
# Full trace for a specific bar
python scripts/diagnose.py --timestamp 1755900000

# Summary breakdown of all bars
python scripts/diagnose.py --summary --since 2026-08-01

# Full T±window stage-by-stage replay of one journaled trade
python scripts/replay_trade.py --list 5
python scripts/replay_trade.py --trade-id <id>
```

### Structured Log Streams

| Stream | Contents |
|---|---|
| `data/logs/system.jsonl` | Every log line as JSON (`setup_logging()`) |
| `data/logs/decisions.jsonl` | One §8-shaped record per bar: regime/expert/opportunity/router/risk/decision (`log_trade_decision()`) |
| `data/logs/decision_traces/` | Stage-by-stage narrative + latency |
| `data/logs/faults.jsonl` | System faults with full context |

### Fault Log

`data/logs/faults.jsonl` — full context, not just the traceback, so a fault from three days ago is actually reproducible.

## Experiment Log (`src/registry/db.py`)

Every backtest and training run is logged to `data/registry.db::experiments`:

```sql
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL,          -- "backtest" | "training"
    started_at TEXT NOT NULL,
    config_hash TEXT NOT NULL,       -- content hash of the config used
    config_snapshot TEXT NOT NULL,   -- the actual YAML config (truncated)
    component TEXT,                  -- component name (e.g. "router", "full_stack")
    metrics_json TEXT,               -- JSON of resulting metrics
    git_commit TEXT                  -- git commit if available
);
```

This is the literal answer to "data collection for future agents": a growing, queryable history of exactly which configurations were tried and what happened, valuable both for manual tuning and as a training signal for a future hyperparameter-search agent or bandit-style tuner — the config space *is* an action space, and this table is its outcome log.

## Status Script (`scripts/status.py`)

One-glance health check:

```
Mode: paper / live
Breakers: max_drawdown=active, daily_loss_limit=inactive
Pending models: router_v2.pkl (pending_approval)
Today's trades: 5, PnL: +$123.45, Win rate: 60%
```

Queries: `risk_state`, `circuit_breaker_halts`, `model_versions`, `trade_journal`.

## Breaker Reset (`scripts/reset_breaker.py`)

Manual CLI to reset a tripped circuit breaker (explicit `--type`, no "reset all"):

```bash
python scripts/reset_breaker.py --type max_drawdown
python scripts/reset_breaker.py --type consecutive_execution_errors
# valid types: consecutive_execution_errors | max_drawdown | consecutive_losses_hard
#              | daily_loss_limit | data_staleness
```

Resetting `consecutive_execution_errors` / `consecutive_losses_hard` also zeroes their underlying counters.
