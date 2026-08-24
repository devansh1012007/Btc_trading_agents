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

Compares reference vs. current feature distributions:

```python
def check_feature_drift(reference, current, psi_alert_threshold=0.25):
    results = {}
    for feature_name in reference:
        if feature_name not in current:
            continue
        psi = population_stability_index(
            np.array(reference[feature_name]),
            np.array(current[feature_name])
        )
        results[feature_name] = {"psi": psi, "alert": psi >= psi_alert_threshold}
    return results
```

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
- Sustained low regime confidence (< 35%)

## Performance Monitor (`src/monitoring/performance_monitor.py`)

Compares live vs. backtest expectancy per component. Detects when live performance deviates significantly from backtest expectations.

```python
def check_expectancy_gap(backtest_expectancy, live_expectancy, threshold_pct=30.0):
    """Alert if live expectancy differs from backtest by > threshold_pct%."""
    if backtest_expectancy == 0:
        return {"alert": False, "gap_pct": 0.0}
    gap_pct = abs(live_expectancy - backtest_expectancy) / abs(backtest_expectancy) * 100
    return {"alert": gap_pct >= threshold_pct, "gap_pct": gap_pct}
```

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

Flushes `structure_engine.lifecycle_events` to Parquet. Append via read-concat-write pattern:

```python
def flush_zone_events(lifecycle_events, path="data/logs/zone_events.parquet"):
    if not lifecycle_events:
        return
    
    new_df = pd.DataFrame(lifecycle_events)
    if Path(path).exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    
    combined.to_parquet(path, compression="snappy")
```

### Event Types

`created`, `touched`, `rejected`, `broke`, `merged`, `decayed`. Each event contains: `event_type`, `zone_id`, `timestamp`, `bar_index`, and event-specific fields.

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

One record per bar — the full stage-by-stage journey through the pipeline. Written to `data/logs/decision_traces/` as partitioned JSONL.

```bash
# Full trace for a specific bar
python scripts/diagnose.py --timestamp 1755900000

# Summary breakdown of all bars
python scripts/diagnose.py --summary --since 2026-08-01
```

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

Manual CLI to reset tripped circuit breakers:

```bash
python scripts/reset_breaker.py max_drawdown
python scripts/reset_breaker.py all
```
