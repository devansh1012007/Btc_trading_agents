# Failures, Fallbacks & Resilience

## Failure Taxonomy

| Category | Components | Strategy |
|---|---|---|
| **Model absence** | Regime, Structure, Experts, Opportunity Filter, Router, Risk | Rule-based fallback |
| **Model mismatch** | All trainable models | Granularity guard → fallback |
| **Feature schema mismatch** | All trainable models | `feature_version` guard → fallback (Phase 5) |
| **Model inference error** | All ML models | Catch exception → fallback |
| **Network failure** | CCXT connector, real orderbook adapter | Retry with backoff, fallback to synthetic |
| **Exchange API failure** | LiveBackend order placement | Record as rejected, increment error counter |
| **State corruption** | SQLite registry | Context-managed connections, auto schema creation |
| **Process crash** | Circuit breakers, positions | SQLite persistence, idempotent order replay |
| **Data staleness** | Live loop | Heartbeat → data_staleness breaker |
| **Manual intervention** | Any component | Kill switch file |
| **Source-tree drift** | `run.py`, backtest harness, latency benchmark | None at HEAD — see "Source-Tree Regression" below; restore newer files from the `tests/src/` mirror |

## Model Fallback Chains

Every trainable model follows the same pattern: try model → if absent/mismatch/error → fallback.

### Regime Detector

```
LightGBM classifier (granularity check, feature_version check)
    → HMM filter (self-contained numpy, no hmmlearn dependency)
    → Rule-based (momentum thresholds)
```

Each step checks `granularity` mismatch. Classifier inference error → falls to HMM. HMM error → falls to rules. Always available even with no trained models.

### Structure Engine

```
Beta-Bernoulli survival (rule-based, always available)
    + zone_survival_blend_weight × LightGBM zone validity model
```

If zone survival model is absent or granularity mismatch → pure rule-based survival. Model inference error → rule-based only.

### Experts

```
propose_candidate() → None → NO_TRADE (model never called)
propose_candidate() → action → score()
    → LightGBM model (granularity check, feature_version check)
    → Config fallback (fallback_confidence, fallback_expected_return, fallback_expected_risk)
```

Model inference error → fallback mode. Each expert has its own fallback config:

| Expert | Fallback Confidence | Fallback Expected Return | Fallback Expected Risk |
|---|---|---|---|
| Trend | 0.6 | 0.08 | 0.12 |
| Mean Reversion | 0.55 | 0.06 | 0.10 |
| Breakout | 0.50 | 0.10 | 0.15 |

### Opportunity Filter

```
Hard rules (always evaluated)
    → If hard rules fail → NOT tradable (model never consulted)
    → If hard rules pass → ML blend (if available)
```

Model absent → pure hard rules. Model inference error → pure hard rules. Model says tradable but hard rules fail → **hard rules always win**.

### Router

```
ML model (if available, granularity check, feature_version check)
    → Highest-confidence expert above confidence_floor
    → no_trade if no expert qualifies
```

Model absent → fallback. Model inference error → fallback.

### Risk Layer

```
Kill switch (file check, zero dependencies)
    → Circuit breakers (SQLite check)
    → Hard rules
    → ML risk blend (if available, granularity check, feature_version check)
    → Liquidity curve + soft pause
```

Model absent → pure hard rules + liquidity curve. Model inference error → rule-based risk score.

## Network Resilience

### CCXT Connector

`enableRateLimit: True` on all exchange instances. CCXT handles rate limiting internally.

### Synthetic Liquidity

Used for backtesting when no real order book exists. Feature-conditioned: spread widens with volatility, depth scales with volume. No network calls.

### Real Orderbook Adapter

PaperLiveBackend falls back gracefully on network failure:

```python
try:
    return fetch_real_liquidity(self.connector, symbol)
except Exception as e:
    logger.warning(f"Real order-book fetch failed, falling back to snapshot's existing liquidity: {e}")
    return LiquidityEstimate(
        spread_pct=market.spread_pct,
        bid_volume=market.bid_volume,
        ask_volume=market.ask_volume,
        source=market.liquidity_source,
    )
```

### Docker Entrypoint Retry

Live mode checks network + exchange reachability before starting:

```bash
retry_network(max_retries=5, base_delay=10, backoff=2)
# Attempts: 10s, 20s, 40s, 80s, 160s
```

If exchange unreachable after retries → aborts live mode with actionable error messages.

## State Persistence

### SQLite Registry

Single file (`data/registry.db`) with 6 tables:

| Table | Purpose |
|---|---|
| `risk_state` | Current risk state (equity, drawdown, etc.) |
| `circuit_breaker_halts` | Tripped breakers (breaker_type, tripped_at, reason, active) |
| `trade_journal` | Open and closed trades |
| `model_versions` | Model registry (pending → approved → superseded) |
| `orders` | Idempotent order tracking |
| `experiments` | Experiment log (config hash, metrics, git commit) — Phase 5 |

Context-managed connections with auto-commit. Schema auto-created on first connection. Survives process restart.

### Circuit Breaker Persistence

Verified: `test_circuit_breaker_persists_across_restart` kills and recreates a `RiskStateStore` against the same DB file — breakers remain tripped.

### Idempotent Order Placement

LiveBackend checks orders table before `create_order`:

```python
existing = self._find_existing_order(client_order_id)
if existing is not None:
    # Return ExecutionRecord with existing exchange_order_id
    # NOT re-placing the order
```

Verified: `test_execution_idempotent_order_placement` — crash-and-restart with mocked exchange, `create_order` called exactly once.

## Graceful Degradation

### OHLCV-Only Replay

When no ticker data is available (every call from training/backtest scripts), `MarketDataLayer.create_from_bar()` synthesizes bid/ask from midpoint with zero depth. Phase 4's `attach_synthetic_liquidity()` replaces this after features are computed.

**Known issue**: Synthetic model's mean spread (~0.09%) may exceed `max_spread_pct: 0.05%`, causing every bar to fail the spread check. This needs real spread data for calibration.

### No Trained Models

If no models are trained yet (empty `data/models/`), the system runs in pure fallback mode:
- Regime: rule-based (momentum thresholds)
- Structure: pure rule-based zones
- Experts: config fallback confidence/EV
- Opportunity Filter: pure hard rules
- Router: highest-confidence expert above floor

The system is **fully functional** in fallback mode — it just won't make trades that pass the opportunity filter (which is correct behavior).

### Pre-Phase-5 Models Fall Back (Phase 5)

All models trained before Phase 5 have `feature_version` pre-rename. They will silently fall back to rule-based mode until manually retrained. This is safe — the fallback paths are tested and work correctly.

## Error Handling Patterns

### Exception Swallowing (Alerting)

`src/alerting/notifier.py` never crashes the caller:

```python
def notify(event_type, message):
    try:
        # Send Telegram message
        return True
    except Exception as e:
        logger.error(f"Alert failed for {event_type}: {e}")
        return False
```

Missing env vars → logs warning, drops notification. Telegram API failure → logs error, returns False.

### SQLite Connection Safety

```python
@contextmanager
def connect(db_path=DEFAULT_DB_PATH):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
```

Always closes, even on exception. Auto-commits on clean exit.

### Parquet Append Safety

```python
try:
    df = pd.read_parquet(f)
    dfs.append(df)
except Exception as e:
    logger.warning(f"Could not load {f}: {e}")
```

Corrupt parquet files are skipped with a warning, not a crash.

## Decision Tracing (Phase 5)

Every bar produces a `DecisionTrace` with `StageResult` per stage. Written to `data/logs/decision_traces/` as partitioned JSONL.

```bash
# Full trace for a specific bar
python scripts/diagnose.py --timestamp 1755900000

# Summary breakdown of all bars
python scripts/diagnose.py --summary --since 2026-08-01

# Stage-by-stage trade replay (journal + traces + OHLCV)
python scripts/replay_trade.py --trade-id <id>
```

Categories of failure:
1. **Decision-layer rejections** — no expert candidate, opportunity filter veto, risk rejection, circuit breaker active. Information, not bugs — but needs to be legible and aggregable.
2. **System faults** — exception, connectivity failure, stale data feed. Logged to `data/logs/faults.jsonl` with full context.

## Known Failure Modes

### volatility_short Always Near Zero

`FeatureEngine._volatility(prices, periods=1)` computes `np.std()` over a single-element array. The standard deviation of one number is zero by definition. Every training script scaled its barriers by `k_tp/k_sl * volatility_short`, so every barrier collapsed to zero width.

**Fix applied**: Triple-barrier labeling uses `barrier_volatility_pct()` which uses `volatility_medium` instead. Applied at all six call sites.

**Left unfixed**: `FeatureEngine.volatility_short` itself is unchanged — fixing it is outside Phase 5's scope.

### Depth Unit Mismatch (Fixed in Phase 4)

`bid_volume`/`ask_volume` are in base currency (BTC), but `min_depth_usdt` is in quote currency (USDT). Direct comparison was wrong.

**Fix**: Both `evaluate()`'s hard rule and `_liquidity_score()` now multiply by `market.close` before comparing.

### Connector Exception Behavior (post-Phase 5)

`src/connectors/data_source.py`'s fetch methods catch bare `Exception`, log the error, and then **re-raise** — nothing is swallowed anymore. Callers (e.g., `run.py`'s live loop) are responsible for catching transient failures; a programming error such as an `AttributeError` surfaces immediately instead of masquerading as "no data".

### consecutive_losses_hard Breaker Is Dead

Configured as `consecutive_losses_hard_halt: 6` in `settings.base.yaml`, but no production code ever calls `trip("consecutive_losses_hard")`. The consistency validator (`validate_config_consistency.py`) catches this and **raises** `ConfigConsistencyError` — one of the reasons preflight fails on shipped defaults (see above). Soft mitigation exists: after `consecutive_losses_soft_pause: 3` losses, position size is halved, but a true hard halt on loss streaks is unimplemented.

### Feature Schema Mismatch (Phase 5)

FeatureVector field names changed in Phase 5: `momentum_5m` → `momentum_medium`, `volatility_5m` → `volatility_medium`, etc. Pre-rename models reference old names. Without the `feature_version` guard, `getattr(fv, "momentum_5m", 0.0)` would silently return 0.0.

**Fix**: All layer model loaders check `feature_version` against `FEATURE_SCHEMA_VERSION`. Pre-rename models fall back to rule-based mode.

### max_concurrent_positions > 1 Raises Error

Configured as `max_concurrent_positions: 1` in `settings.base.yaml`. If set to > 1, `run.py` raises `ValueError` with a clear message. Real multi-position support is a named future item.

### run_live_loop() Is Now Implemented (Phase 5)

`run_live_loop()` in `run.py` is now a real implementation — not a stub. It fetches latest OHLCV bar, runs through the full pipeline, and respects all circuit breakers.

### LiveBackend Never Tested Against Real Exchange

`LiveBackend` and `reconciliation.py` are structurally complete and unit-tested against a mocked exchange. But "works against a mock" and "works against Binance" are different claims. The safety gates (`test_live_mode_requires_both_flags`) and idempotency logic (`test_execution_idempotent_order_placement`) are real tests against real code paths — but they need real credentials to validate end-to-end.

Use `scripts/validate_live_backend.py --testnet` to run the 5-step testnet validation runbook.

## Source-Tree Regression at HEAD (CRITICAL, read first)

The `tests/src/` directory is **not** test code — it is a full mirror of an
older `src/`, committed in `4070b5f` before the test reorganization into
`tests/{unit,integration,e2e}/`. Comparing the two trees shows the mirror is
*newer* than canonical `src/` for ~15 modules: it contains the
presentation-hardening pass (latency instrumentation, paper-sim realism
knobs, structured logging, turnover metric) that `src/` at HEAD is missing.
Several committed entrypoints depend on those missing pieces and therefore
crash on a clean checkout:

| Broken thing | Missing from `src/` | Symptom |
|---|---|---|
| `scripts/run.py:61` import | `src/utils/logging_setup.py` (absent; exists only as `tests/src/utils/logging_setup.py`) | `ModuleNotFoundError` at startup |
| `run.py` trace writes | `DecisionTrace.total_duration_ms` / `stage_timings_ms` fields (`src/diagnostics/decision_trace.py`) | `TypeError` on first `DecisionTrace(...)` |
| `run.py`, `benchmark_latency.py` | `TradingPipeline.last_bar_timings` per-stage timing property (`src/pipeline.py`) | `AttributeError` |
| `full_stack_walk_forward_backtest.py` | `compute_turnover()` (`src/reporting/performance_report.py`) | `ImportError` |
| `tests/unit/test_paper_sim_realism.py` | `simulated_latency_ms` / `order_rejection_rate` knobs (`src/execution/backends/paper_sim_backend.py`) | 2 test failures |

Measured state at HEAD: `pytest tests/unit tests/integration` → 90 passed,
2 failed (`test_paper_sim_realism`), 2 collection errors
(`test_logging_setup.py`, `test_phase5.py` — both import the missing module).

**Recovery path**: restore the five affected files from the `tests/src/`
mirror (they are drop-in newer versions), or re-implement the small diffs
(a per-stage timing property on `TradingPipeline`, two optional latency
fields on `DecisionTrace`, one pure function in `performance_report.py`,
two config-keyed knobs in `PaperSimBackend`). Until then,
`make latency`, `make backtest`, and both `run.py` modes are unusable;
the perception pipeline itself (`pipeline.py`, all layers, experts) is
unaffected and fully tested.

## Built But Not Wired (dead infrastructure)

These modules are complete and unit-tested but have **no production call
site** — nothing instantiates or schedules them:

| Module | Status |
|---|---|
| `src/config/config_watcher.py` (`ConfigWatcher`) | Never instantiated — no loop checks mtimes, so hot-reload does not actually happen anywhere |
| `src/config/regime_adaptive.py` (`resolve_regime_param()`) | Tested but unused; production still reads `labeling.regime_specific_barriers` and `opportunity_filter.regime_overrides` via plain dict lookups |
| `src/monitoring/drift_monitor.py` (`check_feature_drift`, `check_regime_distribution`) | No scheduler/cron wiring exists in the repo |
| `src/monitoring/performance_monitor.py` (`compute_expectancy_gap`, `expectancy_gap_by_component`) | Same — defined, never called by any loop or script |
| `src/alerting/notifier.py` (`Notifier`) | Never constructed; breakers/retraining do not emit alerts yet |

Treat these as "available building blocks", not active behavior.

## Preflight Fails on Shipped Defaults (intentional tripwire)

`scripts/validate_config_consistency.py` (and therefore
`scripts/preflight.py`) **fails out-of-the-box** with the shipped
`settings.base.yaml`: the threshold-plausibility simulation reports ~98% of
simulated bars would fail `max_spread_pct=0.05` given the synthetic
liquidity coefficients (mean synthetic spread ≈ 0.09%). This is Finding #2's
residue wired as a gate — it refuses to bless a config that would reject
nearly every bar. Additionally, `validate_breaker_enforcement_paths()`
raises `ConfigConsistencyError` on `consecutive_losses_hard_halt` (a known
dead breaker, see below), which would also fail preflight if the spread
gate were relaxed first. Resolution requires either calibrating
`liquidity.synthetic.*` against real spreads (paper-live sample log +
`calibrate_synthetic_liquidity.py`) or deliberately raising
`max_spread_pct` — not silently loosening a risk gate to manufacture
activity.

## Model-Payload Key Gaps

Two trained-model pickles are **missing the `granularity` key**, which makes
the granularity guard inert for them (a `None` recorded granularity passes
the check unconditionally):

- `data/models/opportunity_filter.pkl` — `train_opportunity_filter.py` saves only `{model, feature_names, feature_version}`.
- `data/models/router.pkl` — same in `train_router.py`.

All other training scripts (regime HMM/classifier, zone survival, experts,
risk model) record `granularity`. Retraining those two components will not
fix this until their scripts add the key to the payload. The
`feature_version` guard still applies to both.
