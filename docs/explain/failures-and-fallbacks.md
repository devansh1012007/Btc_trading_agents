# Failures, Fallbacks & Resilience

## Failure Taxonomy

| Category | Components | Strategy |
|---|---|---|
| **Model absence** | Regime, Structure, Experts, Opportunity Filter, Router, Risk | Rule-based fallback |
| **Model mismatch** | All trainable models | Granularity guard → fallback |
| **Model inference error** | All ML models | Catch exception → fallback |
| **Network failure** | CCXT connector, real orderbook adapter | Retry with backoff, fallback to synthetic |
| **Exchange API failure** | LiveBackend order placement | Record as rejected, increment error counter |
| **State corruption** | SQLite registry | Context-managed connections, auto schema creation |
| **Process crash** | Circuit breakers, positions | SQLite persistence, idempotent order replay |
| **Data staleness** | Live loop | Heartbeat → data_staleness breaker |
| **Manual intervention** | Any component | Kill switch file |

## Model Fallback Chains

Every trainable model follows the same pattern: try model → if absent/mismatch/error → fallback.

### Regime Detector

```
LightGBM classifier (granularity check)
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
    → LightGBM model (granularity check)
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
ML model (if available, granularity check)
    → Highest-confidence expert above confidence_floor
    → no_trade if no expert qualifies
```

Model absent → fallback. Model inference error → fallback.

### Risk Layer

```
Kill switch (file check, zero dependencies)
    → Circuit breakers (SQLite check)
    → Hard rules
    → ML risk blend (if available, granularity check)
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

Single file (`data/registry.db`) with 5 tables:

| Table | Purpose |
|---|---|
| `risk_state` | Current risk state (equity, drawdown, etc.) |
| `circuit_breaker_halts` | Tripped breakers (breaker_type, tripped_at, reason, active) |
| `trade_journal` | Open and closed trades |
| `model_versions` | Model registry (pending → approved → superseded) |
| `orders` | Idempotent order tracking |

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

## Known Failure Modes

### volatility_30s Always Zero

`FeatureEngine._volatility(prices, periods=1)` computes `np.std()` over a single-element array. The standard deviation of one number is zero by definition. Every training script scaled its barriers by `k_tp/k_sl * volatility_30s`, so every barrier collapsed to zero width.

**Fix applied**: Triple-barrier labeling uses `barrier_volatility_pct()` which uses `volatility_5m` instead. Applied at all six call sites.

**Left unfixed**: `FeatureEngine.volatility_30s` itself is unchanged — fixing it is outside Phase 3's stated scope.

### Depth Unit Mismatch (Fixed in Phase 4)

`bid_volume`/`ask_volume` are in base currency (BTC), but `min_depth_usdt` is in quote currency (USDT). Direct comparison was wrong.

**Fix**: Both `evaluate()`'s hard rule and `_liquidity_score()` now multiply by `market.close` before comparing.

### max_trades_per_hour Not Enforced

Configured in `settings.base.yaml` (`max_trades_per_hour: 10`) but not enforced anywhere in code. `max_concurrent_positions: 1` IS enforced, but implicitly (via `run.py`'s single `open_position` variable), not by reading the config value.

### run_live_loop() Is a Stub

`run_live_loop()` in `run.py` sets up reconciliation and kill-switch check, then explicitly declines to synthesize a working live poll loop. The actual per-bar logic can be lifted from `run_historical()`'s loop body. This is documented honesty, not a bug.

### LiveBackend Never Tested Against Real Exchange

`LiveBackend` and `reconciliation.py` are structurally complete and unit-tested against a mocked exchange. But "works against a mock" and "works against Binance" are different claims. The safety gates (`test_live_mode_requires_both_flags`) and idempotency logic (`test_execution_idempotent_order_placement`) are real tests against real code paths — but they need real credentials to validate end-to-end.
