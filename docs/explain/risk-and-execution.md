# Risk & Execution

## Risk Layer (`src/layers/risk_layer.py`)

### Responsibility

Protect system survival. Can override experts, router, and opportunity filter. Only called for **new entries**, never for position exits.

### Kill Switch (Checked First)

File existence check: `data/control/EMERGENCY_STOP`. If present, returns `approved=False, rejection_reason="manual_kill_switch_active"`. Zero dependencies — no model, no config parsing, no database.

### Circuit Breakers

Five independently tripped/reset breakers, all persisted in SQLite (`data/registry.db`). Schema auto-created on first connection.

| Breaker | Trigger | Reset | Persisted |
|---|---|---|---|
| `consecutive_execution_errors` | 3 consecutive failures | Manual | Yes |
| `max_drawdown` | Drawdown ≥ 8% | Manual | Yes |
| `consecutive_losses_hard` | 6 consecutive losses | Manual | Yes |
| `daily_loss_limit` | Daily loss ≥ 2% | Auto (new day) | Yes |
| `data_staleness` | No heartbeat for 120s | Auto (heartbeat resumes) | Yes |

#### `RiskStateStore`

SQLite-backed state. Context-managed connections with auto-commit.

```python
store.trip("max_drawdown", "drawdown=9.0%")  # Trip a breaker
store.is_tripped("max_drawdown")              # Check status
store.reset("max_drawdown")                   # Manual reset
store.record_trade_result(pnl_pct, pnl_usdt)  # Record trade outcome
store.record_execution_result(success=True)   # Record execution
store.heartbeat()                             # Reset data_staleness
store.update_equity(equity)                   # Update equity, return drawdown %
```

Verified: breakers persist across restarts (`test_circuit_breaker_persists_across_restart` kills and recreates a `RiskStateStore` against the same DB file).

### Hard Rules

| Rule | Default | Notes |
|---|---|---|
| `min_invalidation_distance_pct` | 0.1 | Entry must not be too close to invalidation |
| `max_invalidation_distance_pct` | 3.0 | Entry must not be too far |
| `min_bars_between_switches` | 3 | Soft switching limit (independent from Router) |
| `reject_on_regime` | [PANIC_SPIKE] | Defense in depth vs. Opportunity Filter |
| `liquidity_size_reduction_curve` | thin: 20K, full: 50K | Smooth ramp |

**Defense in depth**: Both Opportunity Filter and Risk Layer independently reject PANIC_SPIKE. If one gate fails, the other catches it.

### Position Sizing

```python
def compute_base_size(equity, expert_output, cfg):
    risk_amount = equity * cfg["risk_pct_per_trade"]  # 0.5%
    invalidation = expert_output.invalidation_distance  # % from config
    if invalidation == 0:
        return 0.0
    return risk_amount / (invalidation / 100)
```

Capped at `max_position_pct_of_equity` (20% of equity).

#### Liquidity Size Reduction

Smooth ramp based on available depth:
- Depth < `thin_threshold_usdt` (20K): multiplier = 0.0 (no trade)
- Depth > `full_size_threshold_usdt` (50K): multiplier = 1.0 (full size)
- Between: linear interpolation

#### Soft Pause

After `consecutive_losses_soft_pause` (3) consecutive losses, size is halved. Resets on a winning trade.

### ML Risk Blend

Optional `P(stop-out)` model. Blend weight: 0.25 (25% model, 75% hard rules + liquidity curve). Same granularity guard as other models.

### Important: Never Called for Exits

In `run.py`, open-position exit checking (TP/SL/timeout) happens **before** `RiskLayer.evaluate()` is called. The Risk Layer is only invoked when `open_position is None` — i.e., for new entry candidates. This is architectural, not just a code path. There is no exit call path to test against.

## Execution Engine (`src/execution/execution_engine.py`)

### Architecture

One interface, three swappable backends selected by `execution.mode` config:

```python
class ExecutionBackend(ABC):
    def get_liquidity(self, symbol, market) -> LiquidityEstimate: ...
    def place_order(self, risk_decision, router_decision, expert_output, market) -> ExecutionRecord: ...
```

Upstream layers (Risk, Router, Opportunity Filter, Experts) call `execution_engine.place_order(...)` and get back `ExecutionRecord` or `None` — they don't know or care which backend ran.

### `simulate_fill()`

Shared fill-simulation math for both paper backends:

```python
# BUY: mid + spread/2 + slippage
# SELL: mid - spread/2 - slippage
fill_price = mid * (1 + spread_frac/2 + slippage_pct/100)  # BUY
fill_price = mid * (1 - spread_frac/2 - slippage_pct/100)  # SELL

max_fillable_usdt = depth_usdt * participation_cap  # 15%
filled_size_usdt = min(requested_size_usdt, max_fillable_usdt)

status = "filled" if filled >= requested else ("partial" if filled > 0 else "rejected")
```

### `make_client_order_id()`

Deterministic hash for idempotent crash-restart recovery:

```python
raw = f"{timestamp}:{symbol}:{side}:{round(size, 6)}"
return hashlib.sha256(raw.encode()).hexdigest()[:32]
```

The same decision always hashes to the same id. On restart, the backend checks if the order already exists before placing.

### ExecutionEngine.place_order()

```python
def place_order(self, risk_decision, router_decision, expert_output, market):
    if not risk_decision.approved:
        return None
    if expert_output is None or expert_output.action == ExpertAction.NO_TRADE:
        return None
    return self.backend.place_order(risk_decision, router_decision, expert_output, market)
```

A rejected `RiskDecision` produces `None` regardless of backend.

## Backends

### PaperSim (`src/execution/backends/paper_sim_backend.py`)

Pure historical/backtest simulation. Liquidity from snapshot's synthetic liquidity model (set by `attach_synthetic_liquidity()`). No network calls.

### PaperLive (`src/execution/backends/paper_live_backend.py`)

Real order book for liquidity (via `fetch_real_liquidity()`), but simulated fills. Meaningfully better validation before touching real capital. Falls back to snapshot's liquidity on network failure.

### Live (`src/execution/backends/live_backend.py`)

Real CCXT `create_order`. Three independent safety gates:

1. **Two-flag gate**: `execution.mode == "live"` AND `execution.confirm_real_money == true` required at construction. Raises `LiveTradingNotConfirmedError` if either is missing.
2. **API keys never in config**: Only env-var names are configured; connector must be authenticated by caller.
3. **Idempotent placement**: `clientOrderId` checked against orders table before `create_order`. Crash-and-restart replays the same decision into the same id.

#### Idempotency

```python
existing = self._find_existing_order(client_order_id)
if existing is not None:
    # Return ExecutionRecord with existing exchange_order_id
    return ExecutionRecord(..., order_id=existing["exchange_order_id"])

# Only place if not found
order = self.connector.exchange.create_order(...)
```

Verified: `test_execution_id_execution_idempotent_order_placement` simulates crash-and-restart with mocked exchange — `create_order` is called exactly once.

#### Order Recording

Orders recorded to SQLite before and after `create_order`:
1. Before: `status="pending", exchange_order_id=""`
2. After: `status=<actual>, exchange_order_id=<exchange_id>`
3. On failure: `status="rejected", exchange_order_id=""`

## Reconciliation (`src/execution/reconciliation.py`)

Run on every process start and periodically in live mode. Compares bot's known open orders against exchange's open orders.

```python
def reconcile(connector, symbol, db_path):
    exchange_order_ids = {o["id"] for o in connector.exchange.fetch_open_orders(symbol)}
    
    with connect(db_path) as conn:
        known_open = conn.execute("SELECT * FROM orders WHERE status NOT IN ('filled', 'cancelled', 'rejected')")
    
    discrepancies = []
    for row in known_open:
        if row["exchange_order_id"] not in exchange_order_ids:
            discrepancies.append({
                "client_order_id": row["client_order_id"],
                "exchange_order_id": row["exchange_order_id"],
                "bot_believed_status": row["status"],
                "issue": "order not found — likely filled or cancelled while bot was down"
            })
```

Tested against mocked exchange (`test_reconciliation_detects_drift`). Never exercised against a real exchange.

## Position Lifecycle (`scripts/run.py`)

### Historical Mode

```python
for snapshot in snapshots:
    # 1. Check open position exits (TP/SL/timeout) — NOT gated by Risk Layer
    if open_position is not None:
        exit_info = check_position_exit(open_position, bar, fee_pct)
        if exit_info:
            record_trade_exit(...)
            equity += exit_info["realized_pnl_usdt"]
            open_position = None
    
    # 2. Only look for NEW entry if nothing open
    if open_position is None and state.router_decision is not None:
        risk_decision = risk_layer.evaluate(...)
        if risk_decision.approved:
            execution_record = execution_engine.place_order(...)
            if execution_record.status in ("filled", "partial"):
                open_position = OpenPosition(...)
```

### Live Mode

`run_live_loop()` is a **documented structural stub** — it sets up reconciliation and kill-switch check, then explicitly declines to synthesize a working live poll loop against a feed that doesn't exist in this environment. The actual per-bar logic can be lifted from `run_historical()`'s loop body.
