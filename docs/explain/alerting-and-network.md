# Alerting & Network

## Alerting System (`src/alerting/notifier.py`)

### Design

Optional, off by default. Telegram backend. Never crashes the caller — errors are logged and the function returns `False`.

### Configuration

```yaml
alerting:
  enabled: false
  backend: "telegram"
  telegram_bot_token_env_var: "BTC_BOT_TELEGRAM_TOKEN"
  telegram_chat_id_env_var: "BTC_BOT_TELEGRAM_CHAT_ID"
  notify_on: [circuit_breaker_tripped, pending_model_approval, execution_error, daily_summary]
```

### Event Types

| Event | Triggered By |
|---|---|
| `circuit_breaker_tripped` | RiskLayer circuit breaker activation |
| `pending_model_approval` | `retrain_all.py` registers a pending model |
| `execution_error` | Execution backend failure |
| `daily_summary` | End-of-day trade summary |

### Implementation

```python
def notify(event_type: str, message: str) -> bool:
    if not config.get("enabled"):
        return False
    
    bot_token = os.environ.get(config["telegram_bot_token_env_var"])
    chat_id = os.environ.get(config["telegram_chat_id_env_var"])
    
    if not bot_token or not chat_id:
        logger.warning(f"Alerting configured but env vars missing for {event_type}")
        return False
    
    try:
        # Send Telegram message
        return True
    except Exception as e:
        logger.error(f"Alert failed for {event_type}: {e}")
        return False
```

### Failure Modes

| Failure | Behavior |
|---|---|
| `alerting.enabled: false` | Returns False immediately |
| Missing env vars | Logs warning, returns False |
| Telegram API failure | Logs error, returns False |
| Network timeout | Exception caught, logged, returns False |

**Never crashes the caller** — this is critical. An alerting failure must never take down the trading system.

## Network Resilience

### CCXT Rate Limiting

All CCXT exchange instances are created with `enableRateLimit: True`. CCXT handles rate limiting internally — automatic delays between requests, retry on 429 responses.

```python
self.exchange = ccxt.binance({
    'enableRateLimit': True,
    'sandbox': sandbox,
})
```

### Local Data Caching

`CCXTConnector.save_ohlcv_local()` and `load_ohlcv_local()` provide a local parquet cache. If the exchange is temporarily unavailable, the system can fall back to cached data.

```python
def save_ohlcv_local(self, df, symbol, data_dir="data/raw"):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    df['date'] = df['timestamp'].dt.date
    for date, group in df.groupby('date'):
        path = f"{data_dir}/{symbol.replace('/', '_')}_{date}.parquet"
        group.to_parquet(path, compression='snappy')

def load_ohlcv_local(self, symbol, date_range, data_dir="data/raw"):
    files = sorted(Path(data_dir).glob(f"{symbol.replace('/', '_')}_*.parquet"))
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not load {f}: {e}")
    # ... concat and return
```

Corrupt parquet files are skipped with a warning.

### Docker Network Retry

Live mode checks network + exchange reachability before starting:

```bash
retry_network(max_retries=5, base_delay=10, backoff=2)
```

Attempts at: 10s, 20s, 40s, 80s, 160s. If exchange unreachable after retries → aborts live mode with actionable error messages:

```
Cannot reach exchange after retries. Aborting live mode.
Options:
  1. Check your network connection
  2. Verify Binance API is accessible from this container
  3. Switch to historical mode: TRADING_MODE=historical
  4. Increase retry limits: NETWORK_RETRY_MAX=10 NETWORK_RETRY_BASE=15
```

### PaperLiveBackend Fallback

When real order book fetch fails:

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

Continues with synthetic liquidity from the snapshot. No trade is missed due to a transient network issue.

## Latency Considerations

### Bar Processing Time

Each bar goes through:
1. Feature computation (rolling buffer operations)
2. Regime detection (HMM filter or LightGBM inference)
3. Structure computation (zone touch detection, survival probability)
4. Expert evaluation (3× LightGBM inference or fallback)
5. Opportunity filter (rule evaluation + optional LightGBM)
6. Router decision (optional LightGBM + fallback)

All operations are in-memory. No I/O during bar processing. Latency is dominated by ML inference (LightGBM is fast — histogram-based, typically < 1ms per prediction).

### Live Loop Timing

Poll interval derived from `system.data_granularity`:

```python
GRANULARITY_SECONDS = {"10s": 10, "30s": 30, "1m": 60, "5m": 300}
poll_seconds = GRANULARITY_SECONDS.get(granularity, 60)
```

The live loop sleeps until the next bar boundary. In practice, the loop should process each bar in < 100ms, well within the granularity window.

### Reconciliation Timing

`reconciliation.py` runs on every process start and periodically thereafter in live mode. It fetches open orders from the exchange and compares against the bot's last-known state. This is a blocking operation — should complete in < 1s under normal conditions.

## Data Staleness

`data_staleness_halt_seconds` (default 120s) circuit breaker. The RiskLayer's `RiskStateStore` maintains a heartbeat:

```python
store.heartbeat()  # Called every bar in run.py
```

If no heartbeat for 120s, the breaker trips. Auto-resets on next heartbeat. This catches cases where the live loop is stuck (e.g., exchange API hanging indefinitely).

## Emergency Stop

File existence check: `data/control/EMERGENCY_STOP`. Checked **first** in `RiskLayer.evaluate()`, zero dependencies.

```python
def evaluate(self, ...):
    if Path(self.kill_switch_path).exists():
        return RiskDecision(approved=False, rejection_reason="manual_kill_switch_active", ...)
    # ... rest of evaluation
```

In Docker, this file can be created by:
```bash
docker exec btc-trading-agent touch /opt/trading-agent/data/control/EMERGENCY_STOP
```

Or on the host:
```bash
touch data/control/EMERGENCY_STOP
```

The live loop checks this file every iteration. No restart needed.
