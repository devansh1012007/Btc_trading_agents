# Configuration & Deployment

## Layered Config System

### Architecture

```
settings.base.yaml (full config)
        ↓ deep merge
settings.{env}.yaml (overlay)
        ↓
load_layered_config(env) → merged dict
```

`load_layered_config(env)` in `src/utils/config_helpers.py`:

```python
def load_layered_config(env="paper", config_dir="config"):
    base = _read_yaml(f"{config_dir}/settings.base.yaml")
    overlay = _read_yaml(f"{config_dir}/settings.{env}.yaml")
    return _deep_merge(base, overlay)

def _deep_merge(base, overlay):
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value  # Lists replaced wholesale
    return merged
```

### Key Behaviors

- **Dicts merge key-by-key** — nested config blocks are deep-merged
- **Lists are replaced wholesale** — overlay's list replaces base's list entirely
- **Non-dict values are replaced** — no special handling for scalars

### Who Uses What

| Script Type | Config Method | Notes |
|---|---|---|
| `run.py`, deployment scripts | `load_layered_config(args.env)` | Always pass `--env` |
| Training/backtest scripts | `--config settings.base.yaml` | Offline, env-agnostic |

## Config Files

### `settings.base.yaml` (318 lines)

Full configuration. All config blocks:

| Block | Purpose |
|---|---|
| `system` | data_granularity, exchange, sandbox, symbol |
| `data` | raw_dir, processed_dir, lookback_bars |
| `feature_engine` | momentum/volatility periods (short/medium/long), RSI period, SMA periods |
| `market` | min/max spread, min depth |
| `structure` | lookback_bars, pivot_window, zone parameters |
| `regime` | n_states, confidence_floor, persistence params, observation_features |
| `labeling` | max_holding_bars, fee_pct, slippage_buffer, k_tp/k_sl, regime_specific_barriers |
| `experts.trend/mr/bo` | Gate params, fallback params, model_path |
| `opportunity_filter` | hard_rules, ml_model, regime_overrides |
| `router` | model, blend_weight, fallback_behavior |
| `models` | zone_model, regime_hmm, regime_classifier, experts_dir |
| `backtest` | train_bars, test_bars, step_bars, embargo_bars |
| `liquidity` | synthetic params, real_order_book depth_limit |
| `risk` | sizing, hard_rules, circuit_breakers, ml_model |
| `execution` | mode, order_type, slippage, confirm_real_money, secrets |
| `retraining` | schedule, components, promotion_margin_pct, min_validation_samples |
| `monitoring` | drift psi_alert_threshold, performance expectancy_gap_alert_threshold |
| `alerting` | enabled, backend, telegram env vars, notify_on |
| `storage` | registry_db, system_state_archive, zone_events |
| `logging` | level, format |

### `settings.paper.yaml`

```yaml
system:
  sandbox: true

execution:
  mode: "paper_sim"
  confirm_real_money: false
```

### `settings.live.yaml`

```yaml
system:
  sandbox: false

execution:
  mode: "live"
  confirm_real_money: false  # <-- YOU flip this, deliberately, not this file
```

**Critical**: `confirm_real_money` is deliberately LEFT FALSE in both overlays. Per §6.4's design, `mode="live"` alone must never be enough to place a real order. You flip `confirm_real_money` to true yourself, explicitly, only when you've actually decided to go live. A single `--env live` on the command line can never be enough by itself.

## Key Config Parameters

### `system.data_granularity`

The **only** switch for timeframe — no code change needed. All layers check it for model-granularity compatibility.

Valid values: `"10s"`, `"30s"`, `"1m"`, `"5m"`. Default: `"1m"`.

### `feature_engine.momentum_periods`

Config-driven periods for momentum features. Phase 5 rename: field names are now period-agnostic (`momentum_short`, `momentum_medium`, `momentum_long`).

```yaml
feature_engine:
  momentum_periods: {short: 1, medium: 5, long: 10}
  volatility_periods: {short: 1, medium: 5}
```

### `labeling.fee_pct`

Binance taker fee. Default: `0.04` (0.04%). Adjust for your tier.

### `labeling.max_holding_bars`

Trade resolution window. Default: `5`. Wall-clock scales with granularity (5 bars at 1m = 5 minutes).

### `labeling.k_tp / k_sl`

Barrier multipliers. Default: `k_tp: 1.5`, `k_sl: 1.0`. Scaled by `barrier_volatility_pct()` (uses `volatility_medium`).

### `labeling.regime_specific_barriers`

Per-regime overrides (Phase 4 mechanism, still in use):

| Regime | k_tp | k_sl | Rationale |
|---|---|---|---|
| TRENDING_UP | 2.0 | 0.8 | Wider TP, tighter SL in trends |
| TRENDING_DOWN | 2.0 | 0.8 | Same |
| CHOP_RANGE | 1.0 | 1.0 | Symmetric |
| PANIC_SPIKE | 0.5 | 0.5 | Very tight — exit fast |

### `opportunity_filter.regime_overrides`

Per-regime overrides for hard rules (Phase 4 mechanism, still in use):

```yaml
opportunity_filter:
  regime_overrides:
    PANIC_SPIKE:
      edge_vs_fees_threshold: 0.05
    TRENDING_UP:
      volatility_ceiling_pct: 2.0
```

### `execution.mode`

Controls execution backend: `paper_sim` | `paper_live` | `live`. Set in env overlay.

### `execution.confirm_real_money`

Second, separate flag from mode. Both required for live orders. Guarded by `LiveTradingNotConfirmedError`.

### `risk.circuit_breakers.*`

| Parameter | Default | Enforced | Meaning |
|---|---|---|---|
| `max_drawdown_pct` | 8.0 | Yes | Halt if drawdown ≥ 8% |
| `consecutive_execution_errors_halt` | 3 | Yes | Halt after 3 consecutive execution failures |
| `daily_loss_limit_pct` | 2.0 | Yes | Halt if daily loss ≥ 2% |
| `consecutive_losses_soft_pause` | 3 | Yes | Halve size after 3 consecutive losses |
| `consecutive_losses_hard_halt` | 6 | **No** | Configured but dead — no code calls `trip("consecutive_losses_hard")` |
| `max_trades_per_hour` | 10 | **Yes** (Phase 5) | Rolling 1-hour window |
| `max_concurrent_positions` | 1 | **Yes** (Phase 5) | Raises error if > 1 |
| `data_staleness_halt_seconds` | 120 | Yes | Halt if no heartbeat for 120s |

### `retraining.*`

| Parameter | Default | Meaning |
|---|---|---|
| `schedule` | `"0 3 * * *"` | Cron syntax |
| `promotion_margin_pct` | 5.0 | Challenger must beat champion by 5% |
| `min_validation_samples` | 500 | Minimum samples to register |

## Hot-Reload (Phase 5)

A `ConfigWatcher` checks `settings.*.yaml`'s mtime each loop iteration and reloads on change — but only for a whitelisted safe subset:

```yaml
config_management:
  hot_reload:
    enabled: true
    check_interval_bars: 10
    safe_reload_paths:
      - "risk.sizing"
      - "risk.circuit_breakers"
      - "labeling"
      - "opportunity_filter.hard_rules"
      - "feature_engine"
    excluded_from_reload:
      - "execution.mode"
      - "execution.confirm_real_money"
      - "system.sandbox"
```

**Safe paths** can be reloaded without restart. **Excluded paths** require a restart — a config typo mid-session should never be able to silently flip the system into live order placement.

## Regime-Adaptive Config (Phase 5)

One unified mechanism for all regime-dependent thresholds:

```python
def resolve_regime_param(param_block: dict, current_regime: RegimeType) -> float:
    return param_block.get("by_regime", {}).get(current_regime.name, param_block["default"])
```

Example config shape:
```yaml
risk:
  sizing:
    risk_pct_per_trade:
      default: 0.5
      by_regime:
        TRENDING_UP: 0.6
        TRENDING_DOWN: 0.6
        CHOP_RANGE: 0.4
        PANIC_SPIKE: 0.0
```

Applied to: risk sizing, circuit breaker sensitivity, TP/SL barriers, opportunity filter thresholds. Replaces the two independently-written patterns (`regime_specific_barriers` in labeling, `regime_overrides` in opportunity filter).

## Deployment Entry Point (`scripts/run.py`)

### CLI Arguments

| Argument | Default | Choices | Required |
|---|---|---|---|
| `--env` | `paper` | `paper`, `live` | No |
| `--mode` | `historical` | `historical`, `live` | No |
| `--input` | — | Path to CSV/Parquet | Yes, for historical |
| `--symbol` | from config | Any CCXT pair | No |
| `--max-bars` | — | Integer | No |
| `--initial-equity` | 10000.0 | Float | No |

### Historical Mode

```bash
python scripts/run.py --mode historical --input data/raw/BTCUSDT.csv --env paper
```

Replays OHLCV bar-by-bar through the full stack. Real position lifecycle tracking (open, check TP/SL/timeout each bar, close, journal). Genuine `paper_sim` backtest through `ExecutionRecord`, not the triple-barrier proxy that `backtest_phase3.py` uses for training-data harvesting.

### Live Mode

```bash
python scripts/run.py --mode live --env live
```

Continuous polling loop. Poll interval derived from `system.data_granularity`. Attaches real order-book liquidity (paper-live/live modes). Runs until killed or `data/control/EMERGENCY_STOP` file is created.

**Requirements**: Real credentials + `confirm_real_money: true` + `execution.mode: "live"`.

## Docker Deployment

### Files

| File | Purpose |
|---|---|
| `Dockerfile` | Multi-stage build (builder + runtime) |
| `docker-compose.yml` | Service definition with volumes, healthcheck, resource limits |
| `.env.example` | Template for environment variables |
| `.dockerignore` | Exclude data dirs, docs, git from build context |
| `scripts/entrypoint.sh` | Network retry, signal handling, graceful shutdown |

### Build & Run

```bash
# Build
docker build -t btc-trading-agent:latest .

# Run historical backtest
docker run --rm \
  -v ./data/raw:/opt/trading-agent/data/raw \
  -v ./data/processed:/opt/trading-agent/data/processed \
  -v ./data/models:/opt/trading-agent/data/models \
  -v ./data/logs:/opt/trading-agent/data/logs \
  -v ./data/control:/opt/trading-agent/data/control \
  -v ./data/registry.db:/opt/trading-agent/data/registry.db \
  -e INPUT_DATA=/opt/trading-agent/data/raw/BTCUSDT.csv \
  btc-trading-agent:latest

# Run live (requires credentials)
docker run --rm \
  -e TRADING_MODE=live \
  -e TRADING_ENV=live \
  -e BTC_BOT_API_KEY=your_key \
  -e BTC_BOT_API_SECRET=your_secret \
  btc-trading-agent:latest
```

### Entrypoint Behavior

1. **Network check** (live mode only): Exponential backoff retry (`NETWORK_RETRY_MAX=5`, base delay `NETWORK_RETRY_BASE_DELAY=10`, backoff `NETWORK_RETRY_BACKOFF=2`)
2. **Input validation**: Historical mode requires `INPUT_DATA` file to exist
3. **Signal handling**: SIGTERM/SIGINT → graceful shutdown (waits up to `SHUTDOWN_TIMEOUT=30` seconds)
4. **Health check**: `python -c "import pandas; import lightgbm; import ccxt; print('OK')"` every 60s

### Docker Compose

- Volumes: raw, processed, models, logs, control, registry.db
- Healthcheck: Python import verification
- Resource limits: 2G memory, 1 CPU (512M/0.25 reserved)
- Logging: json-file driver, 100MB max per file, 3 files, compressed

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `TRADING_MODE` | `historical` | `historical` or `live` |
| `TRADING_ENV` | `paper` | `paper` or `live` |
| `INPUT_DATA` | — | Path to OHLCV file (required for historical) |
| `SYMBOL` | `BTC/USDT` | Trading pair |
| `MAX_BARS` | — | Max bars to process |
| `INITIAL_EQUITY` | `10000` | Starting equity for backtest |
| `BTC_BOT_API_KEY` | — | Binance API key (live mode) |
| `BTC_BOT_API_SECRET` | — | Binance API secret (live mode) |
| `NETWORK_RETRY_MAX` | `5` | Max network retries |
| `NETWORK_RETRY_BASE_DELAY` | `10` | Base retry delay (seconds) |
| `NETWORK_RETRY_BACKOFF` | `2` | Backoff multiplier |
| `SHUTDOWN_TIMEOUT` | `30` | Graceful shutdown timeout (seconds) |

## Secrets Handling

API keys are **never** stored in config files. Only env-var names are configured:

```yaml
execution.secrets:
  api_key_env_var: "BTC_BOT_API_KEY"
  api_secret_env_var: "BTC_BOT_API_SECRET"
```

The connector must be authenticated by the caller (e.g., CCXT reads from environment). This is a design constraint — `LiveBackend` never reads a secret directly.

## Pre-Flight Validation (Phase 5)

Run before starting a session:

```bash
python scripts/preflight.py --env paper
python scripts/validate_config_consistency.py
```

Validates: unit consistency, threshold plausibility, breaker enforcement paths, feature schema version. Refuses to start if anything fails.

## Live Validation Runbook (Phase 5)

Scripted 5-step testnet procedure:

```bash
python scripts/validate_live_backend.py --testnet
```

1. Place minimum-size test order, confirm fill
2. Re-submit identical clientOrderId, confirm idempotency
3. Cancel open order, confirm status reflected
4. Run reconciliation against testnet account
5. Simulate process kill mid-order, restart, confirm no duplicate

Report written to `data/reports/live_validation_<timestamp>.md`.
