# Btc_trading_agents — Complete User Guide

> A hands-on guide to installing, configuring, running, testing and debugging the
> regime-aware multi-expert BTC trading research platform — with worked examples
> and every supported flag combination.

**Audience:** developers, reviewers, and operators evaluating or running the system.
**Platform notes:** commands below are shown for **Windows PowerShell** first, with
`make`/bash equivalents where they exist. All commands run from the repo root
(`C:\Users\devan\Documents\Btc_trading_agents`) unless stated otherwise.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Installation](#2-installation)
3. [Quick Start — 5 Minutes](#3-quick-start--5-minutes)
4. [The Configuration System](#4-the-configuration-system)
5. [Running the System (`run.py`)](#5-running-the-system-runpy)
6. [Backtesting & Ablation](#6-backtesting--ablation)
7. [Diagnostics & Operations](#7-diagnostics--operations)
8. [Interactive Apps (Demo, Trade Map, TUI Dashboard)](#8-interactive-apps)
9. [Models: Training, Approval, Rollback](#9-models-training-approval-rollback)
10. [Testing, Linting & CI](#10-testing-linting--ci)
11. [Ports & Networking](#11-ports--networking)
12. [Environment Variables](#12-environment-variables)
13. [Known Gotchas & Troubleshooting](#13-known-gotchas--troubleshooting)
14. [Appendix A — Complete CLI Reference](#appendix-a--complete-cli-reference)
15. [Appendix B — Makefile Targets](#appendix-b--makefile-targets)
16. [Appendix C — Config Key Reference](#appendix-c--config-key-reference)

---

## 1. System Overview

The pipeline processes one market bar at a time through a fixed stage order
(**this order must not change**):

```
MarketSnapshot → FeatureVector → RegimeState → ZoneMap →
  structural features merged back → ExpertOutputs → OpportunityScore →
  RouterDecision → RiskDecision → ExecutionRecord → SystemState
```

- **Regime is computed before structure** (zones need `RegimeState.regime_relevance`).
- **Three experts** — `trend_expert`, `mean_reversion_expert`, `breakout_expert` —
  each gate themselves on regime and features, then a **router** selects one.
- **Hard rules always override ML**: a rule veto in `OpportunityFilter` or
  `RiskLayer` beats a model saying "tradable".
- **Risk layer is called only for new entries**, never for exits.
- **Every bar produces a `DecisionTrace`** written to `data/logs/decision_traces/`
  — this is what `diagnose.py`, the dashboard and the GUI replay read.
- **No look-ahead**: features at bar *k* use only bars `0..k` (enforced by
  `tests/unit/test_no_lookahead.py`).

Key safety properties:

| Constraint | Where enforced | Value (shipped) |
|---|---|---|
| One concurrent position | `process_bar` | `risk.circuit_breakers.max_concurrent_positions: 1` |
| Trade rate limit | `RiskLayer.evaluate()` | `max_trades_per_hour: 10` |
| Bars between orders | `RiskLayer.evaluate()` | `min_bars_between_orders` (see §13 gotcha) |
| Min trade duration (fill→exit) | `src/labeling/hold_policy.py` | `trade_duration.min_trade_seconds: 10` |
| Circuit breakers persist across restarts | SQLite `data/registry.db` | — |
| Kill switch | file `data/control/EMERGENCY_STOP` | file existence = ON |

---

## 2. Installation

### 2.1 Requirements

- **Python ≥ 3.10** (tested on 3.12; CI runs 3.12)
- `pip` (26.x tested)
- Network access to PyPI (see §2.3 if your connection is poor)
- Optional: Docker, Git Bash/WSL for `make` targets

### 2.2 Standard install (recommended)

The package installs **editable** (`-e`), so source changes take effect without
reinstalling. Extras:

| Extra | Installs | Needed for |
|---|---|---|
| `dev` | ruff, mypy, pytest-cov | lint / typecheck / coverage |
| `demo` | streamlit, matplotlib | Streamlit decision demo |
| `viz` | plotly, streamlit ≥1.36 | Trade Map GUI (charts) |

```powershell
# Everything (what this repo uses):
pip install -e ".[dev,demo,viz]"
```

If pip fails fetching **build dependencies** (a `setuptools>=68.0` error), skip
build isolation — setuptools is already present:

```powershell
pip install -e ".[dev,demo,viz]" --no-build-isolation
```

Linux/macOS equivalent: `pip install -e ".[dev,demo,viz]"` (add
`--no-build-isolation` only if you hit the same error).

### 2.3 Poor/flaky network — install with retries

pip retries each URL (`--retries`) and tolerates slow sockets (`--timeout`),
but a single `IncompleteRead` can abort the run. Wrap it in a retry loop so
cached downloads carry over between attempts (this is exactly what succeeded
on attempt 1 for this checkout):

**PowerShell** — save as `install_with_retries.ps1` and run
`powershell -ExecutionPolicy Bypass -File install_with_retries.ps1`:

```powershell
$maxAttempts = 20
for ($i = 1; $i -le $maxAttempts; $i++) {
    Write-Host "=== Attempt $i of $maxAttempts ==="
    pip install -e ".[dev,demo,viz]" --no-build-isolation --retries 25 --timeout 90
    if ($LASTEXITCODE -eq 0) { Write-Host "=== SUCCESS on attempt $i ==="; break }
    $wait = [Math]::Min(45, $i * 5)
    Write-Host "--- failed; waiting ${wait}s ---"
    Start-Sleep -Seconds $wait
}
```

**Bash** equivalent:

```bash
for i in $(seq 1 20); do
  echo "=== Attempt $i ==="
  pip install -e ".[dev,demo,viz]" --no-build-isolation --retries 25 --timeout 90 && break
  sleep $(( i * 5 < 45 ? i * 5 : 45 ))
done
```

Why this works: pip **caches** every wheel it downloads, so attempt *N* resumes
where attempt *N−1* stopped instead of restarting from zero.

### 2.4 Verify the install

```powershell
python -c "import src, src.gui, src.pipeline; import streamlit, plotly, lightgbm, ccxt, vectorbt; print('imports OK')"
# expected: imports OK

python -m pytest tests/unit -q        # quick sanity test run
```

### 2.5 Install via Makefile (bash only)

```bash
make install-dev    # creates .venv, installs requirements.txt + -e ".[dev]"
```

> ⚠ The Makefile sets `SHELL := /bin/bash` and uses `.venv/bin/*` paths — on
> Windows use **Git Bash** or **WSL**, or skip `make` and run the Python
> commands directly (this guide always shows the direct command).

---

## 3. Quick Start — 5 Minutes

Pick any path; each is self-contained.

### Path A — Streamlit decision demo (no data, no network)

```powershell
python -m streamlit run scripts/demo_app.py --server.port=8571 --server.address=127.0.0.1 --server.headless=true
# open http://127.0.0.1:8571
```

Runs the **real pipeline** over synthetic bars, labeled `ILLUSTRATIVE`.
(`make demo` does the same on Linux.)

### Path B — Trade Map GUI (visual, interactive)

```powershell
pip install -e ".[viz]"                       # once
python scripts/trade_map_app.py               # http://127.0.0.1:8572
```

### Path C — One-shot headless check (CI-friendly)

```powershell
python scripts/dashboard.py --once            # prints all 6 views as text and exits
```

### Path D — Historical replay through the full stack

```powershell
python scripts/run.py --mode historical --input data\raw\BTCUSDT.csv --env paper
```

> `--input` must exist. No real OHLCV file? Generate a backtest report instead:
> ```powershell
> python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity
> ```
> Reports land in `data/reports\backtest_<timestamp>.{json,md}`.

### Path E — Ablation ladder (4 runs, the contribution of each component)

```powershell
python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity
python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity --single-expert trend_expert
python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity --no-risk-layer
python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity --no-regime-gating
```

### Path F — TUI dashboard (live console)

```powershell
python scripts/dashboard.py                   # keys: 1-6 views · r refresh · q quit
```

---

## 4. The Configuration System

### 4.1 Layered YAML — the single most important concept

`load_layered_config(env)` merges two files with a **deep merge; overlay wins**:

```
config/settings.base.yaml          (all defaults, all sections)
        +
config/settings.{env}.yaml         (env = "paper" | "live")
        ↓
      merged dict
```

Rules, precisely:

- **Nested dicts merge key-by-key, recursively.**
- **Scalars and lists are replaced wholesale** (no list concatenation).
- **A missing overlay file → `{}` → base only, silently** (no error).

```python
from src.utils.config_helpers import load_layered_config
config = load_layered_config("paper")     # base + settings.paper.yaml
config = load_layered_config("live")      # base + settings.live.yaml
```

**Always pass `--env`** to deployment scripts — it selects the overlay.
`--env` accepts only `paper` (default) or `live` on every script.

> ⚠ `--config <file>` (on the backtest and retrain scripts) loads a **single
> file, not layered**, and **overrides `--env`**.

### 4.2 What the overlays contain

`config/settings.paper.yaml` (default):
```yaml
system:     { sandbox: true }
execution:  { mode: "paper_sim", confirm_real_money: false }
```

`config/settings.live.yaml`:
```yaml
system:     { sandbox: false }
execution:  { mode: "live", confirm_real_money: false }   # deliberately false!
```

**Two-flag live guard:** real orders require **both**
`execution.mode: "live"` **and** `execution.confirm_real_money: true`.
The live overlay ships with the second flag `false`, so `--env live` alone can
never place a real order — you must edit the config yourself, on purpose.
Violating this raises `LiveTradingNotConfirmedError` at startup.

### 4.3 `execution.mode` — three backends

| Mode | Data | Fills | Use case |
|---|---|---|---|
| `paper_sim` | synthetic liquidity model | simulated from bar liquidity | default; fully offline |
| `paper_live` | **real** order book via CCXT | simulated against the real book; logs paired (synthetic, real) samples for calibration | realism step-up |
| `live` | real | **real orders** (CCXT, idempotent) | requires the second flag too |

Any other value → `ValueError("Unknown execution.mode: ...")`.

**All 6 combinations of `mode × confirm_real_money`:**

| `execution.mode` | `confirm_real_money` | Result |
|---|---|---|
| `paper_sim` | `false` | ✅ simulated trading (default) |
| `paper_sim` | `true` | ✅ simulated (flag ignored for sim) |
| `paper_live` | `false` | ✅ real book, simulated fills |
| `paper_live` | `true` | ✅ real book, simulated fills |
| `live` | `false` | ❌ `LiveTradingNotConfirmedError` — refuses to start |
| `live` | `true` | ⚠ **real money** — only intended combination for live |

### 4.4 `system.data_granularity` — the only timeframe switch

Valid examples: `"10s"`, `"30s"`, `"1m"` (shipped), `"5m"` (also `1h`, `1d`
parseable by the hold policy). Changing it in YAML is **all** that's needed —
no code change. It drives:

1. **Data fetch** — CCXT `timeframe`.
2. **Live poll cadence** — `{"10s":10, "30s":30, "1m":60, "5m":300}` seconds (unknown → 60s).
3. **Hold policy / label horizon** — `granularity × labeling.max_holding_bars`.
4. **Model granularity guard** — a model pickle saved at a different granularity
   is treated as absent → **rule-based fallback, no crash**.

### 4.5 Key config sections (tunables)

Full reference: [Appendix C](#appendix-c--config-key-reference). Highlights:

```yaml
system:
  data_granularity: "1m"
  symbol: "BTC/USDT"
  sandbox: true                # Binance testnet when true

risk:
  sizing: { risk_pct_per_trade: 0.5, max_position_pct_of_equity: 20.0 }
  circuit_breakers:
    max_drawdown_pct: 8.0
    daily_loss_limit_pct: 2.0
    consecutive_execution_errors_halt: 3
    max_trades_per_hour: 10
    max_concurrent_positions: 1        # value < 1 raises ValueError
    data_staleness_halt_seconds: 120

opportunity_filter:
  hard_rules:                  # ALWAYS beat the ML model
    edge_vs_fees_threshold: 0.02
    max_spread_pct: 0.05
    volatility_floor_pct: 0.01
    volatility_ceiling_pct: 1.5
    reject_on_regime: [PANIC_SPIKE]
  regime_overrides:            # per-regime adjustments, keyed by regime NAME
    PANIC_SPIKE: { edge_vs_fees_threshold: 0.05 }

labeling:                      # triple barrier
  k_tp: 1.5                    # TP = 1.5 × volatility
  k_sl: 1.0                    # SL = 1.0 × volatility
  fee_pct: 0.04
  regime_specific_barriers:
    TRENDING_UP: { k_tp: 2.0, k_sl: 0.8 }
    PANIC_SPIKE: { k_tp: 0.5, k_sl: 0.5 }

trade_duration: { min_trade_seconds: 10, require_aligned_models: true }
```

### 4.6 Regime names (used in config values)

`TRENDING_UP`, `TRENDING_DOWN`, `CHOP_RANGE`, `PANIC_SPIKE`
(served lowercase as `trending_up`, `trending_down`, `chop_range`, `panic_spike`).

### 4.7 Instruments registry (`config/instruments.yaml`)

**Not part of the layered config** — a separate contract registry resolved
relative to the **repo root** (never the CWD). Entry shapes:

```yaml
instruments:
  "BTC/USDT":            { asset_class: spot, currency: USDT, display_precision: 2 }
  "BTC/USDT:USDT":       { asset_class: perp, underlying: BTC/USDT, multiplier: 1 }
  "BTC-261220-75000-C":  { asset_class: option, underlying: BTC/USDT,
                           option_type: call, strike: 75000, expiry: "2026-12-20",
                           multiplier: 1, tick_size: 0.1 }
```

`asset_class` ∈ `spot | perp | future | option | equity | fx`. Metadata drives
price precision, notional math, and strike/expiry lines in charts. Unknown
symbol → generic spot fallback; malformed entry → skipped with a warning.

### 4.8 Hot-reload (designed, not yet wired)

Intended hot-reloadable paths: `feature_engine`, `risk.circuit_breakers`,
`risk.sizing`, `labeling`, `opportunity_filter.hard_rules`.
Never hot-reloadable: `execution.mode`, `execution.confirm_real_money`, `system.sandbox`.

> **Reality check:** `ConfigWatcher` exists (`src/config/config_watcher.py`)
> but no production loop instantiates it yet — **today, every config change
> requires a restart.**

### 4.9 Kill switch

```powershell
# ARM (halts live loop + vetoes all new entries):
New-Item -ItemType File -Path data\control\EMERGENCY_STOP -Force
# DISARM:
Remove-Item data\control\EMERGENCY_STOP
```

File existence is the whole protocol — checked first in `RiskLayer.evaluate()`
(rejection reason `manual_kill_switch_active`) and at the top of every live-loop
iteration (logs and exits the loop).

---

## 5. Running the System (`run.py`)

### 5.1 Flags

| Flag | Type | Default | Choices | Meaning |
|---|---|---|---|---|
| `--env` | str | `paper` | `paper`, `live` | selects config overlay |
| `--mode` | str | `historical` | `historical`, `live` | replay a file vs poll exchange |
| `--input` | str | — | — | OHLCV CSV/Parquet (**required for historical**) |
| `--symbol` | str | config | — | overrides `system.symbol` |
| `--max-bars` | int | none | — | cap bars in historical mode |
| `--initial-equity` | float | `10000.0` | — | starting equity |

### 5.2 The 2 × 2 × 2 matrix — every combination

`mode` (2) × `env` (2) × `--input` (present/absent) = **8 states**:

| # | Command | Input? | Result |
|---|---|---|---|
| 1 | `--mode historical --input FILE --env paper` | ✅ | **Standard safe replay.** Simulated fills, sandbox config. |
| 2 | `--mode historical --input FILE --env live` | ✅ | ⚠ Replays a file under the **live overlay** → backend is `live` with `confirm_real_money: false` → **`LiveTradingNotConfirmedError`** at startup (safe by design). With the flag edited to `true`, it would place **real orders** while replaying history — never do this unintentionally. |
| 3 | `--mode historical --env paper` (no input) | ❌ | ❌ `SystemExit("--mode historical requires --input <ohlcv file>")` |
| 4 | `--mode historical --env live` (no input) | ❌ | ❌ same error, before any backend is used |
| 5 | `--mode live --env paper` | ignored | ✅ **Safe live rehearsal**: polls the real exchange for bars, but fills are simulated (`paper_sim`), sandbox on. |
| 6 | `--mode live --env paper --symbol ETH/USDT` | ignored | ✅ same, different symbol |
| 7 | `--mode live --env live` (stock config) | ignored | ❌ `LiveTradingNotConfirmedError` — the two-flag guard firing as designed |
| 8 | `--mode live --env live` **+ `confirm_real_money: true`** | ignored | ⚠ **REAL MONEY.** Requires `BTC_BOT_API_KEY`/`BTC_BOT_API_SECRET` in the environment. Kill switch honored every bar. |

### 5.3 Worked examples

```powershell
# (1) Paper replay of a CSV, capped at 5000 bars
python scripts/run.py --mode historical --input data\raw\BTCUSDT.csv --env paper --max-bars 5000

# (1) Replay with custom equity and symbol
python scripts/run.py --mode historical --input data\raw\BTCUSDT.csv --env paper --symbol BTC/USDT --initial-equity 25000

# (5) Live-data loop, paper fills (safe rehearsal)
python scripts/run.py --mode live --env paper

# (8) Real trading — ONLY after editing settings.live.yaml confirm_real_money: true
python scripts/run.py --mode live --env live
```

### 5.4 What a run writes

| Path | Contents |
|---|---|
| `data/logs/system.jsonl` | all log output as JSON lines |
| `data/logs/decisions.jsonl` | one structured record per bar |
| `data/logs/decision_traces/` | partitioned JSONL `DecisionTrace` (per-stage `duration_ms`) |
| `data/registry.db` | circuit breakers, trade journal, model registry, experiments |
| `data/logs/system_state/` | `SystemState` archive |
| `data/logs/zone_events.parquet` | zone event log |

Live loop specifics: polls on the granularity cadence, checks the kill switch
every bar, and halts the loop (not just entries) when `EMERGENCY_STOP` exists.

---

## 6. Backtesting & Ablation

### 6.1 `full_stack_walk_forward_backtest.py` — flags

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--input` | str | — | OHLCV CSV/Parquet (**XOR** `--synthetic`) |
| `--synthetic` | int | — | generate N synthetic bars instead |
| `--env` | str | `paper` | `paper`\|`live` |
| `--config` | str | — | explicit single file; **overrides `--env`** |
| `--symbol` | str | config | symbol override |
| `--window-bars` | int | `1440` | evaluation window length |
| `--initial-equity` | float | `10000.0` | starting equity |
| `--relax-liquidity` | flag | off | relax spread/depth/edge gates **in memory only** |
| `--skip-baselines` | flag | off | skip the 4 baseline comparisons (faster) |
| `--single-expert` | str | — | ablation: keep only this expert |
| `--no-risk-layer` | flag | off | ablation: auto-approve at base size |
| `--no-regime-gating` | flag | off | ablation: no regime gating anywhere |
| `--refit` | flag | off | ⚠ parsed but currently a **no-op** |
| `--report-dir` | str | `data/reports` | where reports go |

**Rule:** you must pass **`--input FILE` or `--synthetic N`** — otherwise the
script exits with `Provide --input <ohlcv file> or --synthetic <n_bars>`.

**Expert names** for `--single-expert`: `trend_expert`, `mean_reversion_expert`,
`breakout_expert` (registry keys). An unknown name silently keeps **zero** experts.

### 6.2 The canonical ablation ladder (4 runs)

Each run isolates one component's contribution:

```powershell
# 1. Full system (reference)
python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity

# 2. Single expert only (trend)
python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity --single-expert trend_expert

# 3. No risk layer
python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity --no-risk-layer

# 4. No regime gating
python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity --no-regime-gating
```

### 6.3 Full ablation combination space

Two boolean ablations × three expert choices (plus none) = **4 × 2 × 2 = 16
distinct ablation configurations per data source**:

```
--single-expert ∈ {∅, trend_expert, mean_reversion_expert, breakout_expert}   (4)
--no-risk-layer  ∈ {off, on}                                                   (2)
--no-regime-gating ∈ {off, on}                                                 (2)
```

Representative combinations and how to read them:

| Combination | Question it answers |
|---|---|
| *(none)* | full system baseline |
| `--single-expert mean_reversion_expert` | contribution of MR alone |
| `--single-expert breakout_expert --no-regime-gating` | breakout without regime constraints |
| `--no-risk-layer` | what risk controls add (caps/breakers/liquidity sizing) |
| `--no-regime-gating` | what regime gating adds |
| `--no-risk-layer --no-regime-gating` | "raw experts" — upper-bound turnover, no brakes |
| `--single-expert trend_expert --no-risk-layer` | one expert with no brakes |
| all three ablations together | minimal system floor |

Data-source dimension doubles again:

```
--synthetic 900 --relax-liquidity     (offline smoke, MUST relax — see §13)
--input data/raw/BTCUSDT.csv          (real data, no relaxation needed)
```

> **Interpretation caveat:** synthetic + `--relax-liquidity` measures *pipeline
> behavior*, **not production edge** (the flag's help text says exactly this).

### 6.4 Which data source should I use?

| Source | Command fragment | Network | Realism |
|---|---|---|---|
| Synthetic smoke | `--synthetic 900 --relax-liquidity` | none | illustrative only |
| Synthetic strict | `--synthetic 900` | none | ⚠ nearly all bars rejected by strict gates (see §13) |
| Real CSV/Parquet | `--input data\raw\BTCUSDT.csv` | none (file local) | real market data |
| Exchange fetch | `run.py --mode live --env paper` | yes | real bars, simulated fills |

### 6.5 Other reporting tools

```powershell
# Latency/memory benchmark (fully synthetic, deterministic)
python scripts/benchmark_latency.py --bars 300
python scripts/benchmark_latency.py --bars 500 --report-dir data\reports

# Documentation figures → docs\images\ (matplotlib required)
python scripts/generate_figures.py --synthetic 900 --relax-liquidity
python scripts/generate_figures.py --input data\raw\BTCUSDT.csv
```

Every backtest/benchmark run is also logged to `data/registry.db::experiments`
(config hash, config snapshot, metrics, git commit).

---

## 7. Diagnostics & Operations

### 7.1 `diagnose.py` — why did/didn't a bar trade?

The four primary flags are **mutually exclusive; exactly one is required**:

```powershell
# Summary breakdown of all bars
python scripts/diagnose.py --summary

# Summary restricted to a date range
python scripts/diagnose.py --summary --since 2026-08-01

# Full stage-by-stage trace for one bar (epoch seconds)
python scripts/diagnose.py --timestamp 1755900000

# All system fault records
python scripts/diagnose.py --faults
```

| Flag | Pairs with | Notes |
|---|---|---|
| `--timestamp INT` | — | exits 1 if no trace matches |
| `--summary` | `--since YYYY-MM-DD` | the "why aren't we trading" view |
| `--faults` | — | reads `data/logs/faults.jsonl` |
| `--trade-id STR` | — | ⚠ currently prints "not yet implemented" |

### 7.2 `replay_trade.py` — replay one trade's decision journey

`--list` or `--trade-id` is **required**:

```powershell
python scripts/replay_trade.py --list 10                      # 10 most recent (default 15 if bare --list)
python scripts/replay_trade.py --trade-id <id>                # full T±window replay
python scripts/replay_trade.py --trade-id <id> --input data\raw\BTCUSDT.csv --window-bars 10
python scripts/replay_trade.py --trade-id <id> --env live --db data\registry.db
```

Reads journal (SQLite) + decision traces + optional OHLCV for market context.
Exits 1 if the trade id is unknown.

### 7.3 `status.py` — one-glance health

```powershell
python scripts/status.py --env paper
```
Read-only: risk state, journal, pending model versions, kill-switch presence.

### 7.4 `preflight.py` — validate before a session

```powershell
python scripts/preflight.py --env paper
python scripts/preflight.py --env live --config-dir config
```

Validates unit consistency, threshold plausibility, breaker enforcement paths,
feature schema version; creates missing `data/logs` dirs; **warns** if the kill
switch is armed; exits 1 with `Pre-flight FAILED` on any inconsistency.

> ⚠ On the **shipped config this FAILS the liquidity gate** ("~98% of simulated
> bars would fail max_spread_pct=0.05") — a known tripwire, see §13.

### 7.5 `validate_config_consistency.py` — single-file check

```powershell
python scripts/validate_config_consistency.py
python scripts/validate_config_consistency.py --config config\settings.base.yaml --db-path data\registry.db
```

Difference vs preflight: this validates **one YAML file as written** (no
layered merge) and takes no `--env`.

### 7.6 `reset_breaker.py` — clear a tripped breaker

```powershell
python scripts/reset_breaker.py --type max_drawdown
python scripts/reset_breaker.py --type daily_loss_limit --db-path data\registry.db
```

`--type` is required; valid values:

```
consecutive_execution_errors | max_drawdown | consecutive_losses_hard
daily_loss_limit | data_staleness
```

Idempotent — prints `was_active=... -> reset`.

### 7.7 `validate_live_backend.py` — testnet runbook

```powershell
# Requires Binance TESTNET credentials in the environment first:
$env:BTC_BOT_API_KEY = "<testnet key>"
$env:BTC_BOT_API_SECRET = "<testnet secret>"
python scripts/validate_live_backend.py --testnet --output data\reports\live_validation.md
```

Runs 5 real testnet steps (place → idempotent re-submit → cancel →
reconcile → kill-mid-order recovery); exits 1 if any step fails.
Note: `--testnet` defaults `True` and cannot be turned off (mainnet is not
offered by this script — by design).

### 7.8 Operational runbook examples

```powershell
# "The system seems quiet — what's wrong?"
python scripts/diagnose.py --summary --since 2026-10-01
python scripts/status.py --env paper
python scripts/dashboard.py            # view ⑤ Risk & Faults

# "A breaker tripped — verify then clear it"
python scripts/status.py --env paper
python scripts/reset_breaker.py --type max_drawdown
python scripts/status.py --env paper    # confirm cleared

# "Before starting tonight's session"
python scripts/preflight.py --env paper
python scripts/validate_config_consistency.py
python scripts/status.py --env paper

# "Emergency stop everything"
New-Item -ItemType File -Path data\control\EMERGENCY_STOP -Force
```

---

## 8. Interactive Apps

### 8.1 Streamlit Decision Demo — `scripts/demo_app.py`

```powershell
pip install -e ".[demo]"     # once
python -m streamlit run scripts/demo_app.py --server.port=8571 --server.address=127.0.0.1 --server.headless=true
# http://127.0.0.1:8571
```

`make demo` (bash) is equivalent; `make demo STREAMLIT_PORT=8600` overrides the port.

**Sidebar — "Window":**

| Control | Type | Options / range | Default |
|---|---|---|---|
| Data source | radio | `Synthetic` \| `Real CSV/Parquet` | `Synthetic` |
| Bars *(synthetic)* | slider | 600–3000 step 100 | 1200 |
| Seed | number input | 0–9999 | 42 |
| Path to OHLCV file *(real)* | text | any path | `data/raw/BTCUSDT.csv` |
| Bars to use *(real)* | slider | 600–5000 step 100 | 1500 |

**The 2 × 2 data permutations:**

| Source | Bars | Header tag | Needs file? |
|---|---|---|---|
| Synthetic | 600–3000 | `ILLUSTRATIVE — synthetic data` | no |
| Synthetic | + different seed | `ILLUSTRATIVE` (different series) | no |
| Real | 600–5000 | `— REAL DATA` | ✅ must exist |
| Real | different path | `— REAL DATA` | ✅ must exist |

**Four tabs:**

1. **Market & Regime** — price line + regime colour bands
   (green `trending_up`, red `trending_down`, gray `chop_range`, purple `panic_spike`)
   + router expert-selection scatter per bar.
2. **Decision Funnel** — bars processed → expert candidates → opportunity filter
   passed → router selected expert → orders executed (metric cards + bar chart).
3. **Trades** — table: entry time, expert, regime, side (LONG/SHORT), size,
   `pnl_pct`, `pnl_usdt`, exit reason. Empty → info box.
4. **Equity** — equity curve from 10,000 USDT + metrics: Trades, Win rate,
   Expectancy/trade, Total PnL.

Direct-python args (only honored when run as `python scripts/demo_app.py`):
`--input <path>` and `--bars <n>`.

> **Honesty note:** synthetic mode's trades/PnL are *real computations on
> invented prices* — mechanics demo, not a profitability signal. No hard-coded
> fake values exist anywhere in the app.

### 8.2 Trade Map GUI — `scripts/trade_map_app.py`

```powershell
pip install -e ".[viz]"      # once (plotly + streamlit)
python scripts/trade_map_app.py            # http://127.0.0.1:8572
$env:TRADE_MAP_PORT = "8600"; python scripts/trade_map_app.py    # port override
```

The GUI is a **viewer**: replays are forced to `paper_sim`, it never places
orders and never writes tracked `data/`.

**Sidebar — Trade Map section:**

| Control | Options | Default |
|---|---|---|
| Instrument | registry selectbox + free-text symbol | `BTC/USDT` |
| Data source | `artifacts` \| `replay` \| `synthetic` (auto picks recommended) | auto |
| Most recent bars *(replay)* | slider 200–5000 | 1500 |
| Bars *(synthetic)* | slider 300–3000 | 900 |
| Seed *(synthetic)* | 0–99999 | 42 |
| Relax liquidity gates | checkbox | off (replay) / **on** (synthetic) |
| Initial equity | 100–10,000,000 | 10,000 |

**Sidebar — Chart section:** Timeframe (select-slider), Overlays multiselect
(`trades`, `spans`, `barriers`, `decisions`, `zones`, `regimes` — all on),
Decision stages multiselect (all terminal stages except `no_expert_candidate`
and `position_open`), Indicators (default `ema_fast`, `ema_slow`), Panes
(e.g. `rsi`, `macd`; default none), Volume (on), Chart height 480–1000 (720).

**Main area:** header metric strip (Trades, Win rate, Expectancy, Total PnL,
Profit factor, Max drawdown) + clickable Plotly chart + **4 tabs**:

1. **Why this bar?** — Prev/Next/Next-trade navigation; the bar's terminal
   stage chip + reason; facts table (regime, expert, decision, opportunity
   score, router confidence, risk score, risk approved); full per-stage trace
   (stage, ✅/⛔, detail, ms).
2. **Decisions** — decision chart + stage funnel side by side.
3. **Equity** — equity chart.
4. **Trades** — PnL-coloured clickable table (click a row → jumps to entry bar).

**Data-source × control matrix:**

| Source | Bars controls | Relax gate | Needs existing artifacts? |
|---|---|---|---|
| `artifacts` | none | n/a | ✅ journal + traces from a prior run |
| `replay` | 200–5000 bars | optional (default off) | ❌ replays from OHLCV |
| `synthetic` | 300–3000 + seed | default **on** | ❌ |
| `auto` | — | — | tries artifacts → replay → synthetic |

### 8.3 Trade Map HTML export — `scripts/trade_map.py`

```powershell
python scripts/trade_map.py --source synthetic --out trade_map.html
python scripts/trade_map.py --source artifacts --symbol BTC/USDT --timeframe 5m --out data\reports\trade_map.html
python scripts/trade_map.py --source replay --max-bars 2000 --relax-liquidity --indicators ema_fast,ema_slow --panes rsi,macd --out data\reports\map.html
python scripts/trade_map.py --source auto    # text summary only (no --out)
```

| Flag | Values | Default |
|---|---|---|
| `--source` | `auto` \| `artifacts` \| `replay` \| `synthetic` | `auto` |
| `--symbol` | any | `BTC/USDT` |
| `--timeframe` | display TF | auto (≲1500 candles) |
| `--out` | `.html`/`.htm` path | none (text summary) |
| `--max-bars` | int | 1500 (replay) |
| `--bars` | int | 900 (synthetic) |
| `--seed` | int | 42 |
| `--relax-liquidity` | flag | off (forced on for synthetic) |
| `--indicators` | comma list | `ema_fast,ema_slow` (`''` = none) |
| `--panes` | comma list | none (e.g. `rsi,macd`) |
| `--env` | `paper`\|`live` | `paper` |

Refuses to write into protected paths (`data/registry.db`, `data/logs`,
`data/raw`, `data/models`); needs plotly else exit 2; embedded plotly.js so the
HTML opens offline. `make trade-map-export` → `--source auto --out data/reports/trade_map.html`.

### 8.4 TUI Dashboard — `scripts/dashboard.py`

```powershell
python scripts/dashboard.py                      # live TUI (needs textual)
python scripts/dashboard.py --once               # headless one-shot to stdout
python scripts/dashboard.py --refresh-seconds 2.0
```

**Keys:** `1`–`6` views · `r` refresh now · `q` quit.

| Key | View | Shows |
|---|---|---|
| `1` | ① Signal Path | per-stage pass rates + terminal funnel ("where bars stop") |
| `2` | ② Trades | journal tail, today's PnL/win-rate, open positions |
| `3` | ③ Performance | latest backtest report vs success criteria + baselines |
| `4` | ④ Calibration | confidence reliability buckets (experts + router) |
| `5` | ⑤ Risk & Faults | breakers, risk state, fault tail |
| `6` | ⑥ Registry | model versions (30) + experiments (15) |

Status line: kill-switch state, active breakers, open trades, trace count.
Same data sources as `diagnose.py`; render errors never kill the TUI.

---

## 9. Models: Training, Approval, Rollback

### 9.1 Training scripts (offline utilities)

Each reads `config/settings.base.yaml` (single-file `--config`, no `--env`):

| Script | Trains |
|---|---|
| `train_triple_barrier.py` | labeled samples per expert + harness exit rule (one Parquet + `.meta.json`) |
| `train_trend_expert.py` | trend expert scoring model |
| `train_mean_reversion_expert.py` | mean-reversion expert scoring model |
| `train_breakout_expert.py` | breakout expert scoring model |
| `train_opportunity_filter.py` | tradability ML model |
| `train_router.py` | router hindsight multiclass model |
| `train_risk_model.py` | P(stop-out) risk model |
| `train_regime_models.py` | regime HMM + LightGBM classifier |
| `train_zone_survival_model.py` | zone-survival classifier (walk-forward CV) |
| `train_pipeline.py` | harness-ordered training DAG (`--plan` / `--execute --bundle-out …`) |

### 9.2 `retrain_all.py` — the cron entry point

```powershell
# All components
python scripts/retrain_all.py --input data\raw\BTCUSDT.csv --config config\settings.base.yaml

# Layered config instead of single-file
python scripts/retrain_all.py --input data\raw\BTCUSDT.csv --env paper

# Subset only (valid names below)
python scripts/retrain_all.py --input data\raw\BTCUSDT.csv --env paper --components trend_expert router
```

`--input` is **required**. Valid `--components` values (unknown → skipped with
a warning, not an error):

```
zone_survival | regime_hmm | regime_classifier | trend_expert |
mean_reversion_expert | breakout_expert | opportunity_filter | router | risk_model
```

Flow: validate hold policy → run each step as an isolated subprocess → compare
challenger vs champion walk-forward metric with a 5% promotion margin →
register as **`pending_approval`**. **Never auto-promotes.**

### 9.3 Approval & rollback

```powershell
# Interactive review of everything pending
python scripts/approve_model.py

# Targeted, non-interactive
python scripts/approve_model.py --component trend_expert --yes

# Force-promote a version that isn't harness-aligned (use with care)
python scripts/approve_model.py --component router --force

# List available versions (no --to-version = dry list)
python scripts/rollback_model.py --component trend_expert

# Roll back to a specific version
python scripts/rollback_model.py --component trend_expert --to-version v3
```

Recognized `--component` names: the nine above plus `barrier_sizing`,
`risk_sizing`, `exit_model` (aligned components).

### 9.4 Model quirks that affect behavior

- Models are **pickle LightGBM / hmmlearn HMM** under `data/models/`.
- **Granularity mismatch** → rule-based fallback (no crash).
- **Feature schema mismatch** (`FEATURE_SCHEMA_VERSION`, now `"v2_renamed"`) →
  rule-based fallback. This catches pre-Phase-5 models after the field rename.
- **`trade_duration.require_aligned_models: true`**: models trained under a
  different min-trade-duration policy are refused (use `--force` to promote anyway).
- Walk-forward splits use **purge + embargo** — never random/TimeSeriesSplit.

### 9.5 Full retrain → promote example

```powershell
python scripts/retrain_all.py --input data\raw\BTCUSDT.csv --env paper
python scripts/status.py --env paper                    # see what's pending
python scripts/approve_model.py --component trend_expert --yes
python scripts/rollback_model.py --component trend_expert --to-version v2   # if needed
```

---

## 10. Testing, Linting & CI

### 10.1 Test tiers

Markers defined in `pyproject.toml`: `unit`, `integration`, `e2e`.

```powershell
# Everything
python -m pytest tests/ -v

# By tier
python -m pytest tests/unit -v
python -m pytest tests/integration -v
python -m pytest tests/e2e -v

# By marker
python -m pytest -m unit tests/ -v

# Single file / single test
python -m pytest tests/unit/test_phase3.py -v
python -m pytest tests/unit/test_phase3.py::test_triple_barrier_tp_hit_first -v

# Fast fail-fast
python -m pytest tests/ -x -q --tb=short

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
```

`make` equivalents (bash): `make test`, `make unit`, `make integration`,
`make e2e`, `make coverage`, `make fast-test`.

### 10.2 The invariant & leakage suites (most important to trust)

**`tests/unit/test_invariants.py`** — must hold for ANY input:
- position never exceeds `max_position_pct_of_equity`
- risk layer only *shrinks* size (liquidity multiplier bounded 0–1)
- kill switch always blocks (`manual_kill_switch_active`)
- tripped breakers block **and persist across restart**
- daily loss limit halts entries; `no_trade` → zero-size rejection
- `max_trades_per_hour` + `min_bars_between_orders` enforced
- duplicate orders impossible even across restarts (idempotency)
> Extend this file when adding any new risk control.

**`tests/unit/test_no_lookahead.py`** — leakage guards:
- features at bar *k* = pure function of bars `0..k`
- poisoning *future* bars changes no *past* features (bit-identical)
- walk-forward splits: no overlap, embargo respected
- triple-barrier labels depend only on their forward window

### 10.3 Lint & typecheck

```powershell
ruff check src/ scripts/ tests/                 # lint (make lint)
ruff check --fix src/ scripts/ tests/           # autofix
ruff format src/ scripts/ tests/                # format (make format)
mypy src/state/ src/reporting/                  # strict modules (make typecheck)
mypy src/                                       # global lenient pass (also in CI)
```

Config: line-length 120, target py310, rules E/W/F/I/B (E501, B008 ignored;
E402 ignored under `scripts/**` and `tests/**`). `src/state` + `src/reporting`
are **strict** (`disallow_untyped_defs`, `warn_return_any`); everything else
lenient.

`make check` = lint + typecheck + tests (what CI runs).

### 10.4 CI (`.github/workflows/ci.yml`)

On push/PR to `main`, three jobs on `ubuntu-latest` / Python 3.12 /
`pip install -e ".[dev]"`:
1. **Lint** — `ruff check src/ scripts/ tests/`
2. **Type check** — `mypy src/state/ src/reporting/` then `mypy src/`
3. **Tests** — `pytest tests/ -v --cov=src ...` + coverage artifact

---

## 11. Ports & Networking

| Port | App | Override | Notes |
|---|---|---|---|
| **8571** | Streamlit decision demo | `STREAMLIT_PORT` / `--server.port` | bound to `127.0.0.1` |
| **8572** | Trade Map GUI | `TRADE_MAP_PORT` | runs alongside the demo |
| **8590** | handoff webhook endpoint | `handoff.endpoint_url` | off by default (`handoff.mode: off`) |
| ~~8501~~ | — | — | **never used** — reserved to avoid collisions |

All Streamlit apps bind `127.0.0.1` only (local research tool). **Never stop
another project's container/server to free a port — pick a different one:**

```powershell
$env:STREAMLIT_PORT = "8600"; python -m streamlit run scripts/demo_app.py --server.port=8600
$env:TRADE_MAP_PORT = "8600"; python scripts/trade_map_app.py
```

---

## 12. Environment Variables

Secrets live **only** in the environment — never in YAML.

| Variable | Used by | Purpose |
|---|---|---|
| `BTC_BOT_API_KEY` | CCXT, `validate_live_backend.py` | exchange API key |
| `BTC_BOT_API_SECRET` | CCXT, `validate_live_backend.py` | exchange API secret |
| `BTC_BOT_TELEGRAM_TOKEN` | `src/alerting/notifier.py` | Telegram bot token |
| `BTC_BOT_TELEGRAM_CHAT_ID` | `src/alerting/notifier.py` | Telegram chat id |
| `BTC_BOT_AGENT_SECRET` | `src/handoff/agent_sink.py` | webhook signing secret |
| `TRADE_MAP_PORT` | `trade_map_app.py` | GUI port (default 8572) |
| `TRADE_MAP_FORCE_RENDER` | `trade_map_app.py` | `1` = run `main()` outside Streamlit (tests) |
| `DECISIONS_JSONL` | logging/GUI | override path of `decisions.jsonl` |
| `REPLAY_BUDGET_SECONDS` | GUI replay test only | test budget |

Docker/compose (translated to CLI flags by `scripts/entrypoint.sh`):

| Variable | Default | Becomes |
|---|---|---|
| `TRADING_MODE` | `historical` | `--mode` |
| `TRADING_ENV` | `paper` | `--env` |
| `INPUT_DATA` | — | `--input` |
| `SYMBOL` | `BTC/USDT` | `--symbol` |
| `MAX_BARS` | — | `--max-bars` |
| `INITIAL_EQUITY` | `10000` | `--initial-equity` |
| `NETWORK_RETRY_MAX` / `_BASE_DELAY` / `_BACKOFF` | 5 / 10 / 2 | live-mode retry policy |
| `SHUTDOWN_TIMEOUT` | 30 | graceful shutdown seconds |

PowerShell example:

```powershell
$env:BTC_BOT_API_KEY = "your-key"
$env:BTC_BOT_API_SECRET = "your-secret"
python scripts/validate_live_backend.py --testnet
```

> Note: there is **no `load_dotenv()`** in the code — a `.env` file is read by
> **docker-compose** only. Locally, set variables in your shell.

---

## 13. Known Gotchas & Troubleshooting

### 13.1 Install

| Symptom | Cause | Fix |
|---|---|---|
| `Could not find setuptools>=68.0` | build isolation can't reach PyPI | `pip install -e ".[...]" --no-build-isolation` |
| `IncompleteRead` mid-download | flaky network | retry loop (§2.3) — pip's cache resumes progress |
| Dependency-conflict warnings about `crewai`, `wandb`, `opentelemetry` | unrelated packages sharing global site-packages | harmless for this project; use a venv to isolate |
| `import src.gui` fails after install | flat-layout auto-discovery | reinstall — `pyproject.toml` sets `py-modules = []` + `include = ["src*"]` precisely for this |

### 13.2 Running

| Symptom | Cause | Fix |
|---|---|---|
| `--mode historical requires --input` | missing file flag | pass `--input <file>` |
| `LiveTradingNotConfirmedError` | two-flag guard | expected unless you *deliberately* set `confirm_real_money: true` |
| Preflight FAILS: "~98% of bars would fail max_spread_pct=0.05" | shipped synthetic liquidity is wider than the shipped gate | **known issue**; use `--relax-liquidity` for smoke runs, or calibrate via `calibrate_synthetic_liquidity.py` / `paper_live` paired samples |
| Backtest exits "Provide --input ... or --synthetic ..." | neither source flag given | add `--input FILE` or `--synthetic N` |
| `--single-expert typo` produces zero trades | unknown names are silently ignored | use exactly `trend_expert` / `mean_reversion_expert` / `breakout_expert` |
| `--refit` seems to do nothing | **it is currently a no-op** | don't rely on it |
| `diagnose --trade-id` says not implemented | stub | use `replay_trade.py --trade-id` |
| Empty Trade Map / demo | no artifacts yet | pick `synthetic`/`replay` source, or run a backtest first |
| `make` fails on Windows | Makefile needs bash | use Git Bash/WSL, or run the `python ...` command directly |
| Models fall back to rules | granularity or feature-version mismatch | retrain under current config (`retrain_all.py`) |

### 13.3 Config footguns (verified in code)

1. **`confirm_real_money` is `false` in BOTH overlays** — including
   `settings.live.yaml`. You must edit it yourself to go live.
2. **`risk.circuit_breakers.consecutive_losses_hard_halt: 6` is dead** — no
   code trips it; `validate_config_consistency.py` hard-errors on it.
3. **`execution.min_bars_between_orders` is invisible to the RiskLayer** —
   `run.py` passes only the `risk:` block, so the effective value comes from
   `risk.circuit_breakers.min_bars_between_orders` (absent → hardcoded default
   2, same number by coincidence). Move the key under `risk.circuit_breakers`
   to make it authoritative.
4. **`max_concurrent_positions`**: `< 1` raises `ValueError`; `> 1` is accepted
   but `process_bar` still opens at most **one** position (real multi-position
   support is future work).
5. **`settings.local.yaml` is mounted by docker-compose but never read** — the
   loader only merges `settings.base.yaml` + `settings.{env}.yaml`.
6. **`ConfigWatcher` and the Telegram `Notifier` are built but not wired** —
   config edits need a restart; alert events aren't dispatched yet.
7. **Lists replace, never merge** in overlays — a one-key list in an overlay
   overwrites the entire base list.
8. **Legacy `config/settings.yaml`** is a Phase-1–3 artifact; the layered
   loader never reads it.

### 13.4 Safety checklist before live trading

```powershell
# 1. Credentials present (never in YAML)
#    $env:BTC_BOT_API_KEY / $env:BTC_BOT_API_SECRET

# 2. Config deliberately edited: execution.confirm_real_money: true
#    and execution.mode: "live" in config/settings.live.yaml

# 3. Validation passes
python scripts/preflight.py --env live
python scripts/validate_config_consistency.py

# 4. Testnet rehearsal passed
python scripts/validate_live_backend.py --testnet

# 5. Kill switch understood and reachable
New-Item -ItemType File -Path data\control\EMERGENCY_STOP -Force   # stops everything
Remove-Item data\control\EMERGENCY_STOP                            # re-arm

# 6. Start
python scripts/run.py --mode live --env live
```

---

## Appendix A — Complete CLI Reference

### `run.py`
```
--env {paper,live}         default paper
--mode {historical,live}   default historical
--input PATH               OHLCV CSV/Parquet (required for historical)
--symbol SYMBOL            overrides system.symbol
--max-bars INT             cap bars (historical)
--initial-equity FLOAT     default 10000.0
```

### `full_stack_walk_forward_backtest.py`
```
--input PATH | --synthetic N        (one required)
--env {paper,live}                  default paper
--config PATH                       single-file; overrides --env
--symbol SYMBOL
--window-bars INT                   default 1440
--initial-equity FLOAT              default 10000.0
--relax-liquidity                   in-memory gate relaxation
--skip-baselines                    skip 4 baselines
--single-expert {trend_expert,mean_reversion_expert,breakout_expert}
--no-risk-layer
--no-regime-gating
--refit                             (no-op today)
--report-dir PATH                   default data/reports
```

### `diagnose.py`  *(exactly one primary flag required)*
```
--timestamp INT | --trade-id STR | --summary | --faults      (mutually exclusive)
--since YYYY-MM-DD             pairs with --summary
```

### `replay_trade.py`  *(`--list` or `--trade-id` required)*
```
--list [N]                     bare = 15
--trade-id ID
--window-bars INT              default 6
--db PATH                      default data/registry.db
--input PATH                   OHLCV for market context
--env {paper,live}             default paper
```

### `dashboard.py`
```
--once                         one static frame to stdout, then exit
--refresh-seconds FLOAT        default 5.0
keys: 1..6 views · r refresh · q quit
```

### `trade_map.py`
```
--source {auto,artifacts,replay,synthetic}   default auto
--symbol SYMBOL                default BTC/USDT
--timeframe TF                 default auto
--out PATH.html                no default (text summary)
--max-bars INT                 default 1500 (replay)
--bars INT                     default 900 (synthetic)
--seed INT                     default 42
--relax-liquidity
--indicators LIST              default ema_fast,ema_slow
--panes LIST                   default none
--env {paper,live}             default paper
```

### `benchmark_latency.py`
```
--bars INT                     default 300 (+110 warmup)
--report-dir PATH              default data/reports
```

### `generate_figures.py`
```
--input PATH | --synthetic N   (one required)
--env {paper,live}             default paper
--relax-liquidity
output: <repo>/docs/images/*.png   (absolute, not CWD-relative)
```

### `preflight.py` / `validate_config_consistency.py` / `status.py`
```
preflight.py:                     --env {paper,live}   --config-dir DIR (default config)
validate_config_consistency.py:   --config PATH (default config/settings.base.yaml)
                                  --db-path PATH (default data/registry.db)
status.py:                        --env {paper,live}
```

### `reset_breaker.py`
```
--type {consecutive_execution_errors,max_drawdown,consecutive_losses_hard,
        daily_loss_limit,data_staleness}      REQUIRED
--db-path PATH                    default data/registry.db
```

### `validate_live_backend.py`
```
--testnet                         default True (cannot be disabled)
--output PATH                     default data/reports/live_validation_<UTC>.md
requires: BTC_BOT_API_KEY, BTC_BOT_API_SECRET
```

### `retrain_all.py`
```
--input PATH                      REQUIRED
--config PATH                     single-file; overrides --env
--env {paper,live}                default paper
--components [NAME ...]           subset; default all from config
valid: zone_survival regime_hmm regime_classifier trend_expert
       mean_reversion_expert breakout_expert opportunity_filter router risk_model
```

### `approve_model.py` / `rollback_model.py`
```
approve_model.py:
  --component NAME     default: all components with pending versions
  --yes                auto-confirm (scripted)
  --force              bypass harness-alignment gate
  --db-path PATH       default data/registry.db

rollback_model.py:
  --component NAME     REQUIRED
  --to-version ID      omit = list available versions (dry run)
  --db-path PATH       default data/registry.db
```

### `demo_app.py`
```
--input PATH            direct-python use only
--bars INT              default 1200        (streamlit run uses sidebar instead)
```

---

## Appendix B — Makefile Targets

> Requires **bash** (`SHELL := /bin/bash`) and creates `.venv` via `install-dev`.

| Target | Runs |
|---|---|
| `help` | list targets |
| `test` | `python3 -m pytest tests/ -v` |
| `unit` / `integration` / `e2e` | `.venv/bin/python -m pytest tests/<tier> -v` |
| `coverage` | pytest + `--cov=src` term + html |
| `fast-test` | `pytest tests/ -x -q --tb=short` |
| `lint` | `ruff check src scripts tests` |
| `format` | `ruff check --fix` + `ruff format` |
| `typecheck` | `mypy src/state/ src/reporting/` |
| `check` | lint + typecheck + test *(what CI runs)* |
| `backtest` | full-stack backtest on `data/raw/BTCUSDT.csv --env paper` |
| `benchmark-smoke` | 4-run ablation ladder → `data/reports/benchmark_<date>` |
| `preflight` | `scripts/preflight.py --env paper` |
| `pipeline` | `python3 -m src.pipeline` dry-run (100 sandbox bars) |
| `dashboard` / `dashboard-headless` | TUI / `--once` |
| `latency` | `benchmark_latency.py --bars 300` |
| `figures` | `generate_figures.py --synthetic 900 --relax-liquidity` |
| `demo` | Streamlit demo, `STREAMLIT_PORT ?= 8571` |
| `trade-map` | Trade Map GUI, `TRADE_MAP_PORT ?= 8572` |
| `trade-map-export` | `trade_map.py --source auto --out data/reports/trade_map.html` |
| `build-image` | `docker compose build` |
| `install-dev` | venv + requirements + `pip install -e ".[dev]"` |
| `clean` | remove caches/build artifacts (keeps `.venv`) |
| `golden-update` | `python3 tests/golden/digest.py` |

---

## Appendix C — Config Key Reference

Top-level sections of `config/settings.base.yaml` (24): `system`, `data`,
`feature_engine`, `market`, `structure`, `regime`, `labeling`, `trade_duration`,
`experts`, `opportunity_filter`, `router`, `models`, `backtest`, `liquidity`,
`risk`, `execution`, `retraining`, `monitoring`, `alerting`, `storage`,
`logging`, `pipeline`, `handoff`.

> Note: the **regime detector** config lives under `regime:` and the
> **structure engine** under `structure:`. There is no `feedback:` block —
> feedback paths live under `storage:`.

### `system`
| Key | Shipped | Notes |
|---|---|---|
| `data_granularity` | `"1m"` | `10s`/`30s`/`1m`/`5m` — the only timeframe switch |
| `exchange` | `"binance"` | |
| `sandbox` | `true` | `false` only in live overlay |
| `symbol` | `"BTC/USDT"` | |

### `feature_engine`
| Key | Shipped |
|---|---|
| `momentum_periods` | `{short: 1, medium: 5, long: 10}` |
| `volatility_periods` | `{short: 5, medium: 5}` |
| `rsi_period` | `14` |
| `sma_periods` | `[9, 21]` |
| `volume_zscore_period` | `20` |

### `structure` / `regime`
| Section | Key | Shipped |
|---|---|---|
| `structure` | `lookback_bars` / `pivot_window` / `min_zone_separation_pct` / `zone_decay_bars` | 50 / 5 / 1.0 / 100 |
| `regime` | `n_states` | 3 (+ panic handled separately) |
| `regime` | `confidence_floor` / `persistence_window_bars` / `panic_z_threshold` | 0.35 / 20 / 3.0 |

### `labeling` / `trade_duration`
| Key | Shipped | Meaning |
|---|---|---|
| `max_holding_bars` | 5 | label resolves within 5 bars |
| `fee_pct` / `slippage_buffer_pct` | 0.04 / 0.02 | |
| `k_tp` / `k_sl` | 1.5 / 1.0 | barriers = k × volatility |
| `regime_specific_barriers` | per regime | overrides of k_tp/k_sl |
| `trade_duration.min_trade_seconds` | 10 | fill→exit minimum; `0` disables |
| `trade_duration.require_aligned_models` | true | refuse misaligned models |

### `experts` (per expert)
| Key pattern | Meaning |
|---|---|
| `regime_requirement` | list of regimes where the expert may fire |
| `regime_confidence_floor` | min regime confidence (0.35–0.4 shipped) |
| `fallback_confidence/expected_return/expected_risk` | rule-based fallback when model unavailable |
| `model_path` | `data/models/experts/<name>.pkl` |
| trend: `momentum_threshold: 0.05`, `persistence_requirement: 0.5` | |
| mean_reversion: `zone_proximity_pct: 0.3`, `rsi_extremity_threshold: [20,80]` | |
| breakout: `compression_pct_threshold: 10`, `volume_zscore_surge_threshold: 1.0` | |

### `opportunity_filter`
| Key | Shipped | Meaning |
|---|---|---|
| `hard_rules.edge_vs_fees_threshold` | 0.02 | min edge over fees |
| `hard_rules.max_spread_pct` | **0.05** | the gate that fails preflight (§13.2) |
| `hard_rules.min_depth_usdt` | 10000 | |
| `hard_rules.volatility_floor/ceiling_pct` | 0.01 / 1.5 | |
| `hard_rules.reject_on_regime` | `[PANIC_SPIKE]` | |
| `hard_rules.no_consensus_ceiling` | 0.7 | |
| `hard_rules.reject_if_best_confidence_below` | 0.3 | |
| `ml_model.blend_weight` | 0.3 | 30% model / 70% rules; **rules always win conflicts** |
| `regime_overrides` | keyed by regime | layered on top of hard_rules |

### `risk`
| Key | Shipped | Enforced in |
|---|---|---|
| `sizing.risk_pct_per_trade` | 0.5 | sizing formula |
| `sizing.max_position_pct_of_equity` | 20.0 | invariant test |
| `hard_rules.min/max_invalidation_distance_pct` | 0.1 / 3.0 | |
| `circuit_breakers.max_drawdown_pct` | 8.0 | persistent (SQLite) |
| `circuit_breakers.daily_loss_limit_pct` | 2.0 | |
| `circuit_breakers.consecutive_execution_errors_halt` | 3 | |
| `circuit_breakers.consecutive_losses_soft_pause/hard_halt` | 3 / 6 | ⚠ hard_halt is dead (§13.3) |
| `circuit_breakers.max_trades_per_hour` | 10 | `RiskLayer.evaluate()` |
| `circuit_breakers.max_concurrent_positions` | 1 | `process_bar` |
| `circuit_breakers.data_staleness_halt_seconds` | 120 | |
| `ml_model.blend_weight` | 0.25 | |

### `execution`
| Key | Shipped | Notes |
|---|---|---|
| `mode` | `paper_sim` | `paper_sim`/`paper_live`/`live` |
| `order_type` | `limit` | |
| `limit_timeout_bars` | 2 | |
| `partial_fill_policy` | `cancel_remainder` | alt `market_fill_remainder` |
| `slippage_pct` | 0.01 | |
| `confirm_real_money` | false | **second live flag** |
| *(optional)* `simulated_latency_ms` | 0.0 | paper realism knob |
| *(optional)* `latency_adverse_pct_per_100ms` | 0.0 | |
| *(optional)* `order_rejection_rate` | 0.0 | deterministic, seed 42 |

### `monitoring` / `retraining` / `alerting` / `storage`
| Section | Key | Shipped |
|---|---|---|
| `monitoring.drift` | `psi_alert_threshold` | 0.25 |
| `monitoring.performance` | `rolling_window_trades` | 50 |
| `retraining` | `promotion_margin_pct` / `min_validation_samples` | 5.0 / 500 |
| `retraining` | `schedule` | `"0 3 * * *"` (03:00 daily) |
| `alerting` | `enabled` | false |
| `storage` | `registry_db` / `zone_events` / `system_state_archive` | `data/registry.db`, `data/logs/zone_events.parquet`, `data/logs/system_state/` |

---

## Document maintenance

This guide reflects the code as of **October 2026** (v5.0.0). Where the
shipped config and the docs disagree (dead keys, unwired features), the code
was checked directly and the discrepancy is documented in §13.3 rather than
hidden. Update this file in the same commit as any CLI or config change.
