# Model Registry & Feedback Loops

## Model Registry (`src/registry/model_registry.py`)

### Architecture

SQLite-backed version lifecycle. Single file (`data/registry.db`), shared with circuit breakers, trade journal, order tracking, and experiment log.

### Schema

```sql
CREATE TABLE model_versions (
    version_id TEXT PRIMARY KEY,
    component TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    data_granularity TEXT,
    file_path TEXT NOT NULL,
    walk_forward_metrics_json TEXT,
    status TEXT NOT NULL,           -- pending_approval | approved_active | superseded | rejected
    promoted_at TEXT,
    promoted_by TEXT
);

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

### Lifecycle

```
pending_approval → approved_active → superseded
       ↓
    rejected
```

### Functions

| Function | Purpose |
|---|---|
| `register_version(component, file_path, granularity, metrics, status)` | Register new model version |
| `get_active(component, db_path)` | Current approved model for component |
| `get_pending(component, db_path)` | List of pending models |
| `get_champion(component, db_path)` | Champion model (highest avg_auc) |
| `approve(version_id, db_path)` | Promote pending → approved_active, supersede previous |
| `reject(version_id, db_path)` | Mark as rejected |
| `rollback(component, version_id, db_path)` | Restore previous version as approved_active |
| `list_versions(component, db_path)` | All versions for component |

### Key Constraint: Never Auto-Promotes

`retrain_all.py` **never** sets `approved_active`. It only creates `pending_approval` rows. Promotion is exclusively `approve_model.py`'s job. Verified by `test_challenger_never_auto_promotes`.

### Promotion Margin

`retrain_all.py` compares challenger's `avg_auc` against champion's. If challenger clears `promotion_margin_pct` (default 5%), it's registered as `pending_approval`. The margin prevents marginal improvements from constantly displacing stable models.

## Retraining Flow (`scripts/retrain_all.py`)

### Design

One-shot script (invoked by cron/systemd timer, not long-running). For each component:

1. Run training steps (some have multi-step pipelines)
2. Read `.meta.json` for walk-forward metrics
3. Compare against champion's `avg_auc`
4. If challenger clears `promotion_margin_pct`, register as `pending_approval`
5. **Never touches `approved_active`**

### Component Steps

| Component | Steps | Output |
|---|---|---|
| `zone_survival` | `label_zone_events.py` → `train_zone_survival_model.py` | `data/models/zone_validity_model.pkl` |
| `regime_hmm` | `train_regime_models.py` | `data/models/regime_hmm.pkl` |
| `regime_classifier` | `train_regime_models.py` | `data/models/regime_classifier.pkl` |
| `trend_expert` | `train_triple_barrier.py` → `train_trend_expert.py` | `data/models/experts/trend_expert.pkl` |
| `mean_reversion_expert` | `train_mean_reversion_expert.py` | `data/models/experts/mean_reversion_expert.pkl` |
| `breakout_expert` | `train_breakout_expert.py` | `data/models/experts/breakout_expert.pkl` |
| `opportunity_filter` | `train_opportunity_filter.py` | `data/models/opportunity_filter.pkl` |
| `router` | `train_router.py` | `data/models/router.pkl` |
| `risk_model` | `train_risk_model.py` | `data/models/risk_model.pkl` |

### CLI

```bash
# Retrain all components
python scripts/retrain_all.py --input data/raw/BTCUSDT.csv --config config/settings.base.yaml

# Retrain specific components
python scripts/retrain_all.py --input data/raw/BTCUSDT.csv --components router risk_model

# Override config
python scripts/retrain_all.py --input data/raw/BTCUSDT.csv --config config/settings.base.yaml --components trend_expert
```

## Model Promotion (`scripts/approve_model.py`)

```bash
# Promote a pending model
python scripts/approve_model.py router router_v2.pkl

# Promote with reviewer name
python scripts/approve_model.py router router_v2.pkl --reviewer devansh
```

Promotes `pending_approval` → `approved_active`, marks previous as `superseded`.

## Model Rollback (`scripts/rollback_model.py`)

```bash
# Rollback to a previous version
python scripts/rollback_model.py router router_v1.pkl
```

Restores previous version as `approved_active`, marks current as `superseded`.

## Experiment Log (Phase 5)

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

## Feedback Loop Architecture

### What Gets Logged

| Artifact | Location | Content |
|---|---|---|
| Trade journal | SQLite (`trade_journal` table) | Entry/exit, PnL, expert, regime, confidence |
| Zone events | Parquet (`data/logs/zone_events.parquet`) | Zone lifecycle events |
| System state archive | Parquet (`data/logs/system_state/{symbol}/{date}.parquet`) | Every bar's full SystemState |
| Decision traces | JSONL (`data/logs/decision_traces/traces_YYYY-MM-DD.jsonl`) | Stage-by-stage bar journey |
| Faults | JSONL (`data/logs/faults.jsonl`) | Exception records with full context |
| Experiments | SQLite (`experiments` table) | Config hash, metrics, git commit |

### Retraining Targets

Each layer retrains on its own failure mode:

| Layer | Training Target | Label Source |
|---|---|---|
| Zone Engine | Zone survival/failure | Zone touch events → rejection/breakout |
| Regime Layer | Regime usefulness | HMM posteriors + panic override |
| Experts | Trade success/expectancy | Triple-barrier net return > 0 |
| Opportunity Filter | Tradability prediction | Best expert's triple-barrier return > 0 |
| Router | Expert selection reward | Hindsight: which expert would have been best |
| Risk Layer | Adverse excursion/loss | Stop-loss vs (take-profit/timeout) |

### Walk-Forward Splits

All training uses `WalkForwardSplitter` with purge + embargo. Never random or `TimeSeriesSplit` — it leaks labels.

```python
splitter = WalkForwardSplitter(
    train_bars=5000,
    test_bars=500,
    step_bars=100,
    embargo_bars=50,
)
```

The embargo prevents label leakage: a label whose outcome window extends into the test set doesn't contaminate training data in the next fold.

## Monitoring Integration

### Drift Detection

`drift_monitor.py` compares recent live feature distributions against training-time distributions via PSI. Alert threshold: 0.25 (significant shift).

```python
results = check_feature_drift(
    reference=train_features,
    current=live_features,
    psi_alert_threshold=0.25
)
# {"momentum_medium": {"psi": 0.32, "alert": True}, ...}
```

### Performance Gap Detection

`performance_monitor.py` compares live vs. backtest expectancy per component. Alert threshold: 30% gap.

```python
result = check_expectancy_gap(
    backtest_expectancy=0.05,
    live_expectancy=0.02,
    threshold_pct=30.0
)
# {"alert": True, "gap_pct": 60.0}
```

### Alerting Integration

When drift or performance gaps are detected, `alerting/notifier.py` sends Telegram notifications (if enabled). Event types: `pending_model_approval`, `circuit_breaker_tripped`, `execution_error`.

## Health Check (`scripts/status.py`)

One-glance health:

```
Mode: paper / live
Breakers: max_drawdown=active, daily_loss_limit=inactive
Pending models: router_v2.pkl (pending_approval)
Today's trades: 5, PnL: +$123.45, Win rate: 60%
```

Queries: `risk_state`, `circuit_breaker_halts`, `model_versions`, `trade_journal`.

## Breaker Reset (`scripts/reset_breaker.py`)

```bash
# Reset a specific breaker
python scripts/reset_breaker.py max_drawdown

# Reset all breakers
python scripts/reset_breaker.py all
```

Manual reset required for: `max_drawdown`, `consecutive_execution_errors`, `consecutive_losses_hard`. Auto-resets on: `daily_loss_limit` (new day), `data_staleness` (heartbeat resumes).
