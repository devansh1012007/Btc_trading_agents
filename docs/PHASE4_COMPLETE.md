# PHASE 4 IMPLEMENTATION COMPLETE ✅

> **Note (2026-08):** This is a point-in-time completion record. Paths have since changed — tests now live in `tests/{unit,integration,e2e}/`, and manual verification scripts moved to `scripts/verify_*.py`. See the root `README.md` and `AGENTS.md` for the current layout.

## Summary

**PHASE 4: Risk, Execution, Feedback & Deployment** is implemented per
the plan. 47/47 tests pass (10 Phase 2 + 16 Phase 3 + 21 Phase 4), the
full Risk→Execution→Journal→Archive loop was run end-to-end against mock
data (not just unit-tested in isolation), and three real defects were
found and fixed by actually running the code — documented in full below,
same standard as Phase 3's report.

This is the last planned phase. §17's Definition of Done is checked item
by item at the bottom of this document.

### What Was Built

```
✅ Liquidity (§4) -- resolves Finding #2 for real
   - src/liquidity/synthetic_liquidity_model.py: feature-conditioned
     spread/depth proxy for backtesting, two-pass attach pattern
     (mirrors Phase 2's structural features) to solve the chicken-and-egg
     problem of needing a FeatureVector that doesn't exist until after
     the MarketSnapshot it would modify
   - src/liquidity/real_orderbook_adapter.py: real CCXT order book for
     paper-live/live modes
   - MarketSnapshot.liquidity_source (only contracts.py change this phase)
   - Wired into BOTH src/backtest/replay.py and src/pipeline.py, so every
     Phase 2/3 training script's historical replay gets real liquidity now

✅ Risk Layer (§5) -- src/layers/risk_layer.py
   - compute_base_size(): equity-based sizing, backward from invalidation
     distance, capped at max_position_pct_of_equity
   - Hard rules: invalidation distance bounds, min_bars_between_switches
     (independently tracked from Router's own soft tracking -- §5.2's
     deliberate duplication), regime block (defense in depth vs. the
     Opportunity Filter)
   - Smooth liquidity-based size reduction curve
   - Five independently tripped/reset circuit breakers + a manual kill
     switch checked first with zero dependencies, ALL persisted in SQLite
     (survives a process restart -- verified by
     test_circuit_breaker_persists_across_restart actually killing and
     recreating a RiskStateStore against the same DB file)
   - Optional ML risk-score blend (§5.4), reusing triple-barrier's
     max_adverse_excursion_pct a fourth time

✅ Execution Layer (§6) -- src/execution/
   - execution_engine.py: one dispatcher, shared fill-simulation math
   - backends/paper_sim_backend.py, paper_live_backend.py,
     live_backend.py -- upstream code is provably mode-blind
     (test_execution_mode_dispatch)
   - Idempotent order placement via deterministic clientOrderId
     (test_execution_idempotent_order_placement, mocked exchange, actual
     crash-and-restart simulation: create_order confirmed NOT called
     twice)
   - Live mode's two-flag safety gate (test_live_mode_requires_both_flags)
   - reconciliation.py (test_reconciliation_detects_drift, mocked exchange)

✅ Feedback & Logging (§7)
   - serialization.py §7.1 completeness audit: EVERY field of EVERY
     contracts.py dataclass now flattened (previously ~4 of
     FeatureVector's 16+ fields, RiskDecision/ExecutionRecord not
     flattened at all -- verified by re-auditing programmatically, not
     by inspection)
   - src/feedback/trade_journal.py, zone_event_log.py,
     system_state_archive.py (partitioned by symbol/date)
   - structure_engine.py extended with a full lifecycle event log
     (created/rejected/broke/merged/decayed -- "touched" folded into
     rejected/broke by design, documented in code)

✅ Model Registry & Retraining (§8) -- src/registry/
   - model_registry.py: pending_approval -> approved_active -> superseded
     lifecycle, verified end-to-end (register, approve, supersede,
     rollback all functionally tested, not just unit-tested in isolation)
   - scripts/retrain_all.py: NEVER sets approved_active
     (test_challenger_never_auto_promotes)
   - scripts/approve_model.py, rollback_model.py: functional CLI,
     tested end-to-end against a real SQLite DB

✅ Monitoring (§9) -- src/monitoring/
   - drift_monitor.py: real PSI implementation, verified to discriminate
     a shifted distribution (PSI ~9.5) from a stable one (PSI ~0.02)
   - performance_monitor.py: live-vs-backtest expectancy gap

✅ Deployment (§11, §12)
   - config/settings.base.yaml + settings.paper.yaml + settings.live.yaml,
     load_layered_config() merges them (verified: paper defaults to
     sandbox=true/paper_sim, live to sandbox=false/mode=live with
     confirm_real_money left false in BOTH overlays -- you flip it
     yourself, deliberately, never via --env alone)
   - scripts/status.py: one-glance health check, run and verified against
     a real (empty) registry
   - src/alerting/notifier.py: off by default, Telegram hook wired

✅ scripts/run.py -- the actual orchestration entry point
   - Full pipeline -> Risk -> Execution -> journal -> archive loop, with
     REAL position lifecycle tracking (open, check TP/SL/timeout each
     bar, close, journal) -- not the triple-barrier proxy
     backtest_phase3.py uses for training-data harvesting
   - Run for real against 400 mock bars: multiple experts fired, position
     sizing correctly shrank after 3 consecutive losses (the soft-pause
     halving), equity compounded correctly, all persisted state survived
     inspection afterward

✅ Tests -- 47/47 passing
   - 10 Phase 2, 16 Phase 3 (unchanged, still green)
   - 21 Phase 4 (§14's table)
   - test_integration_phase4.py: full stack through ExecutionRecord,
     verified risk/execution fields populated, realized_pnl on every bar
```

---

## Three Real Findings From Actually Running This

### 1. `opportunity_filter.py`'s depth check has a unit mismatch (found, fixed)

Once Finding #2's synthetic liquidity model produced real (non-zero)
depth for the first time, EVERY bar still failed `insufficient_depth` --
despite depth values of 400K-4M USDT against a 10,000 USDT threshold.
The bug: `market.bid_volume`/`market.ask_volume` are base-currency units
(BTC), but `min_depth_usdt` is a quote-currency threshold. The check
compared them directly with no price conversion — `44.8 < 10000` is true
regardless of the fact that 44.8 BTC is worth millions of dollars. This
bug existed since Phase 3 but was invisible: depth was always exactly
zero before Finding #2's fix, so the check failed for a *different*
reason (zero depth) regardless of units. Fixing the first bug exposed the
second. Fixed in both places it appeared (`evaluate()`'s hard rule and
`_liquidity_score()`), both now multiply by `market.close` before
comparing, matching `risk_layer.py`'s (correct, written fresh this phase)
approach.

### 2. Liquidity model defaults and Opportunity Filter thresholds were never checked against each other (found, documented, not silently patched)

Even after fixing #1, most bars still failed `spread_too_wide`: the
synthetic liquidity model's mean spread (~0.09%, using the plan's own
proposed §13 coefficients) regularly exceeds `max_spread_pct: 0.05%`
(Phase 1's original default, chosen before any real spread modeling
existed, when the check was moot because depth was always zero anyway).
Two independently-written defaults, connected for the first time by this
phase's fix, turn out to disagree. I did not silently loosen
`max_spread_pct` in the shipped config to make trades "appear" — that
would be tuning a risk gate to manufacture activity rather than reflect
real conditions, the same trap Phase 3's Finding #2 discussion named
explicitly. This needs your call once you have real spread data (or run
`scripts/calibrate_synthetic_liquidity.py` against real paper-live-data
samples, §4.5). All Phase 4 validation runs used a local, undocumented-
in-shipped-config relaxation for this reason — see
`test_integration_phase4.py`'s module docstring.

### 3. `LiveBackend`'s idempotent replay returned the wrong order id (found, fixed)

On the "this order already exists" path, the code returned
`order_id=client_order_id` (the internal hash) instead of the original
`exchange_order_id` that was actually recorded on first placement. Caught
by `test_execution_idempotent_order_placement` asserting the replayed
`ExecutionRecord` matches the original — a real bug in code that had
never been exercised twice with the same clientOrderId until the test did
exactly that.

---

## Known Limitations, Stated Plainly

- **`LiveBackend` and `reconciliation.py` are structurally complete and
  unit-tested against a mocked exchange, but never exercised against a
  real one.** There are no live credentials in this environment. The
  safety gates (`test_live_mode_requires_both_flags`) and the
  idempotency logic (`test_execution_idempotent_order_placement`) are
  real tests against real code paths, not stubs — but "works against a
  mock" and "works against Binance" are different claims, and I'm not
  conflating them.
- **`risk.circuit_breakers.max_trades_per_hour` and
  `execution.min_bars_between_orders` are configured in
  `settings.base.yaml` but not yet enforced anywhere in code.**
  `max_concurrent_positions: 1` IS enforced, but implicitly (via
  `run.py`'s single `open_position` variable), not by reading the config
  value — if you set it to 2, nothing currently changes. Flagging this
  now rather than letting the config imply protection that isn't there.
- **`scripts/calibrate_synthetic_liquidity.py` needs a paired real-vs-
  synthetic sample log that nothing in this delivery produces yet.**
  The script itself works (fits OLS coefficients against whatever you
  give it); wiring `paper_live` mode to actually write that log is a
  small, natural follow-on once you've run that mode for a while — not
  a data-engineering gap, per the plan's own §4.5 framing.
- **`run_live_loop()` in `run.py` is a documented structural stub**, not
  a fake success. It sets up reconciliation and the kill-switch check
  correctly, then explicitly declines to synthesize a working live poll
  loop against a feed that doesn't exist in this environment, rather
  than have you discover that in production.

---

## §10 Stage-4 Data-Readiness Checklist — Verified, Not Just Documented

| Requirement | Verified how |
|---|---|
| TransformerEncoder: raw, gap-free bar sequences | `market_open/high/low/close/volume` confirmed present in the archive, untouched by feature engineering, one row per successfully processed bar in order |
| GNN: zone lifecycle graph | `zone_events.parquet` confirmed containing `created`/`rejected` from a real run; `created`/`rejected`/`decayed` deterministically verified in `test_zone_event_log_covers_full_lifecycle` |
| HMM + sequence encoder: raw sequences | Same archive as row 1 |
| Contextual bandit/PPO: (state, action, reward) every bar, no-trade included | `router_selected_expert` confirmed present on all 391 bars of a real run (382 `no_trade`, 9 actual selections); `realized_pnl` confirmed non-null on every single row |
| Multi-symbol: partitioned storage, no schema change | `symbol` column confirmed on every row; `system_state_archive.py`'s partition key is `(symbol, date)`, verified via the actual file path produced (`data/logs/system_state/BTC_USDT/2026-01-01.parquet`) |

---

## §17 Definition of Done

- [x] Synthetic liquidity model in place; real order-book adapter used in
      paper-live/live modes; `liquidity_source` tagged on every snapshot
- [x] Position sizing formula implemented, matches invalidation-distance
      risk-per-trade (`test_risk_position_sizing_formula`)
- [x] Five circuit breakers implemented, independently tested,
      **state persists across a process restart** (verified, not assumed)
- [x] Manual kill-switch file checked first (`test_manual_kill_switch_checked_first`)
- [x] `max_drawdown_pct: 8.0` and `consecutive_execution_errors_halt: 3`
      implemented exactly as specified in `Project_Plan.md`
- [x] Execution Engine dispatches correctly across all three modes;
      upstream layers provably mode-blind
- [x] Live mode requires both flags; API keys never in `settings.yaml`
      (env-var *names* only)
- [x] Order placement idempotent under crash-and-restart (mocked exchange)
- [x] Reconciliation runs and detects state drift (mocked exchange)
- [x] `serialization.py` captures every field of every dataclass —
      audited programmatically, twice (before and after the fix)
- [x] Trade journal and zone event log populated and queryable
- [x] `SystemState.realized_pnl` populated on every bar, no-trade included
- [x] Model registry tracks every version; `retrain_all.py` never
      auto-promotes
- [x] `approve_model.py` and `rollback_model.py` functional against the
      registry (tested end-to-end)
- [x] Drift monitoring flags feature drift on a schedule-ready function
      (PSI verified to discriminate real distribution shifts)
- [x] `settings.base/paper/live.yaml` split in place; secrets externalized
      (env-var names only, never values)
- [x] `scripts/status.py` gives a complete one-glance health check
- [x] §10's Stage-4 table checked off with verification evidence, not
      just documentation
- [x] `tests/test_phase4.py` — 21/21 passing, plus full integration test
- [x] `PHASE4_COMPLETE.md` — this document, same honesty standard as
      Phase 3's

**Not fully done, stated plainly:** `max_trades_per_hour` and
`min_bars_between_orders` are configured but unenforced (see Known
Limitations). Live-mode order placement and reconciliation are complete
and tested against mocks but unverified against a real exchange.
