# TODO — deferred work requiring real data / measured numbers

Infrastructure for every item below already exists and is tested. What's
missing is **real market data and real measured results** — per
`docs/PRESENTATION.md` §28 ("only report numbers you actually measured"),
none of these get published until they're genuinely run. Personal
scratch notes live in `docs/todo.md`; this file is the engineering queue.

## 1. Real-data benchmark + ablation publication (highest priority)

- [ ] Acquire BTCUSDT OHLCV history (target: 2024-01-01 → 2026-01-01, 1m)
      into `data/raw/BTCUSDT.csv`.
- [ ] Run `make backtest` (full harness, 4 baselines) on the real data.
- [ ] Run the full ablation ladder on real data:
      `--single-expert {trend_expert, mean_reversion_expert, breakout_expert}`,
      `--no-risk-layer`, `--no-regime-gating`.
- [ ] Populate README's benchmark table + `docs/research-report.md` §11/§12
      from the generated reports (never hand-typed).
- [ ] Retrain all components on real data (`scripts/retrain_all.py`), review
      walk-forward metrics vs champions, promote via `approve_model.py`.

## 2. Figures from real data

- [ ] `python scripts/generate_figures.py --input data/raw/BTCUSDT.csv`
- [ ] Replace the ILLUSTRATIVE (synthetic) images in `docs/images/` +
      README with real-data renders (keep the synthetic set labeled as such
      if useful for the demo).

## 3. Calibration on real data

- [ ] Expert reliability diagrams + router calibration from the real run's
      report (`calibration` block); publish miscalibration scores in the
      research report.

## 4. Failure analysis report (PRESENTATION §11)

- [ ] Run a multi-week paper-trading session (`run.py --mode historical`
      replay first, then `paper_live`).
- [ ] Aggregate `data/logs/faults.jsonl` + losing-trade decision traces
      into the failure taxonomy (false breakouts / regime misclassification
      / slippage / filter gaps / execution errors).
- [ ] Write the Problem → Root Cause → Evidence → Fix → Before/After
      narrative into `docs/research-report.md` §13.

## 5. Paper-trading campaign (gate for any live claim)

- [ ] ≥ 4 weeks of `paper_live` sessions; journal + decisions.jsonl as
      evidence.
- [ ] Compare paper-live fills vs synthetic-liquidity predictions
      (paired samples already logged) → recalibrate
      `liquidity.synthetic.*` via `scripts/calibrate_synthetic_liquidity.py`.
- [ ] Only after the campaign: revisit `execution.mode` progression
      (testnet → limited live), per `validate_live_backend.py` runbook.

## 6. Software performance under real load

- [ ] Re-run `make latency` against a real-data session (current numbers
      are synthetic-harness measurements on dev hardware) and record the
      hardware block in the research report.

## 7. Harness upgrades unlocked by real data

- [ ] Per-fold model refitting in `full_stack_walk_forward_backtest.py`
      (currently fixed models + window segmentation — documented
      simplification).
- [ ] Turnover/cost sensitivity sweep: rerun ladder across
      `simulated_latency_ms` × `order_rejection_rate` grids.

## 8. Model upgrades (post-evidence)

- [ ] Contextual-bandit router; keep supervised router as the baseline it
      must beat (ADR-0005).
- [ ] Sequence encoder behind the existing `ExpertOutput` contract.
- [ ] WebSocket ingestion to replace REST polling staleness window.

---

*Rule of the road (PRESENTATION §28): never publish a number that wasn't
produced by a command in this repo, with its config hash + git commit in
`data/registry.db::experiments`.*
