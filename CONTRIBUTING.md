# Contributing to Btc_trading_agents

Thanks for contributing. This is a trading research system with real-money
potential downstream, so a few rules are non-negotiable.

## Quick start

```bash
git clone https://github.com/devansh1012007/Btc_trading_agents.git
cd Btc_trading_agents

make install-dev     # creates .venv, installs runtime + dev deps
source .venv/bin/activate

make check           # lint + typecheck + tests (must pass before any PR)
```

## Development workflow

1. Create a feature branch from `main`.
2. Make your change. Keep diffs focused — one concern per PR.
3. Run `make check` locally. CI runs exactly: **ruff → mypy → pytest**.
4. Update documentation when behavior changes (`docs/explain/` holds the
   deep-dive docs; keep them truthful).
5. Open a PR describing *what* and *why*, not just *how*.

## Commit message format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```text
feat(risk): add exposure-based sizing gate
fix(router): guard against None model before predict_proba
test(backtest): validate transaction-cost modeling
docs(architecture): document state contract invariants
refactor(features): extract structural feature adapter
perf(pipeline): cache rolling buffers between bars
chore(ci): add coverage reporting
```

Scope is the layer/module: `data`, `features`, `structure`, `regime`,
`experts`, `opportunity`, `router`, `risk`, `execution`, `liquidity`,
`backtest`, `registry`, `monitoring`, `diagnostics`, `config`, `ci`, `docs`.

## Hard rules (do not break these)

- **Pipeline order is fixed**: `MarketSnapshot → FeatureVector → RegimeState →
  ZoneMap → Experts → OpportunityScore → RouterDecision → RiskDecision →
  ExecutionRecord → SystemState`. Regime runs before structure within a bar.
- **No look-ahead anywhere.** Every feature must be computable from data with
  timestamp ≤ bar close. If you add a feature, prove it's causal.
- **Hard rules always override ML** in `OpportunityFilter` and `RiskLayer`.
  Never let a model veto a hard-rule decision or vice versa.
- **Models are never auto-promoted.** Retraining produces
  `pending_approval` versions; promotion is manual via
  `scripts/approve_model.py`.
- **Walk-forward only.** New evaluation code must use `WalkForwardSplitter`
  (purge + embargo). Random or `TimeSeriesSplit` splits leak labels.
- **One concurrent position max** (`max_concurrent_positions: 1`).
- **Live trading requires both** `execution.mode: "live"` AND
  `execution.confirm_real_money: true`. Never loosen this guard.
- **Never commit secrets or API keys.** `.env` is gitignored; use
  `.env.example` as the template.
- **Never commit datasets or model binaries.** `data/` is gitignored.

## Testing expectations

- Every bug fix ships with a regression test that fails without the fix.
- New layers/components need unit tests plus at least one integration test
  through `TradingPipeline`.
- Financial-invariant tests live in `tests/unit/test_invariants.py` — extend
  it when adding risk controls (position limits, exposure caps, breaker
  persistence).
- Tests have no `conftest.py`; use inline helper functions like existing
  test files do.

## Type checking

`src/state/` and `src/reporting/` are held to strict mypy
(`disallow_untyped_defs`) because every layer depends on their types.
New modules should be fully annotated; the strict set grows over time.

## Configuration changes

- Runtime defaults live in `config/settings.base.yaml`; env overlays
  (`settings.paper.yaml` / `settings.live.yaml`) win on merge.
- Changing units (pct vs fraction, USDT vs base) requires updating
  `scripts/validate_config_consistency.py` checks too — run it after any
  threshold edit.
- Hot-reloadable paths (`feature_engine`, `risk.circuit_breakers`,
  `labeling`, `opportunity_filter.hard_rules`) must never include
  capital-risk-structural settings.

## Reporting issues

Include: what you ran, what you expected, what happened, and if it's a
pipeline issue, attach the relevant decision trace from
`data/logs/decision_traces/` (redact anything account-specific).
