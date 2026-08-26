# Limitations — Where This System Does Not Work

A serious engineer can say where their system fails. This page is the
honest list; the failure-handling mechanics live in
[`explain/failures-and-fallbacks.md`](explain/failures-and-fallbacks.md).

## Market-level

- **Non-stationarity.** The edge any component measures decays. Weekly
  retraining is a patch, not a cure; the feature set itself can go stale
  (see `reasoning_for_decisions.md` — "dynamic features or no features").
- **Regime-shift lag.** Detection is backward-looking (HMM filter +
  persistence windows). The first bars of every regime change are
  misclassified by construction; the panic-spike override catches only
  the violent ones.
- **Regime misclassification cascades.** Experts are regime-gated; a
  wrong regime label suppresses the right expert and may surface the
  wrong one. The router can only rank what the gate lets through.

## Data & simulation fidelity

- **Synthetic liquidity.** Historical replay has no order-book history;
  spread/depth come from a feature-conditioned proxy. Real fills may be
  worse. Paper-live mode logs paired (synthetic, real) samples for
  calibration — until that dataset matures, absolute fill quality is
  unverified.
- **OHLCV-only bars.** No trades/tape/depth-of-book. Orderflow features
  are proxies (delta of bar volume vs range), not true aggressor flow.
- **Latency is modeled, not measured, in backtests.** The
  `simulated_latency_ms` knob is a linear stress test, not a market
  microstructure model.

## ML-level

- **Label bias in supervised routing.** The router learns from hindsight
  "which expert would have won" — a biased teacher. Contextual-bandit
  routing is the planned correction (ADR-0005).
- **Tree-model extrapolation.** LightGBM outputs are unreliable outside
  the training distribution; PSI drift monitoring alerts but cannot
  prevent degradation between retrains.
- **Feature-schema coupling.** Renamed features force retraining; the
  schema-version guard falls back to rule mode rather than silently
  mis-scoring, which caps damage but also caps capability.
- **No walk-forward model refitting inside the full-stack harness yet**
  (models are fixed per run; windows segment the timeline). Documented
  in the harness docstring as a deliberate simplification.

## Systems-level

- **One laptop.** Single process, one symbol, one concurrent position.
  No horizontal scale; a host crash halts everything (breakers and
  idempotent order IDs make the halt safe, not productive).
- **CCXT/REST polling.** No WebSocket yet: worst-case data staleness is
  one poll interval; the staleness breaker halts entries beyond it.
- **Single-exchange assumption.** Symbol, fees, and tick size conventions
  assume Binance; porting needs config + connector review.

## Evidence gaps (what we do NOT claim)

- **No live profitability claim.** All published numbers are backtest or
  paper results under stated assumptions; a multi-week paper-trading
  campaign is a prerequisite before any live capital (tracked in `TODO.md`).
- **Small-scale latency figures** (~1 ms/bar) come from the synthetic
  benchmark harness on developer hardware, not from production load.
