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

## Trade Map GUI

What the GUI does **not** show, stated rather than implied:

- **No Greeks, no volatility surface.** Options are drawn as a price chart
  plus contract metadata (strike, expiry, type, multiplier); there is no
  Greeks UI.
- **No order-book depth.** The bars are OHLCV; liquidity is the synthetic
  proxy or a logged sample, never a depth ladder.
- **One symbol at a time.** There is no multi-symbol comparison view.
- **Replayed sessions use a throw-away risk store.** A replay's circuit
  breakers are in-memory and start clean, so a replay cannot reproduce a
  halt that a long-running production session had already tripped.
- **Equity from artifacts is derived.** The runtime writes no equity curve;
  the GUI rebuilds it from the trade journal (initial equity + realised
  PnL), so it ignores unrealised PnL between entry and exit.
- **Zones are levels, not boxes.** The zone log stores a price level per
  event, not the zone's edges; the GUI draws a level from creation to the
  event that ended it and invents no bounds.
- **A higher-timeframe candle shows the decision that got furthest in it**
  (``executed`` beats ``risk_layer`` beats ``opportunity_filter``...); the
  per-bar view (the "why" panel) is always at the base granularity.
- **Relaxed or synthetic sessions say nothing about edge.** The GUI labels
  them loudly; do not read expectancy from them.

## Minimum trade duration

- **No model has been trained under the new contract.** Existing pickles are
  legacy; with `require_aligned_models: true` they fall back to rule-based mode
  until `scripts/retrain_all.py` produces aligned versions.
- **Intrabar timing is not observable.** The 10 s rule is enforced on bar
  boundaries plus modelled latency. Live fills are stamped with the bar label, so
  live entry latency is the configured assumption, not a measurement.
- **The labeler is one expert at a time.** Router selection and the risk layer's
  size/veto are not replayed per sample; `evaluate_training_alignment.py`
  quantifies the remaining gap.

