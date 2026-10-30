# Evidence-Based Architecture Map & Traceability Matrix

## 1. End-to-End Execution Flow
1. **Ingestion (`MarketDataLayer` / `TickSource` / `BarBuilder`):** Converts raw OHLCV frames or tick/quote streams into immutable `MarketSnapshot` contracts.
2. **Perception (`TradingPipeline._process_bar`):**
   - `FeatureEngine`: Causal rolling features (`FeatureVector`).
   - `RegimeDetector`: Causal regime probability distribution and primary regime (`RegimeState`).
   - `DynamicStructureEngine`: Support/resistance zones (`ZoneMap`) + touch/lifecycle events (`confirmed_at_bar` look-ahead guard).
   - `attach_structural_features` & `attach_synthetic_liquidity`: Enriches `FeatureVector` and `MarketSnapshot`.
3. **Decision (`TradingPipeline` + `process_bar`):**
   - `ExpertLayer`: Evaluates enabled `BaseExpert` / `ExpertInstance` gates first, then model scorers (`ExpertOutputs`).
   - `OpportunityFilter`: Hard rules (spread, depth, volatility, regime vetoes, consensus) + optional ML tradability score (`OpportunityScore`). In `staged` order, hard rules stay pre-router and ML confirmation (`OpportunityConfirm`) runs post-router.
   - `Router`: Selects one expert instance or `no_trade` (`RouterDecision`), using Router A (with exit models), Router B (without exit models), or confidence fallback.
4. **Lifecycle, Sizing, Risk & Execution (`position_lifecycle.process_bar`):**
   - **Exit Pass (never gated by RiskLayer):** Iterates over open positions in `PositionBook`, applying the precedence ladder (Hard loss cap → Per-expert `ExitModel` → Expert `ExitPlan` incl. trailing stop → Hard config limits) after the `HoldPolicy` minimum-hold lockout, priced through `ExitFillModel` and `CostModel`, and routed through `ExecutionEngine.close_position`.
   - **Entry Pass:** Checked against `HaltManager`, `CooldownManager`, `PositionBook.can_open`, `RiskGovernor` (tighten-only), and `RiskLayer.evaluate` (kill switch, circuit breakers, rate limits, invalidation bounds, `ExitPlan`-unified base sizing, and ML risk multiplier `∈ [min_multiplier, 1.0]`).
   - **Order Placement & Hand-off:** `AgentSink` (modes `off | shadow | advisory | execute`) and `ExecutionEngine.place_order` (`PaperSimBackend`, `PaperLiveBackend`, `LiveBackend`).
   - **Observability:** `DecisionTrace`, `ModelLedger`, `trade_journal`, `system_state_archive`.

## 2. Core Invariants
- **ADR-0001 (Amended):** LightGBM for supervised tabular models; pluggable unsupervised/state-space backends for regime detection with NumPy-only serving filters.
- **ADR-0002:** Walk-forward validation with purge + embargo covering the full label horizon (`max_holding_bars` + exit delay); out-of-fold (OOF) stacking for downstream stages.
- **ADR-0004:** Hard rules always veto ML. ML sizing multipliers, the `RiskGovernor`, and `AgentSink` advisory replies may only reduce size, tighten limits, or veto — never enlarge risk or override a hard rejection.
- **ADR-0005:** Every learned component defaults to OFF and is promoted only when an out-of-sample ablation gate passes.
- **ADR-0006:** `@dataclass(frozen=True)` contracts in `src/state/contracts.py` (`mypy --strict`).
- **Single Exit & Cost Truth:** `barrier_exit_step` and `CostModel` are the sole implementations of exit and cost arithmetic across training labels, the walk-forward harness, live/paper loops, and GUI replay.
- **No Wall Clock in Decision Code:** All time-boxed halts, cool-downs, and daily accumulators read an injected `Clock` (`BarClock` in simulation, `WallClock` in live mode).

## 3. Request-to-Code Traceability Matrix

| Req # | Capability | Phase | Primary Modules | Config Keys | Tests |
|---|---|---|---|---|---|
| 1, 8 | Dynamic K regimes + 5 backends + legacy4 first fallback | P2, P5 | `src/state/contracts.py`, `src/regime/*`, `src/layers/regime_detector.py` | `regime.backend`, `regime.k.*`, `regime.serving_chain` | `test_regime_identity.py`, `test_regime_backends.py`, `test_regime_k_selection.py`, `test_regime_alignment.py`, `test_regime_serving_chain.py` |
| 2 | Zone survival → dynamic barrier & risk sizing ML models | P6, P7 | `src/labeling/excursions.py`, `src/sizing/barrier_model.py`, `src/sizing/risk_model.py`, `src/layers/risk_layer.py` | `sizing.barrier_model.*`, `sizing.risk_model.*` | `test_excursions.py`, `test_barrier_sizing_model.py`, `test_risk_sizing_model.py` |
| 3, 4, 5 | Per-expert exit models, `ExitPlan` (SL/TSL/TP/hold), exit & expert switches | P4, P9, P10 | `src/labeling/triple_barrier.py`, `src/exits/*`, `src/experts/*` | `exits.*`, `experts.instances` | `test_exit_plan.py`, `test_trailing_stop.py`, `test_plan_heads.py`, `test_exit_model_*.py`, `test_exit_ladder.py` |
| 6 | Per-regime expert instances + OOF top-N selection | P9 | `src/experts/expert_registry.py`, `src/experts/selection.py`, `src/backtest/harness_harvest.py` | `experts.instances`, `experts.selection.*` | `test_expert_instances.py`, `test_regime_chopped_data.py`, `test_top_n_selection.py` |
| 7, 10, 11 | Stage registry, training in harness order, stage switches, bundle alignment | P1, P11, P12 | `src/pipeline_stages.py`, `src/config/switchboard.py`, `src/registry/bundle.py`, `src/training/plan.py` | `pipeline.order`, `pipeline.stages.*` | `test_stage_registry.py`, `test_switchboard.py`, `test_bundle_manifest.py`, `test_training_plan.py`, `test_oof_stacking.py`, `test_refit_harness.py` |
| 9 | Scheduled zone refresh cadence | P6 | `src/layers/structure_engine.py`, `scripts/label_zone_events.py` | `pipeline.stages.zone_selection.*` | `test_zone_refresh.py`, `test_zone_policy_contract.py` |
| 12, 19 | 2 s entry/exit delay, unified `CostModel`, closing orders | P3 | `src/execution/cost_model.py`, `src/execution/price_path.py`, `src/execution/execution_engine.py`, `src/labeling/hold_policy.py` | `costs.*`, `execution.*` | `test_cost_model.py`, `test_price_path.py`, `test_exit_fill_model.py`, `test_closing_orders.py` |
| 13 | Market-agnostic profiles + ATM options | P15 | `src/market/instrument.py`, `src/market/profile.py`, `src/market/adapters.py`, `src/market/options.py` | `market.profile`, `market.derivatives.*` | `test_market_profile.py`, `test_adapters.py`, `test_atm_selector.py`, `test_option_costs.py` |
| 14 | Concurrent trades (`PositionBook`), cool-downs, time-boxed halts, `RiskGovernor` | P8, P11 | `src/utils/clock.py`, `src/execution/position_book.py`, `src/risk/cooldowns.py`, `src/risk/halts.py`, `src/risk/governor.py` | `positions.*`, `cooldowns.*`, `halts.*`, `governor.*` | `test_clock.py`, `test_position_book.py`, `test_cooldowns.py`, `test_halts.py`, `test_population_rules.py`, `test_governor.py` |
| 15, 16 | `ModelLedger`, rolling scorecards/drift/shadow, `TradeSignal`, `AgentSink` | P14 | `src/monitoring/ledger.py`, `src/monitoring/scorecards.py`, `src/handoff/agent_sink.py` | `monitoring.ledger.*`, `monitoring.scorecards.*`, `handoff.*` | `test_model_ledger.py`, `test_scorecards.py`, `test_trade_attribution.py`, `test_agent_sink.py` |
| 17 | GUI updates across all new features + tick playback | P16 | `src/gui/models.py`, `src/gui/overlays.py`, `src/gui/charts.py`, `src/gui/theme.py`, `src/gui/playback.py`, `scripts/trade_map_app.py` | — | `test_gui_lanes.py`, `test_gui_regimes.py`, `test_gui_models_tab.py`, `test_gui_lod.py`, `test_gui_playback_loader.py`, `test_gui_back_compat.py` |
| 18 | Tick & per-second data path | P13 | `src/data/ticks.py`, `src/data/bar_builder.py`, `src/execution/price_path.py` | `data.source`, `data.ticks.*`, `data.execution_resolution` | `test_tick_validation.py`, `test_bar_builder.py`, `test_tick_path.py`, `test_chunked_replay.py`, `test_feature_parity.py` |
