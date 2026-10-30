# Trade Map GUI Verification Checklist (Phase 16)

1. **Multi-Position Lanes (`PositionBook`)**:
   - Overlapping concurrent positions are assigned deterministic, non-overlapping `lane_id` indices (`0, 1, ...`) via `assign_trade_lanes`.
   - Sequential non-overlapping positions reuse `lane_id = 0`.
2. **Dynamic K Regimes (`RegimeSet`)**:
   - Arbitrary `K` regime identifiers (`r_0 .. r_{K-1}` or semantic labels) receive deterministic, collision-safe background band colors via `theme.regime_palette` and `theme.regime_colour`.
3. **Dynamic `ExitPlan` & Trailing Stop Overlays**:
   - `TrailLine`, `DelayConnector`, and `HaltBand` descriptors render without `NaN` or `None` values.
   - `EntryMarker` never leaks exit outcomes (`exit_ts`, `exit_price`, `pnl_pct`, `exit_reason`).
4. **Model Ledger & Scorecards Tab**:
   - `build_models_scorecard_chart` renders per-model Brier score and realized expectancy from `SessionBundle.model_scorecards` (and degrades gracefully to an empty-state message when no predictions were logged).
5. **1s / Tick Playback & Level-of-Detail (`downsample_lod`)**:
   - `downsample_lod` caps rendered points at `max_points` while preserving exact bucket open, high, low, close, and volume.
   - `load_playback_slice` loads bounded `[start_ts, start_ts + window_seconds)` windows on demand.
6. **Backward Compatibility & Viewer Isolation**:
   - Legacy `SessionBundle` fixtures without `model_scorecards`, `halts`, `cooldowns`, or `lane_id` render identically to pre-Phase-16.
   - No GUI module references exchange order placement or mutates `data/`.
