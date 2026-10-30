# Phase 0 — Baseline Verification, Triage, and Performance Reference

**Recorded at:** 2026-10-29 UTC
**Base commit snapshot:** `7eb7492` (`Btc_trading_agents-7eb74927c9b1b5dc234f28353b4c922932a62a53`)
**Branch:** `feat/dynamic-regimes-exits-sizing-harness`
**Environment:** Linux-4.19.0-gvisor-x86_64-with-glibc2.36 · 2 CPU cores · Python 3.11.2 (offline sandbox)

## 1. Test Suite Triage (T0.1)

Prior to this phase, the two previous change sets (`Trade Map GUI` and `minimum trade duration`) had not been verified against the test suite. Running `make unit`, `make integration`, `make e2e`, `make lint`, and `make typecheck` uncovered two attributable defects in the unverified Trade Map GUI code:

1. **`src/gui/__init__.py` forbidden token defect** (`tests/unit/test_gui_app.py::test_no_gui_module_can_reach_an_exchange_or_write_the_journal[__init__.py]`):
   - *Root cause:* The lazy-import docstring in `src/gui/__init__.py` contained the literal substring `ccxt`, violating the strict safety assertion in `test_gui_app.py` that no file in `src/gui/*.py` mentions `ccxt`.
   - *Fix:* Reworded the docstring comment in `src/gui/__init__.py` without altering runtime behavior.
2. **900-bar GUI replay latency budget** (`tests/integration/test_gui_replay.py::test_a_900_bar_replay_is_not_sqlite_bound`):
   - *Root cause:* `src/gui/replay.py` created its temporary directory in the default working directory, called recursive `dataclasses.asdict()` (`copy.deepcopy`) on every bar's `DecisionTrace`, and `FeatureEngine._macd_signal` constructed two pandas `Series` per bar.
   - *Fix:* Pointed `tempfile.mkdtemp` in `replay.py` to `/tmp`, serialized `DecisionTrace` directly to its exact dict schema, and computed the 26-bar EWM in `_macd_signal` with pure NumPy (`0.0` float difference vs pandas). 900-bar replay time dropped to **2.21 s** (budget: 3.0 s).

### Verified Baseline Counts
- **Unit (`tests/unit`):** 882 passed, 0 failed
- **Integration (`tests/integration`):** 37 passed (+ 5 golden tests added in T0.2 = 42 passed), 0 failed
- **End-to-End (`tests/e2e`):** 13 passed, 0 failed
- **Lint (`ruff check src scripts tests`):** 0 errors
- **Typecheck (`mypy src/state/ src/reporting/`):** 0 errors (strict mode)

## 2. Deterministic Golden Regression Oracle (T0.2)

Tracked golden digests (`tests/golden/paper_sim_1m_900.json` and `tests/golden/paper_sim_10s_900.json`, ~41 KB each, well under the 200 KB cap) capture the exact trade list, equity curve, per-bar `terminal_stage` sequence, and `trade_duration` report.
- `tests/integration/test_default_config_is_golden.py` asserts exact equality across both scenarios, proves 3-run consecutive determinism, and verifies oracle sensitivity to parameter changes.

## 3. Performance Baseline (T0.3)

Measured over 300 post-warmup bars (+110 warmup bars) on seeded synthetic OHLCV (`seed=42`):

| Stage / Metric | 1m Mean (ms) | 1m P50 (ms) | 1m P95 (ms) | 1m P99 (ms) | 10s Mean (ms) | 10s P50 (ms) | 10s P95 (ms) | 10s P99 (ms) |
|---|---|---|---|---|---|---|---|---|
| **End-to-end decision** | **1.737** | **1.556** | **3.065** | **3.519** | **1.779** | **1.481** | **3.309** | **3.888** |
| `features_ms` | 0.564 | 0.495 | 1.016 | 1.596 | 0.576 | 0.480 | 1.058 | 1.474 |
| `regime_ms` | 0.253 | 0.206 | 0.481 | 0.821 | 0.269 | 0.204 | 0.605 | 0.808 |
| `structure_ms` | 0.579 | 0.493 | 1.113 | 1.409 | 0.583 | 0.461 | 1.213 | 1.758 |
| `liquidity_ms` | 0.057 | 0.045 | 0.099 | 0.125 | 0.065 | 0.046 | 0.118 | 0.403 |
| `expert_layer_ms` | 0.092 | 0.074 | 0.172 | 0.313 | 0.093 | 0.070 | 0.208 | 0.332 |
| `opportunity_filter_ms` | 0.156 | 0.136 | 0.307 | 0.592 | 0.158 | 0.130 | 0.336 | 0.439 |
| `router_ms` | 0.011 | 0.007 | 0.025 | 0.076 | 0.009 | 0.007 | 0.016 | 0.055 |
| `decisions_ms` (sum) | 0.258 | 0.224 | 0.505 | 0.862 | 0.260 | 0.219 | 0.530 | 0.658 |

- **Throughput:** **575.8 bars/sec** (`1m`), **562.0 bars/sec** (`10s`).
- **Model `predict` calls per bar:** `0` in default rule-based fallback mode (no pre-trained models shipped in `data/models/`).

## 4. Empirical Pipeline Measurements on Synthetic Series (T0.4)

- **Fraction of runtime in `compute_zones` (`structure_ms`):** **33.36%** (`1m`) / **32.77%** (`10s`) — confirms F7 that per-bar zone regeneration/merging is the largest single perception cost center alongside feature rolling stats.
- **Active zone count per bar:** mean **3.16**, max **6**.
- **Regime dwell & flip rate (rule-based detector with persistence window 20):** **3 flips** across 300 bars (**1.0% flip rate**), mean dwell **75.0 bars**.
- **Expert gate fire rates:** `trend_expert` **54.33%**, `breakout_expert` **6.00%**, `mean_reversion_expert` **0.00%**.
- **Real-data blocker (U1):** No historical market CSV/Parquet is shipped in `data/raw/`. Measurements above are on the deterministic synthetic generator; real-market profitability and empirical confidence distributions remain gated on U1.
