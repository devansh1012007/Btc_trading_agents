# Minimum trade duration + harness-aligned training — plan

**Status:** IMPLEMENTED, NOT VERIFIED — written without running the test suite
and without training any model (by instruction).
**Owner request (verbatim intent):** no ML model may decide to take a trade
shorter than **10 seconds**, in every expert; the training/retraining cycle
must train each expert *exactly* the way it is used in the harness; write the
tests, metrics and evals; do not train.

## 0. Reading the request (assumptions — correct me if wrong)

| Phrase | How I read it |
|---|---|
| "MN model" | the ML models (experts first; also the opportunity filter, router and risk model, because they are trained on trade outcomes too) |
| "winding cycle" | the **retraining + walk-forward windowing cycle** (`retrain_all.py`, `approve_model.py`, `WalkForwardSplitter`). There is no "winding" in the code base. |
| "10 seconds long" | **wall-clock time from FILL to EXIT**, not a bar count. Bars are ≥ 10 s today (`10s`, `30s`, `1m`, `5m`), so at the current granularities a bar-close exit can only violate it through **entry latency** (fill happens after the bar label, exit is observed at a later bar label) — and the rule is written so it also holds for 1 s / 5 s bars. |
| "the harness" | `scripts/full_stack_walk_forward_backtest.py` (the full-stack walk-forward harness) and, by construction, `process_bar` — the one per-bar implementation the live loop, historical loop and GUI replay share. |

**What is and is not new.** A trade is always ≥ 1 bar, so with ≥ 10 s bars the
rule is *structurally* near-satisfied — the real exposure is (a) latency eating
into the first bar, (b) any finer granularity, (c) a `max_holding_bars` too
small for the granularity (a 5-bar timeout at 1 s is 5 s), and (d) **models
trained on labels that assume exits the harness would not take**. This change
turns the rule from "happens to be true" into "enforced and measured".

## 1. Findings that drive the design

1. The exit rule exists **twice** (`label_triple_barrier` and
   `check_position_exit`) and the barrier sizing **five times** (4 training
   scripts, `replay.py`, the harness, `process_bar`).
2. Training labels assume entry at the signal bar's **close**; the harness
   enters at the **simulated fill** (`close ± spread/2 ± slippage`).
3. Mean-reversion and breakout experts are trained on the zone **touch-event**
   stream, but deployed behind `propose_candidate()` — a different population.
   The trend expert is trained on gate candidates but **not** restricted to
   bars the opportunity filter passes, as the harness does.
4. Candidates are labeled on **every** bar, overlapping; the harness holds one
   position at a time.
5. Walk-forward embargo is a row count, never checked against the label
   horizon.
6. Retraining promotes on AUC only.
7. Eight training scripts crash on start (`NameError: sys`) — pre-existing.

## 2. Design

One rule, one implementation, three consumers.

```
config: trade_duration.min_trade_seconds (10), assumed_entry_latency_seconds
            │
            ▼
   HoldPolicy  (src/labeling/hold_policy.py)  ── min_hold_bars = ceil((min+latency)/bar)
            │
   barrier_exit_step()  (src/labeling/triple_barrier.py)   ← THE exit rule
       ▲                ▲                      ▲
 label_triple_barrier   check_position_exit    (harness / live / GUI via process_bar)
 (training labels)      (positions)
```

* **Lockout.** An exit — stop-loss, take-profit **and** timeout — is not
  allowed before `min_hold_bars` bars **and** `min_trade_seconds` of
  wall-clock time since the fill. A stop first reachable after the lockout
  fills at the **worse of the stop level and that bar's open** (no free
  fills through a gap the position could not act on).
* **Fail closed.** An invalid policy (`max_holding_bars × bar < min + latency`)
  raises `HoldPolicyError` at construction/start-up; it never silently trades.
* **Models carry their training contract.** Every model card/pickle gets a
  `training_alignment` block (hold-policy fingerprint, population, sample
  policy, entry model, violation count). A model whose contract differs from
  the running config — or that has none (legacy) — loads in **fallback mode**
  with a warning, exactly like a granularity mismatch.
* **Train as deployed.** One harness-aligned labeler
  (`src/labeling/harness_labels.py`): candidate = the expert's own
  `propose_candidate()` gate; must pass the opportunity filter (as in the
  harness); entry = simulated fill (shared `entry_fill_price`); barriers =
  shared regime-specific sizing; exit = shared step incl. lockout;
  non-overlapping per expert (`sequential`, with the order cool-down).
  `--all-candidates` keeps the old overlapping population (card says so).
* **Windowing.** `WalkForwardSplitter` gains bar-index windows with
  label-horizon **purge** and a hard `embargo ≥ horizon` check.
* **Retrain cycle.** `retrain_all.py` validates the policy first, then gates
  registration on the model card's alignment; `approve_model.py` refuses an
  unaligned version without `--force`.

## 3. Work items

| # | Item | Files |
|---|---|---|
| 1 | `HoldPolicy`, granularity parser, model-contract helpers | `src/labeling/hold_policy.py` |
| 2 | Shared exit step, barrier sizing, lockout in labeler | `src/labeling/triple_barrier.py` |
| 3 | Shared entry-fill math | `src/execution/execution_engine.py` |
| 4 | Lockout in positions; `process_bar` wiring; hold seconds on exits | `src/execution/position_lifecycle.py` |
| 5 | Harness-aligned labeler + harvest + sequential selection | `src/labeling/harness_labels.py`, `src/backtest/harness_harvest.py` |
| 6 | Splitter horizon/purge | `src/backtest/walk_forward.py` |
| 7 | Experts + the 3 other model loaders enforce the contract | `src/experts/*.py`, `src/layers/{opportunity_filter,router,risk_layer}.py` |
| 8 | Training scripts on the shared labeler (+ `import sys` fixes) | `scripts/train_*.py` |
| 9 | Retrain/approve gates | `scripts/retrain_all.py`, `scripts/approve_model.py` |
| 10 | Metrics + eval script + harness report section | `src/evaluation/trade_duration.py`, `scripts/evaluate_training_alignment.py`, harness |
| 11 | Config + validator + preflight | `config/settings.base.yaml`, `src/config/consistency_validator.py` |
| 12 | Tests | `tests/unit/test_min_trade_duration.py`, `tests/unit/test_harness_alignment.py`, `tests/integration/test_min_hold_harness.py` |
| 13 | Docs | `docs/explain/min-trade-duration.md`, AGENTS.md, limitations |

## 4. Metrics & evals (what "good" means)

| Metric | Where | Pass |
|---|---|---|
| `min_duration_violations` (trades with hold < min) | harness report, model card, eval script | **0** |
| hold-time distribution (min / p5 / median / p95) per expert | same | min ≥ 10 s |
| `pinned_rate` (exits on the first eligible bar) | same | reported; a high value means barriers are too tight for the lockout |
| `lockout_cost` (expectancy with vs without lockout, share of outcomes changed) | eval script | reported |
| `label_harness_agreement` (exit reason, bars held, net PnL of every harness trade vs its label) | eval script | ≥ 99 % coverage & agreement |
| `population_coverage` (candidates → passed filter → sampled) | model card | reported |

## 5. Explicit non-goals / limits

* No model is trained; no model files exist or are produced.
* Intrabar timing is unknowable from OHLC: "10 s" is enforced on bar
  boundaries plus modelled latency, not on tick data.
* An exit blocked by the lockout cannot be overridden by a stop: the cost is
  measured (`lockout_cost`, `lockout_adverse_excursion_pct`), not hidden.
* The labeler reproduces the harness for ONE expert at a time; selection
  between experts (router) and the risk layer's size/veto are not replayed
  per sample. The eval script quantifies the remaining gap.
