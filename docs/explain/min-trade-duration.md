# Minimum trade duration (10 s) and harness-aligned training

**Rule.** No trade may be held shorter than `trade_duration.min_trade_seconds`
(default **10 s**), measured in wall-clock time from the simulated FILL to the
EXIT. It applies to every expert and to every ML component that decides, sizes
or selects a trade. `0` disables it.

**Why it is not just "bars ≥ 10 s".** A position is opened at the fill, which
happens `entry latency` after the signal bar, and can only be exited when a
later bar closes. With 10 s bars and 2 s of latency, a one-bar trade lasts 8 s.
Finer granularities (5 s, 1 s) make one-bar trades shorter still. The rule is
therefore expressed in seconds and converted to bars by `HoldPolicy`:

    min_hold_bars = ceil((min_trade_seconds + entry_latency_seconds) / bar_seconds)

## One rule, one implementation

| Concern | Where |
|---|---|
| the policy and its arithmetic | `src/labeling/hold_policy.py::HoldPolicy` |
| the exit rule, incl. the lockout | `src/labeling/triple_barrier.py::barrier_exit_step` |
| barrier sizing | `src/labeling/triple_barrier.py::barriers_for_regime` |
| entry price | `src/execution/execution_engine.py::entry_fill_price` |
| training labels | `src/labeling/harness_labels.py::label_candidate` |
| live positions / harness / GUI | `position_lifecycle.check_position_exit`, via `process_bar` |
| models served | `base_expert`, `opportunity_filter`, `router`, `risk_layer` |

The labeler and the live position call **the same function**, so a label and a
trade cannot disagree about when or where a position exits. A test checks this
over 400 random paths and 4 lockout lengths.

**The lockout.** While `bars_held < min_hold_bars` nothing exits — not a stop,
not a target, not the timeout. On the first legal bar, a stop-loss that the
position could not act on fills at the **worse** of the stop level and that
bar's open: a gap through the stop is not filled for free.

**Fail closed.** If the config cannot satisfy the rule (a timeout shorter than
the minimum, or latency longer than the longest hold), `HoldPolicy` raises. Every
entry point — `process_bar`, the expert registry, the pipeline, the training
scripts, `validate_config_consistency` — raises before any trade happens.

## Models carry their training contract

Every trained model (pickle and card) records a `training_alignment` block: the
hold-policy fingerprint (minimum, granularity, max hold, min hold bars, latency),
population, sample policy, the number of labels that broke the rule, and the
hold-time statistics. A loader compares it with the running config:

* a model trained under a **different policy** is refused (fallback mode);
* a **legacy** model with no contract is refused when
  `trade_duration.require_aligned_models` is true (the shipped default);
* a model whose labels contained **any** short trade is refused;
* a model trained on the **wrong population** is refused (experts: the gate
  population, not zone touch events).

Refusal means the component runs in its rule-based mode and logs why. It never
silently serves a model trained on trades the harness does not take.

## Training runs as the harness uses each expert

`scripts/train_triple_barrier.py` runs one replay and, for each expert, keeps:

1. **population** — bars where that expert's own `propose_candidate()` fired
   **and** the Opportunity Filter passed (the harness never routes a rejected
   bar);
2. **labels** — the harness's own outcome: simulated-fill entry, regime-specific
   barriers, the exit rule with the lockout, round-trip fees;
3. **sampling** — one position at a time per expert, with the order cool-down
   (`select_sequential`), because the harness holds one position at a time.

It writes `<output>.meta.json` recording the contract. The expert trainers
(`train_{trend,mean_reversion,breakout}_expert.py`), the opportunity-filter,
router and risk-model trainers refuse a dataset whose contract does not match
the running config, a label shorter than the minimum, or an embargo shorter
than the label horizon.

Walk-forward validation is measured in **bars**, not rows (`WalkForwardSplitter
.split_by_bar_index`): candidate rows are sparse, so a row-count window is an
unknown amount of time. Training rows whose label is still being decided inside
the test window are purged.

`--all-candidates` and `--include-filtered-out` are available for research. They
are **not** how the harness trades, and the model card records that they were
used.

## Retrain and promotion

`scripts/retrain_all.py` validates the hold policy before any subprocess starts,
labels once with the shared labeler, and stamps every registered version with
`alignment_ok` / `alignment_problem`. `scripts/approve_model.py` refuses a
version that is not harness-aligned unless `--force` is given.

## Metrics and evals

| Metric | Meaning | Must be |
|---|---|---|
| `violations` | harness trades held shorter than the minimum | **0** |
| hold min / p05 / median / p95 | distribution of fill-to-exit seconds | min ≥ minimum |
| `pinned_rate` | share of barrier exits on the first legal bar | reported |
| `coverage` | harness trades that have a training label | ≥ 99 % |
| `exit_reason_agreement`, `bars_held_agreement` | label vs harness, per trade | ≥ 99 % |
| `pnl_max_abs_error_pct` | label vs harness net PnL | ≈ 0 |
| `lockout_cost` | expectancy with vs without the lockout, same candidates | reported |

Run it with:

    python scripts/evaluate_training_alignment.py --input data/raw/BTC_USDT.parquet
    python scripts/evaluate_training_alignment.py --synthetic 3000 --bar-seconds 5 \
        --granularity 5s --relax-liquidity

It exits 0 on pass, 1 on a failed check (usable as a CI or retrain gate), and 2 if
the policy itself is unsatisfiable. It writes JSON and Markdown to `data/reports/`.
The harness report (`full_stack_walk_forward_backtest.py`) gains a
**Minimum Trade Duration** section.

## What this does not guarantee

* **Intrabar timing is not observable from OHLC.** "10 s" is enforced on bar
  boundaries plus the modelled latency, not on ticks. A live fill is stamped with
  the bar label, so live latency is the configured assumption, not a measurement.
* **The labeler replays one expert at a time.** Routing between experts and the
  risk layer's size and veto are not replayed per sample. The alignment eval
  measures the gap; it does not close it.
* **No model was trained by this change.** Existing pickles are legacy and are
  refused under the shipped config until retrained.
