# States & Contracts

## Frozen Dataclasses

All shared state lives in `src/state/contracts.py` as frozen dataclasses. Immutability is enforced — layers cannot mutate state produced by upstream layers.

### FEATURE_SCHEMA_VERSION

```python
FEATURE_SCHEMA_VERSION = "v2_renamed"
```

Phase 5 schema version. All models carry this in their pickle payload. Pre-rename models (trained before Phase 5) will fall back to rule-based mode.

### RegimeType (Enum)

```
TRENDING_UP | TRENDING_DOWN | CHOP_RANGE | PANIC_SPIKE | UNKNOWN
```

Used in config lists (e.g., `experts.trend.regime_requirement: [TRENDING_UP, TRENDING_DOWN]`). `parse_regime_list()` converts YAML strings to enum members via `RegimeType[s.upper()]`.

### ExpertAction (Enum)

```
BUY=1 | SELL=-1 | NO_TRADE=0
```

### MarketSnapshot

Raw market state at a timestamp. NOT predictive.

| Field | Type | Notes |
|---|---|---|
| `timestamp` | int | Unix seconds |
| `symbol` | str | e.g., "BTC/USDT" |
| `open/high/low/close/volume` | float | OHLCV |
| `bid/ask` | float | Microstructure |
| `bid_volume/ask_volume` | float | Filled by synthetic liquidity or real order book |
| `spread_pct` | float | `(ask - bid) / mid * 100` |
| `funding_rate` | Optional[float] | Exotic data |
| `open_interest` | Optional[float] | Exotic data |
| `liquidation_volume` | Optional[float] | Exotic data |
| `hour_utc` | int | Session info |
| `day_of_week` | int | 0=Monday, 6=Sunday |
| `liquidity_source` | str | "synthetic" (backtesting) or "real_order_book" (live) |

### FeatureVector

Engineered features. No look-ahead bias — computed from past data only.

**Phase 5 rename**: Field names are now period-agnostic (`short`/`medium`/`long`) instead of timeframe-specific (`30s`/`5m`/`10m`).

| Field | Category |
|---|---|
| `momentum_short/medium/long` | Short-horizon |
| `volatility_short/medium/long` | Short-horizon (note: `volatility_short` is always 0.0) |
| `returns_5` | Short-horizon |
| `rsi_14`, `macd_signal` | Short-horizon |
| `sma_9`, `sma_21` | Medium-horizon |
| `momentum_long`, `volatility_long` | Medium-horizon |
| `imbalance_ratio`, `orderflow_momentum` | Orderflow |
| `spread_pct`, `depth_10_level` | Liquidity |
| `distance_to_nearest_support/resistance` | Structural (filled by `attach_structural_features`) |
| `support_strength`, `resistance_strength` | Structural |
| `raw_features` | Dict[str, float] | Debug |

### Zone

Support/resistance zone. Zones are **ranges**, not single prices.

| Field | Type | Notes |
|---|---|---|
| `zone_id` | str | Unique identifier |
| `lower_bound/upper_bound/mid` | float | Zone bounds |
| `strength` | float | 0–1, how strong is this level |
| `age` | int | Bars since creation |
| `touch_count` | int | How many times tested |
| `rejection_count/breakout_count` | int | Performance tracking |
| `rejection_score/breakout_score` | float | Likelihood of future rejection/breakout |
| `liquidity_score` | float | Based on volume at zone |
| `survival_probability` | float | Beta-Bernoulli × time_decay × optional ML blend |
| `regime_relevance` | Dict[str, float] | Per-regime importance |
| `zone_type` | str | "support" or "resistance" |
| `created_at/last_touched_at` | int | Timestamps |

### ZoneMap

Active zones at a timestamp.

| Field | Type |
|---|---|
| `zones` | List[Zone] |
| `support_zones` | List[Zone] |
| `resistance_zones` | List[Zone] |
| `nearest_support` | Optional[Zone] |
| `nearest_resistance` | Optional[Zone] |

### RegimeState

Market regime probabilities. Sum to ~1.0.

| Field | Type | Notes |
|---|---|---|
| `prob_trending_up/down/chop_range/panic_spike` | float | 4 probabilities |
| `primary_regime` | RegimeType | Best guess |
| `regime_confidence` | float | 0–1 |
| `hmm_hidden_state` | Optional[int] | From HMM |
| `hmm_probabilities` | List[float] | From HMM |
| `regime_age` | int | Bars since last regime change |
| `regime_persistence_score` | float | How sticky is current regime |

### ExpertOutput

Single expert's decision.

| Field | Type | Notes |
|---|---|---|
| `expert_name` | str | "trend_expert", "mean_reversion_expert", etc. |
| `action` | ExpertAction | BUY/SELL/NO_TRADE |
| `confidence` | float | 0–1 |
| `no_trade_probability` | float | 1 - confidence |
| `expected_return/risk` | float | Predicted % return/risk |
| `reward_to_risk` | float | `expected_return / max(expected_risk, 1e-6)` |
| `invalidation_distance` | float | % away from invalidation level |
| `zone_aware` | bool | Expert considered nearby zones |
| `model_output` | float | Raw model score before thresholding |
| `regime_used` | RegimeType | Which regime the model was trained on |

### ExpertOutputs

All experts' decisions at one timestamp.

| Method | Returns |
|---|---|
| `all()` | List[ExpertOutput] |
| `get_by_name(name)` | Optional[ExpertOutput] |
| `get_best_by_confidence()` | Optional[ExpertOutput] (least-unconfident if all NO_TRADE) |
| `get_consensus_action()` | ExpertAction (majority buy/sell, else NO_TRADE) |

### OpportunityScore

"Is there enough edge to justify any trade?"

| Field | Type | Notes |
|---|---|---|
| `tradable` | bool | Final verdict |
| `tradable_probability` | float | 0–1 |
| `environment_quality` | float | 0–1 |
| `opportunity_strength` | float | 0–1 |
| `failed_rules` | List[str] | Which hard rules rejected this |
| `edge_vs_fees` | float | expected_edge - (fees + slippage) |
| `liquidity_score` | float | Breakdown |
| `volatility_score` | float | Breakdown |
| `expert_consensus_score` | float | Breakdown |
| `regime_acceptability` | float | Breakdown |

### RouterDecision

Which expert should control this trade? Assumes opportunity filter already validated.

| Field | Type | Notes |
|---|---|---|
| `selected_expert` | str | e.g., "trend_expert" |
| `selection_confidence` | float | 0–1 |
| `no_trade_probability` | float | 0–1 |
| `expert_ranking` | Dict[str, float] | All options ranked |
| `reasoning` | Optional[str] | Human-readable |

### RiskDecision

"Is this trade acceptable from a risk perspective?" Risk layer can override everyone.

| Field | Type | Notes |
|---|---|---|
| `approved` | bool | |
| `risk_score` | float | 0–1, higher = riskier |
| `base_size` | float | Requested size |
| `adjusted_size` | float | After risk limits |
| `size_reduction_pct` | float | How much reduced |
| `rejection_reason` | Optional[str] | |

### ExecutionRecord

What actually happened when we executed.

| Field | Type | Notes |
|---|---|---|
| `timestamp` | int | Bar timestamp |
| `order_timestamp/fill_timestamp` | int | |
| `side` | str | "BUY" or "SELL" |
| `order_price/fill_price` | float | |
| `requested_size/filled_size` | float | |
| `slippage_pct/fees_pct/total_cost_pct` | float | Costs |
| `status` | str | "pending", "filled", "rejected", "cancelled" |
| `order_id` | str | |

### SystemState

Master snapshot. Every bar gets one.

| Field | Type | Notes |
|---|---|---|
| `timestamp/symbol` | int/str | |
| `market_snapshot` | MarketSnapshot | Always present |
| `feature_vector` | Optional[FeatureVector] | |
| `zone_map` | Optional[ZoneMap] | |
| `regime_state` | Optional[RegimeState] | |
| `expert_outputs` | Optional[ExpertOutputs] | |
| `opportunity_score` | Optional[OpportunityScore] | |
| `router_decision` | Optional[RouterDecision] | |
| `risk_decision` | Optional[RiskDecision] | Only for new entries |
| `execution_record` | Optional[ExecutionRecord] | Only on filled orders |
| `realized_pnl` | Optional[float] | 0.0 on no-trade bars, not None |

## Decision Trace (Phase 5)

One record per bar — the full stage-by-stage journey through the pipeline.

### StageResult

```python
@dataclass(frozen=True)
class StageResult:
    stage_name: str          # "trend_expert_gate", "opportunity_filter", "router", "risk_layer"
    passed: bool
    detail: str              # human-readable: "momentum 0.03% below threshold 0.05%"
    values: Dict[str, float] # actual numbers, for reproducibility
    duration_ms: Optional[float] = None  # wall-clock stage latency (PRESENTATION §23)
```

### DecisionTrace

```python
@dataclass(frozen=True)
class DecisionTrace:
    timestamp: int
    symbol: str
    stages: List[StageResult]     # every stage the bar passed through, in order
    terminal_stage: str            # where the journey ended
    terminal_reason: str
    total_duration_ms: Optional[float] = None            # end-to-end decision latency for this bar
    stage_timings_ms: Optional[Dict[str, float]] = None   # pipeline-internal per-stage latencies
```

The latency fields are populated by `TradingPipeline._process_bar()`
(`pipeline.last_bar_timings` exposes the same numbers for live monitoring)
and by `scripts/run.py`, which measures end-to-end decision latency around
`run_live_bar()`. Benchmark harness: `scripts/benchmark_latency.py`
(`make latency`).

Written to `data/logs/decision_traces/` as partitioned JSONL.

## Structured Log Streams

Three append-only JSONL streams, each with a distinct purpose (don't mix them):

| Stream | Writer | Contents |
|---|---|---|
| `data/logs/system.jsonl` | `src/utils/logging_setup.py::setup_logging()` | Every log line as JSON (level, logger, message) |
| `data/logs/decisions.jsonl` | `log_trade_decision()` via `decision_record_from_state()` | One §8-shaped record per bar: regime/expert/opportunity/router/risk/decision |
| `data/logs/decision_traces/traces_<date>.jsonl` | `DecisionTraceWriter` | Full stage-by-stage "why" narrative + latency |

### SystemFault

```python
@dataclass(frozen=True)
class SystemFault:
    timestamp: int
    symbol: str
    exception_type: str
    exception_message: str
    stack_trace: str
    system_state_snapshot: Optional[Dict[str, Any]]
    last_known_good_state_timestamp: Optional[int]
```

Written to `data/logs/faults.jsonl`.

## Serialization

`src/state/serialization.py` flattens every field of every dataclass into a dict. Previously ~4 fields of `FeatureVector` and all fields of `RiskDecision`/`ExecutionRecord` were not flattened — fixed by Phase 4's programmatic audit. The serialization is verified by `test_full_systemstate_archive_completeness` which checks that every field of every dataclass appears in `serialization.py`.

`dataclasses.replace()` is used throughout for immutable updates (e.g., attaching synthetic liquidity to a snapshot, adding a `risk_decision` to a `SystemState`).
