"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================
""" RegimeType might be abstract in futre version if the classification is going to be unsupervised"""
class RegimeType(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CHOP_RANGE = "chop_range"
    PANIC_SPIKE = "panic_spike"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DynamicRegime:
    """Duck-compatible dynamic regime identifier (.name, .value, hashable, orderable)."""
    id: str
    display_name: Optional[str] = None
    semantic_tag: Optional[str] = None

    @property
    def value(self) -> str:
        return self.id

    @property
    def name(self) -> str:
        return (self.display_name or self.id).upper()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DynamicRegime):
            return self.id == other.id
        if isinstance(other, RegimeType):
            return self.id == other.value or self.id.upper() == other.name
        if isinstance(other, str):
            return self.id == other or self.id.upper() == other.upper()
        return False

    def __hash__(self) -> int:
        for member in RegimeType:
            if self.id == member.value or self.id.upper() == member.name:
                return hash(member)
        return hash(self.id)

    def __lt__(self, other: object) -> bool:
        other_val = getattr(other, "value", str(other))
        return self.id < str(other_val)


RegimeLike = RegimeType | DynamicRegime


def coerce_regime(value: "RegimeLike | str") -> RegimeLike:
    """Coerce a string or RegimeLike into RegimeType (if one of the 5 legacy names/values) or DynamicRegime."""
    if isinstance(value, (RegimeType, DynamicRegime)):
        return value
    s = str(value).strip()
    for member in RegimeType:
        if s == member.value or s.upper() == member.name:
            return member
    return DynamicRegime(id=s.lower(), display_name=s.upper())


@dataclass(frozen=True)
class RegimeSet:
    """Ordered set of regime identifiers and display metadata."""
    set_id: str
    regime_ids: tuple[str, ...]
    display_names: Dict[str, str] = field(default_factory=dict)
    semantic_tags: Dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.set_id

    def ids(self) -> tuple[str, ...]:
        return self.regime_ids

    @property
    def regimes(self) -> tuple[RegimeLike, ...]:
        return tuple(coerce_regime(rid) for rid in self.regime_ids)

    @property
    def k(self) -> int:
        return len(self.regime_ids)

    @property
    def version_hash(self) -> str:
        import hashlib, json
        blob = json.dumps(
            {"set_id": self.set_id, "regime_ids": list(self.regime_ids)},
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def prob_feature_names(self, prefix: str = "prob_") -> List[str]:
        return [f"{prefix}{rid}" for rid in self.regime_ids]


LEGACY_REGIME_SET = RegimeSet(
    set_id="legacy4",
    regime_ids=("trending_up", "trending_down", "chop_range", "panic_spike"),
    display_names={
        "trending_up": "TRENDING_UP",
        "trending_down": "TRENDING_DOWN",
        "chop_range": "CHOP_RANGE",
        "panic_spike": "PANIC_SPIKE",
    },
    semantic_tags={
        "trending_up": "trend_up",
        "trending_down": "trend_down",
        "chop_range": "range",
        "panic_spike": "panic",
    },
)


def regime_prob_features(
    regime_set: Optional[RegimeSet] = None,
    prefix: str = "prob_",
) -> List[str]:
    """Ordered probability feature names for ``regime_set`` (defaults to LEGACY_REGIME_SET)."""
    rs = regime_set or LEGACY_REGIME_SET
    return rs.prob_feature_names(prefix=prefix)


class ExpertAction(Enum):
    """Expert decision output."""
    BUY = 1
    SELL = -1
    NO_TRADE = 0


# ============================================================================
# LAYER 1: MARKET SNAPSHOT
# ============================================================================

@dataclass(frozen=True) #class immutable after initialization.
class MarketSnapshot:
    """
    Raw market state at a given timestamp.
    This layer is NOT predictive.
    """
    timestamp: int  # Unix seconds
    symbol: str  # e.g. "BTCUSDT"

    # OHLCV
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Microstructure
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float
    spread_pct: float  # (ask - bid) / mid * 100

    # Exotic data
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    liquidation_volume: Optional[float] = None

    # Session info
    hour_utc: int = 0
    day_of_week: int = 0  # 0=Monday, 6=Sunday

    # Which liquidity estimate produced bid/ask/bid_volume/ask_volume above
    # (Phase 4, §4.4): "synthetic" for backtesting/historical replay (no
    # real order book exists for the past), "real_order_book" when a live
    # CCXT order book was actually queried (paper-live-data / live modes).
    liquidity_source: str = "synthetic"

    def to_dict(self) -> Dict:
        """Convert to dict, handling Enums."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'MarketSnapshot':
        """Create from dict."""
        return cls(**d)


# ============================================================================
# LAYER 2: FEATURE VECTOR
# ============================================================================

# Feature schema version — incremented whenever FeatureVector field names
# change. Model loaders check this against their saved payload to avoid
# silently degrading (getattr on a renamed field returns 0.0, not an error).
FEATURE_SCHEMA_VERSION = "v2_renamed"


@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    Field names are period-agnostic (short/medium/long) because the actual
    periods are config-driven via feature_engine.momentum_periods and
    feature_engine.volatility_periods — a name like "momentum_5m" would
    lie if the config changed that period to 3.
    """
    timestamp: int
    symbol: str

    # Short-horizon features (config-driven period, default 1 bar)
    momentum_short: float
    momentum_medium: float
    volatility_short: float
    volatility_medium: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float

    # Medium-horizon features (config-driven period)
    sma_9: float
    sma_21: float
    momentum_long: float
    volatility_long: float

    # Orderflow features
    imbalance_ratio: float  # aggressive_buy / total_volume
    orderflow_momentum: float

    # Liquidity features
    spread_pct: float
    depth_10_level: float

    # Structural features (computed relative to zones)
    distance_to_nearest_support: Optional[float] = None
    distance_to_nearest_resistance: Optional[float] = None
    support_strength: Optional[float] = None
    resistance_strength: Optional[float] = None

    # Metadata
    feature_computed_at: int = 0  # lag in ms

    # Raw data for debugging
    raw_features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dict."""
        d = asdict(self)
        # Handle nested dicts
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'FeatureVector':
        """Creates class variable from dict."""
        return cls(**d)


# ============================================================================
# LAYER 3: DYNAMIC STRUCTURE
# ============================================================================

@dataclass(frozen=True)
class Zone:
    """
    A support or resistance zone.
    Zones are ranges, not single prices.
    """
    zone_id: str  # unique identifier
    lower_bound: float
    upper_bound: float
    mid: float

    # Metadata
    strength: float  # 0-1, how strong is this level
    age: int  # bars since creation
    touch_count: int  # how many times tested

    # Performance tracking
    rejection_count: int
    breakout_count: int
    rejection_score: float  # likelihood of future rejection
    breakout_score: float  # likelihood of future breakout

    # Quality metrics
    liquidity_score: float  # based on volume at zone
    survival_probability: float  # ML: prob zone stays relevant
    regime_relevance: Dict[str, float] = field(default_factory=dict)  # per-regime importance
    """
    the above one is used to define a field with a mutable default value (in this case, an empty dictionary) while providing strict type hinting.
    """

    zone_type: str = "support"  # "support", "resistance"
    created_at: int = 0
    last_touched_at: int = 0


@dataclass(frozen=True)
class ZoneMap:
    """Active zones at current timestamp."""
    timestamp: int
    symbol: str
    current_price: float

    zones: List[Zone] = field(default_factory=list)
    support_zones: List[Zone] = field(default_factory=list)
    resistance_zones: List[Zone] = field(default_factory=list)

    nearest_support: Optional[Zone] = None
    nearest_resistance: Optional[Zone] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# LAYER 4: REGIME DETECTION
# ============================================================================

@dataclass(frozen=True)
class RegimeState:
    """
    Market regime probabilities.
    Do NOT force a single regime; use probability vector.
    """
    timestamp: int
    symbol: str

    # Regime probabilities (sum to ~1.0)
    prob_trending_up: float = 0.0
    prob_trending_down: float = 0.0
    prob_chop_range: float = 0.0
    prob_panic_spike: float = 0.0

    # Best guess (but use probabilities upstream)
    primary_regime: RegimeLike = RegimeType.UNKNOWN
    regime_confidence: float = 0.0  # 0-1

    # Hidden state tracking (from HMM)
    hmm_hidden_state: Optional[int] = None
    hmm_probabilities: List[float] = field(default_factory=list)

    # Persistence metrics
    regime_age: int = 0  # bars since last regime change
    regime_persistence_score: float = 0.0  # how sticky is current regime

    # Dynamic K regime set support (Phase 2, T2.1)
    regime_set_id: str = "legacy4"
    regime_probs: Dict[str, float] = field(default_factory=dict)
    serving_source_used: str = "rules"

    def __post_init__(self) -> None:
        if not self.regime_probs:
            object.__setattr__(
                self,
                "regime_probs",
                {
                    "trending_up": float(self.prob_trending_up),
                    "trending_down": float(self.prob_trending_down),
                    "chop_range": float(self.prob_chop_range),
                    "panic_spike": float(self.prob_panic_spike),
                },
            )
        else:
            # Populate legacy fields if present in regime_probs and legacy fields were left at 0.0
            probs = dict(self.regime_probs)
            object.__setattr__(self, "regime_probs", probs)
            if (
                self.prob_trending_up == 0.0
                and self.prob_trending_down == 0.0
                and self.prob_chop_range == 0.0
                and self.prob_panic_spike == 0.0
            ):
                for rid, attr in (
                    ("trending_up", "prob_trending_up"),
                    ("trending_down", "prob_trending_down"),
                    ("chop_range", "prob_chop_range"),
                    ("panic_spike", "prob_panic_spike"),
                ):
                    if rid in probs:
                        object.__setattr__(self, attr, float(probs[rid]))

    def get_prob(self, regime_id: str, default: float = 0.0) -> float:
        """Accessor for regime probability by id."""
        return float(self.regime_probs.get(regime_id, default))

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['primary_regime'] = self.primary_regime.value
        return d


# ============================================================================
# LAYER 5: EXPERT OUTPUTS
# ============================================================================

@dataclass(frozen=True)
class TrailSpec:
    """Trailing stop specification attached to an ExitPlan."""
    enabled: bool = False
    activation_pct: float = 0.0
    distance_pct: float = 0.0
    step_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "activation_pct": float(self.activation_pct),
            "distance_pct": float(self.distance_pct),
            "step_pct": float(self.step_pct),
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "TrailSpec":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            activation_pct=float(data.get("activation_pct", 0.0)),
            distance_pct=float(data.get("distance_pct", 0.0)),
            step_pct=float(data.get("step_pct", 0.0)),
        )


@dataclass(frozen=True)
class ExitPlan:
    """Unified exit plan carried by ExpertOutput and OpenPosition."""
    stop_loss_pct: float
    take_profit_pct: float
    max_holding_bars: int
    trailing: Optional[TrailSpec] = None
    hard_max_loss_pct: Optional[float] = None
    source: str = "regime_specific"

    @property
    def effective_stop_loss_pct(self) -> float:
        if self.hard_max_loss_pct is not None and self.hard_max_loss_pct > 0:
            return min(float(self.stop_loss_pct), float(self.hard_max_loss_pct))
        return float(self.stop_loss_pct)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stop_loss_pct": float(self.stop_loss_pct),
            "take_profit_pct": float(self.take_profit_pct),
            "max_holding_bars": int(self.max_holding_bars),
            "trailing": self.trailing.to_dict() if self.trailing is not None else None,
            "hard_max_loss_pct": float(self.hard_max_loss_pct) if self.hard_max_loss_pct is not None else None,
            "source": str(self.source),
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> Optional["ExitPlan"]:
        if not data:
            return None
        tr_raw = data.get("trailing")
        return cls(
            stop_loss_pct=float(data["stop_loss_pct"]),
            take_profit_pct=float(data["take_profit_pct"]),
            max_holding_bars=int(data.get("max_holding_bars", 5)),
            trailing=TrailSpec.from_dict(tr_raw) if tr_raw else None,
            hard_max_loss_pct=float(data["hard_max_loss_pct"]) if data.get("hard_max_loss_pct") is not None else None,
            source=str(data.get("source", "regime_specific")),
        )


@dataclass(frozen=True)
class ExpertOutput:
    """
    Single expert's decision.
    All experts output this same format.
    """
    expert_name: str  # "trend_expert", "mean_reversion_expert", etc.
    timestamp: int
    symbol: str

    # Decision
    action: ExpertAction
    confidence: float  # 0-1
    no_trade_probability: float  # expert thinks this is bad

    # Risk/return expectations
    expected_return: float  # predicted % return next window
    expected_risk: float  # predicted % adverse move
    reward_to_risk: float  # expected_return / expected_risk

    # Structural awareness
    invalidation_distance: float  # % away from invalidation level
    zone_aware: bool  # expert considered nearby zones

    # Debug info
    model_output: float  # raw model score before thresholding
    regime_used: RegimeLike = RegimeType.UNKNOWN
    exit_plan: Optional[ExitPlan] = None
    instance_id: Optional[str] = None


@dataclass(frozen=True)
class ExpertOutputs:
    """All experts' decisions at one timestamp."""
    timestamp: int
    symbol: str

    outputs: Dict[str, ExpertOutput] = field(default_factory=dict)

    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)

    def all(self) -> List[ExpertOutput]:
        """All expert outputs, in insertion order (Phase 3)."""
        return list(self.outputs.values())

    def get_by_name(self, expert_name: str) -> Optional[ExpertOutput]:
        """Look up one expert's output by name, or None if absent (Phase 3)."""
        return self.outputs.get(expert_name)

    def get_best_by_confidence(self) -> Optional[ExpertOutput]:
        """The single highest-confidence output, or None if there are no
        experts at all (Phase 3). A NO_TRADE-only set still returns the
        least-unconfident one -- callers that care should also check
        `.action`."""
        if not self.outputs:
            return None
        return max(self.outputs.values(), key=lambda o: o.confidence)

    def get_consensus_action(self) -> ExpertAction:
        """What do experts agree on?"""
        if not self.outputs:
            return ExpertAction.NO_TRADE
        actions = [o.action for o in self.outputs.values()]
        buy_count = sum(1 for a in actions if a == ExpertAction.BUY)
        sell_count = sum(1 for a in actions if a == ExpertAction.SELL)
        if buy_count > sell_count:
            return ExpertAction.BUY
        elif sell_count > buy_count:
            return ExpertAction.SELL
        return ExpertAction.NO_TRADE


# ============================================================================
# LAYER 6: OPPORTUNITY FILTER
# ============================================================================

@dataclass(frozen=True)
class OpportunityScore:
    """
    "Is there enough edge to justify any trade?"
    This layer says YES or NO before routing.
    """
    timestamp: int
    symbol: str

    tradable: bool  # Final verdict: trade or not
    tradable_probability: float  # 0-1

    # Quality metrics
    environment_quality: float  # 0-1, how good are conditions?
    opportunity_strength: float  # 0-1, how strong is edge?

    # Reasoning (for debugging)
    failed_rules: List[str] = field(default_factory=list)  # which hard rules rejected this
    edge_vs_fees: float = 0.0  # expected_edge - (fees + slippage)

    # Breakdown for analysis (Phase 3, §7.3)
    liquidity_score: float = 0.0
    volatility_score: float = 0.0
    expert_consensus_score: float = 0.0
    regime_acceptability: float = 0.0


# ============================================================================
# LAYER 7: ROUTER
# ============================================================================

@dataclass(frozen=True)
class RouterDecision:
    """
    Which expert should control this trade?
    Router assumes opportunity filter already validated.
    """
    timestamp: int
    symbol: str

    selected_expert: str  # e.g. "trend_expert"
    selection_confidence: float  # 0-1

    no_trade_probability: float  # router thinks this is bad

    # All options ranked (for debugging)
    expert_ranking: Dict[str, float] = field(default_factory=dict)
    reasoning: Optional[str] = None


# ============================================================================
# LAYER 8: RISK LAYER
# ============================================================================

@dataclass(frozen=True)
class RiskDecision:
    """
    "Is this trade acceptable from a risk perspective?"
    Risk layer can override everyone.
    """
    timestamp: int
    symbol: str

    approved: bool
    risk_score: float  # 0-1, higher = riskier

    # Adjustments
    base_size: float  # requested size
    adjusted_size: float  # after risk limits
    size_reduction_pct: float  # how much reduced

    # Reasoning
    rejection_reason: Optional[str] = None


# ============================================================================
# LAYER 9: EXECUTION
# ============================================================================

@dataclass(frozen=True)
class ExecutionRecord:
    """
    What actually happened when we executed.
    """
    timestamp: int
    order_timestamp: int  # when order was placed
    fill_timestamp: int  # when it filled
    symbol: str

    side: str  # "BUY" or "SELL"
    order_price: float
    fill_price: float
    requested_size: float
    filled_size: float

    # Costs
    slippage_pct: float
    fees_pct: float
    total_cost_pct: float

    status: str  # "pending", "filled", "rejected", "cancelled"
    order_id: str = ""


# ============================================================================
# MASTER: SYSTEM STATE
# ============================================================================

@dataclass(frozen=True)
class SystemState:
    """
    Complete system state at one timestamp.
    This is the immutable, serializable snapshot.
    Every bar gets one SystemState entry.
    """
    timestamp: int
    symbol: str

    # Layer outputs (flow through system)
    market_snapshot: MarketSnapshot
    feature_vector: Optional[FeatureVector] = None
    zone_map: Optional[ZoneMap] = None
    regime_state: Optional[RegimeState] = None
    expert_outputs: Optional[ExpertOutputs] = None
    opportunity_score: Optional[OpportunityScore] = None
    router_decision: Optional[RouterDecision] = None
    risk_decision: Optional[RiskDecision] = None
    execution_record: Optional[ExecutionRecord] = None

    # Results
    realized_pnl: Optional[float] = None  # actual PnL after execution
    """
    this can cause coz issue if this is not Async coz realised PnL make be know 2 candles later (other option is to save the PnL at that given point of time which will cause wrong data collecetion but dont do that )
    """

    def to_dict(self) -> Dict:
        """Convert to dict for serialization."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'SystemState':
        """Create from dict."""
        # Would need custom deserialization for nested objects
        # For now, keep it simple
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
