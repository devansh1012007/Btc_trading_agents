"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional

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
    prob_trending_up: float
    prob_trending_down: float
    prob_chop_range: float
    prob_panic_spike: float

    # Best guess (but use probabilities upstream)
    primary_regime: RegimeType
    regime_confidence: float  # 0-1

    # Hidden state tracking (from HMM)
    hmm_hidden_state: Optional[int] = None
    hmm_probabilities: List[float] = field(default_factory=list)

    # Persistence metrics
    regime_age: int = 0  # bars since last regime change
    regime_persistence_score: float = 0.0  # how sticky is current regime

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['primary_regime'] = self.primary_regime.value
        return d


# ============================================================================
# LAYER 5: EXPERT OUTPUTS
# ============================================================================

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
    regime_used: RegimeType = RegimeType.UNKNOWN


@dataclass(frozen=True)
class ExpertOutputs:
    """All experts' decisions at one timestamp."""
    timestamp: int
    symbol: str

    outputs: Dict[str, ExpertOutput] = field(default_factory=dict)

    def add_expert(self, expert_output: ExpertOutput) -> "ExpertOutputs":
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
