# Code Citations

## License: unknown
https://github.com/0xbigz/drift-v2-streamlit/blob/fdec260c518dc11a8fdd0735119260a340477e55/tabs/backtester.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/PpNutthanon/TestGit/blob/a1bb20de4d3064ded081e2aa8edf5a5633c4e4dc/Pythonpp/MarkTrade.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/sisandla/binance_analytics/blob/aff5a40950616788a86ffb30dd9dd080208b9593/test2.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/0xbigz/drift-v2-streamlit/blob/fdec260c518dc11a8fdd0735119260a340477e55/tabs/backtester.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/PpNutthanon/TestGit/blob/a1bb20de4d3064ded081e2aa8edf5a5633c4e4dc/Pythonpp/MarkTrade.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/sisandla/binance_analytics/blob/aff5a40950616788a86ffb30dd9dd080208b9593/test2.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/osdajiba/crypto_trader/blob/ac27678d569435006cfbfea50ac112556bb9d196/src/okx.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```


## License: unknown
https://github.com/0xbigz/drift-v2-streamlit/blob/fdec260c518dc11a8fdd0735119260a340477e55/tabs/backtester.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/PpNutthanon/TestGit/blob/a1bb20de4d3064ded081e2aa8edf5a5633c4e4dc/Pythonpp/MarkTrade.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/sisandla/binance_analytics/blob/aff5a40950616788a86ffb30dd9dd080208b9593/test2.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/osdajiba/crypto_trader/blob/ac27678d569435006cfbfea50ac112556bb9d196/src/okx.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```


## License: unknown
https://github.com/0xbigz/drift-v2-streamlit/blob/fdec260c518dc11a8fdd0735119260a340477e55/tabs/backtester.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/PpNutthanon/TestGit/blob/a1bb20de4d3064ded081e2aa8edf5a5633c4e4dc/Pythonpp/MarkTrade.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/sisandla/binance_analytics/blob/aff5a40950616788a86ffb30dd9dd080208b9593/test2.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/osdajiba/crypto_trader/blob/ac27678d569435006cfbfea50ac112556bb9d196/src/okx.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```


## License: unknown
https://github.com/0xbigz/drift-v2-streamlit/blob/fdec260c518dc11a8fdd0735119260a340477e55/tabs/backtester.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/PpNutthanon/TestGit/blob/a1bb20de4d3064ded081e2aa8edf5a5633c4e4dc/Pythonpp/MarkTrade.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/sisandla/binance_analytics/blob/aff5a40950616788a86ffb30dd9dd080208b9593/test2.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/osdajiba/crypto_trader/blob/ac27678d569435006cfbfea50ac112556bb9d196/src/okx.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```


## License: unknown
https://github.com/0xbigz/drift-v2-streamlit/blob/fdec260c518dc11a8fdd0735119260a340477e55/tabs/backtester.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/PpNutthanon/TestGit/blob/a1bb20de4d3064ded081e2aa8edf5a5633c4e4dc/Pythonpp/MarkTrade.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/sisandla/binance_analytics/blob/aff5a40950616788a86ffb30dd9dd080208b9593/test2.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/osdajiba/crypto_trader/blob/ac27678d569435006cfbfea50ac112556bb9d196/src/okx.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```


## License: unknown
https://github.com/0xbigz/drift-v2-streamlit/blob/fdec260c518dc11a8fdd0735119260a340477e55/tabs/backtester.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/PpNutthanon/TestGit/blob/a1bb20de4d3064ded081e2aa8edf5a5633c4e4dc/Pythonpp/MarkTrade.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/sisandla/binance_analytics/blob/aff5a40950616788a86ffb30dd9dd080208b9593/test2.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger
```


## License: unknown
https://github.com/osdajiba/crypto_trader/blob/ac27678d569435006cfbfea50ac112556bb9d196/src/okx.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```


## License: unknown
https://github.com/0xbigz/drift-v2-streamlit/blob/fdec260c518dc11a8fdd0735119260a340477e55/tabs/backtester.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```


## License: unknown
https://github.com/PpNutthanon/TestGit/blob/a1bb20de4d3064ded081e2aa8edf5a5633c4e4dc/Pythonpp/MarkTrade.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```


## License: unknown
https://github.com/osdajiba/crypto_trader/blob/ac27678d569435006cfbfea50ac112556bb9d196/src/okx.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```


## License: unknown
https://github.com/sisandla/binance_analytics/blob/aff5a40950616788a86ffb30dd9dd080208b9593/test2.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```


## License: unknown
https://github.com/0xbigz/drift-v2-streamlit/blob/fdec260c518dc11a8fdd0735119260a340477e55/tabs/backtester.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```


## License: unknown
https://github.com/PpNutthanon/TestGit/blob/a1bb20de4d3064ded081e2aa8edf5a5633c4e4dc/Pythonpp/MarkTrade.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```


## License: unknown
https://github.com/osdajiba/crypto_trader/blob/ac27678d569435006cfbfea50ac112556bb9d196/src/okx.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```


## License: unknown
https://github.com/sisandla/binance_analytics/blob/aff5a40950616788a86ffb30dd9dd080208b9593/test2.py

```
I'll generate comprehensive PHASE 1 code. Here's the complete implementation structure:

## PHASE 1: Foundation & Data Pipeline - Complete Implementation

### 1. **Project Structure Setup**

Create this directory structure:
```
Btc_trading_agents/
├── src/
│   ├── __init__.py
│   ├── state/
│   │   ├── __init__.py
│   │   ├── contracts.py          ← Start here
│   │   └── serialization.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── data_source.py        ← CCXT wrapper
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── market_data.py        ← Market snapshots
│   │   └── feature_engine.py     ← Feature engineering
│   └── pipeline.py               ← Orchestrator
├── config/
│   ├── settings.yaml
│   └── exchange.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── tests/
│   ├── __init__.py
│   └── test_phase1.py
├── requirements.txt
└── README.md
```

### 2. **requirements.txt**

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data & Exchange
ccxt>=3.0.0
python-binance>=1.0.17

# ML & Modeling (Phase 2+)
lightgbm>=4.0.0
hmmlearn>=0.3.0
scikit-learn>=1.3.0

# Backtesting
vectorbt>=0.25.0

# Config
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
pydantic>=2.0.0
structlog>=23.0.0
```

### 3. **src/state/contracts.py** (STATE SCHEMA - CRITICAL)

```python
"""
Shared state contract for the trading system.
All layers read/write through these immutable dataclasses.
This is the single source of truth for system state.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from datetime import datetime
import numpy as np

# ============================================================================
# REGIME & STRUCTURE ENUMS
# ============================================================================

class RegimeType(Enum):
    """Market regime classification."""
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FeatureVector:
    """
    Engineered features for all downstream layers.
    No look-ahead bias; computed from past data only.
    """
    timestamp: int
    symbol: str
    
    # Short-horizon features (30s-2m)
    momentum_30s: float
    momentum_5m: float
    volatility_30s: float
    volatility_5m: float
    returns_5: float  # 5-bar return
    rsi_14: float
    macd_signal: float
    
    # Medium-horizon features (5-10m)
    sma_9: float
    sma_21: float
    momentum_10m: float
    volatility_10m: float
    
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
        """Create from dict."""
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
    regime_relevance: Dict[RegimeType, float] = field(default_factory=dict)  # per-regime importance
    
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
    
    def add_expert(self, expert_output: ExpertOutput):
        """Add an expert output (returns new ExpertOutputs)."""
        new_outputs = self.outputs.copy()
        new_outputs[expert_output.expert_name] = expert_output
        return ExpertOutputs(self.timestamp, self.symbol, new_outputs)
    
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
    reasoning: str = ""


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
```

### 4. **src/state/serialization.py** (Serialization Helpers)

```python
"""
Serialize/deserialize SystemState to/from Parquet, JSON.
"""
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .contracts import SystemState, MarketSnapshot, FeatureVector, RegimeState

def system_state_to_dict(state: SystemState) -> Dict[str, Any]:
    """Flatten SystemState to dict (compatible with Parquet)."""
    d = {
        'timestamp': state.timestamp,
        'symbol': state.symbol,
    }
    
    # Flatten market snapshot
    if state.market_snapshot:
        ms = state.market_snapshot
        d.update({
            f'market_{k}': v for k, v in {
                'open': ms.open,
                'high': ms.high,
                'low': ms.low,
                'close': ms.close,
                'volume': ms.volume,
                'bid': ms.bid,
                'ask': ms.ask,
                'spread_pct': ms.spread_pct,
            }.items()
        })
    
    # Flatten feature vector
    if state.feature_vector:
        fv = state.feature_vector
        d.update({
            f'feature_{k}': v for k, v in {
                'momentum_30s': fv.momentum_30s,
                'momentum_5m': fv.momentum_5m,
                'volatility_30s': fv.volatility_30s,
                'spread_pct': fv.spread_pct,
            }.items()
        })
    
    # Flatten regime
    if state.regime_state:
        rs = state.regime_state
        d.update({
            f'regime_{k}': v for k, v in {
                'primary': rs.primary_regime.value,
                'confidence': rs.regime_confidence,
                'prob_up': rs.prob_trending_up,
                'prob_down': rs.prob_trending_down,
                'prob_chop': rs.prob_chop_range,
                'prob_panic': rs.prob_panic_spike,
            }.items()
        })
    
    # Flatten PnL
    if state.realized_pnl is not None:
        d['realized_pnl'] = state.realized_pnl
    
    return d


def dataframe_to_parquet(states: list, path: str) -> None:
    """Convert list of SystemStates to Parquet file."""
    records = [system_state_to_dict(s) for s in states]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.to_parquet(path, compression='snappy')


def parquet_to_dataframe(path: str) -> pd.DataFrame:
    """Load Parquet file back to DataFrame."""
    return pd.read_parquet(path)
```

### 5. **src/connectors/data_source.py** (CCXT Wrapper)

```python
"""
CCXT connector for fetching market data.
Handles normalization, caching, retry logic.
"""
import ccxt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CCXTConnector:
    """Wrapper around CCXT for BTC data fetching."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        """
        Initialize CCXT exchange connection.
        
        Args:
            exchange_name: 'binance', 'bybit', etc.
            sandbox: Use testnet/sandbox if available
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'sandbox': sandbox,
        })
        
        logger.info(f"Initialized {exchange_name} (sandbox={sandbox})")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: '1m', '5m', '1h', etc.
            limit: Number of bars to fetch
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            logger.info(f"Fetche
```

