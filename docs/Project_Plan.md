# BTC Scalping Multi-Expert Trading Agent - Full Project Plan

## 1. Project Brief
**Problem Statement:** Build a profitable, adaptive BTC scalping agent (30s & 1m) that dynamically selects and executes strategies across different market regimes using specialized experts.

**Target Users:** Solo trader running on laptop (Binance/USDT-BTC).

**Success Metrics:**
- Positive expectancy after fees/slippage vs baseline LightGBM model
- Win rate + risk-adjusted return better than simple momentum
- Max drawdown < 8% in backtests
- Survives chop, trend, and high-vol regimes

**Constraints:**
- Runs on 12GB RAM + integrated graphics (no GPU training)
- Real-time capable on laptop
- Monolith architecture
- Must be highly modular and debuggable

**Core Philosophy:** Layered decision making with multiple safety gates. Never let any single model control the full decision.

## 2. High-Level Architecture

**Global Flow:**
Market Data → Feature Engine → Dynamic Structure → Regime Detection → Expert Layer → Opportunity Filter → Router → Risk Layer → Execution → Feedback/Logging

**Tech Stack:**
- **Language:** Python 3.11+
- **Backtesting:** VectorBT Pro (or basic VectorBT)
- **Data:** CCXT + Binance WebSocket
- **ML:** LightGBM (primary), hmmlearn, scikit-learn
- **Async:** asyncio + concurrent.futures for experts
- **State:** Pydantic models (strict schemas)
- **Logging:** structlog + JSON logs
- **Simulation:** Custom event-driven backtester

## 3. Module Breakdown (Implementation Order)

### Phase 1: Foundation (Must be rock solid)
1. **Shared State Contract** (Most Important)
   - Define `MarketState`, `FeatureVector`, `ZoneMap`, `RegimeState`, etc. using Pydantic.
   
2. **Market Data Layer**
   - WebSocket manager (Binance)
   - OHLCV + orderbook + trades synchronization
   - Multi-timeframe alignment

3. **Feature Engine**
   - All engineered features (no leakage)
   - Modular feature groups

### Phase 2: Core Intelligence
4. **Dynamic Structure Engine** (Support/Resistance Zones)
5. **Regime Detection Layer** (HMM + LightGBM)
6. **Expert Layer** (Start with 3 experts: Trend, Mean Reversion, Breakout)
7. **Opportunity Filter**
8. **Router** (Start with LightGBM ranking → later Contextual Bandit)

### Phase 3: Safety & Execution
9. **Risk Layer** (Strongest gate)
10. **Execution Layer** (CCXT order management)
11. **Feedback + Logging + Walk-forward Retraining System**

## 4. Data & Training Strategy
- Use historical data (at least 1-2 years of tick/1s data)
- Walk-forward optimization (no future leakage)
- Train each component on its relevant regime only
- Generate labels from simulated outcomes

## 5. Risk & Failure Management
**Highest Risk Areas:**
- WebSocket disconnects / latency
- Memory leaks with growing zone map
- Overfitting in experts/router
- Execution slippage in live

**Mitigations:**
- Circuit breaker: Stop trading after 3 consecutive errors
- Detailed structured logging
- Multiple validation layers
- Paper trading mode mandatory before live

## 6. Development Rules (Anti-Slop)
- Every module must have its own test suite
- Strict input/output contracts
- No direct data sharing between layers — only through state object
- Manual code review after each major module
- Document trade-offs in ADR (Architecture Decision Records)

## 7. Milestones
1. Data pipeline + features working (backtest only)
2. Structure engine + regime detection
3. 3 Experts + Opportunity Filter
4. Router + Risk Layer
5. Full backtesting engine with VectorBT
6. Paper trading on Binance testnet
7. Live trading with kill switch

## 8. Future Upgrades (Nice-to-have)
- Telegram alerts
- PostgreSQL + TimescaleDB
- Hierarchical RL (after stable Stage 3)
- More experts
- Dynamic timeframe switching

**Current Priority:** Get to Milestone 4 (Router + Risk) with 3 experts working reliably in backtest.
