# ADR-0012: Causal `ModelLedger` Scorecards, `MarketProfile`, and `AgentSink` Handoff Contract

- **Status:** Accepted

## Context
Tracking real per-model performance without look-ahead, trading non-crypto instruments (equities and ATM options), and handing attributed expert trades to an external agent require explicit contracts.

## Decision
1. Record predictions and outcomes in `ModelLedger` (`src/monitoring/ledger.py`) and join them strictly where `outcome.resolved_timestamp <= as_of_timestamp` (L8).
2. Parameterize market conventions via `MarketProfile` (`src/market/profile.py`) and select ATM option contracts via `ATMSelector` (`src/market/options.py`).
3. Emit HMAC-signed, idempotent `TradeSignal` payloads through `AgentSink` (`off | shadow | advisory | execute` in `src/handoff/agent_sink.py`).
