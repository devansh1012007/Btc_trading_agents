"""
Unit tests for Phase 13 T13.5: `run_chunked_replay` produces identical `SystemState`
sequences regardless of `chunk_size`.
"""
import pytest

from src.backtest.chunked_replay import run_chunked_replay
from src.backtest.replay import replay_full_stack
from src.utils.synthetic import synthetic_ohlcv

pytestmark = [pytest.mark.unit]


def test_chunked_replay_matches_full_stack_replay_bar_for_bar(paper_config):
    df = synthetic_ohlcv(n_bars=160, bar_seconds=60)
    full = replay_full_stack(df, "BTC/USDT", paper_config, run_experts=True)
    chunked = run_chunked_replay(df, "BTC/USDT", paper_config, chunk_size=45, run_experts=True)

    assert chunked.chunks_processed == 4
    assert len(chunked.retained_states) == len(full.system_states)
    for s_full, s_chunk in zip(full.system_states, chunked.retained_states, strict=True):
        assert s_full.timestamp == s_chunk.timestamp
        assert s_full.regime_state.primary_regime == s_chunk.regime_state.primary_regime
        assert s_full.router_decision.selected_expert == s_chunk.router_decision.selected_expert
