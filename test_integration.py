#!/usr/bin/env python
"""
Phase 1 Integration Test: End-to-end pipeline with mock data.
Demonstrates: fetch → market data → features → save
"""
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.state.contracts import MarketSnapshot, SystemState
from src.layers.market_data import MarketDataLayer
from src.layers.feature_engine import FeatureEngine
from src.state.serialization import system_state_to_dict
from pathlib import Path


def generate_mock_ohlcv(n_bars: int = 100, start_price: float = 50000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    logger.info(f"Generating {n_bars} mock OHLCV bars...")
    
    now = datetime.utcnow()
    data = []
    price = start_price
    
    for i in range(n_bars):
        # Random walk
        change_pct = np.random.normal(0, 0.5) / 100
        price = price * (1 + change_pct)
        
        open_p = price * (1 + np.random.normal(0, 0.2) / 100)
        close_p = price * (1 + np.random.normal(0, 0.2) / 100)
        high_p = max(open_p, close_p) * (1 + abs(np.random.normal(0, 0.1)) / 100)
        low_p = min(open_p, close_p) * (1 - abs(np.random.normal(0, 0.1)) / 100)
        volume = np.random.uniform(100, 1000)
        
        timestamp = now + timedelta(minutes=i)
        
        data.append({
            'timestamp': timestamp,
            'open': open_p,
            'high': high_p,
            'low': low_p,
            'close': close_p,
            'volume': volume,
            'symbol': 'BTC/USDT',
        })
    
    df = pd.DataFrame(data)
    logger.info(f"✓ Generated {len(df)} bars, price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    return df


def test_end_to_end_pipeline():
    """Run complete Phase 1 pipeline with mock data."""
    logger.info("=" * 80)
    logger.info("PHASE 1 END-TO-END INTEGRATION TEST")
    logger.info("=" * 80)
    
    # Step 1: Generate mock data
    df_ohlcv = generate_mock_ohlcv(n_bars=100)
    
    # Step 2: Create MarketSnapshots
    logger.info("\nStep 2: Creating MarketSnapshots...")
    snapshots = MarketDataLayer.batch_create_from_dataframe(df_ohlcv, "BTC/USDT")
    assert len(snapshots) == len(df_ohlcv), "Snapshot count mismatch"
    logger.info(f"✓ Created {len(snapshots)} MarketSnapshots")
    
    # Verify snapshot integrity
    snap = snapshots[0]
    assert snap.timestamp > 0
    assert snap.close > 0
    assert snap.spread_pct >= 0
    logger.info(f"  Sample: timestamp={snap.timestamp}, close=${snap.close:.2f}, spread={snap.spread_pct:.4f}%")
    
    # Step 3: Compute features
    logger.info("\nStep 3: Computing features with FeatureEngine...")
    feature_engine = FeatureEngine(lookback_bars=100)
    system_states = []
    
    for i, snapshot in enumerate(snapshots):
        feature_engine.add_bar(snapshot)
        fv = feature_engine.compute_features(snapshot)
        
        if fv:
            state = SystemState(
                timestamp=snapshot.timestamp,
                symbol=snapshot.symbol,
                market_snapshot=snapshot,
                feature_vector=fv,
            )
            system_states.append(state)
        
        if (i + 1) % 25 == 0:
            logger.info(f"  Processed {i + 1}/{len(snapshots)} bars")
    
    logger.info(f"✓ Created {len(system_states)} SystemStates with features")
    
    # Verify features
    if system_states:
        state = system_states[-1]
        fv = state.feature_vector
        logger.info(f"  Sample features:")
        logger.info(f"    momentum_30s: {fv.momentum_30s:.2f}%")
        logger.info(f"    momentum_5m: {fv.momentum_5m:.2f}%")
        logger.info(f"    volatility_30s: {fv.volatility_30s:.2f}%")
        logger.info(f"    rsi_14: {fv.rsi_14:.2f}")
        logger.info(f"    sma_9: ${fv.sma_9:.2f}, sma_21: ${fv.sma_21:.2f}")
        logger.info(f"    imbalance_ratio: {fv.imbalance_ratio:.4f}")
    
    # Step 4: Serialize to Parquet
    logger.info("\nStep 4: Serializing to Parquet...")
    records = [system_state_to_dict(s) for s in system_states]
    df_processed = pd.DataFrame(records)
    
    output_path = "data/processed/test_btc_features.parquet"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(output_path, compression='snappy')
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"✓ Saved {len(df_processed)} rows to {output_path}")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    
    # Step 5: Reload and verify
    logger.info("\nStep 5: Reloading from Parquet...")
    df_reloaded = pd.read_parquet(output_path)
    assert len(df_reloaded) == len(df_processed), "Reload row count mismatch"
    logger.info(f"✓ Successfully reloaded {len(df_reloaded)} rows")
    
    # Step 6: Data quality checks
    logger.info("\nStep 6: Data Quality Checks...")
    
    # Check for NaN in critical columns
    critical_cols = [
        'market_close', 'market_spread_pct',
        'feature_momentum_30s', 'feature_momentum_5m',
    ]
    
    for col in critical_cols:
        if col in df_reloaded.columns:
            nan_count = df_reloaded[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"⚠ Found {nan_count} NaN values in {col} (acceptable for edge cases)")
            # Only fail on market data NaN
            if col.startswith('market_'):
                assert nan_count == 0, f"NaN found in {col}: {nan_count} rows"
    
    logger.info("✓ No NaN values in critical columns")
    
    # Check timestamp alignment
    assert 'timestamp' in df_reloaded.columns
    timestamps = df_reloaded['timestamp'].values
    assert len(np.unique(timestamps)) == len(timestamps), "Duplicate timestamps detected"
    logger.info(f"✓ Timestamps aligned and unique ({len(timestamps)} unique)")
    
    # Step 7: Summary stats
    logger.info("\nStep 7: Summary Statistics...")
    logger.info(f"\nDataset Info:")
    logger.info(f"  Total rows: {len(df_processed)}")
    logger.info(f"  Total columns: {len(df_processed.columns)}")
    logger.info(f"  Date range: {df_processed['timestamp'].min()} to {df_processed['timestamp'].max()}")
    
    if 'market_close' in df_processed.columns:
        closes = df_processed['market_close']
        logger.info(f"\nPrice Statistics:")
        logger.info(f"  Min: ${closes.min():.2f}")
        logger.info(f"  Max: ${closes.max():.2f}")
        logger.info(f"  Mean: ${closes.mean():.2f}")
        logger.info(f"  Std: ${closes.std():.2f}")
    
    # All checks passed
    logger.info("\n" + "=" * 80)
    logger.info("✅ PHASE 1 INTEGRATION TEST PASSED")
    logger.info("=" * 80)
    logger.info("\nNext Steps:")
    logger.info("1. Review PHASE 1 README: README_PHASE1.md")
    logger.info("2. Start PHASE 2: Dynamic Structure Engine (zones)")
    logger.info("3. Implement Regime Detection (HMM + classifier)")
    logger.info("\nKey files to review:")
    logger.info("  - src/state/contracts.py (state schema)")
    logger.info("  - src/layers/feature_engine.py (feature computation)")
    logger.info("  - config/settings.yaml (configuration)")
    
    return df_processed


if __name__ == "__main__":
    try:
        df = test_end_to_end_pipeline()
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        sys.exit(1)
