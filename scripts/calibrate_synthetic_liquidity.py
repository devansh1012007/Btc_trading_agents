"""
Calibration for the synthetic liquidity model (§4.5). NOT required to
ship Phase 4 -- the synthetic model's coefficients (liquidity.synthetic.*
in settings.base.yaml) are hand-picked starting points. Once you've run
in paper-live-data mode for a while, every bar processed in that mode has
logged BOTH a real order-book observation (via the real order-book
adapter) and, implicitly, the features that would have driven the
synthetic estimate for that same bar -- this script fits the synthetic
model's coefficients against those paired observations via ordinary
least squares, rather than leaving them as guesses forever.

Expects a Parquet log of paired samples with columns: volatility_medium,
volume_zscore, bar_volume, bar_close, real_spread_pct, real_depth_usdt.
This log is not produced automatically by anything in Phase 4 as shipped
-- wiring paper-live-data mode to write it is a small, natural follow-on
once you're actually running that mode for a while, not a data-engineering
gap (the real and synthetic estimates are already computed side by side
in that mode; this script only needs them written to disk).
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fit_spread_model(df: pd.DataFrame) -> dict:
    """
    real_spread_pct ~ base + k_volatility * volatility_5m + k_illiquidity / max(|volume_zscore|, floor)
    Fit base and k_volatility via OLS on the volatility term; k_illiquidity
    and the floor are harder to fit jointly via plain OLS (nonlinear in
    the floor), so this fits a fixed floor and solves the remaining linear
    system -- adequate for periodic recalibration, not a general nonlinear
    solver.
    """
    floor = 0.1
    illiquidity_term = 1.0 / np.maximum(np.abs(df["volume_zscore"]), floor)
    X = np.column_stack([np.ones(len(df)), df["volatility_medium"].values, illiquidity_term])
    y = df["real_spread_pct"].values

    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    base_spread_pct, k_volatility, k_illiquidity = coeffs
    predicted = X @ coeffs
    r2 = 1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2) if len(y) > 1 else None

    return {
        "base_spread_pct": float(base_spread_pct), "k_volatility": float(k_volatility),
        "k_illiquidity": float(k_illiquidity), "min_volume_zscore_floor": floor, "r2": float(r2) if r2 is not None else None,
    }


def fit_depth_model(df: pd.DataFrame) -> dict:
    """real_depth_usdt ~ k_depth * bar_volume * bar_close * participation_rate.
    Fits k_depth * participation_rate as one product (they're not
    separately identifiable from depth alone) and reports it as k_depth
    with depth_participation_rate held at its configured default."""
    bar_notional = df["bar_volume"].values * df["bar_close"].values
    combined = np.linalg.lstsq(bar_notional.reshape(-1, 1), df["real_depth_usdt"].values, rcond=None)[0][0]
    return {"k_depth_times_participation_rate": float(combined)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Parquet log of paired real-vs-synthetic liquidity samples")
    parser.add_argument("--output", default="data/processed/liquidity_calibration.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"{input_path} not found. This script needs a log of paired real-vs-synthetic "
            "liquidity samples from paper-live-data mode -- see this file's module docstring."
        )

    df = pd.read_parquet(input_path)
    required = {"volatility_medium", "volume_zscore", "bar_volume", "bar_close", "real_spread_pct", "real_depth_usdt"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input is missing required columns: {missing}")

    spread_fit = fit_spread_model(df)
    depth_fit = fit_depth_model(df)
    logger.info(f"Fitted spread model (R^2={spread_fit['r2']}): {spread_fit}")
    logger.info(f"Fitted depth model: {depth_fit}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"spread": spread_fit, "depth": depth_fit, "n_samples": len(df)}, f, indent=2)
    logger.info(f"Saved calibration to {args.output} -- update liquidity.synthetic in settings.base.yaml by hand with these values")


if __name__ == "__main__":
    main()
