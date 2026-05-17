"""
indicators/pipeline.py
-----------------------
IndicatorPipeline: applies a list of indicators sequentially to a DataFrame.

This is the only class strategy/backtest modules need to import from the
indicators package. It keeps strategies decoupled from individual indicator
implementations.

Usage
-----
from indicators.pipeline import IndicatorPipeline

pipeline = IndicatorPipeline()           # loads defaults from config
df_with_indicators = pipeline.run(df)   # returns enriched DataFrame

Or custom:
    pipeline = IndicatorPipeline(indicators=[RSI(period=9), MACD(), VWAP(window=10)])
    df_out = pipeline.run(df)

Columns produced (defaults):
    sma_20, sma_50, sma_200
    ema_20, ema_50, ema_200
    rsi_14
    macd_line, macd_signal, macd_hist
    vwap, vwap_upper_1, vwap_lower_1, vwap_upper_2, vwap_lower_2
"""

from __future__ import annotations

import pandas as pd

from indicators.base import BaseIndicator
from indicators.moving_averages import SMA, EMA
from indicators.rsi import RSI
from indicators.macd import MACD
from indicators.vwap import VWAP
from utils.logger import get_logger

log = get_logger(__name__)

# Default set — matches exactly what the strategy engine expects
_DEFAULT_INDICATORS: list[BaseIndicator] = [
    SMA(),
    EMA(),
    RSI(),
    MACD(),
    VWAP(),
]


class IndicatorPipeline:

    def __init__(self, indicators: list[BaseIndicator] | None = None):
        self.indicators = indicators or _DEFAULT_INDICATORS

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all indicators in sequence.

        Parameters
        ----------
        df : raw OHLCV DataFrame

        Returns
        -------
        Enriched DataFrame with all indicator columns appended.
        NaN rows (warm-up period) are preserved — strategies handle filtering.
        """
        if df.empty:
            raise ValueError("Cannot run pipeline on empty DataFrame.")

        log.info(f"[Pipeline] Running {len(self.indicators)} indicators on {len(df)} rows")

        result = df.copy()
        for ind in self.indicators:
            try:
                result = ind.compute(result)
                log.debug(f"[Pipeline] ✓ {ind.name}")
            except Exception as e:
                log.error(f"[Pipeline] ✗ {ind.name} failed: {e}")
                raise

        # Report warm-up cost (rows with any NaN in indicator columns)
        indicator_cols = [c for c in result.columns if c not in df.columns]
        warmup_rows = result[indicator_cols].isna().any(axis=1).sum()
        log.info(
            f"[Pipeline] Complete | "
            f"Total rows: {len(result)} | "
            f"Warm-up (NaN) rows: {warmup_rows} | "
            f"Usable rows: {len(result) - warmup_rows} | "
            f"New columns: {indicator_cols}"
        )

        return result

    def run_all(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Convenience: run pipeline over a dict of {symbol: df}."""
        return {sym: self.run(df) for sym, df in data.items()}

    def add(self, indicator: BaseIndicator) -> "IndicatorPipeline":
        """Fluent method to append an indicator: pipeline.add(RSI(9)).add(VWAP(10))"""
        self.indicators.append(indicator)
        return self
