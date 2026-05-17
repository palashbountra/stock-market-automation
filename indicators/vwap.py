"""
indicators/vwap.py
------------------
Volume Weighted Average Price (VWAP).

VWAP = cumulative(typical_price × volume) / cumulative(volume)
  where typical_price = (high + low + close) / 3

Two modes:
  session  : resets every trading day  (standard intraday VWAP)
  rolling  : rolling N-day VWAP        (used with daily data — what we have now)

Since Phase 1 data is daily OHLCV (not tick/minute), we use ROLLING VWAP
by default. When we add intraday data in later phases, session VWAP becomes
meaningful.

Columns added:
  vwap              → rolling VWAP (default 20-day window)
  vwap_upper_1      → VWAP + 1 standard deviation band
  vwap_lower_1      → VWAP - 1 standard deviation band
  vwap_upper_2      → VWAP + 2 standard deviation bands
  vwap_lower_2      → VWAP - 2 standard deviation bands

Standard deviation bands are critical for options:
  Price at VWAP ± 2σ → mean reversion trades (iron condors, straddles)
  Price breaking VWAP + 1σ → trending, avoid selling premium

Usage in strategies:
  - Price crosses above VWAP → bullish bias (buy calls / bull put spread)
  - Price crosses below VWAP → bearish bias (buy puts / bear call spread)
  - Price hugging VWAP with low IV → short straddle candidate
"""

import numpy as np
import pandas as pd
from indicators.base import BaseIndicator
from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)


class VWAP(BaseIndicator):

    def __init__(self, window: int = 20, bands: list[int] | None = None):
        """
        Parameters
        ----------
        window : rolling window in bars (days for daily data)
        bands  : list of σ-multiples for deviation bands, default [1, 2]
        """
        self.window = window
        self.bands  = bands or [1, 2]

    @property
    def name(self) -> str:
        return f"VWAP_{self.window}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(df, ["high", "low", "close", "volume"])
        out = df.copy()

        # Typical price
        tp = (out["high"] + out["low"] + out["close"]) / 3
        tp_vol = tp * out["volume"]

        # Rolling VWAP
        rolling_tp_vol = tp_vol.rolling(window=self.window, min_periods=self.window).sum()
        rolling_vol    = out["volume"].rolling(window=self.window, min_periods=self.window).sum()

        out["vwap"] = (rolling_tp_vol / rolling_vol).round(4)

        # Standard deviation of typical price over same window (for bands)
        tp_std = tp.rolling(window=self.window, min_periods=self.window).std()

        for b in self.bands:
            out[f"vwap_upper_{b}"] = (out["vwap"] + b * tp_std).round(4)
            out[f"vwap_lower_{b}"] = (out["vwap"] - b * tp_std).round(4)

        log.debug(
            f"[VWAP] Computed vwap (window={self.window}) + "
            f"bands {self.bands} | NaN: {out['vwap'].isna().sum()}"
        )
        return out
