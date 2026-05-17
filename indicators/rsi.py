"""
indicators/rsi.py
-----------------
Relative Strength Index (RSI) — Wilder's smoothing method.

Column added: rsi_<period>   (default: rsi_14)

Wilder's RSI uses a modified EMA (SMMA / RMA) with alpha = 1/period.
This matches TradingView, Zerodha Kite charts, and most professional platforms.

Key levels (stored as constants for use in strategies):
  RSI_OVERSOLD  = 30   → potential buy zone
  RSI_OVERBOUGHT= 70   → potential sell zone
  RSI_NEUTRAL   = 50   → trend confirmation

For options strategies specifically:
  RSI extremes are used to judge directional bias before selling premium.
  e.g. RSI > 70 + high IV → consider bear call spread
"""

import numpy as np
import pandas as pd
from indicators.base import BaseIndicator
from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

# Standard RSI threshold constants — imported by strategy modules
RSI_OVERSOLD   = 30
RSI_OVERBOUGHT = 70
RSI_NEUTRAL    = 50


class RSI(BaseIndicator):

    def __init__(self, period: int | None = None):
        self.period = period or cfg["indicators"]["rsi_period"]

    @property
    def name(self) -> str:
        return f"RSI_{self.period}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(df, ["close"])
        out = df.copy()

        col = f"rsi_{self.period}"
        out[col] = self._wilder_rsi(out["close"], self.period)

        nan_count = out[col].isna().sum()
        log.debug(f"[RSI] Computed {col} | NaN count: {nan_count}")
        return out

    @staticmethod
    def _wilder_rsi(series: pd.Series, period: int) -> pd.Series:
        """
        Wilder's smoothed RSI — identical to TradingView's default RSI.
        Uses RMA (Running Moving Average) = EMA with alpha=1/period.
        """
        delta = series.diff()

        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # First average: simple mean over first `period` bars
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi.round(4)
