"""
indicators/moving_averages.py
------------------------------
Simple Moving Average (SMA) and Exponential Moving Average (EMA).

Columns added to DataFrame:
  SMA → sma_<period>        e.g. sma_20, sma_50, sma_200
  EMA → ema_<period>        e.g. ema_20, ema_50

Both classes accept a list of periods so one instantiation handles all windows.

Design note: EMA uses pandas ewm() with adjust=False — this matches the
standard definition used by most charting platforms (TradingView, Zerodha Kite).
"""

import pandas as pd
from indicators.base import BaseIndicator
from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)


class SMA(BaseIndicator):
    """Simple Moving Average over one or more periods."""

    def __init__(self, periods: list[int] | None = None):
        self.periods = periods or cfg["indicators"]["moving_averages"]

    @property
    def name(self) -> str:
        return "SMA"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(df, ["close"])
        out = df.copy()
        for p in self.periods:
            col = f"sma_{p}"
            out[col] = out["close"].rolling(window=p, min_periods=p).mean().round(4)
            log.debug(f"[SMA] Computed sma_{p} | NaN count: {out[col].isna().sum()}")
        return out


class EMA(BaseIndicator):
    """Exponential Moving Average over one or more periods."""

    def __init__(self, periods: list[int] | None = None):
        self.periods = periods or cfg["indicators"]["moving_averages"]

    @property
    def name(self) -> str:
        return "EMA"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(df, ["close"])
        out = df.copy()
        for p in self.periods:
            col = f"ema_{p}"
            out[col] = (
                out["close"]
                .ewm(span=p, adjust=False, min_periods=p)
                .mean()
                .round(4)
            )
            log.debug(f"[EMA] Computed ema_{p} | NaN count: {out[col].isna().sum()}")
        return out
