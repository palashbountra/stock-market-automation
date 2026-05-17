"""
indicators/macd.py
------------------
Moving Average Convergence Divergence (MACD).

Standard parameters: fast=12, slow=26, signal=9  (Gerald Appel's original)

Columns added:
  macd_line     = EMA(fast) - EMA(slow)
  macd_signal   = EMA(macd_line, signal_period)
  macd_hist     = macd_line - macd_signal   ← histogram (momentum bars)

Reading MACD for equity + options:
  - Histogram crossing zero → momentum shift (directional bias)
  - Divergence (price makes new high, histogram does not) → exhaustion signal
  - For options: MACD crossovers confirm direction before buying directional spreads
  - Zero-line rejection on MACD histogram → mean reversion signal for straddles
"""

import pandas as pd
from indicators.base import BaseIndicator
from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)


class MACD(BaseIndicator):

    def __init__(
        self,
        fast: int | None = None,
        slow: int | None = None,
        signal: int | None = None,
    ):
        macd_cfg = cfg["indicators"]["macd"]
        self.fast   = fast   or macd_cfg["fast"]
        self.slow   = slow   or macd_cfg["slow"]
        self.signal = signal or macd_cfg["signal"]

    @property
    def name(self) -> str:
        return f"MACD_{self.fast}_{self.slow}_{self.signal}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(df, ["close"])
        out = df.copy()

        ema_fast = out["close"].ewm(span=self.fast, adjust=False, min_periods=self.fast).mean()
        ema_slow = out["close"].ewm(span=self.slow, adjust=False, min_periods=self.slow).mean()

        out["macd_line"]   = (ema_fast - ema_slow).round(4)
        out["macd_signal"] = (
            out["macd_line"]
            .ewm(span=self.signal, adjust=False, min_periods=self.signal)
            .mean()
            .round(4)
        )
        out["macd_hist"] = (out["macd_line"] - out["macd_signal"]).round(4)

        log.debug(
            f"[MACD] Computed macd_line / macd_signal / macd_hist "
            f"({self.fast}/{self.slow}/{self.signal}) | "
            f"NaN: {out['macd_line'].isna().sum()}"
        )
        return out
