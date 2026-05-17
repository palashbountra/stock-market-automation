"""
strategies/base.py
------------------
Abstract base class every strategy must inherit from.

Contract:
  - name         : unique string ID  (used in config, logging, reports)
  - required_indicators : list of column names this strategy needs
  - generate_signals()  : accepts enriched DataFrame, returns it with signal columns

Signal column spec (added to DataFrame):
  signal      : int   →  1 = BUY, -1 = SELL, 0 = HOLD/flat
  position    : int   →  1 = long, -1 = short, 0 = no position (forward-filled signal)
  entry_price : float →  price at which trade was entered (NaN when no trade)
  exit_price  : float →  price at which trade was exited  (NaN when no exit)
  trade_reason: str   →  human-readable explanation of why signal fired

This contract is what the Backtesting engine (Phase 4) and Paper Trader
(Phase 5) will consume — they never look inside strategy logic.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier."""
        ...

    @property
    def required_indicators(self) -> list[str]:
        """
        List of DataFrame columns this strategy requires.
        Override in subclass to get automatic pre-flight checks.
        """
        return []

    def validate(self, df: pd.DataFrame) -> None:
        """Raises ValueError if any required indicator column is missing."""
        missing = set(self.required_indicators) - set(df.columns)
        if missing:
            raise ValueError(
                f"[{self.name}] Missing indicator columns: {missing}. "
                f"Run IndicatorPipeline first."
            )

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : enriched OHLCV DataFrame with indicator columns

        Returns
        -------
        Same DataFrame with added columns:
            signal, position, entry_price, exit_price, trade_reason
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers available to all strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _init_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add blank signal columns to a copy of df."""
        out = df.copy()
        out["signal"]       = 0
        out["position"]     = 0
        out["entry_price"]  = float("nan")
        out["exit_price"]   = float("nan")
        out["trade_reason"] = ""
        return out

    @staticmethod
    def _forward_fill_position(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert point-in-time signals into a continuous position column.
        BUY  signal → position stays 1 until SELL signal
        SELL signal → position stays -1 until BUY signal
        """
        position = 0
        positions = []
        for sig in df["signal"]:
            if sig == 1:
                position = 1
            elif sig == -1:
                position = -1
            positions.append(position)
        df["position"] = positions
        return df
