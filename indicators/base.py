"""
indicators/base.py
------------------
Abstract base class every indicator must inherit from.

Contract:
- __init__  : accept config parameters
- compute() : accept a DataFrame, return same DataFrame with new column(s) added
- name      : string identifier used for logging and output column naming

This enforces a plug-and-play pattern — the strategy engine can call
indicator.compute(df) without knowing which indicator it is.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseIndicator(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string identifier for this indicator."""
        ...

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : DataFrame with at minimum columns [date, open, high, low, close, volume]

        Returns
        -------
        Same DataFrame with one or more new columns appended in-place (copy).
        Never mutates the original df.
        """
        ...

    def _require_columns(self, df: pd.DataFrame, cols: list[str]) -> None:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"[{self.name}] DataFrame missing required columns: {missing}")
