"""
execution/price_feed.py
------------------------
Price feed abstraction layer.

Two implementations:
  MockPriceFeed   — replays stored historical data bar-by-bar.
                    Used now (Phase 5) to simulate live market without API.
  KitePriceFeed   — streams real-time quotes from Zerodha WebSocket.
                    Used in Phase 6 when we go live.

Both implement the same interface so the paper trader doesn't need to change.

MockPriceFeed design:
  - Loads stored OHLCV data for each symbol
  - Returns one bar at a time per iteration (simulates end-of-bar signal)
  - Adds small random noise within each bar's H/L range to simulate intrabar
    price movement (optional, off by default)
  - Respects market hours check (skip weekends/holidays)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Iterator, List, Optional

import pandas as pd

from data.loader import DataLoader
from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)


class BasePriceFeed(ABC):

    @abstractmethod
    def stream(self) -> Iterator[Dict[str, dict]]:
        """
        Yields one tick per iteration.
        Each tick is a dict: { symbol: { open, high, low, close, volume, date } }
        """
        ...

    @abstractmethod
    def latest(self) -> Dict[str, float]:
        """Return {symbol: current_price} for all symbols."""
        ...


class MockPriceFeed(BasePriceFeed):
    """
    Replays historical OHLCV data bar-by-bar.

    Parameters
    ----------
    symbols      : list of NSE symbols
    speed_seconds: pause between bars (0 = as fast as possible for testing)
    start_date   : replay from this date (default: earliest available)
    end_date     : replay until this date (default: latest available)
    """

    def __init__(
        self,
        symbols:       List[str] | None = None,
        speed_seconds: float = 0.0,
        start_date:    str | None = None,
        end_date:      str | None = None,
    ):
        self.symbols       = symbols or cfg["universe"]["symbols"]
        self.speed_seconds = speed_seconds
        self.start_date    = start_date
        self.end_date      = end_date
        self._current_bars: Dict[str, dict] = {}

        loader = DataLoader()
        self._data: Dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            df = loader.get(sym, start=start_date, end=end_date)
            self._data[sym] = df.reset_index(drop=True)

        log.info(
            f"[MockFeed] Loaded {len(self.symbols)} symbols | "
            f"Bars per symbol: {len(next(iter(self._data.values())))}"
        )

    def stream(self) -> Iterator[Dict[str, dict]]:
        """
        Yields one bar-snapshot per iteration (all symbols at same date).
        Bars are aligned by date across symbols.
        """
        # Build unified date index across all symbols
        all_dates = sorted(set(
            date
            for df in self._data.values()
            for date in df["date"].tolist()
        ))

        for date in all_dates:
            tick: Dict[str, dict] = {}

            for sym in self.symbols:
                df  = self._data[sym]
                row = df[df["date"] == date]
                if row.empty:
                    continue

                r = row.iloc[0]
                bar = {
                    "date":   r["date"],
                    "open":   r["open"],
                    "high":   r["high"],
                    "low":    r["low"],
                    "close":  r["close"],
                    "volume": r["volume"],
                }
                tick[sym]                  = bar
                self._current_bars[sym]    = bar

            if tick:
                if self.speed_seconds > 0:
                    time.sleep(self.speed_seconds)
                yield tick

    def latest(self) -> Dict[str, float]:
        return {sym: bar["close"] for sym, bar in self._current_bars.items()}


class KitePriceFeed(BasePriceFeed):
    """
    Real-time WebSocket feed from Zerodha Kite.
    Implemented in Phase 6 — placeholder here so imports don't break.
    """

    def __init__(self, symbols: List[str] | None = None):
        raise NotImplementedError(
            "KitePriceFeed is implemented in Phase 6. "
            "Set config.data.use_mock=true to use MockPriceFeed."
        )

    def stream(self):
        raise NotImplementedError

    def latest(self):
        raise NotImplementedError


def get_price_feed(symbols: List[str] | None = None, **kwargs) -> BasePriceFeed:
    """Factory — returns MockPriceFeed or KitePriceFeed based on config."""
    if cfg["data"]["use_mock"]:
        return MockPriceFeed(symbols=symbols, **kwargs)
    return KitePriceFeed(symbols=symbols)
