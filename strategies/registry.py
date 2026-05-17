"""
strategies/registry.py
-----------------------
Strategy Registry — the plug-in loader for the strategy engine.

To add a new strategy (e.g. from a trading advisor):
  1. Create strategies/my_strategy.py inheriting BaseStrategy
  2. Register it here with a unique key
  3. Reference that key in config.yaml  strategy.default

The rest of the system (backtest, paper trader, live) never imports
strategies directly — always goes through the registry.

Usage
-----
from strategies.registry import StrategyRegistry

strategy = StrategyRegistry.get("rsi_dma_crossover")
df_signals = strategy.generate_signals(df_with_indicators)

# List all available strategies
StrategyRegistry.list()
"""

import pandas as pd
from strategies.base import BaseStrategy
from strategies.rsi_dma import RSI_DMA_Strategy
from strategies.macd_crossover import MACD_Crossover_Strategy
from strategies.vwap_reversion import VWAP_Reversion_Strategy
from utils.logger import get_logger

log = get_logger(__name__)

# ── Registry map ──────────────────────────────────────────────────────────
# key (used in config.yaml)  →  strategy class
_REGISTRY: dict[str, type[BaseStrategy]] = {
    "rsi_dma_crossover": RSI_DMA_Strategy,
    "macd_crossover":    MACD_Crossover_Strategy,
    "vwap_reversion":    VWAP_Reversion_Strategy,
}


class StrategyRegistry:

    @staticmethod
    def get(name: str, **kwargs) -> BaseStrategy:
        """
        Instantiate and return a strategy by name.

        Parameters
        ----------
        name   : registry key, e.g. "rsi_dma_crossover"
        kwargs : passed to the strategy constructor (override defaults)
        """
        if name not in _REGISTRY:
            available = list(_REGISTRY.keys())
            raise KeyError(
                f"Strategy '{name}' not found. Available: {available}"
            )
        strategy = _REGISTRY[name](**kwargs)
        log.info(f"[Registry] Loaded strategy: {strategy.name}")
        return strategy

    @staticmethod
    def list() -> list[str]:
        """Return names of all registered strategies."""
        return list(_REGISTRY.keys())

    @staticmethod
    def register(name: str, strategy_class: type[BaseStrategy]) -> None:
        """
        Dynamically register a new strategy at runtime.
        Useful when trading advisors provide custom strategy modules.
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(f"{strategy_class} must inherit from BaseStrategy")
        _REGISTRY[name] = strategy_class
        log.info(f"[Registry] Registered new strategy: {name}")

    @staticmethod
    def run_all(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Run every registered strategy on the same DataFrame. Returns dict of results."""
        results = {}
        for name in _REGISTRY:
            try:
                strategy = StrategyRegistry.get(name)
                results[name] = strategy.generate_signals(df)
            except Exception as e:
                log.error(f"[Registry] Strategy '{name}' failed: {e}")
        return results
