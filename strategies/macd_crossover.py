"""
strategies/macd_crossover.py
-----------------------------
Strategy: MACD Line / Signal Line Crossover
--------------------------------------------
A pure momentum strategy — no DMA filter, works in both trending and
recovering markets.

ENTRY (BUY):
  1. macd_line crosses ABOVE macd_signal (bullish crossover)
  2. macd_line is below zero (avoids chasing late in a move)
     → buying early in a new upswing, not at the top

EXIT (SELL):
  1. macd_line crosses BELOW macd_signal (bearish crossover)
  OR
  2. macd_hist turns negative after being positive (early exit on momentum loss)

Long-only (same reasoning as rsi_dma — for options: triggers bull debit spreads).

Why two strategies?
  RSI_DMA   = trend follower (needs 50+200 DMA alignment) — fewer but cleaner trades
  MACD Cross = momentum      (no DMA requirement) — more trades, works in recovery legs
  Together in backtest they show which regime fits a given stock better.
"""

import pandas as pd
from strategies.base import BaseStrategy
from utils.logger import get_logger

log = get_logger(__name__)


class MACD_Crossover_Strategy(BaseStrategy):

    def __init__(self, require_below_zero: bool = True):
        """
        Parameters
        ----------
        require_below_zero : if True, only enter when macd_line < 0
                             (avoids late entries in extended rallies)
        """
        self.require_below_zero = require_below_zero

    @property
    def name(self) -> str:
        return "macd_crossover"

    @property
    def required_indicators(self) -> list[str]:
        return ["macd_line", "macd_signal", "macd_hist"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate(df)
        out = self._init_signal_columns(df)

        valid_mask = out[["macd_line", "macd_signal", "macd_hist"]].notna().all(axis=1)

        prev_line   = None
        prev_signal = None
        prev_hist   = None
        in_trade    = False

        for idx in out.index:
            if not valid_mask[idx]:
                prev_line = prev_signal = prev_hist = None
                continue

            macd_line   = out.at[idx, "macd_line"]
            macd_signal = out.at[idx, "macd_signal"]
            macd_hist   = out.at[idx, "macd_hist"]
            close       = out.at[idx, "close"]

            if prev_line is not None and prev_signal is not None:

                # ── ENTRY ──────────────────────────────────────────────
                bullish_cross = (prev_line <= prev_signal) and (macd_line > macd_signal)
                below_zero_ok = (not self.require_below_zero) or (macd_line < 0)

                if not in_trade and bullish_cross and below_zero_ok:
                    out.at[idx, "signal"]       = 1
                    out.at[idx, "entry_price"]  = close
                    out.at[idx, "trade_reason"] = (
                        f"BUY | MACD crossed above signal "
                        f"({macd_line:.2f} > {macd_signal:.2f})"
                        f"{' | macd<0' if self.require_below_zero else ''}"
                    )
                    in_trade = True
                    log.debug(f"[{self.name}] BUY  @ {out.at[idx,'date']} | close={close}")

                # ── EXIT ───────────────────────────────────────────────
                elif in_trade:
                    bearish_cross    = (prev_line >= prev_signal) and (macd_line < macd_signal)
                    hist_turned_neg  = (prev_hist is not None and prev_hist > 0 and macd_hist < 0)

                    if bearish_cross or hist_turned_neg:
                        reason = "bearish MACD cross" if bearish_cross else "MACD hist turned negative"
                        out.at[idx, "signal"]       = -1
                        out.at[idx, "exit_price"]   = close
                        out.at[idx, "trade_reason"] = f"SELL | {reason}"
                        in_trade = False
                        log.debug(f"[{self.name}] SELL @ {out.at[idx,'date']} | close={close}")

            prev_line   = macd_line
            prev_signal = macd_signal
            prev_hist   = macd_hist

        out = self._forward_fill_position(out)

        buys  = (out["signal"] ==  1).sum()
        sells = (out["signal"] == -1).sum()
        log.info(f"[{self.name}] Signals generated → BUY: {buys} | SELL: {sells}")

        return out
