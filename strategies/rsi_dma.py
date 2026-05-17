"""
strategies/rsi_dma.py
----------------------
Strategy: RSI + DMA Crossover
------------------------------
Classic trend-following strategy suitable for Indian large-caps.

ENTRY (BUY) rules — ALL must be true:
  1. RSI < rsi_oversold (default 40) — stock is in pullback/oversold zone
  2. close > sma_50                  — price is above 50-day MA (uptrend intact)
  3. close > sma_200                 — price is above 200-day MA (macro uptrend)
  4. macd_hist > 0                   — momentum is positive / turning up

EXIT (SELL) rules — ANY is sufficient:
  1. RSI > rsi_overbought (default 65) — stock is in overbought zone
  2. close < sma_50                    — price breaks below 50 DMA (trend break)
  3. macd_hist < 0 AND was > 0 last bar — MACD histogram turns negative (momentum loss)

Rationale:
  - Buying during RSI pullback while the trend (50 + 200 DMA) is intact
    captures mean-reversion bounces within uptrends — high win-rate in trending
    markets like Nifty 50 components.
  - MACD histogram confirmation reduces false signals in sideways markets.
  - This is a long-only strategy (suitable for equity, no shorting).
    For options: this signal can trigger bull call spreads or cash-secured puts.

Parameters (all in config.yaml or passed directly):
  rsi_period     : 14
  rsi_oversold   : 40   (relaxed from 30 — catches more setups in strong uptrends)
  rsi_overbought : 65
  fast_ma        : sma_50
  slow_ma        : sma_200
"""

import pandas as pd
from strategies.base import BaseStrategy
from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)


class RSI_DMA_Strategy(BaseStrategy):

    def __init__(
        self,
        rsi_oversold: float   = 40.0,
        rsi_overbought: float = 65.0,
        fast_ma: str          = "sma_50",
        slow_ma: str          = "sma_200",
        rsi_col: str          = "rsi_14",
    ):
        self.rsi_oversold   = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.fast_ma        = fast_ma
        self.slow_ma        = slow_ma
        self.rsi_col        = rsi_col

    @property
    def name(self) -> str:
        return "rsi_dma_crossover"

    @property
    def required_indicators(self) -> list[str]:
        return [self.rsi_col, self.fast_ma, self.slow_ma, "macd_hist"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate(df)

        out = self._init_signal_columns(df)

        # Drop warm-up NaN rows before signal generation
        valid_mask = out[[self.rsi_col, self.fast_ma, self.slow_ma, "macd_hist"]].notna().all(axis=1)

        prev_hist = None
        in_trade  = False

        for idx in out.index:
            if not valid_mask[idx]:
                prev_hist = None
                continue

            rsi      = out.at[idx, self.rsi_col]
            close    = out.at[idx, "close"]
            fast     = out.at[idx, self.fast_ma]
            slow     = out.at[idx, self.slow_ma]
            hist     = out.at[idx, "macd_hist"]

            # ── ENTRY ──────────────────────────────────────────────────
            if (
                not in_trade
                and rsi   < self.rsi_oversold
                and close > fast
                and close > slow
                and hist  > 0
            ):
                out.at[idx, "signal"]       = 1
                out.at[idx, "entry_price"]  = close
                out.at[idx, "trade_reason"] = (
                    f"BUY | RSI={rsi:.1f}<{self.rsi_oversold} "
                    f"& close>{self.fast_ma} & close>{self.slow_ma} "
                    f"& MACD_hist={hist:.2f}>0"
                )
                in_trade = True
                log.debug(f"[{self.name}] BUY  @ {out.at[idx,'date']} | close={close}")

            # ── EXIT ───────────────────────────────────────────────────
            elif in_trade:
                hist_turned_neg = (prev_hist is not None and prev_hist > 0 and hist < 0)
                if (
                    rsi   > self.rsi_overbought
                    or close < fast
                    or hist_turned_neg
                ):
                    reason_parts = []
                    if rsi > self.rsi_overbought:
                        reason_parts.append(f"RSI={rsi:.1f}>{self.rsi_overbought}")
                    if close < fast:
                        reason_parts.append(f"close<{self.fast_ma}")
                    if hist_turned_neg:
                        reason_parts.append("MACD_hist turned negative")

                    out.at[idx, "signal"]      = -1
                    out.at[idx, "exit_price"]  = close
                    out.at[idx, "trade_reason"] = "SELL | " + " & ".join(reason_parts)
                    in_trade = False
                    log.debug(f"[{self.name}] SELL @ {out.at[idx,'date']} | close={close}")

            prev_hist = hist

        out = self._forward_fill_position(out)

        buys  = (out["signal"] ==  1).sum()
        sells = (out["signal"] == -1).sum()
        log.info(f"[{self.name}] Signals generated → BUY: {buys} | SELL: {sells}")

        return out
