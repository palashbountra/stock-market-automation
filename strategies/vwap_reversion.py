"""
strategies/vwap_reversion.py
-----------------------------
Strategy: VWAP Mean Reversion
-------------------------------
Mean reversion strategy — trades the "snap back" to VWAP after an overextension.
Particularly relevant for options later (iron condors, short straddles) since
it identifies when a stock is stretched and likely to revert.

ENTRY (BUY — long reversion):
  1. close < vwap_lower_1    — price has dipped 1σ below VWAP (oversold vs VWAP)
  2. rsi_14 < 45             — momentum not strongly bearish (avoid falling knives)
  3. close > sma_50          — macro trend is still up (don't fade a real downtrend)

EXIT (SELL):
  1. close >= vwap            — price reverted back to VWAP (target hit)
  OR
  2. close < vwap_lower_2    — price breaks 2σ band (stop loss — trend has changed)
  OR
  3. rsi_14 > 60             — overbought after reversion (take profit)

Why this matters for options (Phase 10+):
  - When price is between vwap_lower_1 and vwap_upper_1, IV is typically lower
    → ideal for selling premium (straddles, iron condors)
  - When price pierces vwap_lower_2 or vwap_upper_2, volatility is elevated
    → ideal for buying options (straddles, directional plays)
  - This strategy builds the intuition/data for those decisions later.

This is a long-only reversion strategy.
"""

import pandas as pd
from strategies.base import BaseStrategy
from utils.logger import get_logger

log = get_logger(__name__)


class VWAP_Reversion_Strategy(BaseStrategy):

    def __init__(
        self,
        rsi_col: str    = "rsi_14",
        trend_ma: str   = "sma_50",
        rsi_entry: float = 45.0,
        rsi_exit: float  = 60.0,
    ):
        self.rsi_col   = rsi_col
        self.trend_ma  = trend_ma
        self.rsi_entry = rsi_entry
        self.rsi_exit  = rsi_exit

    @property
    def name(self) -> str:
        return "vwap_reversion"

    @property
    def required_indicators(self) -> list[str]:
        return [
            "vwap", "vwap_upper_1", "vwap_lower_1",
            "vwap_upper_2", "vwap_lower_2",
            self.rsi_col, self.trend_ma,
        ]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate(df)
        out = self._init_signal_columns(df)

        req_cols = self.required_indicators
        valid_mask = out[req_cols].notna().all(axis=1)

        in_trade = False

        for idx in out.index:
            if not valid_mask[idx]:
                continue

            close      = out.at[idx, "close"]
            vwap       = out.at[idx, "vwap"]
            lower_1    = out.at[idx, "vwap_lower_1"]
            lower_2    = out.at[idx, "vwap_lower_2"]
            rsi        = out.at[idx, self.rsi_col]
            trend      = out.at[idx, self.trend_ma]

            # ── ENTRY ──────────────────────────────────────────────────
            if (
                not in_trade
                and close  < lower_1          # price below 1σ VWAP band
                and rsi    < self.rsi_entry    # not in freefall
                and close  > trend             # macro uptrend intact
            ):
                out.at[idx, "signal"]       = 1
                out.at[idx, "entry_price"]  = close
                out.at[idx, "trade_reason"] = (
                    f"BUY | close={close:.2f} < vwap_lower_1={lower_1:.2f} "
                    f"& RSI={rsi:.1f}<{self.rsi_entry} "
                    f"& close>{self.trend_ma}"
                )
                in_trade = True
                log.debug(f"[{self.name}] BUY  @ {out.at[idx,'date']} | close={close}")

            # ── EXIT ───────────────────────────────────────────────────
            elif in_trade:
                target_hit  = close >= vwap
                stop_hit    = close < lower_2
                overbought  = rsi > self.rsi_exit

                if target_hit or stop_hit or overbought:
                    if target_hit:
                        reason = f"VWAP target hit (close={close:.2f} >= vwap={vwap:.2f})"
                    elif stop_hit:
                        reason = f"STOP: close={close:.2f} < vwap_lower_2={lower_2:.2f}"
                    else:
                        reason = f"RSI overbought: {rsi:.1f}>{self.rsi_exit}"

                    out.at[idx, "signal"]       = -1
                    out.at[idx, "exit_price"]   = close
                    out.at[idx, "trade_reason"] = f"SELL | {reason}"
                    in_trade = False
                    log.debug(f"[{self.name}] SELL @ {out.at[idx,'date']} | close={close} | {reason}")

        out = self._forward_fill_position(out)

        buys  = (out["signal"] ==  1).sum()
        sells = (out["signal"] == -1).sum()
        log.info(f"[{self.name}] Signals generated → BUY: {buys} | SELL: {sells}")

        return out
