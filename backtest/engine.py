"""
backtest/engine.py
------------------
Core backtesting engine — simulates trades from strategy signals on
historical OHLCV data.

Design principles:
  - Event-driven row-by-row simulation (no look-ahead bias)
  - Executes at next-bar OPEN after a signal (realistic — you can't trade
    at the close that triggered the signal)
  - Applies commission and slippage from config
  - Long-only for now (Phase 8 adds shorting + options P&L model)

Trade lifecycle:
  signal=1 on bar N  → buy at open of bar N+1
  signal=-1 on bar N → sell at open of bar N+1

Output:
  - trade_log   : list of completed trade dicts
  - equity_curve: pd.Series indexed by date
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List

from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Trade:
    """Represents a single completed round-trip trade."""
    symbol:       str
    strategy:     str
    entry_date:   pd.Timestamp
    exit_date:    pd.Timestamp
    entry_price:  float
    exit_price:   float
    shares:       int
    gross_pnl:    float = 0.0
    commission:   float = 0.0
    net_pnl:      float = 0.0
    return_pct:   float = 0.0
    outcome:      str   = ""   # WIN | LOSS | BREAKEVEN

    def __post_init__(self):
        self.gross_pnl  = (self.exit_price - self.entry_price) * self.shares
        self.commission = (self.entry_price + self.exit_price) * self.shares * (cfg["backtest"]["commission_pct"] / 100)
        self.net_pnl    = round(self.gross_pnl - self.commission, 2)
        self.return_pct = round((self.net_pnl / (self.entry_price * self.shares)) * 100, 4)
        self.outcome    = "WIN" if self.net_pnl > 0 else ("LOSS" if self.net_pnl < 0 else "BREAKEVEN")


class BacktestEngine:
    """
    Simulates trades from a signal DataFrame produced by any BaseStrategy.

    Parameters
    ----------
    initial_capital : starting cash (INR), default from config
    commission_pct  : per-leg commission as % of trade value
    slippage_pct    : slippage as % of execution price
    position_size_pct : % of current capital deployed per trade (default 100%)
    """

    def __init__(
        self,
        initial_capital:    float | None = None,
        commission_pct:     float | None = None,
        slippage_pct:       float | None = None,
        position_size_pct:  float = 100.0,
    ):
        bt_cfg = cfg["backtest"]
        self.initial_capital   = initial_capital   or bt_cfg["initial_capital"]
        self.commission_pct    = commission_pct    or bt_cfg["commission_pct"]
        self.slippage_pct      = slippage_pct      or bt_cfg["slippage_pct"]
        self.position_size_pct = position_size_pct

    def run(
        self,
        signal_df: pd.DataFrame,
        symbol:    str,
        strategy:  str,
    ) -> dict:
        """
        Run backtest simulation.

        Parameters
        ----------
        signal_df : output of strategy.generate_signals() — must have columns:
                    date, open, close, signal
        symbol    : e.g. "RELIANCE"
        strategy  : strategy name string

        Returns
        -------
        dict with keys: trades, equity_curve, summary
        """
        df = signal_df.reset_index(drop=True).copy()
        self._validate(df)

        capital      = self.initial_capital
        cash         = capital
        shares_held  = 0
        entry_price  = 0.0
        entry_date   = None
        in_trade     = False

        trade_log:   List[Trade]     = []
        equity_dates: List[pd.Timestamp] = []
        equity_vals:  List[float]    = []

        for i, row in df.iterrows():
            date  = row["date"]
            open_ = row["open"]
            close = row["close"]

            # Use next-bar open for execution (i > 0)
            # Signal from previous bar executes at this bar's open
            if i > 0:
                prev_signal = df.at[i - 1, "signal"]

                # ── BUY EXECUTION ──────────────────────────────────────
                if prev_signal == 1 and not in_trade:
                    exec_price = self._apply_slippage(open_, side="buy")
                    deploy     = cash * (self.position_size_pct / 100)
                    shares     = int(deploy / exec_price)

                    if shares > 0:
                        cost        = shares * exec_price
                        commission  = cost * (self.commission_pct / 100)
                        cash       -= (cost + commission)
                        shares_held = shares
                        entry_price = exec_price
                        entry_date  = date
                        in_trade    = True
                        log.debug(f"[BT] BUY  {symbol} @ {exec_price:.2f} x{shares} on {date.date()}")

                # ── SELL EXECUTION ─────────────────────────────────────
                elif prev_signal == -1 and in_trade:
                    exec_price = self._apply_slippage(open_, side="sell")
                    commission = shares_held * exec_price * (self.commission_pct / 100)
                    proceeds   = shares_held * exec_price - commission
                    cash      += proceeds

                    trade = Trade(
                        symbol      = symbol,
                        strategy    = strategy,
                        entry_date  = entry_date,
                        exit_date   = date,
                        entry_price = entry_price,
                        exit_price  = exec_price,
                        shares      = shares_held,
                    )
                    trade_log.append(trade)
                    log.debug(
                        f"[BT] SELL {symbol} @ {exec_price:.2f} x{shares_held} on {date.date()} | "
                        f"P&L: ₹{trade.net_pnl:.2f} ({trade.return_pct:.2f}%) [{trade.outcome}]"
                    )

                    shares_held = 0
                    entry_price = 0.0
                    entry_date  = None
                    in_trade    = False

            # ── Mark-to-market equity ──────────────────────────────────
            portfolio_value = cash + (shares_held * close)
            equity_dates.append(date)
            equity_vals.append(round(portfolio_value, 2))

        # If trade is still open at end — mark to last close (unrealised)
        if in_trade and shares_held > 0:
            last_close  = df.iloc[-1]["close"]
            last_date   = df.iloc[-1]["date"]
            open_trade_value = shares_held * last_close
            log.info(
                f"[BT] Open trade at end: {shares_held} shares @ entry {entry_price:.2f} | "
                f"current close {last_close:.2f} | unrealised P&L: "
                f"₹{(last_close - entry_price) * shares_held:.2f}"
            )

        equity_curve = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_dates), name="equity")

        summary = self._compute_summary(trade_log, equity_curve, symbol, strategy)

        return {
            "trades":       trade_log,
            "equity_curve": equity_curve,
            "summary":      summary,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_slippage(self, price: float, side: str) -> float:
        """Buy slightly higher, sell slightly lower."""
        slip = self.slippage_pct / 100
        return round(price * (1 + slip) if side == "buy" else price * (1 - slip), 4)

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        required = {"date", "open", "close", "signal"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"[BacktestEngine] signal_df missing columns: {missing}")

    def _compute_summary(
        self,
        trades: List[Trade],
        equity: pd.Series,
        symbol: str,
        strategy: str,
    ) -> dict:
        """Compute all performance metrics."""
        n = len(trades)

        if n == 0:
            return {
                "symbol": symbol, "strategy": strategy,
                "total_trades": 0, "message": "No completed trades.",
            }

        net_pnls    = [t.net_pnl for t in trades]
        returns_pct = [t.return_pct for t in trades]
        wins        = [t for t in trades if t.outcome == "WIN"]
        losses      = [t for t in trades if t.outcome == "LOSS"]

        win_rate    = round(len(wins) / n * 100, 2)
        avg_win     = round(np.mean([t.net_pnl for t in wins]),   2) if wins   else 0.0
        avg_loss    = round(np.mean([t.net_pnl for t in losses]), 2) if losses else 0.0
        profit_factor = round(
            sum(t.net_pnl for t in wins) / abs(sum(t.net_pnl for t in losses)), 4
        ) if losses and sum(t.net_pnl for t in losses) != 0 else float("inf")

        total_pnl   = round(sum(net_pnls), 2)
        total_return = round((equity.iloc[-1] - self.initial_capital) / self.initial_capital * 100, 2)

        # Max drawdown
        rolling_max  = equity.cummax()
        drawdown     = (equity - rolling_max) / rolling_max * 100
        max_drawdown = round(drawdown.min(), 2)

        # Sharpe ratio (annualised, risk-free = 6.5% for India)
        daily_returns = equity.pct_change().dropna()
        rf_daily      = 0.065 / 252
        excess        = daily_returns - rf_daily
        sharpe        = round((excess.mean() / excess.std()) * np.sqrt(252), 4) if excess.std() > 0 else 0.0

        # Longest winning / losing streak
        outcomes   = [1 if t.outcome == "WIN" else -1 for t in trades]
        max_win_streak  = self._max_streak(outcomes,  1)
        max_loss_streak = self._max_streak(outcomes, -1)

        return {
            "symbol":           symbol,
            "strategy":         strategy,
            "initial_capital":  self.initial_capital,
            "final_capital":    round(equity.iloc[-1], 2),
            "total_pnl":        total_pnl,
            "total_return_pct": total_return,
            "total_trades":     n,
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate_pct":     win_rate,
            "avg_win":          avg_win,
            "avg_loss":         avg_loss,
            "profit_factor":    profit_factor,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio":     sharpe,
            "max_win_streak":   max_win_streak,
            "max_loss_streak":  max_loss_streak,
            "best_trade_pnl":   round(max(net_pnls), 2),
            "worst_trade_pnl":  round(min(net_pnls), 2),
        }

    @staticmethod
    def _max_streak(outcomes: list[int], target: int) -> int:
        max_s = cur_s = 0
        for o in outcomes:
            if o == target:
                cur_s += 1
                max_s = max(max_s, cur_s)
            else:
                cur_s = 0
        return max_s
