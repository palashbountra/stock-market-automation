"""
backtest/reporter.py
--------------------
Formats and exports backtest results in multiple forms:

1. Console report   — clean tabular summary for quick reading
2. Trade log CSV    — every trade with entry/exit/P&L detail
3. Equity curve CSV — day-by-day portfolio value
4. Summary CSV      — one row per (symbol, strategy) comparison table

All files saved to data/processed/backtest/
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import List

from backtest.engine import Trade
from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

_ROOT    = Path(__file__).resolve().parent.parent
_OUT_DIR = _ROOT / cfg["data"]["processed_dir"] / "backtest"
_OUT_DIR.mkdir(parents=True, exist_ok=True)


class BacktestReporter:

    def __init__(self, results: list[dict]):
        """
        Parameters
        ----------
        results : list of dicts returned by BacktestEngine.run()
                  Each dict has: trades, equity_curve, summary
        """
        self.results = results

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a clean comparison table of all backtest runs."""
        summaries = [r["summary"] for r in self.results]

        # Filter runs that actually had trades
        active = [s for s in summaries if s.get("total_trades", 0) > 0]
        empty  = [s for s in summaries if s.get("total_trades", 0) == 0]

        if empty:
            for s in empty:
                sym_name   = s['symbol']
                strat_name = s['strategy']
                print(f"\n  !  {sym_name} / {strat_name}: No completed trades.")

        if not active:
            print("\n  No trades completed across any strategy/symbol combination.")
            return

        print("\n" + "═" * 100)
        print(f"  BACKTEST RESULTS SUMMARY")
        print("═" * 100)

        header = (
            f"{'Symbol':<10} {'Strategy':<22} {'Trades':>6} {'Win%':>6} "
            f"{'Net P&L':>12} {'Return%':>8} {'MaxDD%':>8} "
            f"{'Sharpe':>7} {'PF':>6} {'AvgWin':>10} {'AvgLoss':>10}"
        )
        print(header)
        print("─" * 100)

        for s in active:
            pnl_str     = "₹" + f"{s['total_pnl']:,.0f}"
            avg_win_str = "₹" + f"{s['avg_win']:,.0f}"
            avg_los_str = "₹" + f"{s['avg_loss']:,.0f}"
            print(
                f"{s['symbol']:<10} {s['strategy']:<22} {s['total_trades']:>6} "
                f"{s['win_rate_pct']:>5.1f}% "
                f"{pnl_str:>12} "
                f"{s['total_return_pct']:>7.2f}% "
                f"{s['max_drawdown_pct']:>7.2f}% "
                f"{s['sharpe_ratio']:>7.3f} "
                f"{s['profit_factor']:>6.2f} "
                f"{avg_win_str:>10} "
                f"{avg_los_str:>10}"
            )

        print("═" * 100)

        # Best strategy by Sharpe
        best = max(active, key=lambda s: s["sharpe_ratio"])
        print(f"\n  🏆 Best Sharpe: {best['strategy']} on {best['symbol']} "
              f"(Sharpe={best['sharpe_ratio']:.3f}, Return={best['total_return_pct']:.2f}%)")
        print()

    def print_trades(self, symbol: str | None = None, strategy: str | None = None) -> None:
        """Print individual trade details, optionally filtered."""
        print(f"\n{'─'*90}")
        print(f"  TRADE LOG{f' | {symbol}' if symbol else ''}{f' | {strategy}' if strategy else ''}")
        print(f"{'─'*90}")
        hdr = f"{'#':>4}  {'Symbol':<10} {'Strategy':<22} {'Entry Date':<13} {'Exit Date':<13} {'Entry₹':>8} {'Exit₹':>8} {'Qty':>5} {'Net P&L':>10} {'Ret%':>7} {'Result':<10}"
        print(hdr)
        print("─" * 90)

        n = 0
        for r in self.results:
            s = r["summary"]
            if symbol   and s.get("symbol")   != symbol:   continue
            if strategy and s.get("strategy") != strategy: continue

            for trade in r["trades"]:
                n += 1
                print(
                    f"{n:>4}  {trade.symbol:<10} {trade.strategy:<22} "
                    f"{str(trade.entry_date.date()):<13} {str(trade.exit_date.date()):<13} "
                    f"{trade.entry_price:>8.2f} {trade.exit_price:>8.2f} "
                    f"{trade.shares:>5} "
                    f"{'₹' + str(f'{trade.net_pnl:,.0f}'):>10} "
                    f"{trade.return_pct:>6.2f}% "
                    f"{trade.outcome:<10}"
                )

        if n == 0:
            print("  No trades found.")
        print()

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_all(self) -> None:
        """Export trade logs, equity curves, and summary CSV."""
        self._export_trade_log()
        self._export_equity_curves()
        self._export_summary()
        log.info(f"[Reporter] All backtest reports saved to: {_OUT_DIR}")
        print(f"\n  📁 Reports saved to: {_OUT_DIR}")

    def _export_trade_log(self) -> None:
        rows = []
        for r in self.results:
            for t in r["trades"]:
                rows.append({
                    "symbol":       t.symbol,
                    "strategy":     t.strategy,
                    "entry_date":   t.entry_date.date(),
                    "exit_date":    t.exit_date.date(),
                    "entry_price":  t.entry_price,
                    "exit_price":   t.exit_price,
                    "shares":       t.shares,
                    "gross_pnl":    t.gross_pnl,
                    "commission":   round(t.commission, 2),
                    "net_pnl":      t.net_pnl,
                    "return_pct":   t.return_pct,
                    "outcome":      t.outcome,
                })

        if rows:
            df = pd.DataFrame(rows)
            path = _OUT_DIR / "trade_log.csv"
            df.to_csv(path, index=False)
            log.info(f"[Reporter] Trade log: {len(rows)} trades → {path}")

    def _export_equity_curves(self) -> None:
        frames = {}
        for r in self.results:
            s   = r["summary"]
            key = f"{s.get('symbol','?')}_{s.get('strategy','?')}"
            frames[key] = r["equity_curve"].rename(key)

        if frames:
            df   = pd.DataFrame(frames)
            path = _OUT_DIR / "equity_curves.csv"
            df.to_csv(path)
            log.info(f"[Reporter] Equity curves → {path}")

    def _export_summary(self) -> None:
        summaries = [r["summary"] for r in self.results if r["summary"].get("total_trades", 0) > 0]
        if summaries:
            df   = pd.DataFrame(summaries)
            path = _OUT_DIR / "backtest_summary.csv"
            df.to_csv(path, index=False)
            log.info(f"[Reporter] Summary → {path}")
