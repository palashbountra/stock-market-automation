"""
execution/paper_trader.py
--------------------------
Paper Trading Engine — the bridge between backtesting and live trading.

What it does:
  1. Streams price bars from MockPriceFeed (or Kite in Phase 6)
  2. Maintains a rolling window of recent bars per symbol
  3. Runs IndicatorPipeline on that window every bar
  4. Passes enriched data to the active strategy
  5. Executes BUY/SELL decisions via PaperState (no real money)
  6. Logs all activity and saves state after every bar

Key difference from backtest:
  Backtest  = process all history at once, in memory, no waiting
  PaperTrader = bar-by-bar, real-time pacing, persistent state

This architecture means switching to live trading (Phase 6) only requires
swapping MockPriceFeed → KitePriceFeed and PaperState → LiveBroker.
The strategy + indicator pipeline stay identical.

Parameters (all from config.yaml or overridable):
  strategy     : which strategy to run (default: config.strategy.default)
  warmup_bars  : bars needed before signals are valid (= max indicator lookback)
  capital      : starting paper capital
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List

import pandas as pd

from execution.paper_state import PaperState
from execution.price_feed import get_price_feed, MockPriceFeed
from indicators.pipeline import IndicatorPipeline
from strategies.registry import StrategyRegistry
from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

# Minimum bars needed for all indicators (SMA-200 is the longest)
_WARMUP_BARS = 205


class PaperTrader:

    def __init__(
        self,
        strategy_name: str | None = None,
        capital:       float | None = None,
        speed_seconds: float = 0.0,    # 0 = fast replay, >0 = real-time pacing
        max_bars:      int | None = None,  # cap replay length (None = all)
        resume:        bool = False,   # resume existing session
    ):
        self.strategy_name = strategy_name or cfg["strategy"]["default"]
        self.capital       = capital or cfg["backtest"]["initial_capital"]
        self.speed_seconds = speed_seconds
        self.max_bars      = max_bars
        self.symbols       = cfg["universe"]["symbols"]

        # Load or create state
        if resume:
            self.state = PaperState.load() or PaperState.new_session(self.capital)
        else:
            self.state = PaperState.new_session(self.capital)

        # Rolling bar buffer per symbol (deque auto-discards old bars)
        self._buffers: Dict[str, Deque[dict]] = {
            sym: deque(maxlen=_WARMUP_BARS + 50) for sym in self.symbols
        }

        self.pipeline = IndicatorPipeline()
        self.strategy = StrategyRegistry.get(self.strategy_name)

        log.info(
            f"[PaperTrader] Init | strategy={self.strategy_name} | "
            f"capital=₹{self.capital:,.0f} | symbols={self.symbols}"
        )

    def run(self) -> PaperState:
        """
        Main loop — streams bars, generates signals, executes paper trades.
        Returns final state when feed is exhausted or max_bars reached.
        """
        feed = get_price_feed(self.symbols, speed_seconds=self.speed_seconds)

        bars_processed = 0
        log.info("[PaperTrader] Starting paper trading session...")
        print(f"\n{'='*65}")
        print(f"  PAPER TRADING SESSION")
        print(f"  Strategy : {self.strategy_name}")
        print(f"  Capital  : ₹{self.capital:,.0f}")
        print(f"  Symbols  : {self.symbols}")
        print(f"{'='*65}\n")

        for tick in feed.stream():
            bars_processed += 1
            if self.max_bars and bars_processed > self.max_bars:
                log.info(f"[PaperTrader] max_bars={self.max_bars} reached. Stopping.")
                break

            # Update price buffers
            for sym, bar in tick.items():
                self._buffers[sym].append(bar)

            # Update mark-to-market
            current_prices = {sym: bar["close"] for sym, bar in tick.items()}
            self.state.update_prices(current_prices)

            # Get the bar date (use first symbol in tick)
            bar_date = list(tick.values())[0]["date"]
            ts       = str(bar_date)[:10]  # YYYY-MM-DD

            # Wait for warmup
            min_buf = min(len(self._buffers[s]) for s in self.symbols if s in tick)
            if min_buf < _WARMUP_BARS:
                if bars_processed % 20 == 0:
                    log.debug(f"[PaperTrader] Warming up: {min_buf}/{_WARMUP_BARS} bars")
                continue

            # Run signals for each symbol
            for sym in self.symbols:
                if sym not in tick:
                    continue

                buf = list(self._buffers[sym])
                df  = pd.DataFrame(buf)
                df["date"]   = pd.to_datetime(df["date"])
                df["symbol"] = sym

                try:
                    enriched  = self.pipeline.run(df)
                    signal_df = self.strategy.generate_signals(enriched)
                except Exception as e:
                    log.error(f"[PaperTrader] Signal error for {sym}: {e}")
                    continue

                # Look at the LAST row (current bar's signal)
                last = signal_df.iloc[-1]
                sig  = last["signal"]
                close = last["close"]

                # ── Execute paper trades ────────────────────────────────
                if sig == 1 and sym not in self.state.positions:
                    # Size: deploy % of capital config
                    deploy = self.state.cash * (cfg["strategy"]["risk_per_trade_pct"] / 100)
                    # Cap at 95% of cash to leave buffer
                    deploy = min(deploy, self.state.cash * 0.95)
                    qty    = int(deploy / close)

                    if qty > 0:
                        self.state.open_position(sym, qty, close, self.strategy_name, ts)
                        reason = last.get("trade_reason", "")[:50]
                        print(f"  [{ts}] 🟢 BUY  {sym:10} | {qty} shares @ ₹{close:.2f} | {reason}")

                elif sig == -1 and sym in self.state.positions:
                    pnl = self.state.close_position(sym, close, self.strategy_name, ts)
                    emoji = "✅" if (pnl or 0) >= 0 else "🔴"
                    reason = last.get("trade_reason", "")[:50]
                    print(f"  [{ts}] {emoji} SELL {sym:10} | ₹{close:.2f} | P&L: ₹{pnl:,.0f} | {reason}")

            # Print periodic status (every 30 bars)
            if bars_processed % 30 == 0:
                self._print_status(ts)

        # Final status
        self._print_status("FINAL")
        self._save_trade_log()
        log.info("[PaperTrader] Session complete.")
        return self.state

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_status(self, label: str) -> None:
        s = self.state
        print(f"\n  ── Status [{label}] ──────────────────────────────────")
        print(f"     Portfolio : ₹{s.portfolio_value:>12,.2f}")
        print(f"     Cash      : ₹{s.cash:>12,.2f}")
        print(f"     Realised  : ₹{s.realised_pnl:>12,.2f}")
        print(f"     Unrealised: ₹{s.unrealised_pnl:>12,.2f}")
        print(f"     Total P&L : ₹{s.total_pnl:>12,.2f} ({s.total_return_pct:+.2f}%)")

        if s.positions:
            print(f"     Open positions:")
            for sym, pos in s.positions.items():
                upnl = (pos["current_price"] - pos["avg_price"]) * pos["qty"]
                print(
                    f"       {sym:10} | {pos['qty']} shares | "
                    f"entry ₹{pos['avg_price']:.2f} | "
                    f"now ₹{pos['current_price']:.2f} | "
                    f"uPnL ₹{upnl:,.0f}"
                )
        else:
            print(f"     Open positions: None")
        print()

    def _save_trade_log(self) -> None:
        from pathlib import Path
        out_dir = Path(__file__).resolve().parent.parent / "data" / "paper_trading"
        out_dir.mkdir(parents=True, exist_ok=True)

        trades = [t for t in self.state.trade_history if t.get("side") == "SELL"]
        if trades:
            df = pd.DataFrame(trades)
            path = out_dir / "paper_trade_log.csv"
            df.to_csv(path, index=False)
            log.info(f"[PaperTrader] Trade log saved → {path}")
