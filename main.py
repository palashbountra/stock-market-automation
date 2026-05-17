"""
main.py
-------
Entry point for the trading system.
Run from the project root:  python3 main.py [command]

Commands
--------
  fetch        – Fetch and store OHLCV data for all configured symbols
  summary      – Show what's in the database
  indicators   – Run indicator pipeline on all symbols and save enriched CSVs
  signals      – Run all strategies on all symbols and print signal tables
  backtest     – Run backtest engine across all symbols x strategies
  paper        – Start a paper trading session (simulated live trading)
  paper-status – Show current paper trading portfolio status
  login        – Zerodha Kite Connect daily login (generates access token)
  server       – Start FastAPI backend server for the dashboard
"""

import sys
from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger("main")


def cmd_fetch():
    """Phase 1: fetch and store market data."""
    from data.loader import DataLoader
    loader = DataLoader()
    symbols = cfg["universe"]["symbols"]
    mode = "MOCK" if cfg["data"]["use_mock"] else "KITE API"
    log.info(f"=== FETCH START | mode={mode} | symbols={symbols} ===")
    loader.refresh(symbols)
    log.info("=== FETCH COMPLETE ===")
    loader.summary()


def cmd_summary():
    """Show DB summary without fetching."""
    from data.loader import DataLoader
    DataLoader().summary()


def cmd_indicators():
    """Phase 2: compute all indicators on stored data and save enriched CSVs."""
    from data.loader import DataLoader
    from indicators.pipeline import IndicatorPipeline
    from pathlib import Path

    loader   = DataLoader()
    pipeline = IndicatorPipeline()
    symbols  = cfg["universe"]["symbols"]
    out_dir  = Path(__file__).parent / cfg["data"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== INDICATORS START | symbols={symbols} ===")

    for sym in symbols:
        df       = loader.get(sym)
        enriched = pipeline.run(df)
        csv_path = out_dir / f"{sym}_indicators.csv"
        enriched.to_csv(csv_path, index=False)
        log.info(f"[INDICATORS] {sym} saved → {csv_path}")

        cols = ["date", "close", "sma_20", "sma_50", "rsi_14",
                "macd_line", "macd_hist", "vwap"]
        available = [c for c in cols if c in enriched.columns]
        print(f"\n── {sym} (last 3 rows) ──")
        print(enriched[available].dropna().tail(3).to_string(index=False))

    log.info("=== INDICATORS COMPLETE ===")


def cmd_signals():
    """Phase 3: run all strategies and print signal summary."""
    from data.loader import DataLoader
    from indicators.pipeline import IndicatorPipeline
    from strategies.registry import StrategyRegistry
    from pathlib import Path

    loader   = DataLoader()
    pipeline = IndicatorPipeline()
    symbols  = cfg["universe"]["symbols"]
    out_dir  = Path(__file__).parent / cfg["data"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy_names = StrategyRegistry.list()
    log.info(f"=== SIGNALS START | symbols={symbols} | strategies={strategy_names} ===")

    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"  {sym}")
        print(f"{'='*60}")

        df       = loader.get(sym)
        enriched = pipeline.run(df)

        for sname in strategy_names:
            strategy = StrategyRegistry.get(sname)
            try:
                result = strategy.generate_signals(enriched)
            except ValueError as e:
                log.warning(f"Skipping {sname} for {sym}: {e}")
                continue

            # Save signals CSV
            csv_path = out_dir / f"{sym}_{sname}_signals.csv"
            signal_cols = ["date", "close", "signal", "position",
                           "entry_price", "exit_price", "trade_reason"]
            result[signal_cols].to_csv(csv_path, index=False)

            # Print all non-zero signals
            trades = result[result["signal"] != 0][signal_cols].copy()
            buys   = (result["signal"] ==  1).sum()
            sells  = (result["signal"] == -1).sum()

            print(f"\n  ── Strategy: {sname} ──")
            print(f"     BUY signals: {buys} | SELL signals: {sells}")

            if not trades.empty:
                print(f"     Last 5 signals:")
                # Shorten trade_reason for display
                display = trades.tail(5).copy()
                display["trade_reason"] = display["trade_reason"].str[:60]
                print(display[["date", "close", "signal", "trade_reason"]]
                      .to_string(index=False))
            else:
                print("     No signals generated on this dataset.")

    log.info("=== SIGNALS COMPLETE ===")
    print(f"\nSignal CSVs saved to: {out_dir}")


def cmd_backtest():
    """Phase 4: run backtest engine on all symbols x all strategies."""
    from data.loader import DataLoader
    from indicators.pipeline import IndicatorPipeline
    from strategies.registry import StrategyRegistry
    from backtest.engine import BacktestEngine
    from backtest.reporter import BacktestReporter

    loader   = DataLoader()
    pipeline = IndicatorPipeline()
    engine   = BacktestEngine()
    symbols  = cfg["universe"]["symbols"]
    strategy_names = StrategyRegistry.list()

    log.info(f"=== BACKTEST START | {symbols} x {strategy_names} ===")

    all_results = []

    for sym in symbols:
        df       = loader.get(sym)
        enriched = pipeline.run(df)

        for sname in strategy_names:
            strategy = StrategyRegistry.get(sname)
            try:
                signal_df = strategy.generate_signals(enriched)
                result    = engine.run(signal_df, symbol=sym, strategy=sname)
                all_results.append(result)
                log.info(
                    f"[BT] {sym}/{sname} | "
                    f"trades={result['summary'].get('total_trades',0)} | "
                    f"P&L=₹{result['summary'].get('total_pnl',0):,.0f} | "
                    f"return={result['summary'].get('total_return_pct',0):.2f}%"
                )
            except Exception as e:
                log.error(f"[BT] {sym}/{sname} failed: {e}")

    reporter = BacktestReporter(all_results)
    reporter.print_summary()
    reporter.print_trades()
    reporter.export_all()

    log.info("=== BACKTEST COMPLETE ===")


def cmd_paper():
    """Phase 5: run a paper trading session (full historical replay)."""
    from execution.paper_trader import PaperTrader

    # Use macd_crossover — it generates more trades for demo (rsi_dma needs real data)
    strategy = "macd_crossover"
    capital  = cfg["backtest"]["initial_capital"]

    log.info(f"=== PAPER TRADING START | strategy={strategy} | capital=₹{capital:,.0f} ===")

    trader = PaperTrader(
        strategy_name  = strategy,
        capital        = capital,
        speed_seconds  = 0.0,
        max_bars       = None,
        resume         = False,
    )
    final_state = trader.run()

    log.info(
        f"=== PAPER TRADING COMPLETE | "
        f"Final portfolio: ₹{final_state.portfolio_value:,.2f} | "
        f"P&L: ₹{final_state.total_pnl:,.2f} ({final_state.total_return_pct:+.2f}%) ==="
    )


def cmd_paper_status():
    """Show current paper trading portfolio status without running a new session."""
    from execution.paper_state import PaperState

    state = PaperState.load()
    if state is None:
        print("\n  No paper trading session found. Run: python3 main.py paper")
        return

    print(f"\n{'='*55}")
    print(f"  PAPER TRADING PORTFOLIO STATUS")
    print(f"{'='*55}")
    print(f"  Session started : {state.session_start[:19]}")
    print(f"  Last updated    : {state.last_updated[:19]}")
    print(f"  Initial capital : ₹{state.initial_capital:>12,.2f}")
    print(f"  Current cash    : ₹{state.cash:>12,.2f}")
    print(f"  Portfolio value : ₹{state.portfolio_value:>12,.2f}")
    print(f"  Realised P&L    : ₹{state.realised_pnl:>12,.2f}")
    print(f"  Unrealised P&L  : ₹{state.unrealised_pnl:>12,.2f}")
    print(f"  Total P&L       : ₹{state.total_pnl:>12,.2f} ({state.total_return_pct:+.2f}%)")
    print(f"{'─'*55}")

    if state.positions:
        print(f"  Open Positions:")
        for sym, pos in state.positions.items():
            upnl = (pos["current_price"] - pos["avg_price"]) * pos["qty"]
            upct = (pos["current_price"] - pos["avg_price"]) / pos["avg_price"] * 100
            print(
                f"    {sym:10} | {pos['qty']:4} shares | "
                f"entry ₹{pos['avg_price']:.2f} | "
                f"now ₹{pos['current_price']:.2f} | "
                f"uPnL ₹{upnl:,.0f} ({upct:+.2f}%)"
            )
    else:
        print(f"  Open Positions  : None")

    sell_trades = [t for t in state.trade_history if t.get("side") == "SELL"]
    wins  = sum(1 for t in sell_trades if (t.get("net_pnl", 0) or 0) > 0)
    total = len(sell_trades)
    print(f"{'─'*55}")
    print(f"  Completed trades: {total}")
    if total > 0:
        print(f"  Win rate        : {wins}/{total} ({wins/total*100:.1f}%)")
    print(f"{'='*55}\n")


def cmd_login():
    """Phase 6: Zerodha Kite Connect daily login."""
    from broker.auth import login_flow
    login_flow()


def cmd_server():
    """Phase 7: Start FastAPI backend + open dashboard in browser."""
    import webbrowser
    import threading
    import uvicorn

    def open_browser():
        import time
        time.sleep(1.5)
        # Open through the server — NOT file:// (file:// blocks API calls)
        webbrowser.open("http://localhost:8000")
        print(f"\n  📊 Dashboard: http://localhost:8000")
        print(f"  🔌 API docs:  http://localhost:8000/docs")
        print(f"\n  Press Ctrl+C to stop the server\n")

    threading.Thread(target=open_browser, daemon=True).start()

    log.info("=== SERVER START | http://localhost:8000 ===")
    uvicorn.run(
        "dashboard.api:app",
        host   = cfg["dashboard"]["host"],
        port   = cfg["dashboard"]["port"],
        reload = False,
        log_level = "warning",
    )


COMMANDS = {
    "fetch":         cmd_fetch,
    "summary":       cmd_summary,
    "indicators":    cmd_indicators,
    "signals":       cmd_signals,
    "backtest":      cmd_backtest,
    "paper":         cmd_paper,
    "paper-status":  cmd_paper_status,
    "login":         cmd_login,
    "server":        cmd_server,
}


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "fetch"

    if command not in COMMANDS:
        print(f"Unknown command: '{command}'")
        print(f"Available: {list(COMMANDS.keys())}")
        sys.exit(1)

    log.info(f"IndiaTrader v{cfg.get('system', {}).get('version', 'unknown')} | cmd={command}")
    COMMANDS[command]()


if __name__ == "__main__":
    main()
