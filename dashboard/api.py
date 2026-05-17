"""
dashboard/api.py
----------------
FastAPI backend — serves all data to the React dashboard.

Endpoints:
  GET /api/status          → system status, mode, connection
  GET /api/portfolio       → paper/live portfolio summary
  GET /api/positions       → open positions
  GET /api/trades          → recent trade history
  GET /api/signals         → latest signals per symbol per strategy
  GET /api/equity-curve    → equity curve data for charting
  GET /api/backtest        → backtest summary table
  GET /api/quotes          → live quotes for watchlist
  GET /api/symbols         → configured symbols list
  POST /api/run-backtest   → trigger a backtest run
  POST /api/run-paper      → trigger a paper trading session

Run with:
  python3 main.py server
  Then open http://localhost:8000 in your browser
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

_ROOT           = Path(__file__).resolve().parent.parent
_DASHBOARD_DIR  = Path(__file__).resolve().parent

app = FastAPI(
    title       = "IndiaTrader API",
    description = "Backend for the IndiaTrader dashboard",
    version     = cfg["system"]["version"],
)

# Allow all origins so the dashboard works both from file:// and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Serve dashboard HTML at root ───────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve the dashboard HTML — open http://localhost:8000 in browser."""
    return FileResponse(_DASHBOARD_DIR / "index.html")


# ── Helper ─────────────────────────────────────────────────────────────────

def _read_csv_safe(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.fillna("").to_dict(orient="records")


# ── System ─────────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    return {
        "system":    cfg["system"]["name"],
        "version":   cfg["system"]["version"],
        "mode":      cfg["system"]["mode"],
        "use_mock":  cfg["data"]["use_mock"],
        "symbols":   cfg["universe"]["symbols"],
        "timestamp": datetime.now().isoformat(),
        "broker":    cfg["broker"]["name"],
    }


@app.get("/api/symbols")
def get_symbols():
    return {"symbols": cfg["universe"]["symbols"]}


# ── Portfolio ───────────────────────────────────────────────────────────────

@app.get("/api/portfolio")
def get_portfolio():
    state_file = _ROOT / "data" / "paper_trading" / "state.json"
    if not state_file.exists():
        return {
            "initial_capital": cfg["backtest"]["initial_capital"],
            "portfolio_value": cfg["backtest"]["initial_capital"],
            "cash":            cfg["backtest"]["initial_capital"],
            "total_pnl":       0,
            "total_return_pct": 0,
            "realised_pnl":    0,
            "unrealised_pnl":  0,
            "positions":       {},
            "session_start":   None,
        }

    with open(state_file) as f:
        state = json.load(f)

    # Compute derived fields
    positions   = state.get("positions", {})
    cash        = state.get("cash", 0)
    pos_value   = sum(p["qty"] * p["current_price"] for p in positions.values())
    portfolio   = cash + pos_value
    initial     = state.get("initial_capital", cfg["backtest"]["initial_capital"])
    total_pnl   = round(portfolio - initial, 2)
    total_ret   = round(total_pnl / initial * 100, 2)

    trades      = state.get("trade_history", [])
    realised    = sum(t.get("net_pnl", 0) for t in trades if t.get("side") == "SELL")
    unrealised  = sum(
        (p["current_price"] - p["avg_price"]) * p["qty"]
        for p in positions.values()
    )

    return {
        "initial_capital":  initial,
        "portfolio_value":  round(portfolio, 2),
        "cash":             round(cash, 2),
        "total_pnl":        total_pnl,
        "total_return_pct": total_ret,
        "realised_pnl":     round(realised, 2),
        "unrealised_pnl":   round(unrealised, 2),
        "positions":        positions,
        "session_start":    state.get("session_start"),
        "last_updated":     state.get("last_updated"),
    }


@app.get("/api/positions")
def get_positions():
    portfolio = get_portfolio()
    positions = portfolio.get("positions", {})
    rows = []
    for sym, pos in positions.items():
        upnl = (pos["current_price"] - pos["avg_price"]) * pos["qty"]
        rows.append({
            "symbol":        sym,
            "qty":           pos["qty"],
            "avg_price":     pos["avg_price"],
            "current_price": pos["current_price"],
            "unrealised_pnl": round(upnl, 2),
            "unrealised_pct": round(upnl / (pos["avg_price"] * pos["qty"]) * 100, 2),
            "strategy":      pos.get("strategy", ""),
            "entry_date":    pos.get("entry_date", ""),
        })
    return {"positions": rows}


@app.get("/api/trades")
def get_trades(limit: int = 50):
    state_file = _ROOT / "data" / "paper_trading" / "state.json"
    if not state_file.exists():
        return {"trades": []}

    with open(state_file) as f:
        state = json.load(f)

    trades = [t for t in state.get("trade_history", []) if t.get("side") == "SELL"]
    trades = sorted(trades, key=lambda t: t.get("timestamp", ""), reverse=True)
    return {"trades": trades[:limit]}


# ── Signals ────────────────────────────────────────────────────────────────

@app.get("/api/signals")
def get_signals():
    """Returns latest signal per symbol per strategy from processed CSVs."""
    processed = _ROOT / "data" / "processed"
    result    = []
    symbols   = cfg["universe"]["symbols"]
    strategies = ["rsi_dma_crossover", "macd_crossover", "vwap_reversion"]

    for sym in symbols:
        for strat in strategies:
            path = processed / f"{sym}_{strat}_signals.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            # Get last non-zero signal
            active = df[df["signal"] != 0]
            if active.empty:
                last = df.iloc[-1].to_dict()
                last["signal"] = 0
            else:
                last = active.iloc[-1].to_dict()

            # trade_reason can be NaN in CSV for HOLD rows — clean it
            reason = last.get("trade_reason", "")
            if not isinstance(reason, str) or reason.lower() == "nan":
                reason = "No signal — monitoring"

            # For HOLD, show latest close price (last row), not last signal price
            display_close = df.iloc[-1]["close"] if last.get("signal", 0) == 0 else last.get("close", 0)
            display_date  = str(df.iloc[-1]["date"]) if last.get("signal", 0) == 0 else str(last.get("date", ""))

            result.append({
                "symbol":       sym,
                "strategy":     strat,
                "date":         display_date,
                "close":        display_close,
                "signal":       int(last.get("signal", 0)),
                "position":     int(last.get("position", 0)),
                "trade_reason": reason[:80],
            })

    return {"signals": result}


# ── Equity Curve ───────────────────────────────────────────────────────────

@app.get("/api/equity-curve")
def get_equity_curve():
    path = _ROOT / "data" / "processed" / "backtest" / "equity_curves.csv"
    if not path.exists():
        return {"curves": {}, "dates": []}

    df    = pd.read_csv(path, index_col=0)
    dates = df.index.tolist()
    curves = {col: df[col].fillna("").tolist() for col in df.columns}
    return {"dates": dates, "curves": curves}


# ── Backtest ───────────────────────────────────────────────────────────────

@app.get("/api/backtest")
def get_backtest_summary():
    path = _ROOT / "data" / "processed" / "backtest" / "backtest_summary.csv"
    return {"summary": _read_csv_safe(path)}


@app.get("/api/trades/backtest")
def get_backtest_trades():
    path = _ROOT / "data" / "processed" / "backtest" / "trade_log.csv"
    return {"trades": _read_csv_safe(path)}


# ── Live quotes ────────────────────────────────────────────────────────────

@app.get("/api/quotes")
def get_quotes():
    try:
        from broker.kite_broker import get_broker
        broker = get_broker()
        broker.connect()
        symbols = cfg["universe"]["symbols"]
        exchange = cfg["universe"]["exchange"]
        instruments = [f"{exchange}:{sym}" for sym in symbols]
        ltp = broker.get_ltp(instruments)
        return {"quotes": ltp}
    except Exception as e:
        log.error(f"[API] /api/quotes failed: {e}")
        return {"quotes": {}, "error": str(e)}


# ── Trigger actions ────────────────────────────────────────────────────────

@app.post("/api/run-backtest")
def trigger_backtest():
    """Run backtest in background and refresh results."""
    try:
        from data.loader import DataLoader
        from indicators.pipeline import IndicatorPipeline
        from strategies.registry import StrategyRegistry
        from backtest.engine import BacktestEngine
        from backtest.reporter import BacktestReporter

        loader   = DataLoader()
        pipeline = IndicatorPipeline()
        engine   = BacktestEngine()
        symbols  = cfg["universe"]["symbols"]

        all_results = []
        for sym in symbols:
            df       = loader.get(sym)
            enriched = pipeline.run(df)
            for sname in StrategyRegistry.list():
                strategy  = StrategyRegistry.get(sname)
                signal_df = strategy.generate_signals(enriched)
                result    = engine.run(signal_df, symbol=sym, strategy=sname)
                all_results.append(result)

        reporter = BacktestReporter(all_results)
        reporter.export_all()
        return {"status": "success", "message": f"Backtest complete. {len(all_results)} runs."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run-signals")
def trigger_signals():
    """Re-generate signals for all symbols."""
    try:
        from data.loader import DataLoader
        from indicators.pipeline import IndicatorPipeline
        from strategies.registry import StrategyRegistry
        from pathlib import Path

        loader   = DataLoader()
        pipeline = IndicatorPipeline()
        out_dir  = _ROOT / cfg["data"]["processed_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        for sym in cfg["universe"]["symbols"]:
            df       = loader.get(sym)
            enriched = pipeline.run(df)
            for sname in StrategyRegistry.list():
                strategy  = StrategyRegistry.get(sname)
                result    = strategy.generate_signals(enriched)
                signal_cols = ["date", "close", "signal", "position",
                               "entry_price", "exit_price", "trade_reason"]
                result[signal_cols].to_csv(out_dir / f"{sym}_{sname}_signals.csv", index=False)

        return {"status": "success", "message": "Signals regenerated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
