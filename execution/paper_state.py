"""
execution/paper_state.py
-------------------------
Paper trading state manager.

Tracks everything about a paper trading session:
  - Cash balance
  - Open positions (symbol → qty, avg entry price)
  - Completed trades
  - Real-time P&L

State is persisted to a JSON file so sessions survive process restarts.
This same interface will be implemented by the live broker in Phase 6 —
paper trader and live trader are drop-in replacements.

File: data/paper_trading/state.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

_ROOT      = Path(__file__).resolve().parent.parent
_STATE_DIR = _ROOT / "data" / "paper_trading"
_STATE_DIR.mkdir(parents=True, exist_ok=True)
_STATE_FILE = _STATE_DIR / "state.json"


@dataclass
class Position:
    symbol:      str
    qty:         int
    avg_price:   float
    entry_date:  str
    strategy:    str
    current_price: float = 0.0

    @property
    def unrealised_pnl(self) -> float:
        return round((self.current_price - self.avg_price) * self.qty, 2)

    @property
    def unrealised_pct(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return round((self.current_price - self.avg_price) / self.avg_price * 100, 2)


@dataclass
class PaperTrade:
    symbol:      str
    strategy:    str
    side:        str   # BUY | SELL
    qty:         int
    price:       float
    timestamp:   str
    net_pnl:     float = 0.0
    outcome:     str   = ""


@dataclass
class PaperState:
    initial_capital: float
    cash:            float
    positions:       Dict[str, dict] = field(default_factory=dict)
    trade_history:   List[dict]      = field(default_factory=list)
    session_start:   str             = ""
    last_updated:    str             = ""

    # ── Serialisation ──────────────────────────────────────────────────

    def save(self) -> None:
        self.last_updated = datetime.now().isoformat()
        with open(_STATE_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)
        log.debug(f"[PaperState] Saved to {_STATE_FILE}")

    @classmethod
    def load(cls) -> Optional["PaperState"]:
        if _STATE_FILE.exists():
            with open(_STATE_FILE) as f:
                data = json.load(f)
            log.info(f"[PaperState] Loaded existing session from {_STATE_FILE}")
            return cls(**data)
        return None

    @classmethod
    def new_session(cls, initial_capital: float | None = None) -> "PaperState":
        capital = initial_capital or cfg["backtest"]["initial_capital"]
        state = cls(
            initial_capital = capital,
            cash            = capital,
            session_start   = datetime.now().isoformat(),
        )
        state.save()
        log.info(f"[PaperState] New session started | Capital: ₹{capital:,.0f}")
        return state

    # ── Position management ────────────────────────────────────────────

    def open_position(self, symbol: str, qty: int, price: float,
                      strategy: str, timestamp: str) -> bool:
        commission = qty * price * (cfg["backtest"]["commission_pct"] / 100)
        cost       = qty * price + commission

        if cost > self.cash:
            log.warning(f"[Paper] Insufficient cash for {symbol}: need ₹{cost:.0f}, have ₹{self.cash:.0f}")
            return False

        self.cash -= cost
        self.positions[symbol] = {
            "symbol":       symbol,
            "qty":          qty,
            "avg_price":    price,
            "entry_date":   timestamp,
            "strategy":     strategy,
            "current_price": price,
        }

        trade = PaperTrade(
            symbol=symbol, strategy=strategy, side="BUY",
            qty=qty, price=price, timestamp=timestamp
        )
        self.trade_history.append(asdict(trade))
        self.save()

        log.info(f"[Paper] BUY  {qty} x {symbol} @ ₹{price:.2f} | "
                 f"Cost: ₹{cost:.0f} | Cash remaining: ₹{self.cash:.0f}")
        return True

    def close_position(self, symbol: str, price: float,
                       strategy: str, timestamp: str) -> Optional[float]:
        if symbol not in self.positions:
            log.warning(f"[Paper] No open position for {symbol}")
            return None

        pos        = self.positions[symbol]
        qty        = pos["qty"]
        entry      = pos["avg_price"]
        commission = qty * price * (cfg["backtest"]["commission_pct"] / 100)
        proceeds   = qty * price - commission
        net_pnl    = round(proceeds - (qty * entry), 2)

        self.cash += proceeds
        del self.positions[symbol]

        trade = PaperTrade(
            symbol=symbol, strategy=strategy, side="SELL",
            qty=qty, price=price, timestamp=timestamp,
            net_pnl=net_pnl,
            outcome="WIN" if net_pnl > 0 else "LOSS"
        )
        self.trade_history.append(asdict(trade))
        self.save()

        log.info(f"[Paper] SELL {qty} x {symbol} @ ₹{price:.2f} | "
                 f"P&L: ₹{net_pnl:.0f} ({'WIN' if net_pnl>0 else 'LOSS'}) | "
                 f"Cash: ₹{self.cash:.0f}")
        return net_pnl

    def update_prices(self, prices: Dict[str, float]) -> None:
        for sym, price in prices.items():
            if sym in self.positions:
                self.positions[sym]["current_price"] = price

    # ── Portfolio metrics ──────────────────────────────────────────────

    @property
    def portfolio_value(self) -> float:
        position_value = sum(
            p["qty"] * p["current_price"]
            for p in self.positions.values()
        )
        return round(self.cash + position_value, 2)

    @property
    def total_pnl(self) -> float:
        return round(self.portfolio_value - self.initial_capital, 2)

    @property
    def total_return_pct(self) -> float:
        return round(self.total_pnl / self.initial_capital * 100, 2)

    @property
    def unrealised_pnl(self) -> float:
        return round(sum(
            (p["current_price"] - p["avg_price"]) * p["qty"]
            for p in self.positions.values()
        ), 2)

    @property
    def realised_pnl(self) -> float:
        return round(sum(
            t.get("net_pnl", 0)
            for t in self.trade_history
            if t.get("side") == "SELL"
        ), 2)
