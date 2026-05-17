"""
broker/kite_broker.py
----------------------
Zerodha Kite Connect broker implementation.

Wraps the Kite API for:
  - Fetching live quotes
  - Placing / modifying / cancelling orders
  - Fetching positions and holdings
  - Fetching order history

This is the LIVE version — only used when config.system.mode = "live"
In mock/paper mode, none of this is called.

All order types supported by Kite are available but the system
currently uses MARKET orders for simplicity. Limit orders will
be added in Phase 8 (Risk Management).

NSE F&O (options) order support is pre-wired here for Phase 10.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from utils.config_loader import cfg
from utils.logger import get_logger

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
log = get_logger(__name__)


class KiteBroker:
    """
    Live broker interface for Zerodha Kite Connect.

    Usage:
        broker = KiteBroker()
        broker.connect()
        quote = broker.get_quote("NSE:RELIANCE")
        order_id = broker.place_market_order("RELIANCE", "BUY", 10)
    """

    def __init__(self):
        self._kite = None
        self._connected = False

    def connect(self) -> bool:
        """Authenticate and establish Kite session."""
        try:
            from broker.auth import get_kite_client
            self._kite = get_kite_client()
            profile = self._kite.profile()
            self._connected = True
            log.info(
                f"[Kite] Connected | User: {profile['user_name']} "
                f"({profile['user_id']}) | Email: {profile['email']}"
            )
            return True
        except Exception as e:
            log.error(f"[Kite] Connection failed: {e}")
            self._connected = False
            return False

    def is_connected(self) -> bool:
        return self._connected

    # ── Market Data ────────────────────────────────────────────────────

    def get_quote(self, instruments: list[str]) -> dict:
        """
        Fetch live quotes for a list of instruments.

        Parameters
        ----------
        instruments : e.g. ["NSE:RELIANCE", "NSE:TCS", "NFO:NIFTY24APR22500CE"]

        Returns
        -------
        dict keyed by instrument string with OHLC, volume, LTP etc.
        """
        self._check_connected()
        quotes = self._kite.quote(instruments)
        log.debug(f"[Kite] Quotes fetched for {instruments}")
        return quotes

    def get_ltp(self, instruments: list[str]) -> dict[str, float]:
        """Returns {instrument: last_traded_price} — lightweight quote."""
        self._check_connected()
        raw = self._kite.ltp(instruments)
        return {k: v["last_price"] for k, v in raw.items()}

    def get_ohlc(self, instruments: list[str]) -> dict:
        """Returns today's OHLC for instruments."""
        self._check_connected()
        return self._kite.ohlc(instruments)

    # ── Orders ────────────────────────────────────────────────────────

    def place_market_order(
        self,
        symbol:   str,
        side:     str,        # "BUY" or "SELL"
        qty:      int,
        exchange: str = "NSE",
        product:  str = "CNC",  # CNC=delivery, MIS=intraday, NRML=F&O
    ) -> Optional[str]:
        """
        Place a market order. Returns order_id on success.

        product codes:
          CNC  — Cash and Carry (delivery equity, hold overnight)
          MIS  — Margin Intraday Square-off (auto-squared at 3:20 PM)
          NRML — Normal (F&O positions, can hold overnight)
        """
        self._check_connected()

        try:
            from kiteconnect import KiteConnect
            order_id = self._kite.place_order(
                tradingsymbol  = symbol,
                exchange       = exchange,
                transaction_type = self._kite.TRANSACTION_TYPE_BUY if side == "BUY"
                                   else self._kite.TRANSACTION_TYPE_SELL,
                quantity       = qty,
                order_type     = self._kite.ORDER_TYPE_MARKET,
                product        = product,
                variety        = self._kite.VARIETY_REGULAR,
            )
            log.info(
                f"[Kite] ORDER PLACED | {side} {qty} x {exchange}:{symbol} "
                f"| MARKET | product={product} | order_id={order_id}"
            )
            return order_id
        except Exception as e:
            log.error(f"[Kite] Order failed: {e}")
            return None

    def place_limit_order(
        self,
        symbol:   str,
        side:     str,
        qty:      int,
        price:    float,
        exchange: str = "NSE",
        product:  str = "CNC",
    ) -> Optional[str]:
        """Place a limit order at a specific price."""
        self._check_connected()
        try:
            order_id = self._kite.place_order(
                tradingsymbol    = symbol,
                exchange         = exchange,
                transaction_type = self._kite.TRANSACTION_TYPE_BUY if side == "BUY"
                                   else self._kite.TRANSACTION_TYPE_SELL,
                quantity         = qty,
                price            = price,
                order_type       = self._kite.ORDER_TYPE_LIMIT,
                product          = product,
                variety          = self._kite.VARIETY_REGULAR,
            )
            log.info(
                f"[Kite] LIMIT ORDER | {side} {qty} x {symbol} @ ₹{price} "
                f"| order_id={order_id}"
            )
            return order_id
        except Exception as e:
            log.error(f"[Kite] Limit order failed: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        self._check_connected()
        try:
            self._kite.cancel_order(
                variety  = self._kite.VARIETY_REGULAR,
                order_id = order_id,
            )
            log.info(f"[Kite] Order cancelled: {order_id}")
            return True
        except Exception as e:
            log.error(f"[Kite] Cancel failed: {e}")
            return False

    # ── Portfolio ─────────────────────────────────────────────────────

    def get_positions(self) -> dict:
        """Returns current day + net positions."""
        self._check_connected()
        positions = self._kite.positions()
        log.debug(f"[Kite] Positions fetched | "
                  f"day={len(positions['day'])} net={len(positions['net'])}")
        return positions

    def get_holdings(self) -> list:
        """Returns long-term equity holdings (demat)."""
        self._check_connected()
        return self._kite.holdings()

    def get_orders(self) -> list:
        """Returns today's order book."""
        self._check_connected()
        return self._kite.orders()

    def get_order_history(self, order_id: str) -> list:
        """Returns full status history of a specific order."""
        self._check_connected()
        return self._kite.order_history(order_id)

    def get_margins(self) -> dict:
        """Returns available margin / funds."""
        self._check_connected()
        return self._kite.margins()

    # ── Options specific (Phase 10) ───────────────────────────────────

    def get_instruments(self, exchange: str = "NFO") -> list:
        """
        Fetch full instrument list for an exchange.
        For NFO (F&O): returns all option/future contracts with
        expiry, strike, instrument_token etc.
        Used in Phase 10 to build the options chain.
        """
        self._check_connected()
        instruments = self._kite.instruments(exchange)
        log.info(f"[Kite] Fetched {len(instruments)} instruments from {exchange}")
        return instruments

    def get_option_chain(self, underlying: str, expiry: str) -> list:
        """
        Returns all CE + PE contracts for an underlying + expiry.
        underlying : e.g. "NIFTY", "BANKNIFTY", "RELIANCE"
        expiry     : e.g. "2024-04-25"
        Phase 10 will build a full chain analyser on top of this.
        """
        all_instruments = self.get_instruments("NFO")
        chain = [
            i for i in all_instruments
            if i["name"] == underlying
            and str(i["expiry"])[:10] == expiry
            and i["instrument_type"] in ("CE", "PE")
        ]
        log.info(f"[Kite] Option chain: {underlying} {expiry} | {len(chain)} strikes")
        return chain

    # ── Internal ──────────────────────────────────────────────────────

    def _check_connected(self):
        if not self._connected or self._kite is None:
            raise RuntimeError(
                "Not connected to Kite. Call broker.connect() first, "
                "or run: python3 main.py login"
            )


class MockBroker:
    """
    Drop-in replacement for KiteBroker when use_mock=True.
    Returns realistic dummy data so the rest of the system
    works identically without an active Kite session.
    """

    def connect(self) -> bool:
        log.info("[MockBroker] Connected (mock mode)")
        return True

    def is_connected(self) -> bool:
        return True

    def get_ltp(self, instruments: list[str]) -> dict[str, float]:
        prices = {"NSE:RELIANCE": 4650.0, "NSE:TCS": 2540.0, "NSE:INFY": 1550.0}
        return {i: prices.get(i, 1000.0) for i in instruments}

    def get_positions(self) -> dict:
        return {"day": [], "net": []}

    def get_holdings(self) -> list:
        return []

    def get_orders(self) -> list:
        return []

    def get_margins(self) -> dict:
        return {"equity": {"available": {"live_balance": 100000.0}}}

    def place_market_order(self, symbol, side, qty, **kwargs) -> str:
        order_id = f"MOCK_{symbol}_{side}_{qty}_{datetime.now().strftime('%H%M%S')}"
        log.info(f"[MockBroker] Order simulated: {side} {qty} x {symbol} → {order_id}")
        return order_id


def get_broker():
    """Factory — returns KiteBroker or MockBroker based on config."""
    if cfg["data"]["use_mock"] or cfg["system"]["mode"] != "live":
        return MockBroker()
    return KiteBroker()
