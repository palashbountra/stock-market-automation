"""
data/loader.py
--------------
The single public interface for the data layer.
All other modules (indicators, strategy, backtest) import from here.

Usage
-----
from data.loader import DataLoader

loader = DataLoader()
df = loader.get("RELIANCE")            # loads from DB / CSV
loader.refresh(["RELIANCE", "TCS"])    # fetch + store fresh data
"""

import sqlite3
from pathlib import Path

import pandas as pd

from data.storage import save_ohlcv, _get_conn, _ensure_table
from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parent.parent


class DataLoader:
    """Unified data access layer for the trading system."""

    def __init__(self):
        self._use_mock = cfg["data"]["use_mock"]
        self._interval  = cfg["data"]["interval"]
        self._days      = cfg["data"]["historical_days"]
        self._symbols   = cfg["universe"]["symbols"]
        self._raw_dir   = _ROOT / cfg["data"]["raw_dir"]

    # ------------------------------------------------------------------
    # Fetch + store
    # ------------------------------------------------------------------

    def refresh(self, symbols: list[str] | None = None) -> None:
        """Fetch fresh data for given symbols (or all in config) and persist."""
        symbols = symbols or self._symbols
        log.info(f"Refreshing data for: {symbols} | mode={'mock' if self._use_mock else 'kite'}")

        for sym in symbols:
            df = self._fetch(sym)
            save_ohlcv(df, self._interval)

        log.info("Data refresh complete.")

    def _fetch(self, symbol: str) -> pd.DataFrame:
        if self._use_mock:
            from data.mock_fetcher import fetch_mock_ohlcv
            return fetch_mock_ohlcv(symbol, self._days, self._interval)
        else:
            from data.kite_fetcher import fetch_kite_ohlcv
            return fetch_kite_ohlcv(symbol, self._days, self._interval)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV from SQLite for a symbol.
        Falls back to CSV if DB is empty (e.g. first run before refresh).

        Parameters
        ----------
        symbol : e.g. "RELIANCE"
        start  : "YYYY-MM-DD"  (inclusive, optional)
        end    : "YYYY-MM-DD"  (inclusive, optional)
        """
        df = self._load_from_db(symbol, start, end)

        if df.empty:
            log.warning(f"[DB] No data for {symbol}. Trying CSV fallback...")
            df = self._load_from_csv(symbol, start, end)

        if df.empty:
            raise ValueError(
                f"No data found for {symbol}. Run loader.refresh() first."
            )

        log.info(f"[LOAD] {symbol}: {len(df)} rows ({df['date'].min().date()} → {df['date'].max().date()})")
        return df

    def get_all(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Return dict of {symbol: df} for all configured symbols."""
        return {sym: self.get(sym, start, end) for sym in self._symbols}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_db(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        try:
            conn = _get_conn()
            _ensure_table(conn)

            query = "SELECT date, open, high, low, close, volume, symbol FROM ohlcv WHERE symbol = ?"
            params: list = [symbol]

            if start:
                query += " AND date >= ?"
                params.append(start)
            if end:
                query += " AND date <= ?"
                params.append(end)

            query += " ORDER BY date ASC"

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
            return df

        except Exception as e:
            log.error(f"DB read failed: {e}")
            return pd.DataFrame()

    def _load_from_csv(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        csv_path = self._raw_dir / f"{symbol}_{self._interval}.csv"
        if not csv_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(csv_path, parse_dates=["date"])
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a quick summary of what's stored in the DB."""
        conn = _get_conn()
        _ensure_table(conn)
        rows = conn.execute("""
            SELECT symbol, COUNT(*) as rows,
                   MIN(date) as first, MAX(date) as last,
                   MIN(close) as min_close, MAX(close) as max_close
            FROM ohlcv
            GROUP BY symbol
            ORDER BY symbol
        """).fetchall()
        conn.close()

        if not rows:
            print("Database is empty. Run loader.refresh() first.")
            return

        print(f"\n{'Symbol':<12} {'Rows':<8} {'First':<14} {'Last':<14} {'Min ₹':<12} {'Max ₹'}")
        print("-" * 72)
        for r in rows:
            print(f"{r[0]:<12} {r[1]:<8} {r[2]:<14} {r[3]:<14} {r[4]:<12.2f} {r[5]:.2f}")
        print()
