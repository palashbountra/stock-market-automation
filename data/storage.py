"""
data/storage.py
---------------
Handles persistence of OHLCV data in two forms:
  1. CSV  → data/raw/<SYMBOL>_<interval>.csv        (human-readable, workspace)
  2. SQLite → data/db/market_data.db  table: ohlcv  (fast querying)

SQLite DB path: resolved from config, but can be overridden via env var
  TRADING_DB_PATH=/tmp/market_data.db python main.py fetch
This is useful when the workspace is on a network/FUSE mount that doesn't
support SQLite file locking (macOS SMB etc). On a real local machine the
default path inside the project folder works fine.
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd

from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR = _ROOT / cfg["data"]["raw_dir"]

# Allow env override for DB path (useful on network-mounted workspaces)
_DB_PATH = Path(os.environ.get("TRADING_DB_PATH", str(_ROOT / cfg["data"]["db_path"])))

_RAW_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

log.debug(f"SQLite DB path: {_DB_PATH}")


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(str(_DB_PATH))


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol  TEXT    NOT NULL,
            date    TEXT    NOT NULL,
            open    REAL    NOT NULL,
            high    REAL    NOT NULL,
            low     REAL    NOT NULL,
            close   REAL    NOT NULL,
            volume  INTEGER NOT NULL,
            UNIQUE(symbol, date)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON ohlcv(symbol, date)")
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_ohlcv(df: pd.DataFrame, interval: str | None = None) -> None:
    """
    Persist a DataFrame (must have columns: date, open, high, low, close, volume, symbol).
    - Upserts into SQLite (INSERT OR REPLACE)
    - Writes / overwrites CSV in raw_dir
    """
    interval = interval or cfg["data"]["interval"]
    _validate_df(df)

    symbol = df["symbol"].iloc[0]

    # --- SQLite ---
    conn = _get_conn()
    _ensure_table(conn)

    records = df[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()
    records["date"] = records["date"].astype(str)

    conn.executemany(
        """INSERT OR REPLACE INTO ohlcv
           (symbol, date, open, high, low, close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        records.itertuples(index=False, name=None),
    )
    conn.commit()
    conn.close()
    log.info(f"[DB]  Saved {len(df)} rows for {symbol} → {_DB_PATH}")

    # --- CSV ---
    csv_path = _RAW_DIR / f"{symbol}_{interval}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"[CSV] Saved {symbol} → {csv_path}")


def _validate_df(df: pd.DataFrame) -> None:
    required = {"date", "open", "high", "low", "close", "volume", "symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty — nothing to save.")
