"""
data/kite_fetcher.py
--------------------
Fetches historical OHLCV data from Zerodha Kite Connect API.
Only used when config.broker.api_key is set and config.data.use_mock = false.

Kite Connect docs: https://kite.trade/docs/connect/v3/historical/
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)


def _get_kite_client():
    """Lazy import + initialise KiteConnect so the system works without
    the kiteconnect package when running in mock mode."""
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise ImportError(
            "kiteconnect package not installed. "
            "Run: pip install kiteconnect"
        )

    api_key = cfg["broker"]["api_key"]
    access_token = cfg["broker"]["access_token"]

    if not api_key or not access_token:
        raise ValueError(
            "broker.api_key and broker.access_token must be set in config.yaml "
            "before using the live fetcher."
        )

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def fetch_kite_ohlcv(
    symbol: str,
    days: Optional[int] = None,
    interval: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV from Kite Connect.

    Parameters
    ----------
    symbol   : NSE symbol string, e.g. "RELIANCE"
    days     : number of calendar days of history (default from config)
    interval : Kite interval string (default from config)

    Returns
    -------
    pd.DataFrame with columns: date, open, high, low, close, volume, symbol
    """
    days = days or cfg["data"]["historical_days"]
    interval = interval or cfg["data"]["interval"]
    exchange = cfg["universe"]["exchange"]

    instrument_token = f"{exchange}:{symbol}"
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    log.info(f"[KITE] Fetching {interval} data for {instrument_token} "
             f"from {start_dt.date()} to {end_dt.date()}")

    kite = _get_kite_client()

    # Kite returns list of dicts with keys: date, open, high, low, close, volume
    raw = kite.historical_data(
        instrument_token=instrument_token,
        from_date=start_dt,
        to_date=end_dt,
        interval=interval,
        continuous=False,
        oi=False,
    )

    if not raw:
        raise ValueError(f"No data returned for {symbol}. Check symbol / credentials.")

    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Standardise column order
    df = df[["date", "open", "high", "low", "close", "volume", "symbol"]]
    log.info(f"[KITE] {symbol}: {len(df)} rows fetched")
    return df
