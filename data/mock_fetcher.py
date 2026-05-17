"""
data/mock_fetcher.py
--------------------
Generates realistic synthetic OHLCV data for Indian stocks.
Used when config.data.use_mock = true or no API credentials are set.

Prices are seeded so results are reproducible per symbol.
"""

import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils.config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

# Approximate real starting prices (INR) for realism
_SEED_PRICES = {
    "RELIANCE": 2800.0,
    "TCS":      3900.0,
    "INFY":     1600.0,
}
_DEFAULT_SEED_PRICE = 1000.0


def _symbol_seed(symbol: str) -> int:
    """Deterministic integer seed from symbol name."""
    return int(hashlib.md5(symbol.encode()).hexdigest(), 16) % (2**31)


def fetch_mock_ohlcv(
    symbol: str,
    days: int | None = None,
    interval: str | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        date, open, high, low, close, volume
    sorted ascending by date, covering `days` trading days ending today.
    """
    days = days or cfg["data"]["historical_days"]
    interval = interval or cfg["data"]["interval"]

    rng = np.random.default_rng(_symbol_seed(symbol))
    base_price = _SEED_PRICES.get(symbol, _DEFAULT_SEED_PRICE)

    log.info(f"[MOCK] Generating {days} days of {interval} data for {symbol}")

    # --- generate business-day date range ----------------------------------
    end_date = datetime.today().date()
    all_dates = pd.bdate_range(end=end_date, periods=days)

    n = len(all_dates)

    # --- simulate price path with GBM (geometric brownian motion) ----------
    mu = 0.0003          # daily drift ~7.5% annualised
    sigma = 0.015        # daily vol ~24% annualised
    returns = rng.normal(mu, sigma, n)
    close_prices = base_price * np.cumprod(1 + returns)

    # Intraday ranges
    daily_range_pct = rng.uniform(0.005, 0.025, n)   # 0.5% – 2.5% intraday range
    high = close_prices * (1 + daily_range_pct)
    low  = close_prices * (1 - daily_range_pct)
    open_ = low + rng.uniform(0, 1, n) * (high - low)

    # Volume: base 1M shares, lognormal noise
    volume = (rng.lognormal(mean=14.0, sigma=0.5, size=n)).astype(int)

    df = pd.DataFrame({
        "date":   pd.to_datetime(all_dates),
        "open":   np.round(open_, 2),
        "high":   np.round(high, 2),
        "low":    np.round(low, 2),
        "close":  np.round(close_prices, 2),
        "volume": volume,
    })
    df["symbol"] = symbol
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info(f"[MOCK] {symbol}: {len(df)} rows | "
             f"price range ₹{df['close'].min():.2f}–₹{df['close'].max():.2f}")
    return df
