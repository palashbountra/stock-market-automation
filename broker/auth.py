"""
broker/auth.py
--------------
Zerodha Kite Connect authentication flow.

Kite login works like this every single day:
  1. You open a login URL in your browser
  2. You log in with your Zerodha credentials
  3. Kite redirects to your redirect_url with ?request_token=XXXX
  4. You copy that request_token and run this script
  5. Script exchanges it for an access_token (valid until midnight)
  6. access_token is saved to .env automatically

Run every morning before market open (9:00 AM IST):
    python3 broker/auth.py

Or just run:
    python3 main.py login
"""

import os
import re
import sys
import webbrowser
from pathlib import Path

from dotenv import load_dotenv, set_key

# Load .env from project root
_ROOT    = Path(__file__).resolve().parent.parent
_ENV_FILE = _ROOT / ".env"
load_dotenv(_ENV_FILE)


def get_login_url() -> str:
    """Return the Kite Connect login URL for this app."""
    api_key = os.getenv("KITE_API_KEY", "")
    if not api_key:
        raise ValueError("KITE_API_KEY not found in .env — please fill it in.")
    return f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"


def generate_access_token(request_token: str) -> str:
    """
    Exchange a request_token for an access_token.
    Requires: kiteconnect package (pip install kiteconnect)
    """
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise ImportError(
            "kiteconnect not installed.\n"
            "Run: pip install kiteconnect"
        )

    api_key    = os.getenv("KITE_API_KEY", "")
    api_secret = os.getenv("KITE_API_SECRET", "")

    if not api_key or not api_secret:
        raise ValueError(
            "KITE_API_KEY or KITE_API_SECRET missing in .env\n"
            "Open .env and fill in your API secret."
        )

    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]

    # Save access_token to .env for the rest of the system to use
    set_key(str(_ENV_FILE), "KITE_ACCESS_TOKEN", access_token)
    print(f"\n✅ Access token generated and saved to .env")
    print(f"   Token: {access_token[:10]}...{access_token[-5:]} (truncated for safety)")
    print(f"   Valid until midnight today.\n")

    return access_token


def login_flow():
    """Interactive login flow — run this every morning."""
    print("\n" + "="*60)
    print("  ZERODHA KITE CONNECT — DAILY LOGIN")
    print("="*60)

    login_url = get_login_url()
    print(f"\n  Step 1: Opening Kite login in your browser...")
    print(f"  URL: {login_url}\n")
    webbrowser.open(login_url)

    print("  Step 2: Log in with your Zerodha credentials.")
    print("  After login, you'll be redirected to:")
    print("    http://127.0.0.1:5000/?request_token=XXXXXX&action=login&status=success")
    print("\n  Step 3: Copy the request_token from that URL.")
    print("  (It's the value after '?request_token=' and before '&action')\n")

    request_token = input("  Paste your request_token here: ").strip()

    if not request_token:
        print("❌ No token entered. Exiting.")
        sys.exit(1)

    # Clean up in case user pasted full URL
    if "request_token=" in request_token:
        match = re.search(r"request_token=([^&]+)", request_token)
        if match:
            request_token = match.group(1)

    print(f"\n  Exchanging request_token for access_token...")
    access_token = generate_access_token(request_token)
    return access_token


def get_access_token() -> str:
    """
    Returns the current access_token from .env.
    Call this from any module that needs an authenticated Kite session.
    """
    load_dotenv(_ENV_FILE, override=True)
    token = os.getenv("KITE_ACCESS_TOKEN", "")
    if not token:
        raise ValueError(
            "No access token found. Run: python3 main.py login\n"
            "You need to login every morning before market hours."
        )
    return token


def get_kite_client():
    """
    Returns a fully authenticated KiteConnect instance.
    Raises if credentials are missing.
    """
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise ImportError("Run: pip install kiteconnect")

    load_dotenv(_ENV_FILE, override=True)
    api_key      = os.getenv("KITE_API_KEY", "")
    access_token = get_access_token()

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


if __name__ == "__main__":
    login_flow()
