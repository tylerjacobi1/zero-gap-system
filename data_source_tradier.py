import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from requests.exceptions import HTTPError
from ib_insync import IB, Contract, util

# ─── Tradier API Setup ───
TRADIER_BASE = "https://api.tradier.com/v1"
API_KEY = st.secrets.get("TRADIER_API_KEY") or os.getenv("TRADIER_API_KEY")
if not API_KEY:
    st.error("TRADIER_API_KEY not set. Please check .streamlit/secrets.toml or your environment.")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}


def get_expirations(symbol: str) -> list[str]:
    try:
        url = f"{TRADIER_BASE}/markets/options/expirations"
        resp = requests.get(url, params={"symbol": symbol}, headers=HEADERS)
        resp.raise_for_status()
        expirations = resp.json().get("expirations", {}).get("date", [])
        if not expirations:
            st.sidebar.warning(f"No expirations found for symbol {symbol}.")
        return expirations
    except Exception as e:
        st.sidebar.error(f"⚠ Could not load expirations: {e}")
        return []


def fetch_option_chain(symbol: str, expiration: str) -> pd.DataFrame:
    try:
        url = f"{TRADIER_BASE}/markets/options/chains"
        resp = requests.get(
            url,
            params={"symbol": symbol, "expiration": expiration, "greeks": "true"},
            headers=HEADERS,
        )
        resp.raise_for_status()

        options = resp.json().get("options", {}).get("option", [])
        if not options:
            st.sidebar.warning(f"⚠ No options data for {symbol} @ {expiration}")
            return pd.DataFrame()

        df = pd.DataFrame(options)

        # Sanitize + normalize data
        df["strike"] = df["strike"].astype(float)
        df[["bid", "ask"]] = df[["bid", "ask"]].astype(float)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        df["option_type"] = df["option_type"].str.lower()

        # Merge greeks into flat structure if present
        if "greeks" in df.columns:
            greeks_df = pd.json_normalize(df.pop("greeks"))
            df = pd.concat([df, greeks_df], axis=1)

        # Ensure IV column exists
        if "iv" not in df.columns:
            df["iv"] = (df["bid"] + df["ask"]) / 2
        else:
            df["iv"] = pd.to_numeric(df["iv"], errors="coerce")

        # Add underlying price snapshot
        df["underlying_price"] = get_underlying_price(symbol)

        return df

    except Exception as e:
        st.sidebar.error(f"⚠ Tradier chain fetch failed: {e}")
        return pd.DataFrame()

def get_underlying_price(symbol: str) -> float:
    try:
        url = f"{TRADIER_BASE}/markets/quotes"
        resp = requests.get(url, params={"symbols": symbol}, headers=HEADERS)
        resp.raise_for_status()
        quote = resp.json().get("quotes", {}).get("quote", {})
        return float(quote.get("last", np.nan))
    except Exception:
        return np.nan


def get_iv_history(symbol: str, interval: str = "1min") -> pd.Series:
    url = f"{TRADIER_BASE}/markets/volatility/history"
    try:
        resp = requests.get(url, params={"symbol": symbol, "interval": interval}, headers=HEADERS)
        resp.raise_for_status()
        history = resp.json().get("history", [])
        df = pd.DataFrame(history)
        if df.empty:
            return pd.Series(dtype=float)
        df["iv"] = df["iv"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp")["iv"]
    except Exception:
        return pd.Series(dtype=float)


def compute_iv_change(symbol: str, interval: str = "1min") -> float:
    try:
        ivs = get_iv_history(symbol, interval)
        if len(ivs) < 2 or ivs.iloc[-2] == 0:
            return 0.0
        return (ivs.iloc[-1] - ivs.iloc[-2]) / ivs.iloc[-2] * 100
    except Exception:
        return 0.0


def compute_iv_skew(chain: pd.DataFrame, delta: float = 0.25) -> float:
    underlying = chain["underlying_price"].iloc[0]
    target_call_strike = underlying * (1 + delta)
    target_put_strike = underlying * (1 - delta)

    idx_call = (chain["strike"] - target_call_strike).abs().idxmin()
    idx_put = (chain["strike"] - target_put_strike).abs().idxmin()

    iv_call = float(chain.at[idx_call, "iv"])
    iv_put = float(chain.at[idx_put, "iv"])
    return iv_call - iv_put


def fetch_realtime_bars_ibkr(symbol: str, duration: str, barSize: str, whatToShow: str, useRTH: bool,
                             host: str, port: int, clientId: int) -> pd.DataFrame:
    try:
        ib = IB()
        ib.connect(host, port, clientId=clientId, timeout=5)
        idx = Contract(symbol=symbol, secType="IND", exchange="CBOE", currency="USD")
        bars = ib.reqHistoricalData(
            contract=idx,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=barSize,
            whatToShow=whatToShow,
            useRTH=useRTH,
            formatDate=1,
        )
        ib.disconnect()
        df = util.df(bars)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()

def get_price_history(symbol: str, interval: str = "1min", periods: int = 20) -> pd.Series:
    """
    Fetch recent price bars for the underlying (e.g. 1-min), return a Series of closes.
    """
    url = f"{TRADIER_BASE}/markets/bars"
    try:
        resp = requests.get(
            url,
            params={"symbol": symbol, "interval": interval},
            headers=HEADERS,
        )
        resp.raise_for_status()
        bars = resp.json().get("history", [])
        df = pd.DataFrame(bars)
        if df.empty or "close" not in df:
            return pd.Series(dtype=float)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df["close"].iloc[-periods:]
    except Exception:
        return pd.Series(dtype=float)
