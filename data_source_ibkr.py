"""
data_source_ibkr.py  •  Zero‑Gap Execution System
──────────────────────────────────────────────────────────────────────────────
✓  Re‑uses ONE global IB() session instead of reconnecting every call
✓  All st.secrets[] look‑ups made safe with .get()
✓  Automatic fall‑back to delayed Tradier on any IBKR failure
✓  Streamlit @st.cache_data on readonly helpers to cut API calls
✓  IBKR barSize / duration strings normalised ("1 min", "30 secs", etc.)
"""

import os, re, time, math, requests, warnings
import pandas as pd
import numpy as np
import streamlit as st
from functools import lru_cache

from ib_insync import IB, Contract, Index, Option, util

# ════════════════════════════════════════════════════════════════════════════
#  Secrets & Config
# ════════════════════════════════════════════════════════════════════════════
TRADIER_BASE = "https://api.tradier.com/v1"
TRADIER_KEY  = st.secrets.get("TRADIER_API_KEY", os.getenv("TRADIER_API_KEY", ""))
TRADIER_HEADERS = {
    "Authorization": f"Bearer {TRADIER_KEY}",
    "Accept": "application/json",
}

IB_HOST   = st.secrets.get("ibkr", {}).get("host",   "127.0.0.1")
IB_PORT   = st.secrets.get("ibkr", {}).get("port",   7497)
IB_CLIENT = st.secrets.get("ibkr", {}).get("client_id", 999)

# ════════════════════════════════════════════════════════════════════════════
#  ONE global IB() instance
# ════════════════════════════════════════════════════════════════════════════
_ib = IB()
_ib_warned = False

def _connect_ibkr() -> bool:
    """Ensure _ib is connected. Return True if live connection works."""
    try:
        if not _ib.isConnected():
            _ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT, timeout=3)
        _ = _ib.reqCurrentTime()          # ping
        return True
    except Exception:
        return False

def _warn_delayed():
    global _ib_warned
    if not _ib_warned:
        st.sidebar.warning("⏱  Live IBKR data unavailable — using delayed Tradier feed.")
        _ib_warned = True

# Decide once per session whether live IBKR is available
USE_DELAYED = not _connect_ibkr()
if USE_DELAYED:
    _warn_delayed()

# ════════════════════════════════════════════════════════════════════════════
#  Helper: safe Tradier request
# ════════════════════════════════════════════════════════════════════════════
def _tradier_request(endpoint: str, params: dict | None = None) -> requests.Response:
    if not TRADIER_KEY:
        raise RuntimeError("TRADIER_API_KEY not found (secrets.toml or env var).")
    resp = requests.get(f"{TRADIER_BASE}{endpoint}", params=params or {}, headers=TRADIER_HEADERS)
    if resp.status_code == 401:
        raise RuntimeError("Tradier API → 401 Unauthorized. Check TRADIER_API_KEY.")
    resp.raise_for_status()
    return resp

# ════════════════════════════════════════════════════════════════════════════
#  1) Expiration Dates
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_expirations(symbol: str) -> list[str]:
    """Return sorted list of YYYY‑MM‑DD expirations."""
    if not USE_DELAYED:
        try:
            prot = Option(symbol, "", 0.0, "C", "SMART")
            details = _ib.reqContractDetails(prot)
            dates = sorted({d.contract.lastTradeDateOrContractMonth for d in details})
            return [
                pd.to_datetime(d, format="%Y%m%d").strftime("%Y-%m-%d")
                for d in dates
            ]
        except Exception:
            pass  # fall through to Tradier
    _warn_delayed()
    resp = _tradier_request("/markets/options/expirations", {"symbol": symbol})
    return resp.json().get("expirations", {}).get("date", [])

# ════════════════════════════════════════════════════════════════════════════
#  2) Option Chain (calls & puts, with greeks if available)
# ════════════════════════════════════════════════════════════════════════════
def _ibkr_chain(symbol: str, expiration: str) -> pd.DataFrame:
    exp_tag = expiration.replace("-", "")
    prot = [Option(symbol, exp_tag, 0.0, r, "SMART") for r in ("C", "P")]
    contracts = [d.contract for p in prot for d in _ib.reqContractDetails(p)]
    _ib.qualifyContracts(*contracts)

    ticks = _ib.reqMktData(contracts, snapshot=True, regulatorySnapshot=True)
    time.sleep(0.6)   # wait for bid/ask

    rows = []
    for t in ticks:
        if t.bid is None or t.ask is None:
            continue
        greek = t.modelGreeks or util.ZeroGreeks()
        rows.append({
            "strike":      t.contract.strike,
            "option_type": "call" if t.contract.right == "C" else "put",
            "bid":         t.bid,
            "ask":         t.ask,
            "volume":      int(t.volume or 0),
            "iv":          greek.impliedVol or np.nan,
            "gamma":       greek.gamma or 0.0,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("IBKR returned empty chain")
    return df

def _tradier_chain(symbol: str, expiration: str) -> pd.DataFrame:
    resp = _tradier_request(
        "/markets/options/chains",
        {"symbol": symbol, "expiration": expiration, "greeks": "true"},
    )
    opts = resp.json().get("options", {}).get("option", [])
    df = pd.DataFrame(opts)
    if df.empty:
        raise RuntimeError("Tradier returned empty chain")

    df["strike"] = df["strike"].astype(float)
    df[["bid", "ask"]] = df[["bid", "ask"]].astype(float)
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)
    df["option_type"] = df["option_type"].str.lower()
    if "greeks" in df.columns:
        greeks = pd.json_normalize(df.pop("greeks"))
        df = pd.concat([df, greeks], axis=1)
    if "iv" not in df.columns:
        df["iv"] = (df["bid"] + df["ask"]) / 2
    df["gamma"] = pd.to_numeric(df.get("gamma", 0), errors="coerce").fillna(0.0)
    return df

def fetch_option_chain(symbol: str, expiration: str) -> pd.DataFrame:
    try:
        df = _ibkr_chain(symbol, expiration) if not USE_DELAYED else pd.DataFrame()
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        _warn_delayed()
        df = _tradier_chain(symbol, expiration)
    df["underlying_price"] = get_underlying_price(symbol)
    return df

# ════════════════════════════════════════════════════════════════════════════
#  3) Underlying Price
# ════════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=32)
def _ibkr_price(symbol: str) -> float:
    idx = Index(symbol, exchange="CBOE", currency="USD")
    _ib.qualifyContracts(idx)
    tick = _ib.reqMktData(idx, "", snapshot=True, regulatorySnapshot=True)
    time.sleep(0.3)
    price = tick.last or tick.close or np.nan
    if math.isnan(price):
        raise RuntimeError("IBKR price N/A")
    return float(price)

def _tradier_price(symbol: str) -> float:
    q = _tradier_request("/markets/quotes", {"symbols": symbol}).json()
    q = q.get("quotes", {}).get("quote", {})
    return float(q.get("last", np.nan))

def get_underlying_price(symbol: str) -> float:
    if not USE_DELAYED:
        try:
            return _ibkr_price(symbol)
        except Exception:
            pass
    _warn_delayed()
    return _tradier_price(symbol)

# ════════════════════════════════════════════════════════════════════════════
#  4) Recent price history (pandas Series)
# ════════════════════════════════════════════════════════════════════════════
def _normalise_interval(interval: str) -> str:
    """Convert '1min' → '1 min', '30sec' → '30 secs' (IBKR style)."""
    m = re.fullmatch(r"\s*(\d+)\s*([a-zA-Z]+)\s*", interval)
    if not m:
        raise ValueError(f"Bad interval {interval!r}")
    num, unit = m.groups()
    unit = unit.lower()
    if unit.startswith("min"):
        return f"{num} min"
    if unit.startswith("sec"):
        return f"{num} secs"
    return interval

def get_price_history(symbol: str, interval: str = "1min", periods: int = 20) -> pd.Series:
    bar_size = _normalise_interval(interval)
    total_secs = periods * (60 if "min" in bar_size else 1)
    if not USE_DELAYED:
        try:
            c = Index(symbol, exchange="CBOE", currency="USD")
            bars = _ib.reqHistoricalData(
                c, "", f"{total_secs} S", bar_size,
                "MIDPOINT", useRTH=False, formatDate=1
            )
            df = util.df(bars)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["date"])
                return df.set_index("timestamp")["close"].iloc[-periods:]
        except Exception:
            pass
    _warn_delayed()
    resp = _tradier_request("/markets/bars", {"symbol": symbol, "interval": interval})
    hist = resp.json().get("history", [])
    df = pd.DataFrame(hist)
    if df.empty:
        return pd.Series(dtype=float)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")["close"].iloc[-periods:]

# ════════════════════════════════════════════════════════════════════════════
#  5) IV 1‑Period Change (%)
# ════════════════════════════════════════════════════════════════════════════
def compute_iv_change(symbol: str, interval: str = "1min") -> float:
    try:
        if not USE_DELAYED:
            idx = Index(symbol, exchange="CBOE", currency="USD")
            bars = _ib.reqHistoricalData(
                idx, "", "2 D", _normalise_interval(interval),
                "HISTORICAL_VOLATILITY", useRTH=False, formatDate=1
            )
            df = util.df(bars)
            prev, last = df.close.iloc[-2:]
            return float((last - prev) / prev * 100)
    except Exception:
        pass
    _warn_delayed()
    resp = _tradier_request("/markets/volatility/history", {"symbol": symbol, "interval": interval})
    hist = resp.json().get("history", [])
    df = pd.DataFrame(hist)
    if len(df) >= 2 and df.iv.iloc[-2] != 0:
        p, l = df.iv.iloc[-2], df.iv.iloc[-1]
        return float((l - p) / p * 100)
    return 0.0

# ════════════════════════════════════════════════════════════════════════════
#  6) Real‑time bars (IBKR only; empty df if delayed)
# ════════════════════════════════════════════════════════════════════════════
def fetch_realtime_bars_ibkr(
    symbol: str,
    duration: str = "1800 S",
    bar_size: str = "1 min",
    what_to_show: str = "MIDPOINT",
    use_rth: bool = False,
) -> pd.DataFrame:
    if USE_DELAYED:
        _warn_delayed()
        return pd.DataFrame()

    idx = Index(symbol, exchange="CBOE", currency="USD")
    bars = _ib.reqHistoricalData(
        idx, "", duration, bar_size, what_to_show,
        useRTH=int(use_rth), formatDate=1
    )
    df = util.df(bars)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
