# ──────────────────────────────────────────────────────────────
# Zero‑Gap Execution System – main2.py  (TOP‑OF‑FILE SECTION)
# ──────────────────────────────────────────────────────────────
import asyncio
import base64
import math
import os
import random

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt      #  used later for quick preview plots
import requests            #  used in data_source_tradier.py, etc.

# ── UI‑state defaults ─────────────────────────────────────────
# Positive offset  → legs shift further OTM
# Negative offset  → legs shift closer ITM
for _key in ("strangle_offset", "spread_offset"):
    if _key not in st.session_state:
        st.session_state[_key] = 0

# ── Async event‑loop guard (ib_insync needs one) ──────────────
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ═════════════════════════════ 2. Secrets / Environment ══════════════════════
def _secret(key: str, default: str = ""):
    return st.secrets.get(key) or os.getenv(key, default)

TRADIER_API_KEY = _secret("TRADIER_API_KEY")
IBKR_HOST       = _secret("IBKR_HOST", "127.0.0.1")
IBKR_PORT       = int(_secret("IBKR_PORT", "7497"))
IBKR_CLIENT_ID  = int(_secret("IBKR_CLIENT_ID", "999"))

HEADERS = {
    "Authorization": f"Bearer {TRADIER_API_KEY}",
    "Accept": "application/json"
}

from ib_insync import IB, Index, Contract, ComboLeg, LimitOrder, util

# ═════════════════════════════ 3. Live vs Delayed Mode ══════════════════════
ib = IB()
try:
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=3)
    ib.reqCurrentTime()
    USE_DELAYED = False
except Exception:
    USE_DELAYED = True
    st.sidebar.warning("⏱  Live IBKR feed unavailable — using delayed Tradier data.")
finally:
    if ib.isConnected():
        ib.disconnect()

if USE_DELAYED:
    # Delayed fallback
    from data_source_tradier import (
        get_expirations        as get_expirations,
        fetch_option_chain     as fetch_option_chain,
        get_underlying_price   as get_underlying_price,
        get_price_history      as get_price_history,
        compute_iv_change      as compute_iv_change,
    )

    def fetch_realtime_bars_ibkr(*_, **__) -> pd.DataFrame:
        return pd.DataFrame()
else:
    # Live data
    from data_source_ibkr import (
        get_expirations,
        fetch_option_chain,
        get_underlying_price,
        fetch_realtime_bars_ibkr,
        get_price_history,
        compute_iv_change,
    )

# ═════════════════════════════ 4. Logo in sidebar ═══════════════════════════
import base64
with open("atomic_3.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <style>
      [data-testid="stSidebar"] > div:first-child {{
        position: relative;
      }}
      [data-testid="stSidebar"] > div:first-child::before {{
        content: "";
        position: absolute;
        top: 24px;
        left: 16px;
        width: 160px;
        height: 64px;
        background-image: url("data:image/png;base64,{b64}");
        background-size: contain;
        background-repeat: no-repeat;
        pointer-events: none;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════ 5. Connection‑mode toggle ════════════════════
USE_TWS = st.sidebar.radio("Connect via", ["TWS (7497)", "IB Gateway (4001)"]) == "TWS (7497)"
HOST, PORT = "127.0.0.1", (7497 if USE_TWS else 4001)
CLIENT_ID_ORDER = random.randint(1000, 9000)

# ═════════════════════════════ Send to IBKR Button ════════════════════
if st.sidebar.button("Send to IBKR"):
    ib = IB()
    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID_ORDER, timeout=5)
    except Exception as e:
        st.sidebar.error(f"IBKR connect failed: {e}")
    else:
        expiry = expiration.replace("-", "")
        sym = symbol

        put   = Contract(sym, "OPT", "SMART", "USD", expiry, metrics["put_strike"],  "P", "100")
        call  = Contract(sym, "OPT", "SMART", "USD", expiry, metrics["call_strike"], "C", "100")
        opt_c = spread["option_type"].upper()
        short = Contract(sym, "OPT", "SMART", "USD", expiry, spread["short_strike"], opt_c, "100")
        long  = Contract(sym, "OPT", "SMART", "USD", expiry, spread["long_strike"],  opt_c, "100")

        ib.qualifyContracts(put, call, short, long)

        combo = Contract(secType="BAG", symbol=sym, exchange="SMART", currency="USD")
        combo.comboLegs = [
            ComboLeg(conId=put.conId,   ratio=1, action="BUY",  exchange="SMART"),
            ComboLeg(conId=call.conId,  ratio=1, action="BUY",  exchange="SMART"),
            ComboLeg(conId=short.conId, ratio=1, action="SELL", exchange="SMART"),
            ComboLeg(conId=long.conId,  ratio=1, action="BUY",  exchange="SMART"),
        ]

        net = round(spread["credit"] - metrics["debit"], 2)
        order = LimitOrder("SELL" if net >= 0 else "BUY", qty, abs(net))
        trade = ib.placeOrder(combo, order)

        st.sidebar.success(f"Order submitted (ID {trade.order.orderId}) @ ${abs(net):.2f}")
        ib.disconnect()


# ═════════════════════════════ 6. Strategy helpers ══════════════════════════

# ── 6·1  ATM strike chooser ─────────────────────────────────
def pick_atm_strikes(chain: pd.DataFrame, underlying: float) -> tuple[float, float]:
    """
    Returns the two strikes that bracket the underlying price.
    """
    strikes = sorted(chain["strike"].unique())
    idx     = np.searchsorted(strikes, underlying)
    lo      = strikes[idx - 1] if idx > 0 else strikes[0]
    hi      = strikes[idx]     if idx < len(strikes) else strikes[-1]
    return lo, hi


# ── 6·2  Strangle builder *with offset support* ─────────────
def build_strangle(
    chain: pd.DataFrame,
    underlying: float,
    iv_rank: float = 100,
    skew_offset: int = 0,
    offset: int = 0,                   # ⬅️  new!
) -> dict:
    """
    offset > 0  → shift both legs further OTM (wider strangle)
    offset < 0  → shift both legs closer ITM (tighter strangle)
    Each unit = one strike 'step' in the option ladder.
    """
    strikes = sorted(chain["strike"].unique())
    idx     = np.searchsorted(strikes, underlying)

    # Base leg indices (pre‑offset), same logic you had before
    if iv_rank < 30 and skew_offset > 0:
        put_idx  = max(idx - 1 - skew_offset, 0)
        call_idx = min(idx + skew_offset, len(strikes) - 1)
    else:
        put_idx  = idx - 1 if idx > 0 else 0
        call_idx = idx     if idx < len(strikes) else len(strikes) - 1

    # Apply user nudge
    put_idx  = max(0,                     put_idx  - offset)
    call_idx = min(len(strikes) - 1,      call_idx + offset)

    put_str,  call_str  = strikes[put_idx], strikes[call_idx]
    put_px   = chain.loc[(chain.strike == put_str)  & (chain.option_type == "put"),  "ask"].iloc[0]
    call_px  = chain.loc[(chain.strike == call_str) & (chain.option_type == "call"), "ask"].iloc[0]

    debit = put_px + call_px
    return {
        "put_strike":      put_str,
        "call_strike":     call_str,
        "put_price":       put_px,
        "call_price":      call_px,
        "debit":           round(debit, 2),
        "breakeven_low":   round(put_str  - debit, 2),
        "breakeven_high":  round(call_str + debit, 2),
        "max_loss":        round(debit, 2),
    }


# ── 6·3  Decide which side gets the credit‑spread hedge ─────
def pick_spread_side(metrics: dict, underlying: float) -> str:
    return (
        "below"
        if (underlying - metrics["breakeven_low"])
        <  (metrics["breakeven_high"] - underlying)
        else "above"
    )


# ── 6·4  Credit‑spread selector *with offset support* ───────
def get_rr_spread(
    options: list[dict],
    metrics: dict,
    direction: str,
    underlying: float,
    iv_rank: float = 100,
    max_spread_width: int | None = None,
    offset: int = 0,                    # ⬅️  new!
) -> dict:
    """
    Finds the risk‑reward‑optimal vertical (credit spread).
    offset shifts the entire spread ladder:
      • direction == 'below' (puts):   +offset → further OTM (lower strikes)
      • direction == 'above' (calls):  +offset → further OTM (higher strikes)
    """

    opt_type = "put" if direction == "below" else "call"
    strikes  = sorted({o["strike"] for o in options if o["option_type"] == opt_type})
    if len(strikes) < 2:
        return {}

    # Strike spacing
    step = min(j - i for i, j in zip(strikes, strikes[1:]) if j > i)

    # Desired breakeven we’re hedging to
    target_be = (
        metrics["breakeven_low"] if direction == "below" else metrics["breakeven_high"]
    )

    best = None
    for i, base_s in enumerate(strikes):
        # Apply user offset (sign depends on direction)
        i_eff = i + (-offset if direction == "below" else offset)
        if not 0 <= i_eff < len(strikes):
            continue

        s = strikes[i_eff]                              # short leg
        j = i_eff - 1 if direction == "below" else i_eff + 1
        if not 0 <= j < len(strikes):
            continue
        l = strikes[j]                                  # long leg

        width = abs(l - s)
        if iv_rank <= 70 and max_spread_width and width > max_spread_width * step:
            continue

        row_s = next(o for o in options if o["strike"] == s and o["option_type"] == opt_type)
        row_l = next(o for o in options if o["strike"] == l and o["option_type"] == opt_type)
        credit = row_s["bid"] - row_l["ask"]
        if credit <= 0:
            continue

        spread_be = (s + credit) if direction == "above" else (l - credit)
        dist      = abs(spread_be - target_be)

        cand = {
            "direction":   direction,
            "short_strike": s,
            "long_strike":  l,
            "credit":       round(credit, 2),
            "max_loss":     round(width - credit, 2),
            "option_type":  opt_type,
            "distance":     dist,
        }
        if best is None or (cand["distance"], -cand["credit"]) < (best["distance"], -best["credit"]):
            best = cand

    # Fallback if nothing meets filters
    spread = best or {
        "direction":    direction,
        "short_strike": strikes[0],
        "long_strike":  strikes[1],
        "credit":       0.0,
        "max_loss":     0.0,
        "option_type":  opt_type,
    }

    # Recommended quantity so spread credit ≈ strangle max‑loss
    spread["qty"] = (
        max(1, math.ceil(metrics["max_loss"] / spread["credit"]))
        if spread["credit"] > 0
        else 1
    )
    return spread


# ── 6·5  Simple helpers ─────────────────────────────────────
def calculate_quantity(metrics: dict, spread: dict) -> int:
    return (
        1
        if spread.get("credit", 0) <= 0
        else math.ceil(metrics["max_loss"] / spread["credit"])
    )


def generate_strangle_curve(metrics: dict, underlying: float) -> pd.DataFrame:
    prices = np.linspace(underlying * 0.5, underlying * 1.5, 200)
    pnl    = [
        max(metrics["put_strike"] - p, 0)   - metrics["put_price"]
        + max(p - metrics["call_strike"], 0) - metrics["call_price"]
        for p in prices
    ]
    return pd.DataFrame({"Price": prices, "P/L": pnl})


def generate_spread_curve(spread: dict, qty: int, underlying: float) -> pd.DataFrame:
    prices = np.linspace(underlying * 0.5, underlying * 1.5, 200)
    pnl    = []
    for p in prices:
        if spread["direction"] == "below":
            short = max(spread["short_strike"] - p, 0)
            long  = max(spread["long_strike"]  - p, 0)
        else:
            short = max(p - spread["short_strike"], 0)
            long  = max(p - spread["long_strike"],  0)
        pnl.append((short - long + spread["credit"]) * qty)
    return pd.DataFrame({"Price": prices, "P/L": pnl})

# ── 6·X  Credit‑spread builder (width + short‑nudge) ────────────────────────
def build_credit_spread_by_width(
    chain: pd.DataFrame,
    underlying: float,
    side: str,                   # "below" for put credit spread, "above" for call
    short_offset: int = 0,       # nudge of the SHORT leg, ± strike steps
    width_steps: int = 1,        # gap between short & long legs in strike steps
) -> dict:
    """
    Constructs a credit spread where:
      • 'short_offset'  choses the SHORT strike relative to ATM.
      • 'width_steps'   chooses how many strikes away the LONG leg is.
    Positive offset  → further OTM
    Negative offset  → closer ITM
    """

    strikes = sorted(chain["strike"].unique())
    step    = min(j - i for i, j in zip(strikes, strikes[1:]) if j > i)

    # --- locate ATM index ----------------------------------------------------
    atm_idx = np.searchsorted(strikes, underlying)
    if side == "below":      # put credit spread
        short_idx = max(0, atm_idx - 1 - short_offset)
        long_idx  = max(0, short_idx - width_steps)
        opt_type  = "put"
    else:                    # call credit spread
        short_idx = min(len(strikes) - 1, atm_idx + short_offset)
        long_idx  = min(len(strikes) - 1, short_idx + width_steps)
        opt_type  = "call"

    short_k = strikes[short_idx]
    long_k  = strikes[long_idx]

    # --- fetch bid/ask -------------------------------------------------------
    short_px = chain.loc[
        (chain.strike == short_k) & (chain.option_type == opt_type), "bid"
    ].iloc[0]
    long_px = chain.loc[
        (chain.strike == long_k) & (chain.option_type == opt_type), "ask"
    ].iloc[0]

    credit   = round(short_px - long_px, 2)
    max_loss = round(abs(long_k - short_k) - credit, 2)

    return {
        "direction":    side,
        "option_type":  opt_type,
        "short_strike": short_k,
        "long_strike":  long_k,
        "credit":       credit,
        "max_loss":     max_loss,
        "qty":          1,   # Qty will be overwritten later by calculate_quantity()
    }

# ═════════════════════════════ 7. UI – controls ═════════════════════════════
st.sidebar.header("Controls")
st.title("Zero‑Gap Execution System")

# — Symbol & expiration pickers —
symbol = st.sidebar.selectbox(
    "Symbol", ["XSP", "SPX"],
    index=st.session_state.get("symbol_index", 0),
    key="symbol",
)

# Reset nudges when the user changes symbol or expiry
if st.session_state.get("last_symbol") != symbol:
    st.session_state.strangle_offset = 0
    st.session_state.spread_offset   = 0
    st.session_state["last_symbol"]  = symbol

try:
    expirations = get_expirations(symbol)
except Exception as e:
    st.sidebar.error(f"⚠ Could not load expirations: {e}")
    st.stop()
if not expirations:
    st.sidebar.error("⚠ No expirations returned; check data subscription.")
    st.stop()

expiration = st.sidebar.selectbox("Expiration", expirations, index=0, key="expiration")

if st.session_state.get("last_expiration") != expiration:
    st.session_state.strangle_offset = 0
    st.session_state.spread_offset   = 0
    st.session_state["last_expiration"] = expiration

# ═════════════════════════════ 8. Option Chain & Strategy Metrics ═══════════

# Slider defaults (loaded from session_state so changes persist across reruns)
iv_rank          = st.sidebar.slider("Current IV Rank (%)", 0, 100,
                                     value=st.session_state.get("iv_rank", 50),
                                     step=1, key="iv_rank")

skew_offset      = st.sidebar.slider("Strangle Skew Offset (OTM strikes)", 0, 2,
                                     value=st.session_state.get("skew_offset", 0),
                                     step=1, key="skew_offset")

max_spread_width = st.sidebar.slider("Max Spread Width (strike steps)", 1, 10,
                                     value=st.session_state.get("max_spread_width", 2),
                                     step=1, key="max_spread_width")

win_rate_target  = st.sidebar.slider("Expected spread win‑rate", 0.0, 1.0,
                                     value=st.session_state.get("win_rate_target", 0.60),
                                     step=0.01, key="win_rate_target")
spread_width_steps = st.sidebar.slider(
    "Credit‑Spread Width (strike steps)",    # label
    1, 10,                                   # min / max
    value=st.session_state.get("spread_width_steps", 1),
    step=1,
    key="spread_width_steps",
)

# --- Fetch chain & underlying ------------------------------------------------
try:
    chain = fetch_option_chain(symbol, expiration)
    if chain is None or chain.empty:
        raise RuntimeError("Option chain is missing or invalid.")
    st.sidebar.success("✓ Option chain fetched")
except Exception as e:
    st.sidebar.error(f"Chain fetch failed: {e}")
    st.stop()

try:
    underlying = get_underlying_price(symbol)
    if underlying is None or np.isnan(underlying):
        raise RuntimeError("Invalid underlying price.")
except Exception as e:
    st.sidebar.error(f"Failed to load underlying: {e}")
    st.stop()

# --- Build strategy ----------------------------------------------------------
try:
    # 1) Strangle
    strangle = build_strangle(
        chain,
        underlying,
        iv_rank=iv_rank,
        skew_offset=skew_offset,
        offset=st.session_state.strangle_offset,
    )

    # 2) Decide hedge side & build credit‑spread
    side = pick_spread_side(strangle, underlying)   # "below" or "above"

    spread = build_credit_spread_by_width(
        chain,
        underlying,
        side=side,                               # "below" or "above"
        short_offset=st.session_state.spread_offset,
        width_steps=spread_width_steps,
    )

    # 3) Recommended qty
    qty = calculate_quantity(strangle, spread)

except Exception as e:
    st.sidebar.error(f"Strategy error: {e}")
    st.stop()

# ═════════════════════════════ 9. Display Metrics & Layout ══════════════════

st.subheader("Underlying Price")
st.metric(label="", value=f"${underlying:.2f}")

# ───── Strangle Metrics header + buttons ─────
col_left, col_mid, col_right = st.columns([8, 1, 1])
with col_left:
    st.markdown("### Strangle Metrics")
with col_mid:
    if st.button("➖", key="strangle_minus"):
        st.session_state.strangle_offset -= 1
with col_right:
    if st.button("➕", key="strangle_plus"):
        st.session_state.strangle_offset += 1

# Strangle metrics grid
c1, c2 = st.columns(2)
c1.metric("Total Cost",      f"${strangle['debit']:.2f}")
c1.metric("Breakeven Low",   f"{strangle['breakeven_low']:.2f}")
c2.metric("Breakeven High",  f"{strangle['breakeven_high']:.2f}")
c2.metric("Max Loss",        f"${strangle['max_loss']:.2f}")

st.markdown("---")

# ───── Credit‑spread Metrics header + buttons ─────
col_left, col_mid, col_right = st.columns([8, 1, 1])
with col_left:
    st.markdown(f"### Credit Spread  ({'Bearish' if side == 'below' else 'Bullish'})")
with col_mid:
    if st.button("➖", key="spread_minus"):
        st.session_state.spread_offset -= 1
with col_right:
    if st.button("➕", key="spread_plus"):
        st.session_state.spread_offset += 1

# Spread metrics grid
c1, c2 = st.columns(2)
c1.metric("Short Strike", f"{spread['short_strike']}")
c1.metric("Long Strike",  f"{spread['long_strike']}")
c2.metric("Credit",       f"${spread['credit']:.2f}")
c2.metric("Qty",          f"{qty}")

c1, c2 = st.columns(2)
c1.metric("Tot Credit",   f"${spread['credit'] * qty:.2f}")
c2.metric("Max Loss",     f"${spread['max_loss'] * qty:.2f}")

# ═══════════════════════════ 11. P/L chart (leg table) ══════════════════════
from plot import render_plots

# Build the strangle legs DataFrame
strangle_df = pd.DataFrame(
    [
        {
            "strike":      strangle["put_strike"],
            "option_type": "put",
            "position":    1,
            "premium":     strangle["put_price"],
        },
        {
            "strike":      strangle["call_strike"],
            "option_type": "call",
            "position":    1,
            "premium":     strangle["call_price"],
        },
    ]
)

# Build the credit‑spread legs DataFrame
opt_type = "put" if side == "below" else "call"
short_leg_price = chain.query(
    "strike == @spread['short_strike'] and option_type == @opt_type"
)["bid"].iloc[0]
long_leg_price = chain.query(
    "strike == @spread['long_strike'] and option_type == @opt_type"
)["ask"].iloc[0]

spread_df = pd.DataFrame(
    [
        {
            "strike":      spread["short_strike"],
            "option_type": opt_type,
            "position":    -1,
            "premium":     short_leg_price,
        },
        {
            "strike":      spread["long_strike"],
            "option_type": opt_type,
            "position":    1,
            "premium":     long_leg_price,
        },
    ]
)

# Render the two interactive P/L charts
render_plots(strangle_df, spread_df, underlying)

# ───────────────────────────── Footer ───────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; font-size:0.75rem; color:gray; margin-top:2rem;">
      © 2025 Jacobi Ventures LLC. All rights reserved.  
      “Atomic Zero‑Gap Execution System" is proprietary and confidential.  
      Unauthorized copying, distribution, or modification is strictly prohibited.
    </div>
    """,
    unsafe_allow_html=True,
)
