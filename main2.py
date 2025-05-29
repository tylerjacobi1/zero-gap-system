# 1) Asyncio guard for ib_insync / eventkit
import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# 2) Core imports
import random
import base64, math, os
import streamlit as st            # <- must come *before* any st.session_state usage
import numpy as np, pandas as pd
from ib_insync import IB, Contract, ComboLeg, LimitOrder

# 2.1) One client-ID per session
if 'ibkr_client_id' not in st.session_state:
    st.session_state['ibkr_client_id'] = random.randint(10000, 99999)

# 3) UI-state defaults
for k, default in [
    ("strangle_offset",    0),
    ("spread_offset",      0),
    ("iv_rank",           50),
    ("skew_offset",        0),
    ("max_spread_width",   2),
    ("win_rate_target",  0.6),
    ("spread_width_steps", 1),
]:
    st.session_state.setdefault(k, default)

#Sidebar Atomic Logo
import base64

try:
    with open("atomic.png", "rb") as img:
        b64 = base64.b64encode(img.read()).decode()
    st.sidebar.markdown(
        f"""
        <style>
          [data-testid="stSidebar"] > div:first-child {{ position: relative; }}
          [data-testid="stSidebar"] > div:first-child::before {{
            content: "";
            position: absolute;
            top: 16px;
            left: -10px;
            width: 120px;
            height: 48px;
            background: url(data:image/png;base64,{b64}) no-repeat center/contain;
          }}
        </style>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    pass

# 4) Secrets & environment
def _secret(key: str, default: str = ""):
    return st.secrets.get(key) or os.getenv(key, default)

TRADIER_API_KEY = _secret("TRADIER_API_KEY")
IBKR_HOST       = _secret("IBKR_HOST", "127.0.0.1")
IBKR_PORT       = int(_secret("IBKR_PORT", "7497"))
IBKR_CLIENT_ID  = int(_secret("IBKR_CLIENT_ID", "999"))

# 5) Live vs Delayed data import (uses the one session‐wide client ID)
from ib_insync import IB

ib_test = IB()

async def onError(reqId, code, msg, *args):
    st.sidebar.error(f"[ERR {code}] {msg}")
ib_test.errorEvent += onError

try:
    cid = st.session_state['ibkr_client_id']
    st.sidebar.info(f"[INFO] Testing live IBKR feed (Client ID: {cid})…")
    ib_test.connect(IBKR_HOST, IBKR_PORT, clientId=cid, timeout=3)
    ib_test.reqCurrentTime()
    USE_DELAYED = False
except Exception:
    USE_DELAYED = True
    st.sidebar.warning("⏱ Live IBKR feed unavailable — using delayed Tradier data.")
finally:
    if ib_test.isConnected():
        ib_test.disconnect()

if USE_DELAYED:
    from data_source_tradier import (
        get_expirations,
        fetch_option_chain,
        get_underlying_price,
        get_price_history,
        compute_iv_change,
    )
    def fetch_realtime_bars_ibkr(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()
else:
    # IB gives you everything except expirations, so import its funcs…
    from data_source_ibkr import (
        fetch_option_chain,
        get_underlying_price,
        fetch_realtime_bars_ibkr,
        get_price_history,
        compute_iv_change,
    )
    # …and pull expirations from Tradier instead
    from data_source_tradier import get_expirations

# 6) Strategy helpers
def pick_atm_strikes(chain: pd.DataFrame, underlying: float) -> tuple[float,float]:
    strikes = sorted(chain["strike"].unique())
    idx     = np.searchsorted(strikes, underlying)
    return strikes[max(idx-1,0)], strikes[min(idx,len(strikes)-1)]

def build_strangle(chain: pd.DataFrame, underlying: float,
                   iv_rank: float=100, skew_offset: int=0, offset: int=0) -> dict:
    put_str, call_str = pick_atm_strikes(chain, underlying)
    row_put  = chain[(chain.strike==put_str)  & (chain.option_type=="put") ].iloc[0]
    row_call = chain[(chain.strike==call_str) & (chain.option_type=="call")].iloc[0]
    put_px   = (row_put.get("bid",0)  + row_put.get("ask",0))  / 2
    call_px  = (row_call.get("bid",0) + row_call.get("ask",0)) / 2
    debit    = put_px + call_px
    return {
        "put_strike":     put_str,
        "call_strike":    call_str,
        "put_price":      put_px,
        "call_price":     call_px,
        "debit":          round(debit,2),
        "breakeven_low":  round(put_str   - debit,2),
        "breakeven_high": round(call_str  + debit,2),
        "max_loss":       round(debit,2),
    }

def pick_spread_side(metrics: dict, underlying: float) -> str:
    lo = underlying - metrics["breakeven_low"]
    hi = metrics["breakeven_high"] - underlying
    return "below" if lo < hi else "above"

def build_credit_spread_by_width(chain: pd.DataFrame, underlying: float,
                                 side: str, short_offset: int=0, width_steps: int=1) -> dict:
    opt_type = "put" if side=="below" else "call"
    strikes  = sorted(chain[chain.option_type==opt_type]["strike"].unique())
    if not strikes:
        raise RuntimeError("No strikes available for spread")
    atm_idx = np.searchsorted(strikes, underlying)
    if side=="below":
        s_idx = max(0, atm_idx-1-short_offset)
        l_idx = max(0, s_idx-width_steps)
    else:
        s_idx = min(len(strikes)-1, atm_idx+short_offset)
        l_idx = min(len(strikes)-1, s_idx+width_steps)
    s, l = strikes[s_idx], strikes[l_idx]
    row_s = chain[(chain.strike==s) & (chain.option_type==opt_type)].iloc[0]
    row_l = chain[(chain.strike==l) & (chain.option_type==opt_type)].iloc[0]
    credit   = max(0, row_s.get("bid",0) - row_l.get("ask",0))
    max_loss = abs(l-s) - credit
    qty      = max(1, math.ceil(metrics["max_loss"]/credit)) if credit>0 else 1
    return {"short_strike":s, "long_strike":l,
            "credit":round(credit,2), "max_loss":round(max_loss,2),
            "option_type":opt_type, "qty":qty}

def calculate_quantity(metrics: dict, spread: dict) -> int:
    return 1 if spread["credit"]<=0 else math.ceil(metrics["max_loss"]/spread["credit"])

# 7) UI – Controls
st.sidebar.header("Controls")
st.title("Zero-Gap Execution System")

# Underlying symbol & expiration
symbol     = st.sidebar.text_input("Underlying Symbol", value="XSP").upper()
as_of      = st.sidebar.date_input("As-of Date", value=pd.Timestamp.today().date())
if st.session_state.get("last_symbol") != symbol:
    st.session_state.update({"strangle_offset":0, "spread_offset":0})
    st.session_state["last_symbol"] = symbol

try:
    expirations = get_expirations(symbol)
except Exception as e:
    st.sidebar.error(f"Could not load expirations: {e}")
    st.stop()
if not expirations:
    st.sidebar.error("No expirations returned; check data source.")
    st.stop()
expiration = st.sidebar.selectbox("Expiration", expirations, key="expiration")
if st.session_state.get("last_expiration") != expiration:
    st.session_state.update({"strangle_offset":0, "spread_offset":0})
    st.session_state["last_expiration"] = expiration

# Historical mini-charts
try:
    hist = get_price_history(symbol, as_of - pd.Timedelta(days=30), duration=30)
    st.sidebar.line_chart(pd.DataFrame({"Close":hist["close"]}).set_index(hist["date"]))
    ivchg = compute_iv_change(symbol, as_of)
    st.sidebar.line_chart(pd.DataFrame({"IV Δ":ivchg["iv_change"]}).set_index(ivchg["date"]))
    if not USE_DELAYED:
        bars = fetch_realtime_bars_ibkr(symbol, duration=1)
        st.sidebar.line_chart(pd.DataFrame({"Last":bars["close"]}).set_index(bars["time"]))
except Exception:
    pass

# Strategy sliders
iv_rank            = st.sidebar.slider("IV Rank (%)",        0,100,   value=st.session_state.iv_rank,            key="iv_rank")
skew_offset        = st.sidebar.slider("Strangle Skew Offset",0,2,     value=st.session_state.skew_offset,         key="skew_offset")
max_spread_width   = st.sidebar.slider("Max Spread Width",   1,10,    value=st.session_state.max_spread_width,    key="max_spread_width")
win_rate_target    = st.sidebar.slider("Spread Win-Rate",    0.0,1.0,  value=st.session_state.win_rate_target,    key="win_rate_target")
spread_width_steps = st.sidebar.slider("Spread Width (steps)",1,10,    value=st.session_state.spread_width_steps, key="spread_width_steps")

# 8) Fetch chain & underlying
try:
    chain = fetch_option_chain(symbol, expiration)
    if chain.empty:
        raise RuntimeError("Empty option chain")
    st.sidebar.success("[OK] Option chain fetched")
except Exception as e:
    st.sidebar.error(f"Chain fetch failed: {e}")
    st.stop()

try:
    underlying = get_underlying_price(symbol)
    if underlying is None or np.isnan(underlying):
        raise RuntimeError("Invalid underlying price")
except Exception as e:
    st.sidebar.error(f"Underlying fetch failed: {e}")
    st.stop()

# 9) Build strategy
try:
    metrics = build_strangle(chain, underlying,
                             iv_rank=iv_rank,
                             skew_offset=skew_offset,
                             offset=st.session_state.strangle_offset)
    side    = pick_spread_side(metrics, underlying)
    spread  = build_credit_spread_by_width(
        chain, underlying,
        side=side,
        short_offset=st.session_state.spread_offset,
        width_steps=spread_width_steps
    )
    qty     = calculate_quantity(metrics, spread)
except Exception as e:
    st.sidebar.error(f"Strategy error: {e}")
    st.stop()

# 10) Display metrics
st.subheader("Underlying Price")
st.metric("", f"${underlying:.2f}")

col_left, col_mid, col_right = st.columns([8,1,1])
with col_left:
    st.markdown("### Strangle Metrics")
with col_mid:
    if st.button("➖", key="strangle_minus"):
        st.session_state.strangle_offset -= 1
with col_right:
    if st.button("➕", key="strangle_plus"):
        st.session_state.strangle_offset += 1

c1, c2 = st.columns(2)
c1.metric("Total Cost",     f"${metrics['debit']:.2f}")
c1.metric("Breakeven Low",  f"{metrics['breakeven_low']:.2f}")
c2.metric("Breakeven High", f"{metrics['breakeven_high']:.2f}")
c2.metric("Max Loss",       f"${metrics['max_loss']:.2f}")

st.markdown("---")

col_left, col_mid, col_right = st.columns([8,1,1])
with col_left:
    st.markdown(f"### Credit Spread ({'Bearish' if side=='below' else 'Bullish'})")
with col_mid:
    if st.button("➖", key="spread_minus"):
        st.session_state.spread_offset -= 1
with col_right:
    if st.button("➕", key="spread_plus"):
        st.session_state.spread_offset += 1

c1, c2 = st.columns(2)
c1.metric("Short Strike", f"{spread['short_strike']}")
c1.metric("Long Strike",  f"{spread['long_strike']}")
c2.metric("Credit",       f"${spread['credit']:.2f}")
c2.metric("Qty",          f"{qty}")

c1, c2 = st.columns(2)
c1.metric("Total Credit",   f"${spread['credit'] * qty:.2f}")
c2.metric("Total Max Loss", f"${spread['max_loss'] * qty:.2f}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Section 11: IBKR send order (audited & fixed)
# ─────────────────────────────────────────────────────────────────────────────

import time, random
from ib_insync import IB, Contract, ComboLeg, LimitOrder

# Sidebar: IBKR connection + send button
st.sidebar.markdown("### IBKR Connection")
use_tws = st.sidebar.radio("Connect via", ["TWS (7497)", "IB Gateway (4001)"], index=1) == "TWS (7497)"
HOST, PORT = IBKR_HOST, (7497 if use_tws else 4001)
st.sidebar.info(f"Routing via {'TWS' if use_tws else 'IB Gateway'} on {PORT}")
st.sidebar.caption("All combo orders use LIMIT per IBKR rules.")

if st.sidebar.button("Send to IBKR", key="send_to_ibkr"):
    # Single IB instance per click
    ib = IB()

    # ① Async error handler (ignore farm-status codes)
    async def handleErr(reqId, code, msg, *args):
        if code < 2000:
            st.sidebar.error(f"[ERR {code}] {msg}")
    ib.errorEvent += handleErr

    # ② One clientId/session → connect
    if 'ibkr_client_id' not in st.session_state:
        st.session_state['ibkr_client_id'] = random.randint(10000, 99999)
    cid = st.session_state['ibkr_client_id']

    connected = False
    try:
        st.sidebar.info(f"[INFO] Connecting to IBKR (Client ID: {cid})…")
        ib.connect(HOST, PORT, clientId=cid, timeout=10)
        st.sidebar.success("[OK] Connected")
        connected = True
    except Exception as e:
        st.sidebar.error(f"[ERR] Connection failed: {e}")

    # ③ Only if connected do we build & send the order
    if connected:
        try:
            # — Format expiry
            expiry = expiration.strftime("%Y%m%d") if hasattr(expiration, "strftime") else str(expiration).replace("-", "")

            # — Build & qualify legs
            EXCH = "SMART"
            def mkOpt(strk, rt):
                return Contract(
                    symbol=symbol, secType="OPT", exchange=EXCH,
                    currency="USD", lastTradeDateOrContractMonth=expiry,
                    strike=strk, right=rt, multiplier="100"
                )

            put   = mkOpt(metrics["put_strike"],  "P")
            call  = mkOpt(metrics["call_strike"], "C")
            short = mkOpt(spread["short_strike"], spread["option_type"][0].upper())
            long_ = mkOpt(spread["long_strike"],  spread["option_type"][0].upper())

            st.sidebar.info("[INFO] Qualifying contracts…")
            put, call, short, long_ = ib.qualifyContracts(put, call, short, long_)
            st.sidebar.success("[OK] Contracts qualified")

            # — Assemble BAG
            combo = Contract(secType="BAG", symbol=symbol, exchange=EXCH, currency="USD")
            combo.comboLegs = [
                ComboLeg(conId=put.conId,   ratio=1, action="BUY",  exchange=EXCH),
                ComboLeg(conId=call.conId,  ratio=1, action="BUY",  exchange=EXCH),
                ComboLeg(conId=short.conId, ratio=qty, action="SELL", exchange=EXCH),
                ComboLeg(conId=long_.conId, ratio=qty, action="BUY",  exchange=EXCH),
            ]

            # — Compute limit price & direction
            net = metrics["debit"] - (spread["credit"] * qty)
            price = round(abs(net) + (0.01 if net > 0 else -0.01), 2)
            price = max(price, 0.01)
            act   = "BUY" if net > 0 else "SELL"

            order = LimitOrder(
                action     = act,
                totalQuantity = 1,
                lmtPrice   = price,
                tif        = 'GTC'         # ← Good-Till-Cancelled (required for combos)
            )

            # — Place & await final status
            trade = ib.placeOrder(combo, order)
            t0    = time.time()
            while trade.orderStatus.status in ("PendingSubmit","ApiPending") and time.time() - t0 < 10:
                ib.sleep(0.5)

            final = trade.orderStatus.status
            if final in ("PreSubmitted","Submitted"):
                st.sidebar.success(f"[OK] Order accepted: {final} (ID {trade.order.orderId})")
            else:
                st.sidebar.error(f"[ERR] Final status: {final}")

        except Exception as orderErr:
            st.sidebar.error(f"[ERR] Order failed: {orderErr}")

        finally:
            if ib.isConnected():
                ib.disconnect()
                st.sidebar.info("[INFO] Disconnected from IBKR")
# ─────────────────────────────────────────────────────────────────────────────

                
# ────────────── 12) P/L charts & leg tables
from plot import render_plots

strangle_df = pd.DataFrame([
    {"strike":metrics["put_strike"],  "option_type":"put",  "position":1, "premium":metrics["put_price"]},
    {"strike":metrics["call_strike"], "option_type":"call", "position":1, "premium":metrics["call_price"]},
])
opt_type    = "put" if side=="below" else "call"
short_price = chain.query("strike==@spread['short_strike'] and option_type==@opt_type")["bid"].iloc[0]
long_price  = chain.query("strike==@spread['long_strike']  and option_type==@opt_type")["ask"].iloc[0]
spread_df   = pd.DataFrame([
    {"strike":spread["short_strike"], "option_type":opt_type, "position":-1, "premium":short_price},
    {"strike":spread["long_strike"],  "option_type":opt_type, "position": 1, "premium":long_price},
])

render_plots(strangle_df, spread_df, underlying)

# ─────────────────────────────────────────────────────────────────────────────
# 13) Footer (moved to absolute bottom, with relaxed spacing)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
      text-align: center;
      font-size: 0.85rem;
      color: #999;
      line-height: 1.6;
      margin-top: 3rem;
      margin-bottom: 1.5rem;
    ">
      <p>© 2025 Jacobi Ventures LLC. All rights reserved.“Atomic Zero-Gap Execution System” is proprietary and confidential. Unauthorized copying or modification is prohibited.</p>
    </div>
    """,
    unsafe_allow_html=True
)
