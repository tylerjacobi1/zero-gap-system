# 🌀 Zero Gap System Tool

[1] User Inputs & Controls (Streamlit UI)
    ├─ Symbol (e.g. SPX)
    ├─ Expiration date
    ├─ Underlying price (auto-fetched or manual override)
    ├─ Expected spread win-rate p (slider) → defines threshold = p/(1–p)
    ├─ ε (price offset from midpoint)
    └─ quantity override (optional)
      ↓
[2] Data Gathering
    ├─ Fetch live option chain from Tradier
    └─ Build IBKR Contracts & qualify (for market data & depth)
      ↓
[3] Strangle Construction
    ├─ pick ATM strikes (nearest strikes around underlying)
    ├─ compute:
    │    • put_price + call_price → total_cost
    │    • breakeven_low = put_strike – total_cost
    │    • breakeven_high = call_strike + total_cost
    │    • max_loss = total_cost
    └─ display strangle metrics (cost, BEs, risk)
      ↓
[4] Credit-Spread Pairing
    ├─ pick side (“below” if lower BE closer, else “above”)
    ├─ generate candidate ATM credit spreads (short & long strikes)
    ├─ compute for each:
    │    • credit = short.bid – long.ask
    │    • max_loss = width – credit
    ├─ FILTER: keep only spreads where  
    │      max_loss ≤ credit × threshold
    │    (i.e. C/L ≥ (1–p)/p)
    └─ size your basket:  
         spread_qty = ceil(strangle.max_loss / spread.credit)
      ↓
[5] Liquidity Gate (Book-Driven Timing)
    ├─ subscribe to DOM for both short & long contracts
    ├─ wait until top-of-book size ≥ spread_qty on both legs
    └─ unsubscribe from DOM
      ↓
[6] Price Computation
    ├─ reqMktData for each leg → get bid & ask
    ├─ mid = (bid+ask)/2
    ├─ baseline: short_limit = mid + ε; long_limit = mid – ε
    └─ if depth at mid insufficient on one leg → asymmetric wiggle:
         – thin_short? bump short_limit + 1 tick  
         – thin_long?  bump long_limit  – 1 tick
      ↓
[7] Display Recommended Spread
    ├─ Show strikes, credit, max_loss, spread_qty  
    ├─ Show computed short_limit & long_limit  
    └─ Let user tweak qty or ε if desired
      ↓
[8] Order Execution (IBKR via ib_insync)
    ├─ Build two Option contracts (strangle legs)
    ├─ Build two Option contracts (spread legs)
    ├─ Connect IB() once (TWS or Gateway)
    ├─ 1) Place Strangle combo:
    │     • legs = [long Call, long Put]  
    │     • LimitOrder(LMT, qty=1, price=total_cost)  
    │     • transmit=True  
    │     → if fail: abort, st.stop()
    ├─ 2) Place Spread combo:
    │     • legs = [short leg, long leg]  
    │     • LimitOrder(SELL/BUY, qty=spread_qty, price=limits…)  
    │     • transmit grouping (leg1 False, leg2 True)  
    │     → if fail:  
    │          cancel strangle order (order1.order), st.stop()
    └─ Success: both combos in, show IDs & monitoring
      ↓
[9] Post-Trade Monitoring
    ├─ Optionally subscribe to orderStatus/fill callbacks  
    ├─ If one combo leg ever fills without its partner,  
    │    • **do not** send offsetting close (avoids PDT)  
    │    • only cancel unfilled working orders  
    │    • alert user for manual management  
    └─ Log all fills/orders for audit & P&L
      ↓
[10] Risk & Reporting
    ├─ Show P/L charts (strangle, spread, net)  
    ├─ Log trades to CSV/DB  
    └─ Remind or schedule follow-up tasks (via automations)

―――――――――――――――――――――――――――――――――――――――――――  
All of these steps combine:  
- **Pairing logic** (steps 3 – 4) keeps your risk perfectly hedged.  
- **Liquidity & pricing tactics** (5 – 6) maximize dual-leg fill probability.  
- **Execution & rollback safety** (8 – 9) ensure you never end up over-exposed or hit with PDT. 
