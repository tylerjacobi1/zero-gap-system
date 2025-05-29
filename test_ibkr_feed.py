# test_ibkr_feed.py
from ib_insync import IB, Contract
import time

# ── CONFIG ──
IB_HOST   = "127.0.0.1"
IB_PORT   = 7497      # or 4001 for IB Gateway
IB_CLIENT = 999       # pick any unused clientId

def main():
    ib = IB()

    print(f"→ Connecting to {IB_HOST}:{IB_PORT} as clientId={IB_CLIENT}…")
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT, timeout=5)

    # build your index contract
    idx = Contract(
        symbol="XSP",
        secType="IND",
        exchange="CBOE",
        currency="USD",
    )

    # qualify to ensure it’s recognized
    ib.qualifyContracts(idx)

    # request a 1-shot market-data snapshot
    ticker = ib.reqMktData(idx, "", False, False)
    # give TWS/Gateway a couple ticks to populate bid/ask
    for _ in range(5):
        if ticker.bid is not None and ticker.ask is not None:
            break
        time.sleep(0.2)

    print(f"XSP index → bid={ticker.bid}, ask={ticker.ask}")

    # clean up
    ib.cancelMktData(ticker)
    ib.disconnect()

if __name__ == "__main__":
    main()
