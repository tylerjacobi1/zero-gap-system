# test_ibkr_index.py
from ib_insync import IB, Index, Contract
import time

IB_HOST, IB_PORT, IB_CLIENT = "127.0.0.1", 7497, 999  # use an unused clientId

def main():
    ib = IB()
    print(f"→ Connecting to {IB_HOST}:{IB_PORT} as clientId={IB_CLIENT}…")
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT, timeout=5)

    # 1) Switch to “live” market-data (1=live, 2=frozen, 3=delayed)
    ib.reqMarketDataType(1)

    # 2) Get the conId
    idx = Index(symbol="SPX", exchange="CBOE", currency="USD")
    ib.qualifyContracts(idx)
    print("Contract details:", idx)

    # 3) Build a bare Contract by conId (necessary for streaming)
    idx = Contract(conId=idx.conId)

    # 4) Request a streaming feed (snapshot=False)
    ticker = ib.reqMktData(idx, "", snapshot=False)  

    # 5) Wait up to 5 seconds for bid/ask
    for _ in range(50):
        if ticker.bid is not None and ticker.ask is not None:
            break
        time.sleep(0.1)

    print(f"SPX index → bid={ticker.bid}, ask={ticker.ask}")

    ib.cancelMktData(ticker)
    ib.disconnect()

if __name__ == "__main__":
    main()
