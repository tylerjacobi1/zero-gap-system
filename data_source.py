import pandas as pd
import numpy as np
import datetime
from ib_insync import IB, Contract, util

# ─── Connection settings (override via Streamlit secrets) ───
HOST      = '127.0.0.1'
PORT      = 7497
CLIENT_ID = 123
_TIMEOUT  = 5.0

def _connect() -> IB:
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=_TIMEOUT)
    return ib

def get_expirations(symbol: str) -> list[str]:
    """
    Return sorted list of expiration dates (YYYY-MM-DD) for this option root.
    """
    ib      = _connect()
    root    = Contract(symbol=symbol, secType='OPT', exchange='SMART', currency='USD')
    details = ib.reqContractDetails(root)
    ib.disconnect()

    exps = {d.contract.lastTradeDateOrContractMonth for d in details}
    return sorted(
        datetime.datetime.strptime(e, '%Y%m%d').strftime('%Y-%m-%d')
        for e in exps
    )

def fetch_option_chain(symbol: str, expiration: str) -> pd.DataFrame:
    """
    Fetch one-expiration option chain via IBKR market data snapshots.
    Columns: strike, option_type ('call'|'put'), bid, ask, volume, underlying_price.
    """
    exp_tag = expiration.replace('-', '')
    ib      = _connect()
    root    = Contract(
        symbol=symbol,
        secType='OPT',
        exchange='SMART',
        currency='USD',
        lastTradeDateOrContractMonth=exp_tag,
    )
    details = ib.reqContractDetails(root)

    rows = []
    for det in details:
        c      = det.contract
        ticker = ib.reqMktData(c, '', False, False)
        ib.sleep(0.1)
        if ticker.bid is None or ticker.ask is None:
            ib.cancelMktData(ticker)
            continue
        rows.append({
            'strike':      c.strike,
            'option_type': c.right.lower(),
            'bid':         ticker.bid,
            'ask':         ticker.ask,
            'volume':      getattr(ticker, 'volume', 0),
        })
        ib.cancelMktData(ticker)

    ib.disconnect()
    df = pd.DataFrame(rows)
    df['underlying_price'] = get_underlying_price(symbol)
    return df

def get_underlying_price(symbol: str) -> float:
    """
    Fetch the last price of the underlying via IBKR.
    """
    ib     = _connect()
    c      = Contract(symbol=symbol, secType='STK', exchange='SMART', currency='USD')
    tic    = ib.reqMktData(c, '', False, False)
    ib.sleep(0.1)
    price  = tic.last or tic.close or np.nan
    ib.cancelMktData(tic)
    ib.disconnect()
    return float(price)

def compute_iv_change(symbol: str, interval: str = '1 min') -> float:
    """
    Compute the % change in IBKR's HISTORICAL_VOLATILITY feed over the last 2 bars.
    """
    ib   = _connect()
    volc = Contract(symbol=symbol, secType='IND', exchange='CBOE', currency='USD')
    bars = ib.reqHistoricalData(
        contract=volc,
        endDateTime='',
        durationStr='2 D',
        barSizeSetting=interval,
        whatToShow='HISTORICAL_VOLATILITY',
        useRTH=False,
        formatDate=1,
    )
    ib.disconnect()
    df = util.df(bars)
    if len(df) < 2 or df.close.iloc[-2] == 0:
        return 0.0
    prev, last = df.close.iloc[-2], df.close.iloc[-1]
    return float((last - prev) / prev * 100)

def get_price_history(symbol: str, interval: str = '1 min', periods: int = 20) -> pd.Series:
    """
    Fetch recent underlying midpoint prices as a pd.Series indexed by timestamp.
    """
    ib   = _connect()
    stk  = Contract(symbol=symbol, secType='STK', exchange='SMART', currency='USD')
    bars = ib.reqHistoricalData(
        contract=stk,
        endDateTime='',
        durationStr=f'{periods * int(interval.split()[0])} S',
        barSizeSetting=interval,
        whatToShow='MIDPOINT',
        useRTH=False,
        formatDate=1,
    )
    ib.disconnect()

    df = util.df(bars)
    if df.empty or 'close' not in df:
        return pd.Series(dtype=float)
    df['timestamp'] = pd.to_datetime(df.date)
    df.set_index('timestamp', inplace=True)
    return df['close']

def fetch_realtime_bars_ibkr(
    symbol: str,
    duration: str = "1800 S",
    barSize: str = "1 min",
    whatToShow: str = "MIDPOINT",
    useRTH: bool = False,
) -> pd.DataFrame:
    """
    Pull 1-min OHLC+volume bars from IBKR. Returns empty df on failure.
    """
    try:
        ib   = _connect()
        idx  = Contract(symbol=symbol, secType="IND", exchange="CBOE", currency="USD")
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
