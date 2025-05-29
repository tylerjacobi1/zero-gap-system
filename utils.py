import math
import numpy as np
import pandas as pd


def calculate_strangle_metrics(options: list[dict], filters: dict) -> list[dict]:
    """
    Generate potential strangles based on strikes offset by:
      • d = 1…max_dist
      • if filters['iv_rank'] < 30, use d + filters['skew_offset'] instead of d
    Returns a list of dicts with keys:
      - 'legs': [('put', opt), ('call', opt)]
      - 'debit': total cost
    """
    if not options:
        return []
    underlying = options[0].get("underlying_price")
    if underlying is None:
        raise ValueError("Missing 'underlying_price' in option data.")

    max_dist = int(filters.get("strangle_distance", 1))
    iv_rank = filters.get("iv_rank", 100)
    skew_offset = filters.get("skew_offset", 0)
    strikes = sorted({o["strike"] for o in options})

    strangles: list[dict] = []
    for d in range(1, max_dist + 1):
        eff_d = d + skew_offset if iv_rank < 30 else d
        put_target, call_target = underlying - eff_d, underlying + eff_d

        # find nearest strikes
        put_strike = min(strikes, key=lambda x: abs(x - put_target))
        call_strike = min(strikes, key=lambda x: abs(x - call_target))
        if put_strike >= underlying or call_strike <= underlying:
            continue

        put_opt = next(
            (
                o
                for o in options
                if o["option_type"] == "put" and o["strike"] == put_strike
            ),
            None,
        )
        call_opt = next(
            (
                o
                for o in options
                if o["option_type"] == "call" and o["strike"] == call_strike
            ),
            None,
        )
        if not put_opt or not call_opt:
            continue

        debit = put_opt["ask"] + call_opt["ask"]
        if debit <= filters.get("strangle_cost", float("inf")):
            strangles.append(
                {
                    "legs": [("put", put_opt), ("call", call_opt)],
                    "debit": round(debit, 2),
                }
            )

    return strangles


def calculate_spread_metrics(options: list[dict], filters: dict) -> dict:
    """
    Find the best credit spread within filters.
    Returns a dict with keys: 'legs', 'credit', 'max_loss', 'option_type'.
    """
    candidates = []
    for o in options:
        typ = o["option_type"]
        if typ == "put" and not filters.get("filter_puts", True):
            continue
        if typ == "call" and not filters.get("filter_calls", True):
            continue

        for l in options:
            if l["option_type"] != typ:
                continue
            if typ == "put" and l["strike"] >= o["strike"]:
                continue
            if typ == "call" and l["strike"] <= o["strike"]:
                continue

            credit = o["bid"] - l["ask"]
            width = abs(o["strike"] - l["strike"])
            max_loss = width - credit
            mid = (o["bid"] + o["ask"]) / 2
            slippage = (o["ask"] - o["bid"]) / mid if mid > 0 else float("inf")

            if (
                credit >= filters.get("min_credit", 0)
                and credit <= filters.get("max_credit", float("inf"))
                and slippage <= filters.get("max_slippage", 1.0)
            ):
                candidates.append(
                    {
                        "option_type": typ,
                        "short": o,
                        "long": l,
                        "credit": round(credit, 2),
                        "max_loss": round(max_loss, 2),
                    }
                )

    if not candidates:
        return {}

    best = max(candidates, key=lambda c: c["credit"])
    return {
        "legs": [
            (f"short_{best['option_type']}", best["short"]),
            (f"long_{best['option_type']}", best["long"]),
        ],
        "credit": best["credit"],
        "max_loss": best["max_loss"],
        "option_type": best["option_type"],
    }


def multiply_spread_to_cover_loss(spread: dict, debit: float) -> list[dict]:
    """
    Return a list of copies of `spread` such that total credit >= debit.
    """
    credit = spread.get("credit", 0)
    if credit <= 0:
        return [spread]
    qty = math.ceil(debit / credit)
    return [spread.copy() for _ in range(qty)]


def check_leg_liquidity(
    opt: dict, spread_threshold: float = 0.10, min_volume: int = 10
) -> bool:
    """
    Returns True if
      1) (ask - bid) <= spread_threshold * mid-price
      2) volume >= min_volume
    """
    bid, ask, vol = opt.get("bid"), opt.get("ask"), opt.get("volume", 0)
    if bid is None or ask is None or vol < min_volume:
        return False
    mid = (bid + ask) / 2
    if mid <= 0 or (ask - bid) > spread_threshold * mid:
        return False
    return True


def compute_premium_efficiency_ratio(
    credit: float, qty: int, max_loss: float, strangle_cost: float
) -> float:
    """
    R = total_credit / (total_spread_loss + strangle_cost)
    """
    total_credit = credit * qty
    total_loss = max_loss * qty
    denom = total_loss + strangle_cost
    return float("inf") if denom <= 0 else total_credit / denom


def compute_strangle_gamma(legs: list[tuple[str, dict]], underlying: float) -> float:
    """
    Sum absolute gamma of each leg (expects each opt dict has 'gamma').
    """
    return sum(abs(opt.get("gamma", 0.0)) for _, opt in legs)


def compute_realized_vol(prices: pd.Series, window: int = 14) -> float:
    """
    Annualized realized vol: std(log returns) * sqrt(252*390).
    """
    if prices is None or len(prices) < 2:
        return 0.0
    rets = np.log(prices / prices.shift(1)).dropna()
    return float(rets.std() * np.sqrt(252 * 390))


def generate_strangle_curve(metrics: dict, underlying: float) -> pd.DataFrame:
    """
    P/L across a range of underlying prices for the strangle.
    """
    prices = np.linspace(underlying * 0.5, underlying * 1.5, 200)
    pnl = [
        max(metrics["put_strike"] - p, 0)
        - metrics["put_price"]
        + max(p - metrics["call_strike"], 0)
        - metrics["call_price"]
        for p in prices
    ]
    return pd.DataFrame({"Price": prices, "P/L": pnl})


def generate_spread_curve(spread: dict, qty: int, underlying: float) -> pd.DataFrame:
    """
    P/L across a range of underlying prices for the credit spread.
    """
    prices = np.linspace(underlying * 0.5, underlying * 1.5, 200)
    pnl = []
    for p in prices:
        if spread["option_type"] == "put":
            se = max(spread["legs"][0][1]["strike"] - p, 0)
            le = max(spread["legs"][1][1]["strike"] - p, 0)
        else:
            se = max(p - spread["legs"][0][1]["strike"], 0)
            le = max(p - spread["legs"][1][1]["strike"], 0)
        pnl.append((se - le + spread["credit"]) * qty)
    return pd.DataFrame({"Price": prices, "P/L": pnl})


def pick_spread_side(metrics: dict, underlying: float) -> str:
    """
    Return "below" for put-side, otherwise "above".
    """
    d_put = underlying - metrics["breakeven_low"]
    d_call = metrics["breakeven_high"] - underlying
    return "below" if d_put < d_call else "above"

def get_rr_spread(
    options: list[dict],
    metrics: dict,
    direction: str,
    underlying: float,
    tol: float = 0.01,
    threshold: float = 1.5,
    iv_rank: float = 100.0,
    max_spread_width: int | None = None,
    skew_bias: float = 0.0,
) -> dict:
    """
    Pick a single optimal credit spread:
      1) try a 1-step “ATM” spread anchored at the strangle strike
      2) prefer credit ≈ width/2 within tol + EV-breakeven credit*threshold ≥ width
      3) if iv_rank ≤ 70, enforce width ≤ max_spread_width * strike_step
      4) apply skew_bias to tilt call vs put
    Fallbacks to highest-credit adjacent spread or a dummy zero-credit spread.
    """
    opt = "put" if direction == "below" else "call"
    strikes = sorted({o["strike"] for o in options if o["option_type"] == opt})
    if not strikes:
        return {}

    # compute strike step
    step = None
    if len(strikes) > 1:
        deltas = [j - i for i, j in zip(strikes, strikes[1:])]
        positive = [d for d in deltas if d > 0]
        step = min(positive) if positive else None

    # 1) ATM-anchored one-step spread at the strangle strike
    anchor = metrics["call_strike"] if direction == "above" else metrics["put_strike"]
    if step is not None:
        try:
            i0 = strikes.index(anchor)
            j0 = i0 + 1 if direction == "above" else i0 - 1
            if 0 <= j0 < len(strikes):
                s0, l0 = strikes[i0], strikes[j0]
                row_s = next(o for o in options if o["strike"] == s0 and o["option_type"] == opt)
                row_l = next(o for o in options if o["strike"] == l0 and o["option_type"] == opt)
                credit0 = row_s["bid"] - row_l["ask"]
                width0 = abs(l0 - s0)
                # EV-breakeven filter
                if credit0 > 0 and credit0 * threshold >= width0:
                    return {
                        "direction":    direction,
                        "short_strike": s0,
                        "long_strike":  l0,
                        "credit":       round(credit0, 2),
                        "max_loss":     round(width0 - credit0, 2),
                        "option_type":  opt,
                    }
        except ValueError:
            # anchor not found or lookup failed → fall through
            pass

    # 2) build search sequence around ideal breakeven index
    be = metrics["breakeven_low"] if direction == "below" else metrics["breakeven_high"]
    idx0 = np.searchsorted(strikes, be)
    seq = [idx0] + [i for d in range(1, len(strikes)) for i in (idx0 - d, idx0 + d)]

    candidates: list[dict] = []
    for i in seq:
        if i < 0 or i >= len(strikes):
            continue
        j = i - 1 if direction == "below" else i + 1
        if j < 0 or j >= len(strikes):
            continue

        s, l = strikes[i], strikes[j]
        width = abs(l - s)

        # width-compression guard
        if (
            step is not None
            and iv_rank <= 70
            and max_spread_width is not None
            and width / step > max_spread_width
        ):
            continue

        row_s = next(o for o in options if o["strike"] == s and o["option_type"] == opt)
        row_l = next(o for o in options if o["strike"] == l and o["option_type"] == opt)
        credit = row_s["bid"] - row_l["ask"]
        if credit <= 0:
            continue

        # EV-breakeven
        if credit * threshold < width:
            continue

        # width/2 tolerance
        if abs(credit - width / 2) <= tol:
            candidates.append({
                "direction":    direction,
                "short_strike": s,
                "long_strike":  l,
                "credit":       round(credit, 2),
                "max_loss":     round(width - credit, 2),
                "option_type":  opt,
            })

    # 3) fallback: any positive credit spread passing width guard
    if not candidates:
        for i in seq:
            if i < 0 or i >= len(strikes):
                continue
            j = i - 1 if direction == "below" else i + 1
            if j < 0 or j >= len(strikes):
                continue

            s, l = strikes[i], strikes[j]
            width = abs(l - s)
            if (
                step is not None
                and iv_rank <= 70
                and max_spread_width is not None
                and width / step > max_spread_width
            ):
                continue

            row_s = next(o for o in options if o["strike"] == s and o["option_type"] == opt)
            row_l = next(o for o in options if o["strike"] == l and o["option_type"] == opt)
            credit = row_s["bid"] - row_l["ask"]
            if credit > 0:
                candidates.append({
                    "direction":    direction,
                    "short_strike": s,
                    "long_strike":  l,
                    "credit":       round(credit, 2),
                    "max_loss":     round(width - credit, 2),
                    "option_type":  opt,
                })

    # 4) apply skew_bias tilt
    if skew_bias and candidates:
        for c in candidates:
            factor = 1 if c["option_type"] == "call" else -1
            c["credit"] += round(skew_bias * 0.05 * factor, 4)

    if candidates:
        return max(candidates, key=lambda c: c["credit"])

    # 5) last-resort dummy spread
    if len(strikes) >= 2:
        return {
            "direction":    direction,
            "short_strike": strikes[0],
            "long_strike":  strikes[1],
            "credit":       0.0,
            "max_loss":     0.0,
            "option_type":  opt,
        }

    return {}


def check_momentum_filter(
    bars: pd.DataFrame, vwap: pd.Series, rsi: pd.Series, threshold: float
) -> bool:
    """
    Block entry when
      • last price < last VWAP, or
      • last RSI < threshold.
    """
    last_price = bars["close"].iat[-1]
    last_vwap = vwap.iat[-1]
    if last_price < last_vwap:
        return False

    last_rsi = rsi.iat[-1]
    if last_rsi < threshold:
        return False
    return True


def compute_vwap(bars: pd.DataFrame) -> pd.Series:
    """
    Compute VWAP per bar: cumulative (Typical * volume) / cumulative volume.
    Typical price = (high + low + close) / 3.
    """
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_vp = (tp * bars["volume"]).cumsum()
    cum_vol = bars["volume"].cumsum().replace(0, pd.NA)
    return cum_vp / cum_vol


def compute_rsi(series: pd.Series, window: int = 5) -> pd.Series:
    """
    Compute classic RSI over `window` periods.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))
