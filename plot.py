import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


def _calc_option_payoff(
    price_grid: np.ndarray, strike: float, option_type: str
) -> np.ndarray:
    """European option payoff at expiry."""
    if option_type.lower() == "call":
        return np.maximum(price_grid - strike, 0.0)
    elif option_type.lower() == "put":
        return np.maximum(strike - price_grid, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def _compute_pl(df: pd.DataFrame, price_grid: np.ndarray) -> np.ndarray:
    """Aggregate P/L across all legs in df for each price in price_grid."""
    pl = np.zeros_like(price_grid, dtype=float)
    for _, row in df.iterrows():
        payoff = _calc_option_payoff(price_grid, row["strike"], row["option_type"])
        pl += row["position"] * payoff - row["position"] * row["premium"]
    return pl


def _find_breakevens(prices: np.ndarray, pl: np.ndarray) -> list[float]:
    """Find approximate breakeven points via sign changes + linear interpolation."""
    idx = np.where(np.diff(np.sign(pl)) != 0)[0]
    bevs: list[float] = []
    for i in idx:
        x0, x1 = prices[i], prices[i + 1]
        y0, y1 = pl[i], pl[i + 1]
        if y1 != y0:
            x_be = x0 - y0 * (x1 - x0) / (y1 - y0)
        else:
            x_be = x0
        bevs.append(x_be)
    return bevs


def _make_chart(
    price_grid: np.ndarray, pl: np.ndarray, underlying: float, title: str
) -> alt.Chart:
    """Build an Altair P/L chart with profit/loss shading, breakevens, and underlying line."""
    df = pd.DataFrame({"Price": price_grid, "P/L": pl})
    base = alt.Chart(df).encode(
        x=alt.X("Price:Q", title="Underlying Price"),
        y=alt.Y("P/L:Q", title="Profit / Loss"),
    )

    # Profit area (>=0)
    profit = (
        base.transform_filter("datum['P/L'] >= 0")
        .mark_area(interpolate="monotone", color="green", opacity=0.1)
        .encode(y2=alt.value(0))
    )

    # Loss area (<0)
    loss = (
        base.transform_filter("datum['P/L'] < 0")
        .mark_area(interpolate="monotone", color="red", opacity=0.1)
        .encode(y2=alt.value(0))
    )

    # P/L line
    line = base.mark_line(interpolate="monotone", color="white", strokeWidth=2)

    # Hover rule + tooltip
    hover = alt.selection_single(
        on="mouseover", nearest=True, fields=["Price"], empty="none"
    )
    hover_rule = (
        line.mark_rule(color="#888")
        .add_selection(hover)
        .encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    )
    tooltip = base.mark_point(size=0).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("Price:Q", format=".2f"),
            alt.Tooltip("P/L:Q", format=".2f"),
        ],
    )

    # Zero line
    zero = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[4, 4], color="#666")
        .encode(y="y:Q")
    )

    # Underlying price line
    now = (
        alt.Chart(pd.DataFrame({"x": [underlying]}))
        .mark_rule(strokeDash=[4, 2], color="#888", opacity=0.7)
        .encode(x="x:Q")
    )

    # Breakeven markers
    bevs = _find_breakevens(price_grid, pl)
    bev_df = pd.DataFrame({"BEV": bevs, "y": [0] * len(bevs)})
    bev_marks = (
        alt.Chart(bev_df)
        .mark_point(color="orange", size=50)
        .encode(
            x="BEV:Q",
            y="y:Q",
            tooltip=[alt.Tooltip("BEV:Q", format=".2f", title="Breakeven")],
        )
    )

    chart = (
        (profit + loss + line + hover_rule + tooltip + zero + now + bev_marks)
        .properties(width=300, height=300, title=title)
        .configure_title(fontSize=14, anchor="start")
        .configure_axis(
            grid=False,
            labelFontSize=12,
            titleFontSize=14,
            labelColor="#ccc",
            titleColor="#eee",
        )
        .configure_view(stroke=None)
        .interactive()
    )

    return chart


def _stat_block(
    df: pd.DataFrame, pl: np.ndarray, price_grid: np.ndarray
) -> dict[str, float | str]:
    # net_premium = cash flow at time 0, positive if you receive money
    net_premium = (-df["position"] * df["premium"]).sum()

    # choose the right label
    header = "Total credit" if net_premium > 0 else "Total debit"

    max_loss = float(np.min(pl))
    max_profit_val = np.max(pl)
    max_profit = (
        "Unlimited"
        if max_profit_val > pl[int(0.95 * len(pl))]
        else f"{max_profit_val:,.2f}"
    )

    # find breakevens
    idx = np.where(np.diff(np.sign(pl)) != 0)[0]
    bevs = []
    for i in idx:
        x1, x2 = price_grid[i], price_grid[i + 1]
        y1, y2 = pl[i], pl[i + 1]
        if y2 != y1:
            bevs.append(x1 - y1 * (x2 - x1) / (y2 - y1))
    bev_display = " / ".join(f"{b:.2f}" for b in bevs) if bevs else "—"

    return {
        header: f"{abs(net_premium):,.2f}",
        "Max profit": max_profit,
        "Breakeven": bev_display,
        "Max loss": f"{max_loss:,.2f}",
    }


def _display_stats(stats: dict[str, float | str]):
    # Just iterate through the stats dict in order
    lines = []
    for label, val in stats.items():
        lines.append(f"<b>{label}:</b> {val}")
    html = (
        "<div style='text-align:center; margin-top:0.5em;'>"
        + "<br>".join(lines)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_plots(
    strangle_df: pd.DataFrame,
    spread_df: pd.DataFrame,
    underlying: float,
    price_pad: float = 0.1,
    n_points: int = 500,
) -> None:
    """Render two side-by-side P/L charts with their stats."""
    if strangle_df.empty or spread_df.empty:
        st.warning("Empty dataframes passed to render_plots()")
        return

    # Price grid
    all_strikes = pd.concat([strangle_df["strike"], spread_df["strike"]])
    lo = float(all_strikes.min() * (1 - price_pad))
    hi = float(all_strikes.max() * (1 + price_pad))
    prices = np.linspace(lo, hi, n_points)

    # P/L arrays
    pl_str = _compute_pl(strangle_df, prices)
    pl_sp = _compute_pl(spread_df, prices)

    # Charts
    chart_str = _make_chart(prices, pl_str, underlying, "Strangle P/L Curve")
    chart_sp = _make_chart(prices, pl_sp, underlying, "Credit Spread P/L Curve")

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_str, use_container_width=True)
        stats = _stat_block(strangle_df, pl_str, prices)
        _display_stats(stats)
    with col2:
        st.altair_chart(chart_sp, use_container_width=True)
        stats = _stat_block(spread_df, pl_sp, prices)
        _display_stats(stats)
