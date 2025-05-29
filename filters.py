import streamlit as st

def get_default_filters():
    return {
        "min_credit": 0.30,
        "max_credit": 1.50,
        "max_spread_width": 2,       
        "strike_gap": 0.1,
        "strangle_distance": 4,
        "strangle_cost": 1.00,
        "max_slippage": 0.08,
        "min_fill_score": 0.70,
        "filter_calls": True,
        "filter_puts": True,
    }

def reset_filters():
    return get_default_filters()

def recommend_filters():
    f = get_default_filters()
    f.update(
        min_credit=0.40,
        max_credit=1.20,
        strike_gap=0.2,
        strangle_distance=10,
        strangle_cost=0.85,
    )
    return f

def render_filter_sliders(st, filters):
    filters["min_credit"] = st.slider(
        "Min Credit", 0.10, 2.00, filters["min_credit"], 0.05
    )
    filters["max_credit"] = st.slider(
        "Max Credit", 0.30, 2.50, filters["max_credit"], 0.05
    )
    filters["strike_gap"] = st.select_slider(
        "Target Spread Width (Gap)",
        options=[round(x * 0.1, 1) for x in range(-2, 11)],
        value=filters["strike_gap"],
    )
    filters["strangle_distance"] = st.slider(
        "Strangle Distance", 1, 200, filters["strangle_distance"]
    )
    filters["strangle_cost"] = st.slider(
        "Max Strangle Cost", 0.20, 20.00, filters["strangle_cost"], 0.05
    )
    filters["max_slippage"] = st.slider(
        "Max Slippage", 0.01, 0.50, filters["max_slippage"], 0.01
    )
    filters["min_fill_score"] = st.slider(
        "Min Fill Score", 0.00, 1.00, filters["min_fill_score"], 0.01
    )
    filters["filter_calls"] = st.checkbox("Filter Calls", filters["filter_calls"])
    filters["filter_puts"] = st.checkbox("Filter Puts", filters["filter_puts"])
    # NEW slider for spread-width compression
    filters["max_spread_width"] = st.slider(
        "Max Spread Width (strike steps)",
        1, 10,
        filters["max_spread_width"],
        1,
    )
    return filters
