import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Macro Regime Tracker",
    page_icon  = "📊",
    layout     = "wide",
)

# ── Regime colour map ──────────────────────────────────────────────────────
# Consistent colours used across all charts and cards
REGIME_COLORS = {
    "Goldilocks":  "#22c55e",
    "Reflation":   "#f59e0b",
    "Stagflation": "#ef4444",
    "Deflation":   "#3b82f6",
}


# ── Data loading ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_panels():
    """
    Load all signal panel data.
    Cached for 1 hour — avoids recomputing on every interaction.
    Cache clears automatically when TTL expires.
    """
    from signals import build_all_panels
    return build_all_panels()


def load_narrative() -> dict:
    """Load pre-generated narrative JSON from disk."""
    with open("data/narrative.json", "r") as f:
        return json.load(f)


# ── Helper: metric card ────────────────────────────────────────────────────
def metric_card(label: str, value: str, delta: str = None,
                color: str = None, help: str = None):
    """Wrapper around st.metric for consistent card styling."""
    st.metric(
        label      = label,
        value      = value,
        delta      = delta,
        delta_color = "normal",
        help       = help,
    )


# ── Chart builders ─────────────────────────────────────────────────────────

def chart_yield_curve(curve_shape: dict) -> go.Figure:
    """
    Plot the current yield curve shape across 4 maturities.
    Upward slope = normal. Flat/inverted = risk signal.
    """
    maturities = list(curve_shape.keys())
    yields     = list(curve_shape.values())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = maturities,
        y    = yields,
        mode = "lines+markers",
        line = dict(color="#3b82f6", width=2.5),
        marker = dict(size=8),
        name = "Yield curve",
    ))

    # Add reference line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title       = "Yield curve shape",
        xaxis_title = "Maturity",
        yaxis_title = "Yield (%)",
        height      = 300,
        margin      = dict(t=40, b=20, l=20, r=20),
        showlegend  = False,
    )
    return fig


def chart_spread_history(spread_hist: pd.DataFrame) -> go.Figure:
    """
    Plot 3M/10Y and 2Y/10Y spread history.
    Shaded red below 0 = inversion zone.
    """
    fig = go.Figure()

    # 3M/10Y spread — primary recession predictor
    fig.add_trace(go.Scatter(
        x    = spread_hist.index,
        y    = spread_hist["spread_3m10y"],
        name = "3M/10Y",
        line = dict(color="#ef4444", width=2),
    ))

    # 2Y/10Y spread — market favourite
    fig.add_trace(go.Scatter(
        x    = spread_hist.index,
        y    = spread_hist["spread_2y10y"],
        name = "2Y/10Y",
        line = dict(color="#f59e0b", width=2),
    ))

    # Zero line — inversion threshold
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)

    # Shade inversion zone
    fig.add_hrect(
        y0=-5, y1=0,
        fillcolor="red", opacity=0.05,
        layer="below", line_width=0,
    )

    fig.update_layout(
        title      = "Yield curve spreads — 5 year history",
        yaxis_title = "Spread (%)",
        height     = 300,
        margin     = dict(t=40, b=20, l=20, r=20),
        legend     = dict(orientation="h", y=1.1),
    )
    return fig


def chart_inflation_history(cpi_hist: pd.Series,
                             core_hist: pd.Series) -> go.Figure:
    """
    Plot CPI YoY and Core CPI YoY trend over 10 years.
    2.5% threshold line shows the regime inflation boundary.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x    = cpi_hist.index,
        y    = cpi_hist.values,
        name = "CPI YoY",
        line = dict(color="#ef4444", width=2),
    ))

    fig.add_trace(go.Scatter(
        x    = core_hist.index,
        y    = core_hist.values,
        name = "Core CPI YoY",
        line = dict(color="#f59e0b", width=2, dash="dash"),
    ))

    # 2.5% threshold — inflation regime boundary
    fig.add_hline(
        y=2.5, line_dash="dot",
        line_color="#22c55e", opacity=0.8,
        annotation_text="2.5% threshold",
        annotation_position="top right",
    )

    fig.update_layout(
        title       = "CPI YoY trend — 10 year history",
        yaxis_title = "% YoY",
        height      = 300,
        margin      = dict(t=40, b=20, l=20, r=20),
        legend      = dict(orientation="h", y=1.1),
    )
    return fig


def chart_recession_prob(prob_hist: pd.Series) -> go.Figure:
    """
    Plot NY Fed recession probability over 10 years.
    30% and 50% threshold lines for risk level context.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x    = prob_hist.index,
        y    = prob_hist.values,
        name = "Recession probability",
        fill = "tozeroy",
        line = dict(color="#3b82f6", width=2),
        fillcolor = "rgba(59,130,246,0.15)",
    ))

    # Threshold lines
    fig.add_hline(y=30, line_dash="dot", line_color="#f59e0b",
                  opacity=0.8, annotation_text="30% elevated",
                  annotation_position="top right")
    fig.add_hline(y=50, line_dash="dot", line_color="#ef4444",
                  opacity=0.8, annotation_text="50% high risk",
                  annotation_position="top right")

    fig.update_layout(
        title       = "NY Fed recession probability — 10 year history",
        yaxis_title = "%",
        height      = 300,
        margin      = dict(t=40, b=20, l=20, r=20),
        showlegend  = False,
    )
    return fig


def chart_regime_timeline(regime_series: pd.Series,
                           color_map: dict) -> go.Figure:
    """
    Colour-coded horizontal bar chart showing regime history since 1990.
    Each bar segment = one month, coloured by regime.
    """
    # Encode regime as numeric for plotting
    regime_to_num = {r: i for i, r in enumerate(color_map.keys())}

    fig = go.Figure()

    for regime, color in color_map.items():
        mask = regime_series == regime
        if mask.any():
            fig.add_trace(go.Scatter(
                x    = regime_series[mask].index,
                y    = [1] * mask.sum(),
                mode = "markers",
                marker = dict(
                    color  = color,
                    size   = 8,
                    symbol = "square",
                ),
                name = regime,
            ))

    fig.update_layout(
        title       = "Historical regime timeline — 1990 to today",
        yaxis       = dict(visible=False),
        height      = 160,
        margin      = dict(t=40, b=20, l=20, r=20),
        legend      = dict(orientation="h", y=1.3),
        showlegend  = True,
    )
    return fig


# ── Main dashboard ─────────────────────────────────────────────────────────

def main():
    # ── Load data ──────────────────────────────────────────────────────────
    panels    = load_panels()
    narrative = load_narrative()

    yc   = panels["yield_curve"]
    gr   = panels["growth"]
    inf  = panels["inflation"]
    rec  = panels["recession"]
    rt   = panels["timeline"]
    anom = panels["anomalies"]

    dominant   = narrative["dominant"]
    confidence = narrative["confidence"]
    probs      = narrative["probs"]
    impl       = narrative["implications"]
    regime_color = REGIME_COLORS.get(dominant, "#888")

    # ── Header ─────────────────────────────────────────────────────────────
    st.title("📊 Macro Regime Tracker")
    st.caption(
        "Finance × AI — Module 02 | "
        "XGBoost classifier trained on 30 years of FRED data | "
        f"Last updated: {narrative.get('generated_at', 'N/A')}"
    )

    st.divider()

    # ── HERO SECTION ───────────────────────────────────────────────────────
    st.subheader("Current macro regime")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Big regime card
        st.markdown(
            f"""
            <div style="
                background-color: {regime_color}22;
                border: 2px solid {regime_color};
                border-radius: 12px;
                padding: 24px;
                text-align: center;
            ">
                <div style="font-size: 2rem; font-weight: 700;
                            color: {regime_color};">
                    {dominant}
                </div>
                <div style="font-size: 1rem; color: #888; margin-top: 4px;">
                    {confidence}% model confidence
                </div>
                <div style="font-size: 0.85rem; color: #666; margin-top: 8px;">
                    {rt['current_date']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Probability distribution")
        for regime, prob in probs.items():
            color = REGIME_COLORS.get(regime, "#888")
            st.markdown(
                f"""
                <div style="display:flex; align-items:center;
                            margin-bottom:6px; gap:8px;">
                    <span style="width:90px; font-size:0.85rem;">{regime}</span>
                    <div style="flex:1; background:#eee; border-radius:4px;
                                height:12px;">
                        <div style="width:{prob}%; background:{color};
                                    border-radius:4px; height:12px;"></div>
                    </div>
                    <span style="width:45px; text-align:right;
                                 font-size:0.85rem;">{prob}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:
        st.markdown("#### AI macro narrative")
        st.markdown(
            f"""
            <div style="
                background: #f8f9fa;
                border-left: 4px solid {regime_color};
                border-radius: 4px;
                padding: 16px 20px;
                font-size: 0.95rem;
                line-height: 1.7;
                color: #333;
            ">
                {narrative['narrative']}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Asset class implications")
        impl_cols = st.columns(2)
        for i, (asset, view) in enumerate(impl.items()):
            with impl_cols[i % 2]:
                # Colour the asset label based on signal strength
                if "Strong" in view:
                    label_color = "#22c55e"
                elif "Weak" in view:
                    label_color = "#ef4444"
                else:
                    label_color = "#f59e0b"
                st.markdown(
                    f"<span style='color:{label_color}; font-weight:600;'>"
                    f"{asset}</span> — {view}",
                    unsafe_allow_html=True
                )

    st.divider()

    # ── ANOMALY ALERT ──────────────────────────────────────────────────────
    if anom["has_anomalies"]:
        for a in anom["anomalies"]:
            st.warning(f"⚠️ **{a['type']}** — {a['message']}")

    # ── SIGNAL PANELS ──────────────────────────────────────────────────────
    st.subheader("Signal panels")

    # Row 1: metric cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("CPI YoY", f"{inf['current_cpi']}%",
                    help="Headline inflation — above 2.5% = high")
    with c2:
        metric_card("Core CPI", f"{inf['current_core_cpi']}%",
                    help="Ex food and energy — stickier inflation measure")
    with c3:
        metric_card("5Y Breakeven", f"{inf['current_breakeven']}%",
                    help="Market-implied inflation expectation")
    with c4:
        metric_card("3M/10Y Spread", f"{yc['current_spreads']['3M/10Y']}%",
                    help="Best recession predictor — negative = inverted")
    with c5:
        metric_card("Unemployment", f"{gr['unemployment']}%",
                    delta=f"{gr['unemp_delta']:+.2f}% (3M)",
                    help="3M delta shown — rising fast = recession signal")
    with c6:
        metric_card("Recession Prob", f"{rec['current_prob']}%",
                    help=f"NY Fed model — risk level: {rec['risk_level']}")

    st.markdown("")

    # Row 2: yield curve charts
    col_yc1, col_yc2 = st.columns(2)
    with col_yc1:
        st.plotly_chart(
            chart_yield_curve(yc["curve_shape"]),
            use_container_width=True
        )
    with col_yc2:
        st.plotly_chart(
            chart_spread_history(yc["spread_history"]),
            use_container_width=True
        )

    # Row 3: inflation + recession
    col_inf, col_rec = st.columns(2)
    with col_inf:
        st.plotly_chart(
            chart_inflation_history(inf["cpi_history"], inf["core_history"]),
            use_container_width=True
        )
    with col_rec:
        st.plotly_chart(
            chart_recession_prob(rec["prob_history"]),
            use_container_width=True
        )

    st.divider()

    # ── REGIME TIMELINE ────────────────────────────────────────────────────
    st.subheader("Historical regime timeline")
    st.plotly_chart(
        chart_regime_timeline(rt["regime_series"], rt["color_map"]),
        use_container_width=True
    )

    st.divider()

    # ── FOOTER ─────────────────────────────────────────────────────────────
    st.caption(
        "Finance × AI Series — Module 02: Macro Regime Tracker | "
        "Data: FRED API + yfinance | "
        "Model: XGBoost trained on 1990–2022 | "
        "Narrative: Groq (Llama 3.1) | "
        "https://github.com/khandelwalharshit1307/finance-ai-stack-02-macro-regime"
    )


if __name__ == "__main__":
    main()