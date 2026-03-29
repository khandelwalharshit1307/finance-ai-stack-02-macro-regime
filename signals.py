import pandas as pd
import numpy as np
import warnings
import json

warnings.filterwarnings("ignore", category=FutureWarning)

# ── File paths ─────────────────────────────────────────────────────────────
RAW_DATA_PATH  = "data/raw_data.csv"    # written by data_pull.py
FEATURES_PATH  = "data/features.csv"   # written by features.py
NARRATIVE_PATH = "data/narrative.json" # written by narrative.py


def load_data() -> tuple:
    """
    Load raw and feature data, forward-fill FRED lags.
    Returns both dataframes ready for panel construction.
    """
    raw  = pd.read_csv(RAW_DATA_PATH,  index_col=0, parse_dates=True)
    feat = pd.read_csv(FEATURES_PATH,  index_col=0, parse_dates=True)

    # Forward-fill publication lags — last known value carries forward
    raw  = raw.ffill()
    feat = feat.ffill()

    return raw, feat


def yield_curve_panel(raw: pd.DataFrame) -> dict:
    """
    Yield curve data for the curve shape chart and spread gauges.

    Returns:
    - curve_shape: current yield at each maturity (for line chart)
    - spread_history: 3M/10Y and 2Y/10Y spreads over time (for historical chart)
    - current_spreads: latest spread values for the metric cards
    - inversion_alert: True if 3M/10Y is currently negative
    """
    # ── Current curve shape ────────────────────────────────────────────────
    # Pull latest available value for each maturity
    maturities = {
        "3M":  "y_3m",
        "5Y":  "y_5y",
        "10Y": "y_10y",
        "30Y": "y_30y",
    }

    curve_shape = {}
    for label, col in maturities.items():
        if col in raw.columns:
            val = raw[col].dropna().iloc[-1]
            curve_shape[label] = round(float(val), 3)

    # ── Spread history — last 5 years ──────────────────────────────────────
    cutoff = raw.index[-1] - pd.DateOffset(years=5)
    spread_hist = raw[["spread_3m10y", "spread_2y10y"]].loc[raw.index >= cutoff].copy()
    spread_hist = spread_hist.dropna(how="all")

    # ── Current spread values ──────────────────────────────────────────────
    latest = raw[["spread_3m10y", "spread_2y10y"]].ffill().iloc[-1]
    current_spreads = {
        "3M/10Y": round(float(latest["spread_3m10y"]), 2),
        "2Y/10Y": round(float(latest["spread_2y10y"]), 2),
    }

    # ── Inversion alert ────────────────────────────────────────────────────
    inversion_alert  = current_spreads["3M/10Y"] < 0
    inversion_months = int(raw["spread_3m10y"].lt(0)
                           .iloc[-24:]  # look back 2 years
                           .sum())

    return {
        "curve_shape":       curve_shape,
        "spread_history":    spread_hist,
        "current_spreads":   current_spreads,
        "inversion_alert":   inversion_alert,
        "inversion_months":  inversion_months,
    }


def growth_panel(raw: pd.DataFrame, feat: pd.DataFrame) -> dict:
    """
    Growth pulse data — GDP trend and unemployment momentum.

    Returns:
    - gdp_history: real GDP YoY over time for trend chart
    - current_gdp: latest GDP YoY value
    - unemployment: latest unemployment rate
    - unemp_delta: 3-month change in unemployment (rising = bad)
    - growth_signal: 'Expansion' or 'Contraction'
    """
    # ── GDP history — last 10 years ────────────────────────────────────────
    cutoff  = raw.index[-1] - pd.DateOffset(years=10)
    gdp_raw = raw["gdp_yoy"].loc[raw.index >= cutoff].dropna()

    # ── Current values ─────────────────────────────────────────────────────
    current_gdp  = round(float(raw["gdp_yoy"].dropna().iloc[-1]), 2)
    unemployment = round(float(raw["unemployment"].dropna().iloc[-1]), 1)

    # ── Unemployment momentum ──────────────────────────────────────────────
    unemp_series = raw["unemployment"].dropna()
    unemp_delta  = round(float(unemp_series.iloc[-1] - unemp_series.iloc[-4]), 2)

    # ── Growth signal ──────────────────────────────────────────────────────
    growth_signal = "Expansion" if current_gdp > 0 else "Contraction"

    return {
        "gdp_history":    gdp_raw,
        "current_gdp":    current_gdp,
        "unemployment":   unemployment,
        "unemp_delta":    unemp_delta,
        "growth_signal":  growth_signal,
    }


def inflation_panel(raw: pd.DataFrame, feat: pd.DataFrame) -> dict:
    """
    Inflation dashboard — CPI, Core CPI, PCE, breakevens, and the gap.

    The breakeven gap (actual CPI minus market-implied) is the most
    interesting signal — tells you whether the market thinks inflation
    will persist or cool from here.

    Returns:
    - inflation_history: CPI YoY and core CPI YoY over time
    - current values for each metric
    - breakeven_gap: CPI YoY minus 5Y breakeven
    - inflation_signal: 'High' or 'Low' based on 2.5% threshold
    """
    # ── Inflation history — last 10 years ─────────────────────────────────
    cutoff = raw.index[-1] - pd.DateOffset(years=10)

    # Use feature-engineered YoY series — already pct_change(12) transformed
    cpi_hist = feat["cpi_yoy"].loc[feat.index >= cutoff].dropna()
    core_hist = feat["core_cpi_yoy"].loc[feat.index >= cutoff].dropna()

    # ── Current values ─────────────────────────────────────────────────────
    current_cpi      = round(float(feat["cpi_yoy"].dropna().iloc[-1]), 2)
    current_core_cpi = round(float(feat["core_cpi_yoy"].dropna().iloc[-1]), 2)
    current_pce      = round(float(feat["pce"].dropna().iloc[-1]), 2)
    current_breakeven = round(float(raw["breakeven_5y"].dropna().iloc[-1]), 2)

    # ── Breakeven gap ──────────────────────────────────────────────────────
    # Positive = actual inflation above market expectations (sticky)
    # Negative = market expects inflation to cool below current level
    breakeven_gap = round(current_cpi - current_breakeven, 2)

    # ── Inflation signal ───────────────────────────────────────────────────
    inflation_signal = "High" if current_cpi > 2.5 else "Low"

    return {
        "cpi_history":        cpi_hist,
        "core_history":       core_hist,
        "current_cpi":        current_cpi,
        "current_core_cpi":   current_core_cpi,
        "current_pce":        current_pce,
        "current_breakeven":  current_breakeven,
        "breakeven_gap":      breakeven_gap,
        "inflation_signal":   inflation_signal,
    }


def recession_panel(raw: pd.DataFrame) -> dict:
    """
    Recession probability from the NY Fed model (RECPROUSM156N).

    The NY Fed model is a logistic regression on the 3M/10Y spread —
    published monthly, highly cited by practitioners.
    Above 30% = elevated risk, above 50% = high risk historically.

    Returns:
    - prob_history: recession probability over time
    - current_prob: latest probability value
    - risk_level: 'Low', 'Elevated', or 'High'
    """
    # ── History — last 10 years ────────────────────────────────────────────
    cutoff    = raw.index[-1] - pd.DateOffset(years=10)
    prob_hist = raw["recession_prob"].loc[raw.index >= cutoff].dropna()

    # ── Current value ──────────────────────────────────────────────────────
    current_prob = round(float(raw["recession_prob"].dropna().iloc[-1]), 1)

    # ── Risk level buckets ─────────────────────────────────────────────────
    if current_prob < 15:
        risk_level = "Low"
    elif current_prob < 30:
        risk_level = "Elevated"
    else:
        risk_level = "High"

    return {
        "prob_history":  prob_hist,
        "current_prob":  current_prob,
        "risk_level":    risk_level,
    }


def regime_timeline(feat: pd.DataFrame) -> dict:
    """
    Historical regime timeline — colour-coded by regime label.
    Used for the 'where are we now' chart going back to 1990.

    Returns regime series and colour map for Plotly chart.
    """
    # Colour map for the 4 regimes
    REGIME_COLORS = {
        "Goldilocks":  "#22c55e",  # green
        "Reflation":   "#f59e0b",  # amber
        "Stagflation": "#ef4444",  # red
        "Deflation":   "#3b82f6",  # blue
    }

    regime_series = feat["regime"].dropna()

    return {
        "regime_series": regime_series,
        "color_map":     REGIME_COLORS,
        "current":       regime_series.iloc[-1],
        "current_date":  regime_series.index[-1].strftime("%B %Y"),
    }


def anomaly_detector(narrative_data: dict, feat: pd.DataFrame) -> dict:
    """
    Cross-signal anomaly detector — flags contradictions between
    the macro regime and other signals.

    This is the feature that connects Module 01 (news sentiment)
    to Module 02 (macro regime). If sentiment says Risk-On but
    macro says Deflation, that's a contradiction worth surfacing.

    For now detects internal contradictions within macro signals.
    Module 01 integration added when both dashboards are live.
    """
    anomalies = []
    dominant  = narrative_data.get("dominant", "")
    indicators = narrative_data.get("indicators", {})

    # ── Yield curve vs recession prob contradiction ────────────────────────
    spread    = indicators.get("spread_3m10y", 0)
    rec_prob  = indicators.get("recession_prob", 0)
    if spread > 0.5 and rec_prob > 25:
        anomalies.append({
            "type":    "Curve vs recession prob",
            "message": (f"Yield curve is positive ({spread}%) but NY Fed "
                        f"recession probability is elevated at {rec_prob}%. "
                        f"Bond market and model disagree on near-term risk.")
        })

    # ── Deflation regime but unemployment still low ────────────────────────
    if dominant == "Deflation" and indicators.get("unemployment", 0) < 5.0:
        anomalies.append({
            "type":    "Deflation + tight labour market",
            "message": (f"Deflation regime signal with unemployment at "
                        f"{indicators.get('unemployment')}% — labour market "
                        f"remains tight, which historically delays deflation onset.")
        })

    # ── Breakeven gap contradiction ────────────────────────────────────────
    gap = indicators.get("breakeven_gap", 0)
    if abs(gap) > 1.0:
        direction = "above" if gap > 0 else "below"
        anomalies.append({
            "type":    "Breakeven gap",
            "message": (f"Actual CPI ({indicators.get('cpi_yoy')}%) is "
                        f"{abs(gap)}% {direction} 5Y breakeven "
                        f"({indicators.get('breakeven_5y')}%). "
                        f"Market and realised inflation diverging significantly.")
        })

    return {
        "anomalies":     anomalies,
        "count":         len(anomalies),
        "has_anomalies": len(anomalies) > 0,
    }


def build_all_panels() -> dict:
    """
    Master function — builds all signal panel data in one call.
    Called by dashboard.py at load time.
    """
    print("\n── Layer 4B: Signal Panels ─────────────────────────────────")

    raw, feat = load_data()

    # Load narrative output from Layer 4A
    with open(NARRATIVE_PATH, "r") as f:
        narrative_data = json.load(f)

    # Build each panel
    yc   = yield_curve_panel(raw)
    gr   = growth_panel(raw, feat)
    inf  = inflation_panel(raw, feat)
    rec  = recession_panel(raw)
    rt   = regime_timeline(feat)
    anom = anomaly_detector(narrative_data, feat)

    print(f"  Yield curve:     3M/10Y={yc['current_spreads']['3M/10Y']}%  "
          f"inversion={yc['inversion_alert']}")
    print(f"  Growth:          GDP={gr['current_gdp']}%  "
          f"unemployment={gr['unemployment']}%  signal={gr['growth_signal']}")
    print(f"  Inflation:       CPI={inf['current_cpi']}%  "
          f"core={inf['current_core_cpi']}%  signal={inf['inflation_signal']}")
    print(f"  Recession prob:  {rec['current_prob']}%  "
          f"risk={rec['risk_level']}")
    print(f"  Regime now:      {rt['current']}  ({rt['current_date']})")
    print(f"  Anomalies:       {anom['count']} detected")

    return {
        "yield_curve":  yc,
        "growth":       gr,
        "inflation":    inf,
        "recession":    rec,
        "timeline":     rt,
        "anomalies":    anom,
        "narrative":    narrative_data,
    }


# ── Run directly to test ───────────────────────────────────────────────────
# python signals.py
if __name__ == "__main__":
    panels = build_all_panels()

    print("\n── Yield curve shape ───────────────────────────────────────")
    for mat, val in panels["yield_curve"]["curve_shape"].items():
        print(f"  {mat:<5} {val}%")

    print("\n── Anomalies detected ──────────────────────────────────────")
    if panels["anomalies"]["has_anomalies"]:
        for a in panels["anomalies"]["anomalies"]:
            print(f"  [{a['type']}] {a['message']}")
    else:
        print("  None")