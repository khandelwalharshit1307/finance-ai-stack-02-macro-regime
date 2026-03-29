import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
import os

# ── Silence pandas FutureWarnings ─────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants ──────────────────────────────────────────────────────────────
RAW_DATA_PATH = "data/raw_data.csv"
FEATURES_PATH = "data/features.csv"

# Inflation threshold — above = high inflation regime
INFLATION_THRESHOLD = 2.5

# Growth threshold — GDP YoY above 0 = expansion, below = contraction
GROWTH_THRESHOLD = 0.0


def load_raw() -> pd.DataFrame:
    """Load raw CSV saved by data_pull.py and parse dates as index."""
    df = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
    print(f"  Loaded raw data: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def compute_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute year-on-year % changes for level series.

    Raw CPI and PCE are index levels, not % changes.
    pct_change(12) on monthly data = YoY %.
    GDP is already a YoY % from FRED so we leave it.
    Unemployment we take a 3-month difference to capture momentum.
    """
    df = df.copy()

    # CPI YoY % — raw series is price index level
    df["cpi_yoy"]      = df["cpi_yoy"].pct_change(12) * 100

    # Core CPI YoY %
    df["core_cpi_yoy"] = df["core_cpi_yoy"].pct_change(12) * 100

    # PCE YoY %
    df["pce"]          = df["pce"].pct_change(12) * 100

    # Unemployment momentum — 3M change captures rising/falling trend
    df["unemp_delta"]  = df["unemployment"].diff(3)

    print("  YoY transforms applied: cpi_yoy, core_cpi_yoy, pce, unemp_delta")
    return df


def add_lags(df: pd.DataFrame, cols: list, lags: list = [1, 3]) -> pd.DataFrame:
    """
    Add lagged versions of key features.

    Lags give the model memory — it can see where each signal
    was 1 month and 3 months ago, capturing momentum and trend.
    This roughly doubles the signal depth without adding new data.
    """
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    print(f"  Lags added: {len(cols)} cols × {lags} = {len(cols)*len(lags)} new features")
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived signals that add analytical edge.

    breakeven_gap: difference between actual CPI and what the bond
    market expects (5Y breakeven). Positive = market more optimistic
    than current data. Negative = market pricing in more inflation.

    inversion_flag: binary 1/0 — is the 3M/10Y curve inverted right now?

    inversion_duration: how many consecutive months has the curve been
    inverted? Sustained inversion is a stronger recession signal than
    a single month dip.
    """
    df = df.copy()

    # Gap between actual inflation and market-implied inflation
    df["breakeven_gap"] = df["cpi_yoy"] - df["breakeven_5y"]

    # Binary inversion flag — 1 if 3M yield > 10Y yield
    df["inversion_flag"] = (df["spread_3m10y"] < 0).astype(int)

    # Count consecutive months of inversion
    duration = []
    count = 0
    for val in df["inversion_flag"]:
        if val == 1:
            count += 1
        else:
            count = 0
        duration.append(count)
    df["inversion_duration"] = duration

    print("  Derived features: breakeven_gap, inversion_flag, inversion_duration")
    return df


def label_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a regime label to each month — this is the training target.

    Primary logic:
        Growth signal    = GDP YoY > 0  (expansion) or < 0 (contraction)
        Inflation signal = CPI YoY > 2.5% (high) or <= 2.5% (low)

    Override logic:
        If yield curve deeply inverted AND unemployment rising fast
        → force Deflation regardless of GDP (GDP is a lagging indicator
        and often stays positive well into a recession)

    Four regimes:
        Goldilocks  = growth up   + inflation low  → risk-on
        Reflation   = growth up   + inflation high → commodities, value
        Stagflation = growth down + inflation high → worst regime
        Deflation   = growth down + inflation low  → recession/safe havens
    """
    df = df.copy()

    def assign(row):
        try:
            growth_up    = row["gdp_yoy"] > GROWTH_THRESHOLD
            inflation_hi = row["cpi_yoy"] > INFLATION_THRESHOLD

            # Override: deep inversion + rising unemployment = Deflation
            # even if GDP hasn't gone negative yet (GDP is a lagging indicator)
            curve_inverted = row["spread_3m10y"] < -0.3
            unemp_rising   = row["unemp_delta"] > 0.3

            if curve_inverted and unemp_rising:
                return "Deflation"

            if growth_up and not inflation_hi:
                return "Goldilocks"
            elif growth_up and inflation_hi:
                return "Reflation"
            elif not growth_up and inflation_hi:
                return "Stagflation"
            else:
                return "Deflation"
        except Exception:
            return "Deflation"

    df["regime"] = df.apply(assign, axis=1)

    print("\n  Regime distribution (training labels):")
    print(df["regime"].value_counts().to_string())
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Select final feature columns, drop NaN rows from lag warm-up,
    and scale continuous features with StandardScaler.

    Returns X (features), y (labels), and the fitted scaler.
    The scaler is needed later to transform live data at prediction time.
    """
    # All columns that go into the model
    FEATURE_COLS = [
        "spread_3m10y",        "spread_3m10y_lag1",   "spread_3m10y_lag3",
        "spread_2y10y",        "spread_2y10y_lag1",   "spread_2y10y_lag3",
        "cpi_yoy",             "cpi_yoy_lag1",        "cpi_yoy_lag3",
        "core_cpi_yoy",        "core_cpi_yoy_lag1",   "core_cpi_yoy_lag3",
        "breakeven_5y",        "breakeven_5y_lag1",   "breakeven_5y_lag3",
        "gdp_yoy",             "gdp_yoy_lag1",        "gdp_yoy_lag3",
        "unemp_delta",
        "breakeven_gap",
        "inversion_flag",
        "inversion_duration",
        "recession_prob",
    ]

    # Keep only columns that actually exist in the dataframe
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"\n  Warning — columns not found, skipped: {missing}")

    X_raw = df[available].copy()
    y     = df["regime"].copy()

    # Drop rows where more than 30% of features are NaN
    # This removes the lag warm-up period at the top
    valid = X_raw.dropna(thresh=int(len(X_raw.columns) * 0.7)).index
    X_raw = X_raw.loc[valid]
    y     = y.loc[valid]

    # Fill remaining NaNs with column median
    # Breakeven data only starts 2003, recession_prob has early gaps
    X_raw = X_raw.fillna(X_raw.median())

    # Scale all features — keeps large-range features (PMI)
    # from dominating small-range ones (unemployment)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X        = pd.DataFrame(X_scaled, index=X_raw.index, columns=available)

    print(f"\n  Final feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  NaN rows dropped: {len(df) - len(X)}")
    return X, y, scaler


def run_pipeline() -> tuple:
    """
    Master function — runs all Layer 1 steps in order.
    Call this from classifier.py to get X, y, scaler ready for training.
    """
    print("\n── Layer 1: Feature Engineering ───────────────────────────")

    df = load_raw()
    df = compute_yoy(df)
    df = add_lags(df, cols=[
        "spread_3m10y", "spread_2y10y",
        "cpi_yoy", "core_cpi_yoy",
        "breakeven_5y", "gdp_yoy"
    ])
    df = add_derived(df)
    df = label_regimes(df)

    X, y, scaler = build_feature_matrix(df)

    # Save enriched dataframe for dashboard signal panels
    df.to_csv(FEATURES_PATH)
    print(f"\n  Saved: {FEATURES_PATH}")

    return X, y, scaler, df


# ── Run directly to test ───────────────────────────────────────────────────
# python features.py
if __name__ == "__main__":
    X, y, scaler, df = run_pipeline()

    print("\n── Regime distribution ─────────────────────────────────────")
    print(y.value_counts())

    print("\n── Last 3 rows of feature matrix ───────────────────────────")
    print(X.tail(3).to_string())

    print("\n── Current regime (latest month) ───────────────────────────")
    print(f"  {y.iloc[-1]}  —  {y.index[-1].strftime('%B %Y')}")