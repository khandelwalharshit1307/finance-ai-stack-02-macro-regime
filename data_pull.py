import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

# ── Load environment variables from .env file ──────────────────────────────
# This reads FRED_API_KEY from your .env so it never appears in the code
load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))


# ── FRED series to pull ────────────────────────────────────────────────────
# Keys are our internal names, values are official FRED series IDs
# These cover all 4 regime dimensions: growth, inflation, rates, labour
FRED_SERIES = {
    "spread_3m10y":   "T10Y3M",           # 3M/10Y yield spread — best recession predictor
    "spread_2y10y":   "T10Y2Y",           # 2Y/10Y spread — market's favourite curve signal
    "cpi_yoy":        "CPIAUCSL",         # Headline CPI — overall inflation level
    "core_cpi_yoy":   "CPILFESL",         # Core CPI (ex food & energy) — stickier inflation
    "pce":            "PCEPI",            # PCE deflator — Fed's preferred inflation gauge
    "unemployment":   "UNRATE",           # Unemployment rate — labour market health
    "gdp_yoy":        "A191RL1Q225SBEA",  # Real GDP YoY — quarterly, actual output growth
    "breakeven_5y":   "T5YIE",            # 5Y breakeven — market-implied inflation expectation
    "recession_prob": "RECPROUSM156N",    # NY Fed recession probability model output
}

# ── Treasury yield tickers from yfinance ──────────────────────────────────
# These give us the full yield curve shape across 4 maturities
# FRED has these too but yfinance is more current for live curve plotting
YIELD_TICKERS = {
    "y_3m":  "^IRX",   # 3-month treasury yield
    "y_5y":  "^FVX",   # 5-year treasury yield
    "y_10y": "^TNX",   # 10-year treasury yield (the benchmark)
    "y_30y": "^TYX",   # 30-year treasury yield (long end)
}


def pull_fred() -> pd.DataFrame:
    """
    Pull all FRED series back to 1990 and align to monthly frequency.

    Raw FRED data comes at mixed frequencies:
    - GDP is quarterly
    - CPI, unemployment are monthly
    - Yield spreads are daily

    We resample everything to month-start (MS) so the dataframe
    has one clean row per month — required for the classifier.
    GDP gaps are forward-filled by resample().last().
    """
    frames = {}

    for name, series_id in FRED_SERIES.items():
        try:
            # Pull full history from 1990 — more data = better classifier training
            s = fred.get_series(series_id, observation_start="1990-01-01")
            s.name = name
            frames[name] = s
            print(f"  OK  {series_id} ({name}): {len(s)} observations")
        except Exception as e:
            # Don't crash if one series fails — log and continue
            print(f"  ERR {series_id}: {e}")

    # Combine all series into one dataframe
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)

    # Resample to month-start — takes the last known value in each month
    # This naturally forward-fills quarterly GDP into monthly rows
    df = df.resample("MS").last()

    return df


def pull_yield_curve() -> pd.DataFrame:
    """
    Pull treasury yield curve from yfinance for the last 5 years.

    yfinance gives us live-ish data without FRED's publication lag.
    We use this for the yield curve shape chart in the dashboard.
    Monthly interval keeps it consistent with the FRED dataframe.
    """
    frames = {}

    for name, ticker in YIELD_TICKERS.items():
        try:
            raw = yf.download(ticker, period="5y", interval="1mo", progress=False)

            # squeeze() converts single-column DataFrame to Series
            s = raw["Close"].squeeze()
            s.name = name
            frames[name] = s
            print(f"  OK  {ticker} ({name}): {len(s)} observations")
        except Exception as e:
            print(f"  ERR {ticker}: {e}")

    df = pd.DataFrame(frames)

    # Normalise index to month-start timestamps so it joins cleanly with FRED
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()

    return df


def pull_all() -> pd.DataFrame:
    """
    Master function — pulls FRED + yfinance, joins them, saves to CSV.

    Call this once each time FRED releases new data (roughly monthly).
    The saved CSV is the input for features.py in Layer 1.
    """
    print("\n── Layer 0: Pulling FRED data ─────────────────────────────")
    fred_df = pull_fred()

    print("\n── Layer 0: Pulling yield curve (yfinance) ────────────────")
    yc_df = pull_yield_curve()

    # Left join — FRED is the master index (1990+), yfinance only has 5 years
    # yfinance columns will be NaN before 2020 which is fine
    df = fred_df.join(yc_df, how="left")

    # Save raw data — features.py reads from this file
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_data.csv")

    print(f"\n── Saved: data/raw_data.csv — {df.shape[0]} rows × {df.shape[1]} cols")
    return df


# ── Run directly to test the data pull ────────────────────────────────────
# python data_pull.py
if __name__ == "__main__":
    df = pull_all()

    print("\n── Last 6 months of data ───────────────────────────────────")
    print(df.tail(6).to_string())

    print("\n── Missing values per column ───────────────────────────────")
    print(df.isnull().sum())