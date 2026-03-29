import os
import json
import time
import warnings
from groq import Groq
from dotenv import load_dotenv

# ── Silence deprecation warnings ──────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Load environment variables from .env ──────────────────────────────────
# Reads GROQ_API_KEY so it never appears hardcoded in source
load_dotenv()

# ── Initialise Groq client ─────────────────────────────────────────────────
# Groq uses LPU hardware — fastest free inference available
# Free tier: 14,400 requests/day, we use ~10/month — no limits in practice
# Model: llama-3.1-8b-instant — fast, capable, free
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── File paths ─────────────────────────────────────────────────────────────
RESULT_PATH    = "data/regime_result.json"  # written by classifier.py
FEATURES_PATH  = "data/features.csv"        # written by features.py
NARRATIVE_PATH = "data/narrative.json"      # written here, read by dashboard.py


# ── Asset class implications per regime ───────────────────────────────────
# Static lookup table — what each regime historically means for asset classes
# Used in two places:
# 1. Injected into the Groq prompt for context
# 2. Displayed as the implication grid in the dashboard
REGIME_IMPLICATIONS = {
    "Goldilocks": {
        "Equities":    "Strong — risk-on, growth stocks lead",
        "Bonds":       "Neutral — yields stable, modest returns",
        "Gold":        "Weak — low inflation reduces safe haven demand",
        "Commodities": "Neutral — demand solid but inflation contained",
        "USD":         "Neutral to weak — risk appetite suppresses dollar",
        "Credit":      "Strong — tight spreads, HY outperforms",
    },
    "Reflation": {
        "Equities":    "Moderate — value and cyclicals outperform growth",
        "Bonds":       "Weak — rising yields hurt duration",
        "Gold":        "Strong — inflation hedge demand rises",
        "Commodities": "Strong — energy and metals lead",
        "USD":         "Weak — commodities priced in dollars",
        "Credit":      "Moderate — spreads tight but rising rates a headwind",
    },
    "Stagflation": {
        "Equities":    "Weak — margin compression, multiple contraction",
        "Bonds":       "Weak — inflation erodes real returns",
        "Gold":        "Strong — best regime for gold historically",
        "Commodities": "Mixed — energy strong, industrial metals weak",
        "USD":         "Mixed — flight to safety vs inflation pressure",
        "Credit":      "Weak — spreads widen, default risk rises",
    },
    "Deflation": {
        "Equities":    "Weak — earnings under pressure, defensive names only",
        "Bonds":       "Strong — duration rallies as yields fall",
        "Gold":        "Moderate — safe haven bid but no inflation premium",
        "Commodities": "Weak — demand destruction hits prices",
        "USD":         "Strong — flight to safety and liquidity",
        "Credit":      "Weak — HY spreads blow out, default cycle begins",
    },
}


def load_current_indicators() -> dict:
    """
    Load the most recent available indicator values.

    FRED has publication lags — unemployment releases with ~1 month lag,
    GDP with ~1 quarter lag, CPI with ~2 week lag.
    We use last valid observation (ffill) to avoid NaN in the narrative.
    CPI YoY comes from features.csv (already transformed) not raw_data.csv.
    """
    import pandas as pd

    # features.csv has the YoY-transformed values — use for CPI, core CPI etc
    feat   = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)

    # raw_data.csv has unemployment and GDP in original form
    raw    = pd.read_csv("data/raw_data.csv", index_col=0, parse_dates=True)

    # Forward-fill both — uses last known value for NaN rows
    # This is the standard approach for lagged macro data
    feat   = feat.ffill()
    raw    = raw.ffill()

    latest_feat = feat.iloc[-1]
    latest_raw  = raw.iloc[-1]
    date        = feat.index[-1].strftime("%B %Y")

    def safe_float(val, fallback=0.0):
        """Convert to float — returns fallback if NaN or missing."""
        try:
            v = float(val)
            return fallback if (v != v) else round(v, 2)
        except Exception:
            return fallback

    indicators = {
        "date":               date,
        "spread_3m10y":       safe_float(latest_feat.get("spread_3m10y")),
        "spread_2y10y":       safe_float(latest_feat.get("spread_2y10y")),
        # CPI YoY from features.csv — already pct_change(12) transformed
        "cpi_yoy":            safe_float(latest_feat.get("cpi_yoy")),
        "core_cpi_yoy":       safe_float(latest_feat.get("core_cpi_yoy")),
        "breakeven_5y":       safe_float(latest_feat.get("breakeven_5y")),
        # Unemployment and GDP from raw — forward-filled to last known value
        "unemployment":       safe_float(latest_raw.get("unemployment")),
        "gdp_yoy":            safe_float(latest_raw.get("gdp_yoy")),
        "recession_prob":     safe_float(latest_feat.get("recession_prob")),
        "inversion_flag":     int(safe_float(latest_feat.get("inversion_flag"))),
        "inversion_duration": int(safe_float(latest_feat.get("inversion_duration"))),
        "breakeven_gap":      safe_float(latest_feat.get("breakeven_gap")),
    }

    return indicators


def build_prompt(result: dict, indicators: dict) -> str:
    """
    Construct the prompt sent to Groq API.

    Design principles:
    - Inject actual numbers not vague descriptions
    - Enforce strict 3-sentence output structure
    - Sentence 1: what the data says RIGHT NOW with real numbers
    - Sentence 2: key tension or contradiction in the signals
    - Sentence 3: one specific data release to watch next 4-6 weeks
    - Write for a portfolio manager — no fluff, no hedging
    - Do not mention AI or the model — write as human analysis
    """
    dominant   = result["dominant"]
    confidence = result["confidence"]
    probs      = result["probs"]
    impl       = REGIME_IMPLICATIONS.get(dominant, {})

    # Format probability distribution as readable inline string
    probs_str = " · ".join([f"{k} {v}%" for k, v in probs.items()])

    # Format asset class implications as readable inline string
    impl_str  = " · ".join([f"{k}: {v}" for k, v in impl.items()])

    # Build yield curve context based on inversion status
    if indicators["inversion_flag"] == 1:
        inversion_str = (
            f"The 3M/10Y yield curve has been inverted for "
            f"{indicators['inversion_duration']} consecutive months "
            f"(current spread: {indicators['spread_3m10y']}%)."
        )
    else:
        inversion_str = (
            f"The 3M/10Y yield curve is not currently inverted "
            f"(spread: {indicators['spread_3m10y']}%)."
        )

    # Handle GDP NaN — use placeholder text if data not available
    gdp_str = (
        f"{indicators['gdp_yoy']}%"
        if indicators["gdp_yoy"] != 0.0
        else "latest print pending"
    )

    prompt = f"""You are a senior macro analyst writing a daily briefing for a portfolio manager.

Current date: {indicators['date']}

MACRO REGIME MODEL OUTPUT:
- Dominant regime: {dominant} ({confidence}% confidence)
- Full probability distribution: {probs_str}

CURRENT INDICATOR VALUES:
- 3M/10Y yield spread: {indicators['spread_3m10y']}%
- 2Y/10Y yield spread: {indicators['spread_2y10y']}%
- CPI YoY: {indicators['cpi_yoy']}%
- Core CPI YoY: {indicators['core_cpi_yoy']}%
- 5Y breakeven inflation: {indicators['breakeven_5y']}%
- Breakeven gap (CPI minus breakeven): {indicators['breakeven_gap']}%
- Unemployment rate: {indicators['unemployment']}%
- Real GDP YoY: {gdp_str}
- NY Fed recession probability: {indicators['recession_prob']}%
- {inversion_str}

ASSET CLASS IMPLICATIONS FOR {dominant.upper()} REGIME:
{impl_str}

TASK:
Write exactly 3 sentences. No more, no less. No bullet points. No headers.

Sentence 1: State what the macro data is saying right now. Use specific numbers.
Sentence 2: Identify the single most important tension or contradiction in the signals.
Sentence 3: Name one specific data release or event to watch in the next 4-6 weeks and explain why it matters for regime transition.

Rules:
- Use the actual numbers provided above. Do not round or generalise.
- Write like a practitioner, not a journalist. No fluff.
- Do not start with "The macro environment" or "Currently" — be direct.
- Do not mention the model, AI, or any classification system.
- Write as if this is your own analysis based on the data.
"""
    return prompt


def generate_narrative(force: bool = False) -> dict:
    """
    Call Groq API to generate the 3-sentence macro narrative.

    Cache logic:
    - If narrative.json exists and was generated for the same regime date
      return cached version — avoids unnecessary API calls
    - force=True bypasses cache — use after running classifier.py
      to force fresh narrative for new regime data

    Groq free tier: 14,400 requests/day
    We use ~10/month — well within limits, no quota issues.
    """

    # ── Load regime result from classifier.py ─────────────────────────────
    with open(RESULT_PATH, "r") as f:
        result = json.load(f)

    # ── Cache check — only call API if regime date has changed ────────────
    if not force and os.path.exists(NARRATIVE_PATH):
        with open(NARRATIVE_PATH, "r") as f:
            cached = json.load(f)
        # Return cache if generated for same regime date
        if cached.get("regime_date") == result["date"]:
            print("  Using cached narrative — regime date unchanged.")
            return cached

    # ── Load raw indicator values for prompt injection ─────────────────────
    indicators = load_current_indicators()

    # ── Build structured prompt ────────────────────────────────────────────
    prompt = build_prompt(result, indicators)

    print(f"\n  Calling Groq API (llama-3.1-8b-instant)...")
    print(f"  Regime: {result['dominant']} ({result['confidence']}%)")

    # ── Call Groq API ──────────────────────────────────────────────────────
    try:
        # llama-3.1-8b-instant — fast, free, capable for structured 3-sentence output
        response = client.chat.completions.create(
            model    = "llama-3.1-8b-instant",
            messages = [{"role": "user", "content": prompt}]
        )
        narrative_text = response.choices[0].message.content.strip()
        print(f"  Narrative generated ({len(narrative_text)} chars)")

    except Exception as e:
        # Fallback — rule-based placeholder if API fails
        # Dashboard still works, just with a simpler narrative
        print(f"  Groq API error: {e}")
        print(f"  Using rule-based fallback narrative.")
        narrative_text = (
            f"Macro indicators signal {result['dominant']} regime with "
            f"{result['confidence']}% confidence as of {indicators['date']}. "
            f"CPI at {indicators['cpi_yoy']}% and the 3M/10Y spread at "
            f"{indicators['spread_3m10y']}% are the primary drivers of this call. "
            f"Watch the next CPI and NFP prints for evidence of regime transition."
        )

    # ── Assemble full output dict ──────────────────────────────────────────
    # This is what dashboard.py reads — contains everything it needs
    output = {
        "regime_date":  result["date"],           # date of regime prediction
        "dominant":     result["dominant"],        # dominant regime name
        "confidence":   result["confidence"],      # model confidence %
        "probs":        result["probs"],           # full probability distribution
        "narrative":    narrative_text,            # 3-sentence Groq narrative
        "indicators":   indicators,                # raw indicator values
        "implications": REGIME_IMPLICATIONS.get(  # asset class grid
                            result["dominant"], {}
                        ),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Save to disk ───────────────────────────────────────────────────────
    # dashboard.py reads this file — no API calls needed at dashboard load time
    with open(NARRATIVE_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {NARRATIVE_PATH}")

    return output


# ── Run directly to test ───────────────────────────────────────────────────
# python narrative.py
if __name__ == "__main__":
    # force=True — always regenerate, bypass cache
    output = generate_narrative(force=True)

    print("\n── Macro Narrative ─────────────────────────────────────────")
    print(output["narrative"])

    print("\n── Asset Class Implications ────────────────────────────────")
    for asset, view in output["implications"].items():
        print(f"  {asset:<12} {view}")

    print("\n── Regime Probabilities ────────────────────────────────────")
    for regime, prob in output["probs"].items():
        bar = "█" * int(prob / 5)
        print(f"  {regime:<12} {prob:>5.1f}%  {bar}")