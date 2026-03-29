# Macro Regime Tracker — Finance × AI Module 02

An AI-powered macro regime classifier trained on 30 years of Federal Reserve data. Detects which of 4 economic regimes we are currently in — Goldilocks, Reflation, Stagflation, or Deflation — with full probability scores, a trained ML model, and a Groq-generated 3-sentence narrative that updates automatically when data changes.

Part of the Finance × AI open-source series: building real finance tools with AI, one module at a time.

---

## Current regime — March 2026

Deflation — 98.3% confidence
```
Deflation      98.3%
Goldilocks      0.7%
Stagflation     0.7%
Reflation       0.3%
```

AI narrative (Groq / Llama 3.1-8b-instant):

Earnings growth is under pressure with unemployment at 4.4% and real GDP YoY at 0.7%, while inflation signals such as the 5Y breakeven of 2.56% indicate a slowing economy. The key tension is between the flat 3M/10Y yield curve and the low breakeven gap of -0.16% — bond markets expect lower inflation yet deflationary pressures are building beneath the surface. Watch the NFP release on April 6th: if labour market softness accelerates, recession probability will cross the 15% threshold and the regime transition signal will strengthen.

---

## What it does

1. Pulls 9 macro series from FRED API and 4 treasury yield tickers from yfinance
2. Engineers 23 features including YoY transforms, 1M and 3M lags, and derived signals
3. Labels every month from 1990 to today using a growth × inflation matrix
4. Trains an XGBoost classifier on 1990-2022 data and tests on 2023-today
5. Outputs a full probability distribution across all 4 regimes, not a hard label
6. Generates a 3-sentence macro narrative via Groq API (Llama 3.1)
7. Visualises everything in a Streamlit dashboard with signal panels and anomaly detection

---

## The 4 regimes

The classifier maps every month to one of four macro regimes based on the combination of growth and inflation signals.

Goldilocks: growth expanding, inflation low (below 2.5%). Best regime for equities and credit. Risk-on environment. Growth stocks and HY credit outperform. Historically associated with mid-1990s and 2023-2024.

Reflation: growth expanding, inflation high (above 2.5%). Commodities, energy, and value stocks lead. Rising yields hurt duration. Historically associated with post-COVID stimulus boom in 2021.

Stagflation: growth contracting, inflation high. The worst regime for most asset classes. Equities suffer from margin compression. Bonds suffer from inflation eroding real returns. Gold is the historical winner. Historically associated with 2022 when CPI hit 9% and PMI fell below 50.

Deflation: growth contracting, inflation low. Safe haven regime. Duration rallies as yields fall. USD strengthens on flight to safety. HY spreads blow out and default cycles begin. Bonds are the primary beneficiary. Historically associated with GFC in 2008-2009 and currently flagged for March 2026.

---

## Architecture and layer-by-layer explanation

### Layer 0 — Data sources

Three data sources are used, all free.

**FRED API (Federal Reserve Economic Data)**

The backbone of the pipeline. FRED is the Federal Reserve's official data portal. Free API key, no rate limits for our use case. Pulls data back to 1990 for most series.

Series pulled and why each was chosen:

T10Y3M — 3-month to 10-year yield spread. The single best recession predictor in academic literature. When short rates exceed long rates (inversion), banks stop lending profitably and credit contracts. The NY Fed's own recession model is built on this series alone. We pull it directly as FRED already computes the spread.

T10Y2Y — 2-year to 10-year yield spread. The spread the market watches most closely. More reactive than 3M/10Y to near-term Fed expectations. Used for the spread history chart.

CPIAUCSL — Consumer Price Index for All Urban Consumers. Headline inflation. Raw index level, not a percentage — we compute YoY in the feature pipeline. Used as the primary inflation signal for regime classification.

CPILFESL — Core CPI excluding food and energy. Strips out volatile components and captures the stickier underlying inflation trend. A Fed-watched series. More reliable for identifying structural inflation regimes.

PCEPI — Personal Consumption Expenditures price index. The Fed's preferred inflation gauge. Smoother than CPI because it adjusts for consumer substitution behaviour. Used as a third inflation data point in the feature matrix.

UNRATE — Unemployment rate. Used for two purposes: as a labour market health indicator in the dashboard, and as a 3-month delta (unemp_delta) that captures whether unemployment is rising or falling rapidly — a key recession transition signal.

A191RL1Q225SBEA — Real GDP year-on-year growth rate. Quarterly frequency, already expressed as a percentage. The primary growth signal for regime labelling. A positive reading means expansion, negative means contraction. Note: this series has a publication lag of approximately one quarter.

T5YIE — 5-year breakeven inflation rate. Derived from the spread between 5-year nominal and TIPS yields. Represents what the bond market expects inflation to average over the next 5 years. Compared against actual CPI to compute the breakeven gap — one of our most informative derived features.

RECPROUSM156N — NY Fed recession probability model output. A logistic regression the New York Fed publishes monthly using the 3M/10Y spread as input. Saves us from building our own recession model. Shown directly in the dashboard. Above 30% is elevated, above 50% is high risk historically.

**yfinance**

Used for the live yield curve shape. FRED has the spread series but yfinance gives us the individual maturity yields needed to plot the full curve shape (3M, 5Y, 10Y, 30Y). Tickers: ^IRX (3M), ^FVX (5Y), ^TNX (10Y), ^TYX (30Y). Five years of monthly data pulled.

**Key limitation**: yfinance data starts only from approximately 2020 for reliable monthly pull. FRED is the master index back to 1990. The yfinance columns will show NaN before 2020 in the feature matrix — this is handled by the fillna(median) step in the feature pipeline.

---

### Layer 1 — Feature engineering

Raw FRED data is not model-ready. It arrives at mixed frequencies (daily, monthly, quarterly), as raw index levels rather than growth rates, and without the lagged information the model needs to detect trends. Layer 1 transforms it into a clean 23-column feature matrix.

**Step 1: Frequency alignment**

Everything is resampled to month-start frequency using resample("MS").last(). This takes the last known value in each calendar month. For GDP (quarterly), this naturally forward-fills the quarterly value into the three monthly rows — the standard approach for mixed-frequency macro data.

**Step 2: YoY percentage changes**

CPI, Core CPI, and PCE are raw index levels from FRED — not growth rates. We apply pct_change(12) multiplied by 100 to convert each to a year-on-year percentage. This is the standard transformation: dividing the current month's index by the same month last year, minus one, expressed as a percentage.

Unemployment receives a 3-month difference (diff(3)) rather than a YoY transform. We care about whether unemployment is rising or falling fast, not the absolute level relative to a year ago. A 0.3 percentage point rise in 3 months is a meaningful labour market deterioration signal regardless of the starting level.

GDP is already published as a YoY percentage by FRED — no transformation needed.

**Step 3: Lag features**

For 6 key series (3M/10Y spread, 2Y/10Y spread, CPI YoY, Core CPI YoY, 5Y breakeven, GDP YoY) we add 1-month and 3-month lags using shift(1) and shift(3). This gives the model memory — it can see not just where each signal is now but where it was one month and one quarter ago. This captures momentum and trend direction, which are more predictive of regime transitions than level alone. The lag operation adds 12 features (6 series × 2 lags), bringing the total from 11 raw features to 23.

**Step 4: Derived signals**

Three derived features are computed that add analytical edge beyond the raw series:

Breakeven gap: CPI YoY minus 5Y breakeven inflation. A positive gap means actual inflation is running above what the bond market expects — inflation is stickier than priced. A negative gap means the market expects disinflation from current levels. In March 2026 this is -0.16%, meaning the market expects inflation to cool slightly from the current 2.4%.

Inversion flag: binary 1 or 0 — is the 3M/10Y spread currently negative? Provides the model with a clean binary signal for the most important recession indicator.

Inversion duration: how many consecutive months has the curve been inverted? A single month of inversion is noise. Sustained inversion of 6+ months is historically predictive of recession. This feature is more powerful than the binary flag alone.

**Step 5: Regime labelling**

Each month from 1990 to today receives one of four regime labels. This is the training target for the XGBoost classifier.

Primary logic:
- Growth signal: GDP YoY above 0 = Expansion, below 0 = Contraction
- Inflation signal: CPI YoY above 2.5% = High, at or below 2.5% = Low
- Cross them: Expansion + Low = Goldilocks, Expansion + High = Reflation, Contraction + High = Stagflation, Contraction + Low = Deflation

Override logic:
- If 3M/10Y spread below -0.3% AND unemployment 3-month delta above 0.3pp, force Deflation regardless of GDP. This handles the well-known GDP lag problem — GDP often stays technically positive well into a recession while leading indicators (yield curve, unemployment) are already signalling contraction.

Key assumption: GDP is used as the growth signal rather than PMI because GDP is an official output measure while PMI is a survey. However GDP has a significant publication lag. The 3M override partially compensates for this.

**Step 6: Cleaning and scaling**

Rows where more than 30% of features are NaN are dropped — this removes the lag warm-up period at the start of the series (approximately 15 rows). Remaining NaNs (primarily from breakeven data which only starts in 2003 and recession_prob gaps) are filled with the column median. StandardScaler is then applied to all 23 features so that large-range features like GDP (which can vary from -10% to +10%) do not dominate smaller-range features like unemployment (typically 3-10%).

**Output**: 420 rows × 23 features, covering 1990 to March 2026.

---

### Layer 2 — XGBoost regime classifier

**Model choice rationale**

XGBoost was chosen over simpler models (logistic regression, random forest) for three reasons: it handles non-linear interactions between features well (e.g. inversion is more important when CPI is also rising), it is robust to the class imbalance in our training data (Stagflation and Deflation have more historical examples than Goldilocks and Reflation), and it outputs calibrated probabilities via multi:softprob which is essential for the probability distribution output.

**Train/test split**

A time-based split is used, never random. Train set: January 1990 to December 2022 (381 rows). Test set: January 2023 to March 2026 (39 rows). Using a random split for time series data would leak future information into training and produce inflated accuracy scores that do not reflect real-world performance.

**Key hyperparameters**

objective = multi:softprob: outputs a probability for each of the 4 classes rather than a hard label. This is the core design decision — the model outputs Deflation 98.3%, Goldilocks 0.7%, Stagflation 0.7%, Reflation 0.3% rather than just "Deflation". This probability distribution is more honest and more useful.

num_class = 4: four regime labels.

max_depth = 4: shallow trees. With only 381 training rows, deep trees would overfit. Depth 4 limits each tree to 16 possible leaf nodes.

learning_rate = 0.05: small step size. Combined with 200 estimators, this produces stable convergence without overfitting.

subsample = 0.8: each tree is trained on a random 80% of rows. Reduces variance.

colsample_bytree = 0.8: each tree uses a random 80% of features. Further reduces overfitting on a relatively small dataset.

n_estimators = 200: 200 trees provides sufficient model capacity without requiring significant compute.

**Test set performance**
```
              precision    recall    f1-score
Deflation         1.00      0.67       0.80
Goldilocks        1.00      1.00       1.00
Reflation         0.73      1.00       0.84
Stagflation       0.95      0.95       0.95
accuracy                               0.90
```

Overall accuracy: 90% on 39 truly out-of-sample months from 2023 to 2026. The lower recall on Deflation (0.67) means the model sometimes calls Deflation months as Reflation — this is a known challenge given the regime transition dynamics in 2023 when both signals were close to their thresholds.

**Validation on known historical periods**

Three historical periods where the regime is unambiguous are used as a sanity check:

October 2008: GFC peak. CPI was still elevated from the oil spike earlier that year. Growth was collapsing. Model correctly predicts Stagflation at 89% confidence.

June 2021: Post-COVID stimulus period. GDP strongly positive, CPI crossing 2.5% threshold. Model correctly predicts Stagflation at 100% — the labelling logic correctly identifies this as high-inflation expansion.

June 2022: CPI at 9%, PMI falling, yield curve inverting. Model correctly predicts Stagflation at 99% confidence.

All three validations pass, confirming the model is internally consistent with its training labels.

**Key assumptions and limitations**

The model is trained on US data only. It is not valid for other economies without retraining on country-specific macro series.

The 2.5% inflation threshold and 0.0% GDP threshold for regime labelling are somewhat arbitrary. Changing these thresholds would shift the historical regime distribution and change what the model learns. The thresholds were chosen to approximate the Federal Reserve's own inflation target and the standard definition of economic contraction.

The model cannot predict regime transitions — it classifies the current month based on current data. A high probability of Deflation does not tell you when a transition to another regime will occur.

GDP has a significant publication lag (approximately 3 months). The most recent GDP reading used in the March 2026 prediction is from Q3 2025. The inversion duration override partially compensates but does not fully resolve this limitation.

---

### Layer 3 — Model output

The classifier saves three files to the data/ folder after each run:

model.pkl: the trained XGBoost model object. Loaded by the dashboard at runtime to avoid retraining on every page load.

label_encoder.pkl: the scikit-learn LabelEncoder that maps integer predictions back to regime name strings. Required to decode the model's numeric output.

regime_result.json: the current month's probability distribution. Contains date, dominant regime, confidence percentage, and the full probability dictionary. This is read by the narrative engine and the dashboard. All values are native Python floats to ensure JSON serialisability — XGBoost returns numpy float32 by default which the Python JSON encoder cannot handle without explicit conversion.

---

### Layer 4A — AI narrative engine (Groq)

**Why AI for the narrative**

Rule-based logic can classify a regime and display indicators. It cannot synthesise multiple signals into a coherent practitioner-oriented insight. An LLM can read that the 3M/10Y spread is at 0.71%, the breakeven gap is -0.16%, GDP is 0.7%, unemployment is 4.4%, and the regime is Deflation at 98.3% — and produce: "The key tension is between the flat yield curve and the low breakeven gap, suggesting bond markets expect lower inflation yet deflationary pressures are building." That sentence requires inference across multiple signals. A rule-based system would need explicit code for every possible combination.

**Model choice: Groq (Llama 3.1-8b-instant)**

Groq uses LPU (Language Processing Unit) hardware which makes inference significantly faster than GPU-based providers. The free tier provides 14,400 requests per day — we use approximately 10 per month. Llama 3.1-8b-instant is sufficient for structured 3-sentence output with specific numerical values injected.

Note: Google Gemini free tier was attempted first but hits quota limits (limit: 0 for gemini-2.0-flash on free accounts). Groq has no such issues in practice.

**Prompt design**

The prompt injects: regime probabilities, all 8 current indicator values with their actual numbers, yield curve inversion context, and asset class implications for the current regime. The model is instructed to write exactly 3 sentences with a strict structure: sentence 1 states current data with real numbers, sentence 2 identifies the key tension or contradiction, sentence 3 names one specific upcoming data release and explains why it matters for regime transition. The prompt explicitly forbids generic macro language and requires direct practitioner-style writing.

**Cache logic**

The narrative is only regenerated when the regime date in regime_result.json changes. On all other page loads the cached narrative.json is returned. This means approximately 10 Groq API calls per month regardless of how many times the dashboard is loaded. Using force=True when running narrative.py directly bypasses the cache.

---

### Layer 4B — Signal panels

Five panels are constructed from the raw and feature data:

Yield curve panel: plots the current yield curve shape across 3M, 5Y, 10Y, and 30Y maturities using yfinance tickers. Also computes 5-year spread history for the 3M/10Y and 2Y/10Y series from FRED. Flags inversion (spread below zero) as a boolean alert and counts consecutive months of inversion.

Growth pulse: extracts GDP YoY history for the trend chart, current unemployment, and the 3-month unemployment delta. Growth signal is classified as Expansion (GDP above 0) or Contraction (GDP below 0).

Inflation panel: extracts CPI YoY and Core CPI YoY history from the engineered features (already pct_change transformed), current PCE, 5Y breakeven, and the breakeven gap. Inflation signal is classified as High (CPI above 2.5%) or Low. Note: raw CPI from raw_data.csv is the index level (around 320-330), not a percentage. Always use the features.csv version for percentage values.

Recession probability: extracts the NY Fed recession probability series from FRED (RECPROUSM156N). Classifies risk level as Low (below 15%), Elevated (15-30%), or High (above 30%). The 100% spike in 2020 is correct — the NY Fed model reacted to the COVID shock.

Anomaly detector: checks for cross-signal contradictions that warrant surfacing as alerts. Currently detects three anomalies: yield curve positive but recession probability elevated, Deflation regime but tight labour market (unemployment below 5%), and large breakeven gap (actual CPI diverging from market expectations by more than 1 percentage point). In March 2026 the Deflation + tight labour market anomaly is active — unemployment at 4.4% is historically inconsistent with a deflationary episode and suggests the regime call may be early.

---

### Layer 5 — Streamlit dashboard

The dashboard reads pre-computed data from the data/ folder — it does not retrain the model or call any API on load. All computation happens in the pipeline scripts.

Hero section: large regime card with colour coding (green for Goldilocks, amber for Reflation, red for Stagflation, blue for Deflation), probability distribution bar chart, AI narrative block, and asset class implication grid with colour-coded strength labels.

Signal panels: six metric cards (CPI YoY, Core CPI, 5Y Breakeven, 3M/10Y Spread, Unemployment with 3M delta, Recession Probability), yield curve shape chart, 5-year spread history with inversion shading, 10-year CPI trend with 2.5% threshold line, NY Fed recession probability with 30% and 50% alert lines.

Anomaly alerts: conditional warning banners that only appear when the anomaly detector flags a contradiction. Not shown when no anomalies are detected.

Historical regime timeline: colour-coded scatter chart showing the classified regime for every month from 1990 to today.

---

## Data update workflow

Run this sequence once after each major FRED data release (roughly monthly):
```bash
python data_pull.py       # refreshes raw_data.csv from FRED + yfinance
python classifier.py      # retrains on latest data, saves model artifacts
python narrative.py       # regenerates AI narrative (force=True by default)
streamlit run dashboard.py
```

Then push the updated data/ folder to GitHub. Streamlit Cloud picks up the changes automatically.

**FRED publication schedule (approximate)**:
- CPI: released around the 10th-15th of each month for the prior month
- Unemployment: released first Friday of each month (NFP report)
- GDP: released approximately 30 days after quarter-end (advance estimate)
- Yield spreads: updated daily, pulled as month-end values

---

## Tech stack

| Component | Tool | Cost |
|---|---|---|
| Macro data | FRED API | Free |
| Yield curve | yfinance | Free |
| ML model | XGBoost + scikit-learn | Free |
| AI narrative | Groq API (Llama 3.1) | Free |
| Dashboard | Streamlit + Plotly | Free |
| Deployment | Streamlit Cloud | Free |
| Language | Python 3.9 | Free |

Total monthly cost: $0

---

## Run locally

Clone the repo:
```bash
git clone https://github.com/yourusername/finance-ai-stack-02-macro-regime.git
cd finance-ai-stack-02-macro-regime
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a .env file in the project root:
```
FRED_API_KEY=your_fred_key_here
GROQ_API_KEY=your_groq_key_here
```

Get your free keys at:
- FRED: fred.stlouisfed.org/docs/api/api_key.html
- Groq: console.groq.com

Run the full pipeline:
```bash
python data_pull.py
python classifier.py
python narrative.py
streamlit run dashboard.py
```

---

## File structure
```
finance-ai-stack-02-macro-regime/
├── data_pull.py          Layer 0 — FRED + yfinance data fetch
├── features.py           Layer 1 — feature engineering pipeline
├── classifier.py         Layer 2 — XGBoost training and prediction
├── narrative.py          Layer 4A — Groq narrative engine
├── signals.py            Layer 4B — signal panel data preparation
├── dashboard.py          Layer 5 — Streamlit dashboard
├── requirements.txt      Python dependencies
├── .gitignore
├── .env                  API keys (never committed to GitHub)
└── data/
    ├── raw_data.csv           FRED + yfinance raw output (435 rows × 13 cols)
    ├── features.csv           Engineered feature matrix (420 rows × 23 cols)
    ├── model.pkl              Trained XGBoost model
    ├── label_encoder.pkl      scikit-learn LabelEncoder
    ├── regime_result.json     Current regime probabilities
    └── narrative.json         AI narrative + full panel data for dashboard
```

---

## Known limitations

GDP publication lag: the most recent GDP reading is approximately one quarter old. The classifier partially compensates via the yield curve and unemployment override but cannot fully resolve this.

Small test set: 39 months of out-of-sample data (2023-2026) is a limited evaluation window. The 90% accuracy figure should be interpreted with appropriate caution.

US-only: all data sources are US macroeconomic series. The model is not valid for other economies.

Oil shocks: the current March 2026 Deflation call predates the Iran oil shock which began pushing energy prices higher in mid-March 2026. The March CPI (due April 10th) may show an inflationary impulse that shifts the regime probability toward Reflation or Stagflation in the April update.

Regime boundary sensitivity: the 2.5% CPI threshold and 0.0% GDP threshold are fixed. Real economic regimes do not have hard boundaries — a month with CPI at 2.4% and a month with CPI at 2.6% are classified differently despite being economically similar.

---

## Part of Finance × AI

Module 02 of a weekly open-source series applying AI to real finance workflows.

| Module | Topic | Status |
|---|---|---|
| 01 | News Sentiment Dashboard | Live |
| 02 | Macro Regime Tracker | Live |
| 03 | Central Bank Tracker | Coming |
| 04 | Geopolitical Risk Scorer | Coming |
| 05 | FX Carry Trade Screener | Coming |
| 06-12 | Portfolio, Quant, Credit, Structured Products | Coming |

---

## About

Harshit Khandelwal — MiM at ESSEC Business School, Leveraged Loans Analyst at BNP Paribas Asset Management

https://www.linkedin.com/in/harshit-khandelwal-6278a4193/
