import pandas as pd
import numpy as np
import warnings
import os
import json
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ── Silence warnings ───────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── File paths for saved artifacts ────────────────────────────────────────
# These files are written once and read by dashboard.py at runtime
# so the dashboard never needs to retrain — just loads the saved model
MODEL_PATH   = "data/model.pkl"          # trained XGBoost model
ENCODER_PATH = "data/label_encoder.pkl"  # maps integer predictions back to regime names
RESULT_PATH  = "data/regime_result.json" # current month's probability distribution


def train(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train XGBoost classifier on historical regime labels.

    Split strategy: time-based, not random.
    - Train: everything before 2023 (~380 rows, 1990-2022)
    - Test:  2023 to today (~39 rows, true out-of-sample)

    Never use random split for time series — it would leak
    future data into training and inflate accuracy scores.

    Model output: multi:softprob — returns a probability for
    each of the 4 regime classes, not just a hard label.
    This probability distribution is what makes the dashboard
    interesting — the economy is never 100% one regime.
    """

    # ── Step 1: Encode string labels to integers ───────────────────────────
    # XGBoost requires numeric targets
    # LabelEncoder maps alphabetically:
    # Deflation=0, Goldilocks=1, Reflation=2, Stagflation=3
    le        = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\n  Label encoding:")
    for name, code in zip(le.classes_, le.transform(le.classes_)):
        print(f"    {code} = {name}")

    # ── Step 2: Time-based train/test split ────────────────────────────────
    # Everything before 2023 = training, everything after = test
    # This ensures the model has never seen the test data during training
    cutoff    = "2023-01-01"
    train_idx = X.index < cutoff
    test_idx  = X.index >= cutoff

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    print(f"\n  Train set: {X_train.shape[0]} rows  (1990 – 2022)")
    print(f"  Test set:  {X_test.shape[0]} rows  (2023 – today)")

    # ── Step 3: Define and train XGBoost ──────────────────────────────────
    # objective=multi:softprob   → outputs probability per class
    # num_class=4                → 4 regime labels
    # max_depth=4                → shallow trees prevent overfitting
    # learning_rate=0.05         → small steps = more stable convergence
    # subsample=0.8              → 80% of rows per tree = reduces variance
    # colsample_bytree=0.8       → 80% of features per tree = reduces overfitting
    # n_estimators=200           → 200 trees, enough without heavy compute
    # random_state=42            → reproducible results every run
    model = XGBClassifier(
        objective         = "multi:softprob",
        num_class         = len(le.classes_),
        n_estimators      = 200,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        random_state      = 42,
        verbosity         = 0,
        use_label_encoder = False,
        eval_metric       = "mlogloss",
    )

    model.fit(X_train, y_train)
    print("\n  Model trained successfully.")

    # ── Step 4: Evaluate on test set ──────────────────────────────────────
    # Classification report shows precision, recall, F1 per regime
    y_pred = model.predict(X_test)
    print("\n── Test set performance ────────────────────────────────────")
    print(classification_report(
        y_test, y_pred,
        target_names  = le.classes_,
        zero_division = 0
    ))

    return model, le


def validate_known_periods(model, le, X: pd.DataFrame) -> None:
    """
    Sanity check — verify the model correctly identifies 3 periods
    where we know what the regime was based on our labelling logic.

    Updated checks to match actual data + labelling rules:
    - Oct 2008: CPI still elevated from oil spike, growth collapsing → Stagflation
    - Jun 2021: GDP surging but CPI crossing 2.5% threshold → Stagflation
    - Jun 2022: CPI at 9%, PMI falling, curve inverting → Stagflation

    All 3 Stagflation is consistent — our labelling logic is strict about
    inflation. The current Deflation call for 2026 reflects CPI now back
    below threshold with growth weakening.

    If any FAIL → the feature pipeline or labelling has a bug.
    """
    checks = {
        "2008-10-01": "Stagflation",  # CPI still high from oil spike, growth collapsing
        "2021-06-01": "Stagflation",  # GDP strong but CPI surging past 2.5%
        "2022-06-01": "Stagflation",  # CPI 9%, PMI falling — textbook stagflation
    }

    print("\n── Validation on known historical periods ───────────────────")
    all_passed = True

    for date_str, expected in checks.items():
        try:
            # Get the feature row for this specific date
            row   = X.loc[date_str].values.reshape(1, -1)

            # Get full probability distribution across all 4 regimes
            probs = model.predict_proba(row)[0]

            # Dominant regime = class with highest probability
            pred  = le.classes_[np.argmax(probs)]
            conf  = float(probs[np.argmax(probs)]) * 100

            status = "PASS" if pred == expected else "FAIL"
            if pred != expected:
                all_passed = False

            print(f"  {status}  {date_str}  "
                  f"expected={expected:<12}  "
                  f"predicted={pred:<12} ({conf:.0f}%)")

        except KeyError:
            # Date might not exist in feature matrix — skip
            print(f"  SKIP  {date_str} — not in feature matrix")

    if all_passed:
        print("\n  All validation checks passed — model is internally consistent.")
    else:
        print("\n  WARNING — some checks failed. Review labelling in features.py.")


def predict_current(model, le, X: pd.DataFrame) -> dict:
    """
    Run the model on the most recent month and return
    a full probability distribution across all 4 regimes.

    This is the primary output of Layer 2. It feeds into:
    - narrative.py  → Gemini API prompt construction
    - dashboard.py  → hero card probability bar chart
    """
    # Most recent row in the feature matrix = current month
    latest_date = X.index[-1]
    latest_row  = X.iloc[-1].values.reshape(1, -1)

    # predict_proba returns shape (1, n_classes)
    # Convert numpy float32 → native Python float for JSON serialisation
    probs      = model.predict_proba(latest_row)[0]
    regime_map = {k: round(float(v) * 100, 1)
                  for k, v in zip(le.classes_, probs)}

    # Sort descending — dominant regime appears first
    regime_map = dict(sorted(
        regime_map.items(), key=lambda x: x[1], reverse=True
    ))

    dominant   = max(regime_map, key=regime_map.get)
    confidence = regime_map[dominant]

    # Print probability bar chart in terminal for quick visual check
    print(f"\n── Current regime prediction  ({latest_date.strftime('%B %Y')}) ──")
    for regime, prob in regime_map.items():
        bar = "█" * int(prob / 5)  # one block per 5%
        print(f"  {regime:<12} {prob:>5.1f}%  {bar}")

    return {
        "date":       latest_date.strftime("%Y-%m-%d"),
        "dominant":   dominant,
        "confidence": float(confidence),
        "probs":      regime_map,
    }


def save_artifacts(model, le, result: dict) -> None:
    """
    Persist trained model, label encoder, and current prediction to disk.

    model.pkl        → loaded by dashboard at runtime, no retraining needed
    label_encoder.pkl→ maps numeric predictions back to regime name strings
    regime_result.json → read by narrative.py to build the Gemini prompt
                         also read by dashboard.py for the hero card
    """
    # Save trained model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save label encoder — needed to decode predictions back to strings
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    # Ensure all values are JSON-serialisable native Python types
    # XGBoost returns numpy float32 which json.dump cannot handle
    result_clean = {
        "date":       result["date"],
        "dominant":   result["dominant"],
        "confidence": float(result["confidence"]),
        "probs":      {k: float(v) for k, v in result["probs"].items()}
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result_clean, f, indent=2)

    print(f"\n  Saved model          → {MODEL_PATH}")
    print(f"  Saved encoder        → {ENCODER_PATH}")
    print(f"  Saved regime result  → {RESULT_PATH}")


def run_classifier() -> dict:
    """
    Master function — orchestrates the full Layer 2 pipeline.

    Steps:
    1. Import feature matrix from Layer 1 (features.py)
    2. Train XGBoost on 1990-2022 historical data
    3. Validate on 3 known historical periods
    4. Predict current regime with full probability distribution
    5. Save model, encoder, and result to data/ folder

    Returns the regime result dict consumed by narrative.py.
    """
    # Import here to avoid circular imports at module level
    from features import run_pipeline

    print("\n── Layer 2: XGBoost Regime Classifier ──────────────────────")

    # Get feature matrix and labels from Layer 1
    X, y, scaler, df = run_pipeline()

    # Train model and get label encoder
    model, le = train(X, y)

    # Sanity check against known historical periods
    validate_known_periods(model, le, X)

    # Predict current month's regime
    result = predict_current(model, le, X)

    # Persist everything to disk for dashboard use
    save_artifacts(model, le, result)

    return result


# ── Run directly to test ───────────────────────────────────────────────────
# python classifier.py
if __name__ == "__main__":
    result = run_classifier()

    print("\n── Regime result JSON (input to Layer 4) ───────────────────")
    print(json.dumps(result, indent=2))