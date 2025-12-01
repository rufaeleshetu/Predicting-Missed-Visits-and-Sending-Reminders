# src/score_new_data.py
from pathlib import Path
import sys
import joblib
import pandas as pd

MODEL_PATH = Path("models/rf_missed_any.joblib")

LOW_THR = 0.2
HIGH_THR = 0.6

def assign_risk(p):
    if p < LOW_THR:
        return "low"
    elif p < HIGH_THR:
        return "medium"
    return "high"

def main(input_csv, output_csv):
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]

    df = pd.read_csv(input_csv)

    X = df[features]
    proba = model.predict_proba(X)[:, 1]

    df["rf_proba"] = proba
    df["risk_band"] = df["rf_proba"].apply(assign_risk)

    df.to_csv(output_csv, index=False)
    print(f"Saved scored data to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.score_new_data <input_csv> <output_csv>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
