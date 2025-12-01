# src/train_rf_model.py
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path("data/child_mom_model_df.csv")
MODEL_PATH = Path("models/rf_missed_any.joblib")

def load_data():
    df = pd.read_csv(DATA_PATH)

    target_col = "missed_any"
    possible_id_cols = ["cluster", "household", "woman_line", "caseid"]

    id_cols = [c for c in possible_id_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]

    X = df[feature_cols]
    y = df[target_col].astype(int)

    return X, y, feature_cols

def train_model():
    X, y, feature_cols = load_data()

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "features": feature_cols}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
