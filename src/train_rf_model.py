# src/train_rf_model.py
from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DEFAULT_DATA_PATH = Path("data/child_mom_model_df.csv")
DEFAULT_MODEL_PATH = Path("models/rf_missed_any.joblib")

def load_data(data_path: Path, target_col: str):
    df = pd.read_csv(data_path)

    possible_id_cols = ["cluster", "household", "woman_line", "caseid"]
    id_cols = [c for c in possible_id_cols if c in df.columns]

    feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]
    X = df[feature_cols]
    y = df[target_col].astype(int)

    return X, y, feature_cols

def train_model(data_path: Path, target_col: str, out_path: Path):
    X, y, feature_cols = load_data(data_path, target_col)

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X, y)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "features": feature_cols}, out_path)
    print(f"Saved model bundle to: {out_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Input CSV path")
    p.add_argument("--target", type=str, default="missed_any", help="Target column name")
    p.add_argument("--out", type=str, default=str(DEFAULT_MODEL_PATH), help="Output .joblib path")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(Path(args.data), args.target, Path(args.out))
