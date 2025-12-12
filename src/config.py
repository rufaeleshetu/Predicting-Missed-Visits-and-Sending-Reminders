from pathlib import Path

# ---------- Paths ----------

# Root = project folder (Predicting missed follow-up)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# Key files used in the pipeline
CHILD_MOM_MODEL_PATH = PROCESSED_DIR / "child_mom_model_df.csv"
CHILD_MOM_SCORED_PATH = DATA_DIR / "child_mom_scored.csv"
RF_MODEL_PATH = MODEL_DIR / "rf_missed_any.joblib"

# ---------- Modelling config ----------

RANDOM_STATE = 42

# Adjust if your actual column names differ
TARGET_COL = "missed_any"
ID_COLS = []  # e.g. ["hhid", "childid"] if you have explicit IDs

# Tuned RF hyperparameters – copy from random_search.best_params_
RF_PARAMS = {
    "n_estimators": 300,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "log2",
    "max_depth": 10,
}

# ---------- Risk band config ----------

# Quantiles for high / medium bands (probability cut points)
# We used 70th and 40th percentiles in the notebook
RISK_QUANTILES = {
    "high": 0.7,   # top 30% = high
    "medium": 0.4, # next 30% = medium
}
