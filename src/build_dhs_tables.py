import pandas as pd
from pathlib import Path

# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ---------- load DHS files ----------
ir_path = DATA_RAW / "ETIR81FL.DTA"   # women
kr_path = DATA_RAW / "ETKR81FL.DTA"   # children 0–4

print(f"Loading IR file: {ir_path}")
ir = pd.read_stata(ir_path, convert_categoricals=False)

print(f"Loading KR file: {kr_path}")
kr = pd.read_stata(kr_path, convert_categoricals=False)

# ---------------------------------------------------
# 1. mothers.csv  (one row per woman)
# ---------------------------------------------------
mothers_cols = [
    "v001",  # cluster
    "v002",  # household
    "v003",  # woman's line number
    "caseid",
    "v012",  # age
    "v025",  # urban/rural
    "v024",  # region
    "v106",  # education level
    "v190",  # wealth index
    "v208",  # number of births in last 5 years
]

mothers = ir[mothers_cols].copy()
mothers.rename(columns={
    "v001": "cluster",
    "v002": "household",
    "v003": "woman_line",
    "v012": "age",
    "v025": "urban_rural",
    "v024": "region",
    "v106": "education",
    "v190": "wealth_quintile",
    "v208": "births_5yrs",
}, inplace=True)

mothers.to_csv(DATA_PROC / "mothers.csv", index=False)
print("Saved data/processed/mothers.csv")

# ---------------------------------------------------
# 2. child_immunization.csv  (one row per child)
# ---------------------------------------------------
child_cols = [
    "v001", "v002", "v003", "caseid",
    "bidx",        # child index in mother's birth history
    "b2",          # date of birth (CMC)
    "b4",          # sex
    "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9",
    # BCG, DPT1, DPT2, DPT3, Polio0–3, Measles
]

children = kr[child_cols].copy()
children.rename(columns={
    "v001": "cluster",
    "v002": "household",
    "v003": "woman_line",
    "b2": "dob_cmc",
    "b4": "sex",
    "h1": "bcg",
    "h2": "dpt1",
    "h3": "dpt2",
    "h4": "dpt3",
    "h5": "polio0",
    "h6": "polio1",
    "h7": "polio2",
    "h8": "polio3",
    "h9": "measles1",
}, inplace=True)

children.to_csv(DATA_PROC / "child_immunization.csv", index=False)
print("Saved data/processed/child_immunization.csv")

# ---------------------------------------------------
# 3. child_immunization_with_mothers.csv
#    (child rows + mother characteristics)
# ---------------------------------------------------
combined = children.merge(
    mothers,
    on=["cluster", "household", "woman_line"],
    how="left",
    suffixes=("", "_mother"),
)

combined.to_csv(DATA_PROC / "child_immunization_with_mothers.csv", index=False)
print("Saved data/processed/child_immunization_with_mothers.csv")

print("✔ All tables built successfully.")
