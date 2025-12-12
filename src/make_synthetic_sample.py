import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def calibrate_bias(scores: np.ndarray, target_rate: float, iters: int = 50) -> float:
    """
    Find bias b such that mean(sigmoid(scores - b)) ~= target_rate via binary search.
    """
    lo, hi = -20.0, 20.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        p = sigmoid(scores - mid).mean()
        if p > target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def make_synthetic_df(n: int, seed: int = 42, target_rate: float = 0.39) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- IDs (fake) ---
    cluster = rng.integers(1, 701, size=n)                 # 1..700
    household = rng.integers(1, 51, size=n)                # 1..50
    woman_line = rng.integers(1, 9, size=n)                # 1..8

    # --- Demographics / SES ---
    sex = rng.choice([1, 2], size=n, p=[0.51, 0.49])        # 1=male,2=female (example coding)
    urban_rural = rng.choice([1, 2], size=n, p=[0.25, 0.75])  # 1=urban,2=rural (example)
    region = rng.integers(1, 12, size=n)                    # 1..11
    education = rng.choice([0, 1, 2, 3], size=n, p=[0.35, 0.35, 0.20, 0.10])
    wealth_quintile = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.22, 0.21, 0.20, 0.19, 0.18])

    # --- Child age / birth timing ---
    age = rng.integers(0, 60, size=n)                       # months, 0..59
    ref_cmc = 1500                                          # arbitrary reference month code
    dob_cmc = ref_cmc - age

    births_5yrs = np.clip(rng.poisson(lam=1.2, size=n), 0, 6)

    # --- Vaccines (binary) ---
    # Make coverage correlated with age (older -> more likely received) and wealth/urban.
    age_factor = np.clip(age / 24, 0, 1)                    # 0..1
    wealth_factor = (wealth_quintile - 1) / 4               # 0..1
    urban_factor = (urban_rural == 1).astype(float)         # 1 if urban

    base_cov = 0.15 + 0.55 * age_factor + 0.15 * wealth_factor + 0.10 * urban_factor
    base_cov = np.clip(base_cov, 0.05, 0.95)

    def draw_vax(p):
        return (rng.random(n) < p).astype(int)

    bcg = draw_vax(np.clip(base_cov + 0.10, 0.05, 0.98))
    dpt1 = draw_vax(np.clip(base_cov + 0.05, 0.05, 0.98))
    dpt2 = draw_vax(np.clip(base_cov + 0.02, 0.05, 0.98))
    dpt3 = draw_vax(np.clip(base_cov - 0.02, 0.05, 0.98))

    polio0 = draw_vax(np.clip(base_cov + 0.08, 0.05, 0.98))
    polio1 = draw_vax(np.clip(base_cov + 0.04, 0.05, 0.98))
    polio2 = draw_vax(np.clip(base_cov + 0.01, 0.05, 0.98))
    polio3 = draw_vax(np.clip(base_cov - 0.03, 0.05, 0.98))

    measles1 = draw_vax(np.clip(base_cov - 0.08, 0.05, 0.98))

    vax_cols = [bcg, dpt1, dpt2, dpt3, polio0, polio1, polio2, polio3, measles1]
    n_received = np.sum(np.vstack(vax_cols), axis=0).astype(int)

    # --- Label: missed_any (synthetic) ---
    # Higher miss risk when fewer doses received, rural, lower wealth, lower education, younger.
    score = (
        1.6 * (1 - (n_received / 9.0)) +
        0.5 * (urban_rural == 2).astype(float) +
        0.35 * (wealth_quintile <= 2).astype(float) +
        0.25 * (education == 0).astype(float) +
        0.15 * (age < 12).astype(float) +
        rng.normal(0, 0.35, size=n)
    )

    bias = calibrate_bias(score, target_rate=target_rate)
    p_missed = sigmoid(score - bias)
    missed_any = (rng.random(n) < p_missed).astype(int)

    df = pd.DataFrame({
        "cluster": cluster,
        "household": household,
        "woman_line": woman_line,
        "dob_cmc": dob_cmc,
        "sex": sex,
        "age": age,
        "urban_rural": urban_rural,
        "region": region,
        "education": education,
        "wealth_quintile": wealth_quintile,
        "births_5yrs": births_5yrs,
        "bcg": bcg,
        "dpt1": dpt1,
        "dpt2": dpt2,
        "dpt3": dpt3,
        "polio0": polio0,
        "polio1": polio1,
        "polio2": polio2,
        "polio3": polio3,
        "measles1": measles1,
        "n_received": n_received,
        "missed_any": missed_any,
    })

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_rate", type=float, default=0.391)
    ap.add_argument("--out", type=str, default="data_sample/child_mom_model_df_synthetic.csv")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = make_synthetic_df(n=args.n, seed=args.seed, target_rate=args.target_rate)
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print("Shape:", df.shape)
    print("missed_any rate:", df["missed_any"].mean().round(3))


if __name__ == "__main__":
    main()
