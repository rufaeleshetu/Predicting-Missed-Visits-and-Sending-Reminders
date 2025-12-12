Synthetic sample data (safe)

This folder is used for synthetic (fully generated) data that matches the
project schema. It enables reproducible runs without including DHS microdata.

Important
- Do NOT commit real DHS microdata to this repository.
- CSV files in this folder are ignored by git by default.
- Generate synthetic data locally using:

python src/make_synthetic_sample.py --n 800 --out data_sample/child_mom_model_df_synthetic.csv
