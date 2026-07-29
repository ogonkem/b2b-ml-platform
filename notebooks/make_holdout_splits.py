"""
notebooks/make_holdout_splits.py
────────────────────────────────────────────────────────────────────────────
One-off, reproducible splitter for exercising the platform end-to-end against
held-out data: a training slice for selastone_kaggle_v4.ipynb, plus separate
single-predict / bulk-predict / labeled-data (drift) slices the champion
never saw during training.

notebooks/archive/Loan_Default.csv itself is left untouched — it stays the
baseline retrain.py and the daily ingestion DAG compare against. Everything
here is written to notebooks/archive/holdout/, which is gitignored: it's
fully regenerable from the DVC-tracked baseline with a fixed random_state,
so there's no reason to version it separately.

Row counts (of 148,670 total), carved sequentially so there is zero overlap:
  100,000  train_slice.csv              -> selastone_kaggle_v4.ipynb training data
      25   single_predict_holdout.csv   -> POST /v1/predict, one row at a time
   2,000   bulk_predict_holdout.csv     -> POST /v1/batch/upload
   1,500   labeled_data_drift.csv       -> POST /v1/labeled-data (engineered to trigger PSI drift)
   1,500   labeled_data_stable.csv      -> POST /v1/labeled-data (negative control, no drift)
  ~43,645  unused buffer                -> left in the source file, written nowhere

The drift slice is the 1,500 rows with the lowest Credit_Score in the
remainder pool — deliberately skewed so PSI vs. the full baseline blows past
the 0.2 retraining threshold (lands around 19, overwhelmingly over
threshold — see the self-check printed at the end of this script). The
stable slice is a plain random sample from the same pool, used to confirm
the daily DAG does *not* false-trigger on ordinary data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from shared.drift import check_drift

RANDOM_STATE = 42

SOURCE_CSV = Path(__file__).resolve().parent / "archive" / "Loan_Default.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "archive" / "holdout"

TRAIN_SIZE          = 100_000
SINGLE_PREDICT_SIZE = 25
BULK_PREDICT_SIZE   = 2_000
LABELED_SIZE        = 1_500   # each of stable + drift


def main():
    df = pd.read_csv(SOURCE_CSV)
    print(f"Loaded {SOURCE_CSV} — {len(df):,} rows")

    train_idx, rest_idx = train_test_split(
        df.index, train_size=TRAIN_SIZE, random_state=RANDOM_STATE, stratify=df["Status"]
    )
    single_idx, rest_idx = train_test_split(
        rest_idx, train_size=SINGLE_PREDICT_SIZE, random_state=RANDOM_STATE,
        stratify=df.loc[rest_idx, "Status"],
    )
    bulk_idx, rest_idx = train_test_split(
        rest_idx, train_size=BULK_PREDICT_SIZE, random_state=RANDOM_STATE,
        stratify=df.loc[rest_idx, "Status"],
    )

    # Deliberately skewed: the lowest-Credit_Score rows left in the pool.
    drift_idx = df.loc[rest_idx].nsmallest(LABELED_SIZE, "Credit_Score").index
    rest_idx  = rest_idx.difference(drift_idx)

    stable_idx, buffer_idx = train_test_split(
        rest_idx, train_size=LABELED_SIZE, random_state=RANDOM_STATE,
        stratify=df.loc[rest_idx, "Status"],
    )

    print("\nSplit sizes:")
    print(f"  train_slice              {len(train_idx):>7,}")
    print(f"  single_predict_holdout   {len(single_idx):>7,}")
    print(f"  bulk_predict_holdout     {len(bulk_idx):>7,}")
    print(f"  labeled_data_drift       {len(drift_idx):>7,}")
    print(f"  labeled_data_stable      {len(stable_idx):>7,}")
    print(f"  unused buffer            {len(buffer_idx):>7,}  (left in source, written nowhere)")

    slices = [set(train_idx), set(single_idx), set(bulk_idx), set(drift_idx), set(stable_idx)]
    for i in range(len(slices)):
        for j in range(i + 1, len(slices)):
            assert not (slices[i] & slices[j]), "overlap detected between holdout slices"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Training slice — full columns, what v4 trains on
    df.loc[train_idx].to_csv(OUTPUT_DIR / "train_slice.csv", index=False)
    print(f"\n✓ train_slice.csv              ({len(train_idx):,} rows)")

    # 2. Single-predict holdout — full columns incl. Status (answer key);
    #    the walkthrough script strips Status when building /v1/predict payloads
    df.loc[single_idx].to_csv(OUTPUT_DIR / "single_predict_holdout.csv", index=False)
    print(f"✓ single_predict_holdout.csv   ({len(single_idx):,} rows)")

    # 3. Bulk-predict holdout — feature columns only, no Status (mirrors a real
    #    tenant CSV); answer key held back separately for scoring after download
    df.loc[bulk_idx].drop(columns=["Status"]).to_csv(OUTPUT_DIR / "bulk_predict_holdout.csv", index=False)
    df.loc[bulk_idx, ["ID", "Status"]].to_csv(OUTPUT_DIR / "bulk_predict_answer_key.csv", index=False)
    print(f"✓ bulk_predict_holdout.csv     ({len(bulk_idx):,} rows)")
    print(f"✓ bulk_predict_answer_key.csv  ({len(bulk_idx):,} rows)")

    # 4/5. Labeled-data slices — /v1/labeled-data requires 'actual_outcome' or 'Status'
    for name, idx in [("labeled_data_drift", drift_idx), ("labeled_data_stable", stable_idx)]:
        df.loc[idx].rename(columns={"Status": "actual_outcome"}).to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        print(f"✓ {name}.csv    ({len(idx):,} rows)")

    (OUTPUT_DIR / ".gitignore").write_text("*.csv\n")

    # Self-verification — confirm the drift slice actually drifts and the
    # stable slice doesn't, against the full untouched baseline.
    print("\n── PSI check vs. full baseline ──")
    for name, idx in [("labeled_data_stable", stable_idx), ("labeled_data_drift", drift_idx)]:
        result = check_drift(df, df.loc[idx])
        tag = "DRIFTED >= 0.2" if result["drifted"] else "stable  < 0.2"
        print(f"  {name:<22} max_psi={result['max_psi']:>10.4f}  [{tag}]")
        print(f"    {result['psi_per_col']}")


if __name__ == "__main__":
    main()
