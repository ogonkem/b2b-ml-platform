"""
lifecycle_tests/02_single_predict.py
────────────────────────────────────────────────────────────────────────────
Stage 2 — SINGLE PREDICT. Loads single_predict_holdout.csv (25 rows the
champion never trained on), strips the Status answer key, POSTs each row to
/v1/predict, and scores the returned probabilities against the retained
answers. Confirms /health reports the same model version stage 1 promoted.
"""
import sys
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BASE_URL, HEADERS, HOLDOUT_DIR,
    require, require_holdout_files, require_api_reachable, require_state,
    wait_for_model_version, score, print_score, banner,
)

HOLDOUT_FILE = "single_predict_holdout.csv"
REQUIRED_FIELDS = ["ID", "year", "loan_amount", "property_value", "income", "Credit_Score"]


def main():
    banner("STAGE 2 — SINGLE PREDICT on single_predict_holdout.csv")
    require_holdout_files(HOLDOUT_FILE)
    require_api_reachable()
    state = require_state("baseline_version")

    # ModelManager polls MLflow for the Production version on an interval
    # (MODEL_POLL_INTERVAL, default 60s) — wait for it to actually pick up
    # what stage 1 just promoted before sending any predictions, otherwise
    # they'd be served by whatever was Production beforehand.
    baseline_version = str(state["baseline_version"])
    served = wait_for_model_version(baseline_version, timeout_s=90)
    require(
        served == baseline_version,
        f"/health still reports model_version={served} after 90s, expected {baseline_version}",
    )
    print(f"  /health model_version = {served}  (matches baseline)")

    df = pd.read_csv(HOLDOUT_DIR / HOLDOUT_FILE)

    # /v1/predict requires these fields (see LoanApplication in app/main.py) —
    # a row missing one is a genuinely malformed application (would 422 for
    # a real caller too), not something the scoring exercise should count.
    skip_mask = df[REQUIRED_FIELDS].isna().any(axis=1)
    if skip_mask.any():
        print(f"  Skipping {skip_mask.sum()} row(s) missing a required field: "
              f"{df.loc[skip_mask, 'ID'].tolist()}")
    df = df.loc[~skip_mask]
    df = df.where(pd.notnull(df), None)

    y_true, y_prob = [], []
    with httpx.Client(timeout=30.0) as client:
        for _, row in df.iterrows():
            payload = row.drop("Status").to_dict()
            payload = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in payload.items()}
            # See app/main.py's predict_default() for why this rename is needed —
            # the trained column name has a hyphen, the JSON field can't.
            payload["co_applicant_credit_type"] = payload.pop("co-applicant_credit_type")

            resp = client.post(f"{BASE_URL}/v1/predict", headers=HEADERS, json=payload)
            require(resp.status_code == 200, f"ID={payload['ID']} -> HTTP {resp.status_code}: {resp.text}")
            body = resp.json()

            y_true.append(int(row["Status"]))
            y_prob.append(float(body["default_probability"]))

    result = score(y_true, y_prob)
    print_score("single_predict_holdout", result)
    print("\n[OK] Stage 2 complete.")


if __name__ == "__main__":
    main()
