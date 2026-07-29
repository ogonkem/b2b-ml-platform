"""
lifecycle_tests/07_verify_hotswap.py
────────────────────────────────────────────────────────────────────────────
Stage 7 — VERIFY HOTSWAP. Closes the loop: whatever stage 6 concluded
(promoted a challenger, or kept the baseline), confirms /health and
/v1/predict actually agree on the live model version. This directly
re-validates the ModelManager hot-swap fix in app/main.py / app/model_manager.py
from earlier this session — the exact bug where a promoted model updated
/health's reported version but never reached live predictions.
"""
import sys
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BASE_URL, HEADERS, HOLDOUT_DIR,
    require, require_holdout_files, require_api_reachable, require_state,
    health, wait_for_model_version, banner,
)

HOLDOUT_FILE    = "single_predict_holdout.csv"
REQUIRED_FIELDS = ["ID", "year", "loan_amount", "property_value", "income", "Credit_Score"]


def main():
    banner("STAGE 7 — VERIFY HOTSWAP (/health and /v1/predict agree on model version)")
    require_holdout_files(HOLDOUT_FILE)
    require_api_reachable()
    state = require_state("baseline_version", "promoted")

    expected_version = str(state.get("production_version") or state["baseline_version"])
    print(f"  Expecting model_version = {expected_version}  "
          f"({'promoted challenger' if state['promoted'] else 'baseline retained'})")

    served = wait_for_model_version(expected_version, timeout_s=90)
    require(
        served == expected_version,
        f"/health reports model_version={served} after 90s, expected {expected_version}",
    )
    print(f"  /health model_version = {served}")

    df = pd.read_csv(HOLDOUT_DIR / HOLDOUT_FILE)
    df = df.loc[~df[REQUIRED_FIELDS].isna().any(axis=1)].head(3)
    df = df.where(pd.notnull(df), None)

    with httpx.Client(timeout=30.0) as client:
        for _, row in df.iterrows():
            payload = row.drop("Status").to_dict()
            payload = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in payload.items()}
            payload["co_applicant_credit_type"] = payload.pop("co-applicant_credit_type")

            resp = client.post(f"{BASE_URL}/v1/predict", headers=HEADERS, json=payload)
            require(resp.status_code == 200, f"ID={payload['ID']} -> HTTP {resp.status_code}: {resp.text}")

    h = health()
    require(
        str(h["model_version"]) == expected_version,
        f"model_version drifted after live predictions: {h['model_version']} != {expected_version}",
    )
    print(f"  {len(df)} /v1/predict calls served by model_version={h['model_version']} — matches /health")
    print("\n[OK] Stage 7 complete — hot-swap verified end to end.")


if __name__ == "__main__":
    main()
