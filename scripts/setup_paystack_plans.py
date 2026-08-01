"""
scripts/setup_paystack_plans.py
One-time setup: creates Paystack Plan objects for Selastone's paid tiers
(app/plans.py) and prints the resulting plan codes to paste into .env.
Run manually from the host with PAYSTACK_SECRET_KEY already set in .env —
never called by the app itself, never scheduled, never committed output.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import requests

from app.plans import PLANS, CHECKOUT_PLANS
from app.payments import PAYSTACK_BASE_URL, PAYSTACK_CURRENCY, usd_to_paystack_subunits

PAYSTACK_SECRET_KEY = os.environ.get("PAYSTACK_SECRET_KEY")


def main():
    if not PAYSTACK_SECRET_KEY:
        print("PAYSTACK_SECRET_KEY is not set — add it to .env first.")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}", "Content-Type": "application/json"}
    print(f"Creating Paystack plans in {PAYSTACK_CURRENCY}...\n")

    results = {}
    for plan_id in CHECKOUT_PLANS:
        details = PLANS[plan_id]
        amount_subunits = usd_to_paystack_subunits(details["price_amount"])
        resp = requests.post(
            f"{PAYSTACK_BASE_URL}/plan",
            headers=headers,
            json={
                "name": f"Selastone {details['label']}",
                "amount": amount_subunits,
                "interval": "monthly",
                "currency": PAYSTACK_CURRENCY,
            },
            timeout=10,
        )
        if not resp.ok:
            print(f"Failed to create plan '{plan_id}': {resp.status_code} {resp.text}")
            print("If this is a currency error, set PAYSTACK_CURRENCY to one your "
                  "account actually supports (commonly NGN) and rerun.")
            sys.exit(1)
        plan_code = resp.json()["data"]["plan_code"]
        results[plan_id] = plan_code
        print(f"  {details['label']:<12} (${details['price_amount']}/mo -> {PAYSTACK_CURRENCY} "
              f"{amount_subunits / 100:,.0f}/mo) -> {plan_code}")

    print("\nAdd these to .env:\n")
    for plan_id, code in results.items():
        print(f"PAYSTACK_PLAN_{plan_id.upper()}={code}")


if __name__ == "__main__":
    main()
