"""
app/payments.py
Thin Paystack REST client — plain `requests` calls, matching this
project's existing style of thin clients (redis-py, minio,
clickhouse-connect, psycopg2 are all used directly, nowhere else wraps an
SDK). Paystack's hosted Checkout means no card data ever reaches this app;
its subscription "manage link" is the closest equivalent to a billing
portal — there's no separate customer-portal product to stand up.
"""
import hashlib
import hmac
import os

import requests

from app.plans import CHECKOUT_PLANS

PAYSTACK_BASE_URL = os.environ.get("PAYSTACK_BASE_URL", "https://api.paystack.co")
PAYSTACK_SECRET_KEY = os.environ.get("PAYSTACK_SECRET_KEY", "")
PAYSTACK_CURRENCY = os.environ.get("PAYSTACK_CURRENCY", "NGN")

# Only tiers with a fixed price (see app.plans.CHECKOUT_PLANS) are sold
# through automated checkout — e.g. PAYSTACK_PLAN_STARTER, PAYSTACK_PLAN_PRO.
PAYSTACK_PLAN_CODES = {
    plan_id: os.environ.get(f"PAYSTACK_PLAN_{plan_id.upper()}")
    for plan_id in CHECKOUT_PLANS
}

# app/plans.py prices are USD-denominated (the displayed "$49/mo" labels);
# Paystack needs an amount in whatever currency the account actually
# supports. Fixed test/demo-grade conversion rate, not a live FX lookup —
# same table scripts/setup_paystack_plans.py used to create these plans in
# the first place, so a checkout amount always matches what the plan was
# created with.
_USD_CONVERSION_RATE = {"USD": 1, "NGN": 1_500, "GHS": 15, "KES": 130, "ZAR": 18}.get(PAYSTACK_CURRENCY, 1)


def usd_to_paystack_subunits(usd_amount: float) -> int:
    return round(usd_amount * _USD_CONVERSION_RATE * 100)


def _headers() -> dict:
    return {"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}", "Content-Type": "application/json"}


def initialize_transaction(email: str, plan_code: str, amount_subunits: int, callback_url: str) -> dict:
    """Starts a hosted-checkout transaction for a subscription plan.
    Paystack requires `amount` even when a `plan` is given — it doesn't
    infer the charge from the plan's own configured price — so it must
    match what the plan was created with (usd_to_paystack_subunits() on
    the same app.plans price_amount). Returns {"authorization_url": ...,
    "reference": ...}."""
    resp = requests.post(
        f"{PAYSTACK_BASE_URL}/transaction/initialize",
        headers=_headers(),
        json={"email": email, "plan": plan_code, "amount": amount_subunits, "callback_url": callback_url},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["data"]


def create_subscription(customer_code: str, plan_code: str, authorization_code: str) -> dict:
    """Attaching a `plan` to /transaction/initialize only charges once for
    that plan's amount — confirmed against a real Paystack account, which
    had no subscription record at all after a successful plan-attached
    charge. The recurring subscription itself has to be created
    explicitly, using the reusable card authorization from that first
    successful charge. Returns {"subscription_code": ..., ...}."""
    resp = requests.post(
        f"{PAYSTACK_BASE_URL}/subscription",
        headers=_headers(),
        json={"customer": customer_code, "plan": plan_code, "authorization": authorization_code},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["data"]


def get_manage_link(subscription_code: str) -> str:
    """A Paystack-hosted page where the customer can update their card or
    cancel the subscription — the billing-portal equivalent."""
    resp = requests.get(
        f"{PAYSTACK_BASE_URL}/subscription/{subscription_code}/manage/link",
        headers=_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["data"]["link"]


def verify_webhook_signature(raw_body: bytes, signature: str) -> bool:
    """Paystack has no separate webhook-signing secret — the API secret
    key doubles as the HMAC-SHA512 signing key for the raw request body,
    compared against the x-paystack-signature header."""
    if not signature or not PAYSTACK_SECRET_KEY:
        return False
    computed = hmac.new(PAYSTACK_SECRET_KEY.encode("utf-8"), raw_body, hashlib.sha512).hexdigest()
    return hmac.compare_digest(computed, signature)
