"""
app/plans.py
Subscription tier catalog. A tenant's plan (app.tenants.plan, see app/db.py)
determines its monthly prediction/batch-row quota. Paid tiers with a fixed
`price_amount` are billed through Paystack (app/payments.py) via a hosted
checkout — self-service, no card data ever touches this app. Enterprise has
no fixed price (`price_amount` is None) and is sales-assisted, not sold
through automated checkout.
"""

PLANS = {
    "free": {
        "label": "Free",
        "monthly_quota": 100,
        "price_label": "$0/mo",
        "price_amount": 0,
        "description": "Evaluate the API and score a small volume of applications.",
    },
    "starter": {
        "label": "Starter",
        "monthly_quota": 1_000,
        "price_label": "$49/mo",
        "price_amount": 49,
        "description": "For a single team running predictions in production.",
    },
    "pro": {
        "label": "Pro",
        "monthly_quota": 5_000,
        "price_label": "$199/mo",
        "price_amount": 199,
        "description": "Higher-volume batch scoring across multiple loan books.",
    },
    "enterprise": {
        "label": "Enterprise",
        "monthly_quota": 25_000,
        "price_label": "Contact us",
        "price_amount": None,
        "description": "Custom quota, dedicated support, and SLA guarantees.",
    },
}

DEFAULT_PLAN = "free"

# Tiers sold through automated Paystack checkout — a fixed price_amount is
# required; enterprise is sales-assisted and never goes through checkout.
CHECKOUT_PLANS = [plan_id for plan_id, details in PLANS.items() if details["price_amount"]]
