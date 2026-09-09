"""
app/plans.py
Subscription tier catalog. A tenant's plan (app.tenants.plan, see app/db.py)
determines its monthly prediction/batch-row quota. Paid tiers with a fixed
`price_amount` are billed through Paystack (app/payments.py) via a hosted
checkout — self-service, no card data ever touches this app. Enterprise has
no fixed price (`price_amount` is None) and is sales-assisted, not sold
through automated checkout.

rag_ingestion_quota / rag_retrieval_quota: bundled monthly allowance for
rag_harness (RAG document ingestion / retrieval), enforced the same way as
monthly_quota is for predictions — see rag_harness/main.py. Not billed as a
separate line item yet; every tier's allowance is included in its existing
price. rag_ingestion_quota counts documents ingested (POST /v1/documents),
not MB — see rag_harness/main.py for why.

"agent" tier: one price covering prediction quota + agent orchestration
(POST /v1/agent/assess), with RAG quota bundled in like every other tier.
There is no separate "agent calls/mo" quota field — agent/graph.py's
predict and retrieve steps call Selastone's own /v1/predict and
rag_harness's /v1/retrieve, which already enforce monthly_quota and
rag_retrieval_quota respectively, so an assessment is already metered
through those two existing counters with no new billing dimension needed.
"""

PLANS = {
    "free": {
        "label": "Free",
        "monthly_quota": 100,
        "rag_ingestion_quota": 5,
        "rag_retrieval_quota": 50,
        "price_label": "$0/mo",
        "price_amount": 0,
        "description": "Evaluate the API and score a small volume of applications.",
    },
    "starter": {
        "label": "Starter",
        "monthly_quota": 1_000,
        "rag_ingestion_quota": 50,
        "rag_retrieval_quota": 1_000,
        "price_label": "$49/mo",
        "price_amount": 49,
        "description": "For a single team running predictions in production.",
    },
    "pro": {
        "label": "Pro",
        "monthly_quota": 5_000,
        "rag_ingestion_quota": 250,
        "rag_retrieval_quota": 5_000,
        "price_label": "$199/mo",
        "price_amount": 199,
        "description": "Higher-volume batch scoring across multiple loan books.",
    },
    "agent": {
        "label": "Agent",
        "monthly_quota": 10_000,
        "rag_ingestion_quota": 500,
        "rag_retrieval_quota": 10_000,
        "price_label": "$349/mo",
        "price_amount": 349,
        "description": "Full agentic loan decisioning (POST /v1/agent/assess) — automated, "
                        "retrieval-grounded, LLM-explained decisions on top of your prediction "
                        "and RAG quota. Agent orchestration itself isn't metered separately: "
                        "each assessment consumes one prediction plus whatever RAG retrieval "
                        "it performs, both already covered by the quotas above.",
    },
    "enterprise": {
        "label": "Enterprise",
        "monthly_quota": 25_000,
        "rag_ingestion_quota": 2_000,
        "rag_retrieval_quota": 50_000,
        "price_label": "Contact us",
        "price_amount": None,
        "description": "Custom quota, dedicated support, and SLA guarantees.",
    },
}

DEFAULT_PLAN = "free"

# Tiers sold through automated Paystack checkout — a fixed price_amount is
# required; enterprise is sales-assisted and never goes through checkout.
CHECKOUT_PLANS = [plan_id for plan_id, details in PLANS.items() if details["price_amount"]]
