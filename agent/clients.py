"""
agent/clients.py
Thin HTTP clients for the three services this graph orchestrates
(Selastone's core API, rag_service, rule_engine — all separate deployments,
so these are real HTTP calls, not in-process function calls), plus the one
LLM call the graph makes (synthesize). No retries or circuit-breaking here —
a failure surfaces immediately as a 502 from POST /v1/agent/assess, since a
partial or guessed loan decision is worse than a clear failure for
something this auditable.

Every call forwards the same bearer token the caller sent to
POST /v1/agent/assess — each downstream service independently resolves it
via its own app.auth.verify_token, so the tenant_id is consistent across
every call without the agent needing to know how auth works at all.
"""
import os

import requests

SELASTONE_API_URL = os.environ.get("SELASTONE_API_URL", "http://api:8000")
RAG_SERVICE_URL   = os.environ.get("RAG_SERVICE_URL",   "http://rag_service:8001")
RULE_ENGINE_URL   = os.environ.get("RULE_ENGINE_URL",   "http://rule_engine:8002")

REQUEST_TIMEOUT = 15  # seconds


class UpstreamServiceError(RuntimeError):
    """A downstream service call failed or an LLM call failed. Raised
    rather than swallowed — every node here feeds an audited lending
    decision, so a silent partial result would be worse than a clear
    failure surfaced as a 502."""


def _post(service_name: str, url: str, json_body: dict, token: str) -> dict:
    try:
        response = requests.post(
            url, json=json_body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise UpstreamServiceError(f"{service_name} request failed: {e}") from e

    if response.status_code >= 400:
        raise UpstreamServiceError(f"{service_name} returned {response.status_code}: {response.text}")
    return response.json()


def call_predict(applicant: dict, token: str) -> dict:
    return _post("Selastone /v1/predict", f"{SELASTONE_API_URL}/v1/predict", applicant, token)


def call_retrieve(tenant_id: str, query: str, token: str, top_k: int = 5) -> list:
    return _post(
        "rag_service /v1/retrieve", f"{RAG_SERVICE_URL}/v1/retrieve",
        {"tenant_id": tenant_id, "query": query, "top_k": top_k}, token,
    )


def call_decide(
    tenant_id: str, risk_score: float, shap_factors: list, loan_type: str,
    applicant_profile: dict, use_fallback: bool, token: str,
) -> dict:
    return _post(
        "rule_engine /v1/decide", f"{RULE_ENGINE_URL}/v1/decide",
        {
            "tenant_id": tenant_id,
            "risk_score": risk_score,
            "shap_factors": shap_factors,
            "loan_type": loan_type,
            "applicant_profile": applicant_profile,
            "use_fallback": use_fallback,
        },
        token,
    )


# ── LLM synthesis (the only LLM call in the graph) ──────────────────────────
# Same provider/tradeoff as rag_service's embeddings (OpenAI text-embedding-
# 3-small) — see rag_service/ingest_task.py's module docstring for the full
# external-API discussion, not repeated here. Model choice specifically for
# this call: a small, cheap chat model is enough because the prompt is
# heavily grounded (every number it may cite is handed to it verbatim in
# the prompt — see _build_synthesis_prompt in agent/graph.py) and the
# output is a short, templated explanation, not open-ended reasoning.
SYNTHESIS_MODEL = "gpt-4o-mini"

_openai_client = None


def _openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def call_llm_synthesis(prompt: str) -> str:
    try:
        response = _openai().chat.completions.create(
            model=SYNTHESIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
    except Exception as e:
        raise UpstreamServiceError(f"LLM synthesis call failed: {e}") from e
    return response.choices[0].message.content.strip()
