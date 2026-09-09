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
# Provider-selectable via LLM_PROVIDER: "openai" (real OpenAI), "groq", or
# "ollama" (local). Groq and Ollama both expose an OpenAI-compatible chat-
# completions endpoint, so all three are reached through the same `openai`
# SDK, just pointed at a different base_url/api_key/model — no extra
# dependency needed. This is independent of rag_service's embeddings
# (still OpenAI-only, see rag_service/ingest_task.py's module docstring) —
# swapping the generation provider here does nothing for that separate
# external-API dependency; a real OPENAI_API_KEY is still required for
# ingestion/retrieval to work at all, regardless of LLM_PROVIDER.
#
# Model choice specifically for this call: a small, cheap chat model is
# enough because the prompt is heavily grounded (every number it may cite
# is handed to it verbatim in the prompt — see _build_synthesis_prompt in
# agent/graph.py) and the output is a short, templated explanation, not
# open-ended reasoning.
LLM_PROVIDER      = os.environ.get("LLM_PROVIDER", "openai").lower()
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY")
GROQ_LLM_MODEL    = os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-20b")
OLLAMA_BASE_URL   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL  = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2")
SYNTHESIS_MODEL   = "gpt-4o-mini"   # used only when LLM_PROVIDER == "openai"

_llm_client = None


def _llm():
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI
        if LLM_PROVIDER == "groq":
            _llm_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        elif LLM_PROVIDER == "ollama":
            # Ollama's API key is unchecked by the server — the OpenAI SDK
            # just requires the field to be non-empty.
            _llm_client = OpenAI(api_key="ollama", base_url=f"{OLLAMA_BASE_URL.rstrip('/')}/v1")
        else:
            _llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _llm_client


def _synthesis_model() -> str:
    return {"groq": GROQ_LLM_MODEL, "ollama": OLLAMA_LLM_MODEL}.get(LLM_PROVIDER, SYNTHESIS_MODEL)


def call_llm_synthesis(prompt: str) -> str:
    try:
        response = _llm().chat.completions.create(
            model=_synthesis_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
    except Exception as e:
        raise UpstreamServiceError(f"LLM synthesis call failed: {e}") from e
    return response.choices[0].message.content.strip()
