"""
agent/graph.py
LangGraph orchestration for POST /v1/agent/assess. Six nodes, fixed
order — this is a loan-decisioning pipeline, not an agent that gets to
choose its own steps, so there is exactly one path through the graph every
time:

    predict -> build_retrieval_query -> retrieve -> decide -> synthesize -> audit

The only "branching" is a data flag (policy_aligned), not a different graph
path: retrieve always runs, decide always runs right after it, and whether
decide uses a tenant's real policy or the generic risk-score-only fallback
(rule_engine/thresholds.py's FALLBACK_RISK_BANDS) is just an argument
passed to it — never a fork in the graph itself.
"""
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from agent.clients import call_decide, call_llm_synthesis, call_predict, call_retrieve

TOP_K_RETRIEVAL = 5
SHAP_FACTORS_IN_PROMPT = 5

# Known finance-jargon feature names get a more natural phrase than a bare
# underscore-to-space swap would produce — e.g. "debt_to_income" needs to
# become "debt to income ratio", not just "debt to income", to read as a
# real search query against policy text. Unlisted features fall back to a
# plain underscore/hyphen-to-space conversion.
#
# dtir1/loan_to_income/loan_to_property are the actual feature names
# shared/features.py's FeaturePipeline produces (confirmed against a real
# trained model, not just the illustrative "debt_to_income" from this
# feature's own spec — see shared/features.py's _DERIVED_FEATURES and the
# raw training columns) — aliased here so real /v1/predict output builds a
# sensible query, not just the spec's own illustrative example.
_FEATURE_ALIASES = {
    "debt_to_income": "debt to income ratio",
    "dti": "debt to income ratio",
    "dtir1": "debt to income ratio",
    "loan_to_value": "loan to value ratio",
    "ltv": "loan to value ratio",
    "loan_to_property": "loan to value ratio",
    "loan_to_income": "loan to income ratio",
    "credit_score": "credit score",
    "credit_type": "credit type",
    "co-applicant_credit_type": "co-applicant credit type",
}


class AssessmentState(TypedDict, total=False):
    # Input
    tenant_id: str
    token: str
    applicant: Dict[str, Any]
    loan_type: str
    applicant_profile: Dict[str, Any]

    # predict
    application_ref: str
    model_version: str
    risk_score: float
    default_prediction: int
    default_probability: float
    shap_factors: List[Dict[str, Any]]

    # build_retrieval_query / retrieve
    retrieval_query: str
    retrieved_chunks: List[Dict[str, Any]]
    policy_aligned: bool
    policy_doc_version: str

    # decide
    decision: str
    triggered_thresholds: List[Dict[str, Any]]
    rule_engine_version: str

    # synthesize
    narrative: str


def _humanize(text: str) -> str:
    return text.replace("_", " ").replace("-", " ").strip()


# ── 1. predict ──────────────────────────────────────────────────────────────

def node_predict(state: AssessmentState) -> dict:
    result = call_predict(state["applicant"], state["token"])
    return {
        "application_ref":      str(result["application_id"]),
        "model_version":        result.get("model_version"),
        "default_prediction":   result["default_prediction"],
        "default_probability":  result["default_probability"],
        "risk_score":           result["risk_score"],
        "shap_factors":         result.get("shap_factors", []),
    }


# ── 2. build_retrieval_query (pure Python, no I/O) ──────────────────────────

def node_build_retrieval_query(state: AssessmentState) -> dict:
    shap_factors = state.get("shap_factors") or []
    loan_type_phrase = _humanize(state.get("loan_type", ""))

    if shap_factors:
        raw_name = shap_factors[0]["feature"]
        top_feature_phrase = _FEATURE_ALIASES.get(raw_name.strip().lower(), _humanize(raw_name))
        query = f"{top_feature_phrase} threshold {loan_type_phrase}".strip()
    else:
        # No SHAP signal at all (e.g. a degenerate/dev-mode model, or the
        # underlying model returned nothing) — fall back to a query keyed
        # only on the loan type rather than failing the whole graph.
        query = f"{loan_type_phrase} lending policy thresholds".strip()

    return {"retrieval_query": query}


# ── 3. retrieve ───────────────────────────────────────────────────────────────

def _resolve_policy_doc_version(chunks: List[Dict[str, Any]]) -> Any:
    """A decision can retrieve chunks from more than one document/version —
    this records the single doc_version the decision is most grounded in
    (the highest-ranked chunk's), not an exhaustive list, since
    agent.agent_decisions.policy_doc_version is one column, not an array.
    None when nothing was retrieved (the fallback path)."""
    return chunks[0]["doc_version"] if chunks else None


def node_retrieve(state: AssessmentState) -> dict:
    chunks = call_retrieve(state["tenant_id"], state["retrieval_query"], state["token"], top_k=TOP_K_RETRIEVAL)
    return {
        "retrieved_chunks":   chunks,
        "policy_aligned":     len(chunks) > 0,
        "policy_doc_version": _resolve_policy_doc_version(chunks),
    }


# ── 4. decide ─────────────────────────────────────────────────────────────────

def node_decide(state: AssessmentState) -> dict:
    result = call_decide(
        tenant_id=state["tenant_id"],
        risk_score=state["risk_score"],
        shap_factors=state.get("shap_factors", []),
        loan_type=state.get("loan_type", ""),
        applicant_profile=state.get("applicant_profile", {}),
        # Zero retrieved chunks -> no policy text to ground a tenant-specific
        # decision in -> tell rule_engine to use the generic fallback bands
        # instead of pretending this decision consulted a policy it didn't.
        use_fallback=not state.get("policy_aligned", True),
        token=state["token"],
    )
    return {
        "decision":             result["decision"],
        "triggered_thresholds": result["triggered_thresholds"],
        "rule_engine_version":  result["rule_engine_version"],
    }


# ── 5. synthesize (the only LLM call) ────────────────────────────────────────

def _build_synthesis_prompt(state: AssessmentState) -> str:
    shap_lines = "\n".join(
        f"  - {f['feature']}: {f['value']}"
        for f in (state.get("shap_factors") or [])[:SHAP_FACTORS_IN_PROMPT]
    ) or "  (none provided)"

    if state.get("policy_aligned"):
        chunk_lines = "\n".join(
            f"  - [{c.get('section_ref') or 'unlabeled section'}] {c['chunk_text']}"
            for c in state.get("retrieved_chunks", [])
        ) or "  (none)"
    else:
        chunk_lines = (
            "  No policy documents have been ingested for this tenant yet — this "
            "decision used a generic, risk-score-only fallback, not this tenant's "
            "specific policy."
        )

    threshold_lines = "\n".join(
        f"  - rule \"{t['rule']}\": applicant value {t['applicant_value']} vs. threshold "
        f"{t['threshold']} (loan_type: {t['loan_type']})"
        for t in state.get("triggered_thresholds", [])
    ) or "  (no threshold was breached)"

    return f"""You are explaining an automated loan decision to the applicant and the underwriter reviewing it.

Decision: {state.get("decision")}
Risk score: {state.get("risk_score")}

Statistical factors (SHAP contributions to the risk score, most impactful first):
{shap_lines}

Policy thresholds that fired — these are the ONLY numeric thresholds you may cite. Quote them verbatim; do not invent, round, or estimate any number not listed here:
{threshold_lines}

Retrieved policy text this decision is grounded in:
{chunk_lines}

Write a short, plain-language explanation (3-5 sentences) of this decision for a non-technical reader. You MUST:
- Reference at least one specific statistical factor from the list above.
- If any policy threshold fired, cite it by name and quote its exact threshold and applicant values from the list above verbatim — never make up a number.
- If a specific policy section is available above, reference it (e.g. "per section X"); if no policy documents were available, say so plainly rather than implying a specific policy clause was consulted.
- Not invent any fact, number, or policy detail that is not present in the information given above.
"""


def node_synthesize(state: AssessmentState) -> dict:
    prompt = _build_synthesis_prompt(state)
    narrative = call_llm_synthesis(prompt)
    return {"narrative": narrative}


# ── 6. audit ──────────────────────────────────────────────────────────────────

def node_audit(state: AssessmentState) -> dict:
    import json

    from agent.db import get_cursor

    chunk_ids = [c["chunk_id"] for c in state.get("retrieved_chunks", [])]

    with get_cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO agent.agent_decisions
                 (tenant_id, application_ref, risk_score, model_version,
                  rule_engine_version, decision, triggered_thresholds,
                  retrieved_chunk_ids, policy_doc_version, policy_aligned, llm_narrative)
               VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::uuid[], %s, %s, %s)""",
            (
                state["tenant_id"],
                state.get("application_ref"),
                state.get("risk_score"),
                state.get("model_version"),
                state.get("rule_engine_version"),
                state.get("decision"),
                json.dumps(state.get("triggered_thresholds", [])),
                chunk_ids,
                state.get("policy_doc_version"),
                state.get("policy_aligned", False),
                state.get("narrative"),
            ),
        )
    return {}


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AssessmentState)
    graph.add_node("predict", node_predict)
    graph.add_node("build_retrieval_query", node_build_retrieval_query)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("decide", node_decide)
    graph.add_node("synthesize", node_synthesize)
    graph.add_node("audit", node_audit)

    graph.set_entry_point("predict")
    graph.add_edge("predict", "build_retrieval_query")
    graph.add_edge("build_retrieval_query", "retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_edge("decide", "synthesize")
    graph.add_edge("synthesize", "audit")
    graph.add_edge("audit", END)

    return graph.compile()


GRAPH = build_graph()
