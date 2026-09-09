"""
rule_engine/engine.py
Deterministic, code-only loan decisioning. No LLM, no external API calls,
no randomness, no wall-clock dependence — decide() is a pure function of
its arguments and rule_engine/thresholds.py's config, so the same input
always produces the exact same output. That determinism is the entire
point: this is the piece an auditor can re-run against a historical
decision and get an identical answer.

shap_factors is accepted but not currently read by any seeded tenant's
rules below — no tenant in thresholds.py defines a SHAP-based threshold
yet. It's part of the interface now so a future rule (e.g. flagging a
protected-class-correlated feature carrying unusually high SHAP weight)
doesn't require a breaking API change, and so today's caller can still
pass it through for its own audit log even though this module doesn't act
on it yet.
"""
from typing import Any, Dict, List, Optional

from rule_engine.thresholds import (
    DECISION_SEVERITY,
    FALLBACK_RISK_BANDS,
    RULE_ENGINE_VERSION,
    TENANT_THRESHOLDS,
)


class UnknownTenantError(ValueError):
    """tenant_id has no entry in TENANT_THRESHOLDS. Raised rather than
    falling back to some default rule set — applying the wrong tenant's
    lending policy to a real applicant is a compliance failure, not a
    degraded-but-safe fallback, so this must fail loudly instead."""


def decide(
    tenant_id: str,
    risk_score: float,
    shap_factors: Optional[List[Dict[str, Any]]] = None,
    loan_type: str = "",
    applicant_profile: Optional[Dict[str, Any]] = None,
    use_fallback: bool = False,
) -> Dict[str, Any]:
    """
    use_fallback=True bypasses tenant-specific config entirely (regardless
    of whether tenant_id is otherwise known) and applies FALLBACK_RISK_BANDS
    instead — a generic, risk-score-only judgment call with no DTI/cap
    rules. Intended for agent/graph.py's case where rag_harness's
    /v1/retrieve found no policy documents for this tenant: applying a
    tenant's specific numeric policy with no retrieved text to back it up
    would misrepresent the decision as grounded in a policy it never
    actually consulted. tenant_id is not validated against TENANT_THRESHOLDS
    in this mode, since the whole point is not to depend on that config.
    """
    if use_fallback:
        return _decide_fallback(risk_score, loan_type)

    if tenant_id not in TENANT_THRESHOLDS:
        raise UnknownTenantError(
            f"No rule configuration for tenant_id={tenant_id!r} — refusing to "
            f"fall back to another tenant's thresholds. Known tenants: "
            f"{sorted(TENANT_THRESHOLDS)}"
        )

    config = TENANT_THRESHOLDS[tenant_id]
    applicant_profile = applicant_profile or {}
    _ = shap_factors  # accepted, not yet consumed by any seeded tenant's rules — see module docstring

    decisions: List[str] = []     # every decision implied by any rule that matched, reported or not
    triggered: List[Dict[str, Any]] = []   # only the ones worth surfacing to the caller

    # ── DTI caps ────────────────────────────────────────────────────────────
    # Checked hard-then-soft so at most one DTI entry is ever reported — the
    # hard cap already implies the soft cap was also breached, and reporting
    # both would just be noise.
    dti = applicant_profile.get("dti")
    if dti is not None:
        if "dti_hard_cap" in config and dti > config["dti_hard_cap"]:
            decisions.append("reject")
            triggered.append(_record("dti_hard_cap", dti, config["dti_hard_cap"], loan_type))
        elif "dti_soft_cap" in config and dti > config["dti_soft_cap"]:
            decisions.append("refer")
            triggered.append(_record("dti_soft_cap", dti, config["dti_soft_cap"], loan_type))

    # ── Risk score: either a full band structure, or a single review cutoff ──
    if "risk_bands" in config:
        band = _match_band(risk_score, config["risk_bands"])
        if band is not None:
            decisions.append(band["decision"])
            if band["report"]:
                threshold_value = 0 if band["low"] is None else band["low"]
                triggered.append(_record(band["name"], risk_score, threshold_value, loan_type))
    elif "risk_score_review_threshold" in config:
        threshold = config["risk_score_review_threshold"]
        if risk_score > threshold:
            decisions.append("refer")
            triggered.append(_record("risk_score_review", risk_score, threshold, loan_type))

    # ── First-time-borrower amount cap ───────────────────────────────────────
    if "first_time_borrower_cap" in config:
        is_first_time = applicant_profile.get("is_first_time_borrower")
        requested_amount = applicant_profile.get("requested_amount")
        cap = config["first_time_borrower_cap"]
        if is_first_time and requested_amount is not None and requested_amount > cap:
            decisions.append("approve")   # capped, not declined — still an approval
            triggered.append(_record("first_time_borrower_cap", requested_amount, cap, loan_type))

    decision = max(decisions, key=lambda d: DECISION_SEVERITY[d]) if decisions else "approve"

    return {
        "decision": decision,
        "triggered_thresholds": triggered,
        "rule_engine_version": RULE_ENGINE_VERSION,
    }


def _decide_fallback(risk_score: float, loan_type: str) -> Dict[str, Any]:
    band = _match_band(risk_score, FALLBACK_RISK_BANDS)
    triggered: List[Dict[str, Any]] = []
    decision = band["decision"] if band is not None else "approve"
    if band is not None and band["report"]:
        threshold_value = 0 if band["low"] is None else band["low"]
        triggered.append(_record(band["name"], risk_score, threshold_value, loan_type))

    return {
        "decision": decision,
        "triggered_thresholds": triggered,
        "rule_engine_version": RULE_ENGINE_VERSION,
    }


def _record(rule: str, applicant_value: Any, threshold: Any, loan_type: str) -> Dict[str, Any]:
    return {
        "rule": rule,
        "applicant_value": applicant_value,
        "threshold": threshold,
        "loan_type": loan_type,
    }


def _match_band(score: float, bands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Classifies `score` into the (low, high] band that contains it — low
    exclusive, high inclusive, so a score exactly on a shared boundary
    belongs to the lower band. The first band's low is None (open at the
    bottom) and the last band's high is None (open at the top), so every
    real number matches exactly one band."""
    for band in bands:
        low, high = band["low"], band["high"]
        if low is None and score <= high:
            return band
        if high is None and score > low:
            return band
        if low is not None and high is not None and low < score <= high:
            return band
    return None
