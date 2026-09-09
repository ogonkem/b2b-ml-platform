"""
rule_engine/thresholds.py
Per-tenant rule configuration — a Python dict, not a DB table, mirroring
app/plans.py's code-not-DB pattern for the same reason: this changes rarely,
is reviewed like code (PR diff, not a runtime CRUD form), and every tenant's
numbers need to be traceable to a specific commit for audit purposes.

Threshold semantics (documented here because they're not self-evident from
the numbers alone — an auditor needs to know exactly which side of a
boundary a rule fires on):

  - Cap-style values (dti_soft_cap, dti_hard_cap, risk_score_review_threshold,
    first_time_borrower_cap) fire when the applicant's value STRICTLY EXCEEDS
    the configured number. Being exactly at the cap is compliant. This
    matches the three tenants' own wording where it was given explicitly
    ("risk score >65", "score >70") and is applied consistently to the caps
    whose comparison operator wasn't spelled out (the DTI caps), rather than
    silently picking a different rule for those.

  - risk_bands are classification buckets, not caps — a score is classified
    into whichever band's (low, high] range contains it (low exclusive, high
    inclusive), with the first band open at the bottom and the last band
    open at the top. This is a different question ("which tier is this
    score in") from "did a limit get breached", so it uses a different
    (inclusive-upper) convention. For example, commercial_bank's stated
    "0-30/31-55/56-70/71-100" integer bands become score<=30 / 30<score<=55
    / 55<score<=70 / score>70 here, since risk_score is a float and the
    literal integers 31/56/71 only matter for whole-number inputs — for a
    continuous score the meaningful boundary is the shared edge (30, 55, 70).

  - Each risk_band entry's "report" flag controls whether it appears in
    /v1/decide's triggered_thresholds output. A band whose own decision is
    "approve" and carries no further consequence (risk_band_low,
    full_approval_band) is not reported — nothing "fired" in any actionable
    sense, so listing it would be noise in an audit trail meant to show what
    triggered a non-default outcome. informal_digital_lender's
    reduced_approval_band is the one approve-decision band that IS reported,
    because unlike the other two it's still actionable: the caller needs to
    know to reduce the loan amount even though the top-level decision stays
    "approve".
"""

RULE_ENGINE_VERSION = "1.0.0"

# Resolves the final decision when more than one rule fires with different
# outcomes — the single most severe outcome across everything that fired
# wins, regardless of evaluation order.
DECISION_SEVERITY = {"approve": 0, "refer": 1, "reject": 2}

TENANT_THRESHOLDS = {
    # Numbers as given in this feature's spec for the "commercial bank" example
    # tenant. Note: tests/fixtures/policy_docs/1_commercial_bank_lending_policy.md
    # (written for the rag_service chunking tests) uses different illustrative
    # numbers (LTV/DSCR/approval-authority tiers, no DTI or risk bands) — the
    # two aren't currently reconciled to state the same numbers.
    "commercial_bank": {
        "dti_soft_cap": 0.35,
        "dti_hard_cap": 0.40,
        "risk_bands": [
            {"name": "risk_band_low",      "low": None, "high": 30,   "decision": "approve", "report": False},
            {"name": "risk_band_moderate", "low": 30,   "high": 55,   "decision": "refer",    "report": True},
            {"name": "risk_band_elevated", "low": 55,   "high": 70,   "decision": "refer",    "report": True},
            {"name": "risk_band_high",     "low": 70,   "high": None, "decision": "reject",   "report": True},
        ],
    },

    # No risk_bands here — SACCO's own policy states only a single review
    # threshold, not a full band structure; anything <= it is simply
    # approved by default (see engine.py's default-to-approve path when
    # neither risk_bands nor risk_score_review_threshold fires — not to be
    # confused with FALLBACK_RISK_BANDS below, a different concept).
    "microfinance_sacco": {
        "dti_soft_cap": 0.30,
        "dti_hard_cap": 0.45,
        "risk_score_review_threshold": 65,
    },

    # No DTI concept at all — this lender's policy is explicitly automated,
    # transaction-history-based scoring only, with no formal underwriting
    # committee and no manual review step (see engine.py: a "reject" here
    # is the harshest outcome this tenant's config can ever produce; there
    # is no "refer" path, matching the fixture doc's own description).
    "informal_digital_lender": {
        "risk_bands": [
            {"name": "full_approval_band",    "low": None, "high": 40,   "decision": "approve", "report": False},
            {"name": "reduced_approval_band", "low": 40,   "high": 70,   "decision": "approve", "report": True},
            {"name": "auto_decline_band",     "low": 70,   "high": None, "decision": "reject",  "report": True},
        ],
        # NGN — a first-time borrower requesting more than this is still
        # approved, just capped at this amount, not declined.
        "first_time_borrower_cap": 50_000,
    },
}

# Used by decide(..., use_fallback=True) — a generic, tenant-agnostic,
# risk-score-only band structure for when the caller (agent/graph.py) has
# no policy documents to ground a tenant-specific decision in (rag_service's
# /v1/retrieve returned zero chunks). Deliberately conservative and with no
# DTI/cap rules at all: applying a specific tenant's numeric policy without
# any retrieved policy text to cite would misrepresent the decision as
# policy-grounded when it isn't — better to fall back to a plain,
# generic judgment call and mark the response as not policy-aligned
# (agent/main.py sets policy_aligned: false whenever this path is used).
FALLBACK_RISK_BANDS = [
    {"name": "fallback_low",  "low": None, "high": 40,   "decision": "approve", "report": False},
    {"name": "fallback_mid",  "low": 40,   "high": 70,   "decision": "refer",   "report": True},
    {"name": "fallback_high", "low": 70,   "high": None, "decision": "reject",  "report": True},
]
