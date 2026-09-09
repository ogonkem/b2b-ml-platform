"""
tests/unit/test_rule_engine.py
Table-driven tests for rule_engine/engine.py — this is the piece that must
be provably deterministic for audit purposes, so it's tested at the same
level of rigor as notebooks/retrain.py: every threshold boundary, every
seeded tenant, and explicit edge cases (exact-boundary values, missing
optional inputs, an unknown tenant_id) rather than a handful of happy-path
checks.

No mocking anywhere in this file — rule_engine/engine.py has zero I/O (no
DB, no network, no filesystem beyond its own already-imported config
module), so every test calls decide() directly against real inputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from rule_engine.engine import UnknownTenantError, decide
from rule_engine.thresholds import RULE_ENGINE_VERSION, TENANT_THRESHOLDS


# ── Unknown tenant_id ───────────────────────────────────────────────────────

def test_unknown_tenant_raises_and_names_the_offending_id():
    with pytest.raises(UnknownTenantError) as exc:
        decide(tenant_id="not-a-real-tenant", risk_score=10, loan_type="x")
    assert "not-a-real-tenant" in str(exc.value)


def test_unknown_tenant_lists_the_known_tenants_in_the_error():
    with pytest.raises(UnknownTenantError) as exc:
        decide(tenant_id="nope", risk_score=10, loan_type="x")
    for real_tenant in TENANT_THRESHOLDS:
        assert real_tenant in str(exc.value)


@pytest.mark.parametrize("bad_tenant", [
    "",
    "COMMERCIAL_BANK",          # wrong case
    "commercial-bank",          # wrong separator
    "commercial_bank ",         # trailing whitespace typo
    " commercial_bank",         # leading whitespace typo
    "commercial_bank_2",
    "microfinance",             # partial match of a real tenant id
])
def test_near_miss_tenant_ids_all_raise_never_silently_match(bad_tenant):
    """A near-miss tenant_id must never silently resolve to a *different*
    real tenant's rules — that's a compliance failure, not a graceful
    fallback, so every one of these must fail loudly."""
    with pytest.raises(UnknownTenantError):
        decide(tenant_id=bad_tenant, risk_score=10, loan_type="x")


def test_valid_tenant_ids_all_work():
    for tenant_id in TENANT_THRESHOLDS:
        result = decide(tenant_id=tenant_id, risk_score=10, loan_type="x")
        assert result["decision"] in {"approve", "reject", "refer"}


# ── commercial_bank: risk score bands ──────────────────────────────────────

COMMERCIAL_BANK_BAND_CASES = [
    # (risk_score, expected_band_name, expected_decision, is_reported)
    (0,      "risk_band_low",      "approve", False),
    (15,     "risk_band_low",      "approve", False),
    (30,     "risk_band_low",      "approve", False),    # exactly at edge -> still low band
    (30.01,  "risk_band_moderate", "refer",   True),
    (45,     "risk_band_moderate", "refer",   True),
    (55,     "risk_band_moderate", "refer",   True),     # exactly at edge -> still moderate
    (55.01,  "risk_band_elevated", "refer",   True),
    (63,     "risk_band_elevated", "refer",   True),
    (70,     "risk_band_elevated", "refer",   True),     # exactly at edge -> still elevated
    (70.01,  "risk_band_high",     "reject",  True),
    (85,     "risk_band_high",     "reject",  True),
    (100,    "risk_band_high",     "reject",  True),
    (150,    "risk_band_high",     "reject",  True),     # out-of-nominal-range, still classified
    (-5,     "risk_band_low",      "approve", False),    # out-of-nominal-range, still classified
]


@pytest.mark.parametrize("risk_score,band_name,decision,is_reported", COMMERCIAL_BANK_BAND_CASES)
def test_commercial_bank_risk_bands(risk_score, band_name, decision, is_reported):
    result = decide(tenant_id="commercial_bank", risk_score=risk_score, loan_type="mortgage")
    assert result["decision"] == decision
    fired_names = {t["rule"] for t in result["triggered_thresholds"]}
    if is_reported:
        assert band_name in fired_names
    else:
        assert band_name not in fired_names
        assert result["triggered_thresholds"] == []
    assert result["rule_engine_version"] == RULE_ENGINE_VERSION


# ── commercial_bank: DTI caps ────────────────────────────────────────────────

COMMERCIAL_BANK_DTI_CASES = [
    # (dti, fired_rule_or_None, decision_contribution)
    (0.0,     None,           "approve"),
    (0.10,    None,           "approve"),
    (0.34,    None,           "approve"),
    (0.35,    None,           "approve"),    # exactly at soft cap -> compliant, does not fire
    (0.3501,  "dti_soft_cap", "refer"),
    (0.36,    "dti_soft_cap", "refer"),
    (0.39,    "dti_soft_cap", "refer"),
    (0.40,    "dti_soft_cap", "refer"),      # exactly at hard cap -> still only soft (not > hard)
    (0.4001,  "dti_hard_cap", "reject"),
    (0.50,    "dti_hard_cap", "reject"),
    (1.00,    "dti_hard_cap", "reject"),
]


@pytest.mark.parametrize("dti,fired_rule,decision", COMMERCIAL_BANK_DTI_CASES)
def test_commercial_bank_dti_caps(dti, fired_rule, decision):
    # risk_score=0 keeps the (unreported) low risk band from adding noise.
    result = decide(
        tenant_id="commercial_bank", risk_score=0, loan_type="mortgage",
        applicant_profile={"dti": dti},
    )
    dti_rules = [t for t in result["triggered_thresholds"] if t["rule"].startswith("dti_")]
    if fired_rule is None:
        assert dti_rules == []
        assert result["decision"] == "approve"
    else:
        assert len(dti_rules) == 1
        assert dti_rules[0]["rule"] == fired_rule
        assert dti_rules[0]["applicant_value"] == dti
        assert result["decision"] == decision


def test_commercial_bank_hard_dti_breach_and_low_risk_band_only_reports_dti():
    """High DTI (reject) + low risk score (approve, unreported) -> overall
    decision is reject, and the DTI breach is the only thing surfaced."""
    result = decide(
        tenant_id="commercial_bank", risk_score=5, loan_type="mortgage",
        applicant_profile={"dti": 0.50},
    )
    assert result["decision"] == "reject"
    assert [t["rule"] for t in result["triggered_thresholds"]] == ["dti_hard_cap"]


def test_commercial_bank_soft_dti_breach_and_elevated_risk_both_refer_both_reported():
    result = decide(
        tenant_id="commercial_bank", risk_score=60, loan_type="mortgage",
        applicant_profile={"dti": 0.36},
    )
    assert result["decision"] == "refer"
    assert {t["rule"] for t in result["triggered_thresholds"]} == {"dti_soft_cap", "risk_band_elevated"}


def test_commercial_bank_reject_wins_even_if_it_fired_after_a_refer():
    """Order of evaluation must not matter — a later reject still overrides
    an earlier refer in the final decision."""
    result = decide(
        tenant_id="commercial_bank", risk_score=40, loan_type="mortgage",  # -> refer (moderate band)
        applicant_profile={"dti": 0.99},                                   # -> reject (hard cap)
    )
    assert result["decision"] == "reject"


def test_commercial_bank_dti_missing_only_risk_band_evaluated():
    result = decide(tenant_id="commercial_bank", risk_score=80, loan_type="mortgage", applicant_profile={})
    assert result["decision"] == "reject"
    assert [t["rule"] for t in result["triggered_thresholds"]] == ["risk_band_high"]


def test_commercial_bank_applicant_profile_entirely_omitted():
    """applicant_profile is optional — omitting it must not crash."""
    result = decide(tenant_id="commercial_bank", risk_score=10, loan_type="mortgage")
    assert result["decision"] == "approve"
    assert result["triggered_thresholds"] == []


def test_commercial_bank_has_no_first_time_borrower_cap():
    result = decide(
        tenant_id="commercial_bank", risk_score=10, loan_type="mortgage",
        applicant_profile={"is_first_time_borrower": True, "requested_amount": 1},
    )
    assert result["triggered_thresholds"] == []


# ── microfinance_sacco ───────────────────────────────────────────────────────

SACCO_RISK_CASES = [
    (0,     "approve", []),
    (40,    "approve", []),
    (65,    "approve", []),                          # exactly at review threshold -> compliant
    (65.01, "refer",   ["risk_score_review"]),
    (80,    "refer",   ["risk_score_review"]),
    (100,   "refer",   ["risk_score_review"]),
]


@pytest.mark.parametrize("risk_score,decision,rules", SACCO_RISK_CASES)
def test_sacco_risk_score_review_threshold(risk_score, decision, rules):
    result = decide(tenant_id="microfinance_sacco", risk_score=risk_score, loan_type="group_loan")
    assert result["decision"] == decision
    assert [t["rule"] for t in result["triggered_thresholds"]] == rules


SACCO_DTI_CASES = [
    (0.0,    None,           "approve"),
    (0.29,   None,           "approve"),
    (0.30,   None,           "approve"),    # exactly at soft cap -> compliant
    (0.3001, "dti_soft_cap", "refer"),
    (0.40,   "dti_soft_cap", "refer"),
    (0.45,   "dti_soft_cap", "refer"),      # exactly at hard cap -> still only soft
    (0.4501, "dti_hard_cap", "reject"),
    (0.70,   "dti_hard_cap", "reject"),
]


@pytest.mark.parametrize("dti,fired_rule,decision", SACCO_DTI_CASES)
def test_sacco_dti_caps(dti, fired_rule, decision):
    result = decide(
        tenant_id="microfinance_sacco", risk_score=0, loan_type="group_loan",
        applicant_profile={"dti": dti},
    )
    dti_rules = [t["rule"] for t in result["triggered_thresholds"] if t["rule"].startswith("dti_")]
    assert dti_rules == ([fired_rule] if fired_rule else [])
    assert result["decision"] == decision


def test_sacco_dti_soft_cap_and_review_threshold_combine_to_refer():
    result = decide(
        tenant_id="microfinance_sacco", risk_score=70, loan_type="group_loan",
        applicant_profile={"dti": 0.31},
    )
    assert result["decision"] == "refer"
    assert {t["rule"] for t in result["triggered_thresholds"]} == {"dti_soft_cap", "risk_score_review"}


def test_sacco_has_no_first_time_borrower_cap():
    result = decide(
        tenant_id="microfinance_sacco", risk_score=10, loan_type="group_loan",
        applicant_profile={"is_first_time_borrower": True, "requested_amount": 1},
    )
    assert result["triggered_thresholds"] == []


# ── informal_digital_lender ───────────────────────────────────────────────────

INFORMAL_LENDER_BAND_CASES = [
    (0,     "approve", []),
    (20,    "approve", []),
    (39.99, "approve", []),
    (40,    "approve", []),                          # exactly at edge -> full approval, not reduced
    (40.01, "approve", ["reduced_approval_band"]),
    (55,    "approve", ["reduced_approval_band"]),
    (70,    "approve", ["reduced_approval_band"]),   # exactly at edge -> still reduced
    (70.01, "reject",  ["auto_decline_band"]),
    (90,    "reject",  ["auto_decline_band"]),
    (100,   "reject",  ["auto_decline_band"]),
]


@pytest.mark.parametrize("risk_score,decision,rules", INFORMAL_LENDER_BAND_CASES)
def test_informal_lender_risk_bands(risk_score, decision, rules):
    result = decide(tenant_id="informal_digital_lender", risk_score=risk_score, loan_type="cash_advance")
    assert result["decision"] == decision
    assert [t["rule"] for t in result["triggered_thresholds"]] == rules


def test_informal_lender_never_produces_refer():
    """This tenant's policy is fully automated with no manual-review step —
    its config has no rule capable of producing "refer" at all."""
    for score in range(0, 101, 5):
        result = decide(tenant_id="informal_digital_lender", risk_score=score, loan_type="cash_advance")
        assert result["decision"] != "refer"


FIRST_TIME_BORROWER_CASES = [
    # (is_first_time, requested_amount, cap_fires)
    (True,  50_000,    False),   # exactly at cap -> compliant, does not fire
    (True,  50_000.01, True),
    (True,  60_000,    True),
    (False, 60_000,    False),   # not a first-timer -> cap never applies regardless of amount
]


@pytest.mark.parametrize("is_first_time,requested_amount,cap_fires", FIRST_TIME_BORROWER_CASES)
def test_informal_lender_first_time_borrower_cap(is_first_time, requested_amount, cap_fires):
    result = decide(
        tenant_id="informal_digital_lender", risk_score=10, loan_type="cash_advance",
        applicant_profile={"is_first_time_borrower": is_first_time, "requested_amount": requested_amount},
    )
    cap_rules = [t for t in result["triggered_thresholds"] if t["rule"] == "first_time_borrower_cap"]
    if cap_fires:
        assert len(cap_rules) == 1
        assert cap_rules[0]["applicant_value"] == requested_amount
        assert cap_rules[0]["threshold"] == 50_000
    else:
        assert cap_rules == []
    # The cap only ever limits the amount — it can never turn an approval
    # into a reject/refer by itself.
    assert result["decision"] == "approve"


def test_informal_lender_cap_requires_requested_amount_present():
    """A first-timer with no requested_amount in the profile can't have the
    cap evaluated at all — it must be silently skipped, not crash and not
    be treated as either a pass or a breach."""
    result = decide(
        tenant_id="informal_digital_lender", risk_score=10, loan_type="cash_advance",
        applicant_profile={"is_first_time_borrower": True},
    )
    assert result["triggered_thresholds"] == []
    assert result["decision"] == "approve"


def test_informal_lender_cap_does_not_override_a_reject_from_risk_band():
    result = decide(
        tenant_id="informal_digital_lender", risk_score=90, loan_type="cash_advance",
        applicant_profile={"is_first_time_borrower": True, "requested_amount": 100_000},
    )
    assert result["decision"] == "reject"
    assert {t["rule"] for t in result["triggered_thresholds"]} == {"auto_decline_band", "first_time_borrower_cap"}


def test_informal_lender_has_no_dti_rule():
    """This tenant's policy has no DTI concept at all (fully automated,
    transaction-history-based scoring only) — a dti key in
    applicant_profile must be silently ignored, never misapplied as if it
    were a different tenant's rule."""
    result = decide(
        tenant_id="informal_digital_lender", risk_score=10, loan_type="cash_advance",
        applicant_profile={"dti": 0.99},   # would be a severe breach under bank/SACCO rules
    )
    assert result["decision"] == "approve"
    assert result["triggered_thresholds"] == []


# ── shap_factors: accepted, currently unused by any seeded tenant's rules ────

@pytest.mark.parametrize("shap_factors", [
    None,
    [],
    [{"feature": "income", "value": -0.12}],
    [{"feature": "income", "value": -0.12}, {"feature": "credit_score", "value": 0.31}],
])
def test_shap_factors_variants_never_affect_the_decision(shap_factors):
    kwargs = dict(tenant_id="commercial_bank", risk_score=10, loan_type="mortgage")
    if shap_factors is not None:
        kwargs["shap_factors"] = shap_factors
    result = decide(**kwargs)
    assert result["decision"] == "approve"
    assert result["triggered_thresholds"] == []


def test_missing_shap_factors_produces_identical_result_to_empty_list():
    without = decide(tenant_id="commercial_bank", risk_score=45, loan_type="mortgage",
                      applicant_profile={"dti": 0.36})
    with_empty = decide(tenant_id="commercial_bank", risk_score=45, loan_type="mortgage",
                         applicant_profile={"dti": 0.36}, shap_factors=[])
    assert without == with_empty


# ── Output shape ──────────────────────────────────────────────────────────────

def test_output_has_exactly_the_documented_top_level_keys():
    result = decide(tenant_id="commercial_bank", risk_score=10, loan_type="mortgage")
    assert set(result.keys()) == {"decision", "triggered_thresholds", "rule_engine_version"}


def test_triggered_threshold_entries_have_exactly_the_documented_keys():
    result = decide(
        tenant_id="commercial_bank", risk_score=80, loan_type="mortgage",
        applicant_profile={"dti": 0.50},
    )
    assert result["triggered_thresholds"]   # sanity: this case does fire something
    for entry in result["triggered_thresholds"]:
        assert set(entry.keys()) == {"rule", "applicant_value", "threshold", "loan_type"}


def test_loan_type_is_echoed_into_every_triggered_threshold():
    result = decide(
        tenant_id="commercial_bank", risk_score=10, loan_type="real_estate_payment_plan",
        applicant_profile={"dti": 0.50},
    )
    assert result["triggered_thresholds"]
    assert all(t["loan_type"] == "real_estate_payment_plan" for t in result["triggered_thresholds"])


def test_decision_is_always_one_of_the_three_documented_values():
    for tenant_id in TENANT_THRESHOLDS:
        for score in [-10, 0, 25, 45, 60, 80, 100, 130]:
            for dti in [None, 0.1, 0.5, 0.9]:
                profile = {} if dti is None else {"dti": dti}
                result = decide(tenant_id=tenant_id, risk_score=score, loan_type="x", applicant_profile=profile)
                assert result["decision"] in {"approve", "reject", "refer"}


def test_rule_engine_version_is_present_and_matches_the_module_constant():
    result = decide(tenant_id="commercial_bank", risk_score=10, loan_type="x")
    assert result["rule_engine_version"] == RULE_ENGINE_VERSION
    assert isinstance(RULE_ENGINE_VERSION, str) and RULE_ENGINE_VERSION


# ── Determinism ────────────────────────────────────────────────────────────────

def test_same_input_always_produces_the_same_output():
    """The whole point of a rule engine for audit purposes: zero randomness,
    zero hidden state, zero time-dependence in the decision itself."""
    kwargs = dict(
        tenant_id="commercial_bank", risk_score=42.5, loan_type="mortgage",
        applicant_profile={"dti": 0.37}, shap_factors=[{"feature": "income", "value": 0.1}],
    )
    results = [decide(**kwargs) for _ in range(50)]
    assert all(r == results[0] for r in results)


def test_calls_do_not_share_or_mutate_state_across_invocations():
    """Confirms decide() isn't accidentally accumulating state (e.g. a
    module-level list being appended to instead of a fresh one per call),
    which would make results depend on call history rather than input."""
    decide(tenant_id="commercial_bank", risk_score=80, loan_type="x", applicant_profile={"dti": 0.9})
    result = decide(tenant_id="commercial_bank", risk_score=10, loan_type="x")
    assert result["triggered_thresholds"] == []
    assert result["decision"] == "approve"


# ── Fallback mode (use_fallback=True) ──────────────────────────────────────────
# Used by agent/graph.py when rag_harness's /v1/retrieve found no policy
# documents for the tenant — bypasses tenant-specific config entirely.

FALLBACK_BAND_CASES = [
    (0,     "approve", []),
    (39.99, "approve", []),
    (40,    "approve", []),                       # exactly at edge -> still low
    (40.01, "refer",   ["fallback_mid"]),
    (70,    "refer",   ["fallback_mid"]),          # exactly at edge -> still mid
    (70.01, "reject",  ["fallback_high"]),
    (100,   "reject",  ["fallback_high"]),
]


@pytest.mark.parametrize("risk_score,decision,rules", FALLBACK_BAND_CASES)
def test_fallback_mode_risk_bands(risk_score, decision, rules):
    result = decide(tenant_id="commercial_bank", risk_score=risk_score, loan_type="x", use_fallback=True)
    assert result["decision"] == decision
    assert [t["rule"] for t in result["triggered_thresholds"]] == rules


def test_fallback_mode_ignores_tenant_specific_dti_entirely():
    """A DTI that would hard-reject under commercial_bank's real policy must
    have zero effect in fallback mode — fallback is risk-score-only."""
    result = decide(
        tenant_id="commercial_bank", risk_score=10, loan_type="x",
        applicant_profile={"dti": 0.99}, use_fallback=True,
    )
    assert result["decision"] == "approve"
    assert result["triggered_thresholds"] == []


def test_fallback_mode_works_for_a_completely_unknown_tenant():
    """The whole point: a tenant not yet onboarded into TENANT_THRESHOLDS at
    all must still get a decision via fallback, not an UnknownTenantError."""
    result = decide(tenant_id="never-configured-tenant", risk_score=50, loan_type="x", use_fallback=True)
    assert result["decision"] == "refer"
    assert result["rule_engine_version"] == RULE_ENGINE_VERSION


def test_fallback_mode_is_independent_of_normal_mode_for_the_same_tenant():
    normal = decide(tenant_id="commercial_bank", risk_score=45, loan_type="x")
    fallback = decide(tenant_id="commercial_bank", risk_score=45, loan_type="x", use_fallback=True)
    assert normal["decision"] == "refer"     # risk_band_moderate
    assert fallback["decision"] == "refer"   # fallback_mid — same outcome here, different rule
    assert normal["triggered_thresholds"][0]["rule"] == "risk_band_moderate"
    assert fallback["triggered_thresholds"][0]["rule"] == "fallback_mid"
