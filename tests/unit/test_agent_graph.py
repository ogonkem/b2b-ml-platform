"""
tests/unit/test_agent_graph.py
Pure-logic tests for agent/graph.py's build_retrieval_query node and prompt
construction — no mocking needed, since node_build_retrieval_query and
_build_synthesis_prompt do no I/O at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.graph import _build_synthesis_prompt, _humanize, node_build_retrieval_query


class TestHumanize:

    def test_underscores_become_spaces(self):
        assert _humanize("real_estate_payment_plan") == "real estate payment plan"

    def test_hyphens_become_spaces(self):
        assert _humanize("cash-advance") == "cash advance"


class TestBuildRetrievalQuery:

    def test_matches_the_spec_example_exactly(self):
        """top factor "debt_to_income" + loan_type "real_estate_payment_plan"
        -> "debt to income ratio threshold real estate payment plan" """
        state = {
            "shap_factors": [{"feature": "debt_to_income", "value": 0.31}],
            "loan_type": "real_estate_payment_plan",
        }
        result = node_build_retrieval_query(state)
        assert result["retrieval_query"] == "debt to income ratio threshold real estate payment plan"

    def test_uses_only_the_top_shap_factor(self):
        state = {
            "shap_factors": [
                {"feature": "credit_score", "value": -0.5},
                {"feature": "debt_to_income", "value": 0.31},
            ],
            "loan_type": "mortgage",
        }
        result = node_build_retrieval_query(state)
        assert result["retrieval_query"] == "credit score threshold mortgage"

    def test_unaliased_feature_name_falls_back_to_plain_humanization(self):
        state = {"shap_factors": [{"feature": "co_applicant_credit_type", "value": 0.1}], "loan_type": "mortgage"}
        result = node_build_retrieval_query(state)
        assert result["retrieval_query"] == "co applicant credit type threshold mortgage"

    def test_no_shap_factors_falls_back_to_loan_type_only_query(self):
        state = {"shap_factors": [], "loan_type": "cash_advance"}
        result = node_build_retrieval_query(state)
        assert result["retrieval_query"] == "cash advance lending policy thresholds"

    def test_missing_shap_factors_key_entirely_does_not_crash(self):
        result = node_build_retrieval_query({"loan_type": "group_loan"})
        assert result["retrieval_query"] == "group loan lending policy thresholds"

    def test_missing_loan_type_does_not_crash(self):
        state = {"shap_factors": [{"feature": "income", "value": 0.1}]}
        result = node_build_retrieval_query(state)
        assert result["retrieval_query"] == "income threshold"


class TestSynthesisPromptGrounding:
    """The prompt must hand the LLM every number it's allowed to cite,
    verbatim, and explicitly forbid inventing others — these tests confirm
    the actual threshold/shap values appear in the constructed prompt
    text, not just that some prompt gets built."""

    def _base_state(self, **overrides):
        state = {
            "decision": "reject",
            "risk_score": 85.5,
            "shap_factors": [{"feature": "debt_to_income", "value": 0.42}],
            "triggered_thresholds": [
                {"rule": "dti_hard_cap", "applicant_value": 0.42, "threshold": 0.4, "loan_type": "mortgage"},
            ],
            "policy_aligned": True,
            "retrieved_chunks": [
                {"chunk_id": "c1", "doc_id": "d1", "doc_version": "v1",
                 "section_ref": "4.2", "chunk_text": "DTI must not exceed 40 percent.", "score": 0.9},
            ],
        }
        state.update(overrides)
        return state

    def test_threshold_values_appear_verbatim(self):
        prompt = _build_synthesis_prompt(self._base_state())
        assert "0.42" in prompt
        assert "0.4" in prompt
        assert "dti_hard_cap" in prompt

    def test_instructs_the_model_not_to_invent_numbers(self):
        prompt = _build_synthesis_prompt(self._base_state())
        lowered = prompt.lower()
        assert "do not invent" in lowered or "never make up" in lowered
        assert "verbatim" in lowered

    def test_shap_factor_included(self):
        prompt = _build_synthesis_prompt(self._base_state())
        assert "debt_to_income" in prompt

    def test_section_ref_included_when_policy_aligned(self):
        prompt = _build_synthesis_prompt(self._base_state())
        assert "4.2" in prompt

    def test_policy_not_aligned_states_fallback_plainly_and_omits_chunks(self):
        state = self._base_state(policy_aligned=False, retrieved_chunks=[])
        prompt = _build_synthesis_prompt(state)
        assert "no policy documents" in prompt.lower()
        assert "fallback" in prompt.lower()

    def test_no_triggered_thresholds_still_produces_a_valid_prompt(self):
        state = self._base_state(triggered_thresholds=[], decision="approve")
        prompt = _build_synthesis_prompt(state)
        assert "no threshold was breached" in prompt.lower()
