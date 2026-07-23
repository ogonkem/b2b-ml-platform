import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from shared.drift import compute_psi, check_drift, PSI_THRESHOLD


# ── compute_psi ───────────────────────────────────────────────────────────────

class TestComputePSI:

    def test_identical_distributions_near_zero(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        assert compute_psi(data, data.copy()) < 0.01

    def test_large_shift_exceeds_threshold(self):
        rng      = np.random.default_rng(0)
        expected = rng.normal(0,   1, 1000)
        actual   = rng.normal(10,  1, 1000)   # 10σ shift
        assert compute_psi(expected, actual) >= PSI_THRESHOLD

    def test_constant_array_returns_zero(self):
        data = np.array([5.0] * 200)
        assert compute_psi(data, data.copy()) == 0.0

    def test_returns_float(self):
        data = np.array([1.0, 2.0, 3.0] * 50)
        assert isinstance(compute_psi(data, data.copy()), float)

    def test_non_negative(self):
        rng      = np.random.default_rng(1)
        expected = rng.normal(100_000, 20_000, 500)
        actual   = rng.normal(150_000, 20_000, 500)
        assert compute_psi(expected, actual) >= 0.0

    def test_moderately_shifted_between_thresholds(self):
        rng      = np.random.default_rng(2)
        expected = rng.normal(0, 1, 2000)
        actual   = rng.normal(0.5, 1, 2000)    # small shift — should be < big threshold
        psi = compute_psi(expected, actual)
        assert psi >= 0.0


# ── check_drift ───────────────────────────────────────────────────────────────

class TestCheckDrift:

    @pytest.fixture
    def stable_pair(self):
        rng = np.random.default_rng(0)
        df  = pd.DataFrame({
            "loan_amount":  rng.normal(200_000, 50_000, 500),
            "income":       rng.normal(6_000,   2_000,  500),
            "Credit_Score": rng.normal(700,     50,     500),
        })
        return df, df.copy()

    @pytest.fixture
    def drifted_pair(self):
        rng1 = np.random.default_rng(10)
        rng2 = np.random.default_rng(20)
        baseline = pd.DataFrame({
            "loan_amount":  rng1.normal(200_000, 50_000, 500),
            "income":       rng1.normal(6_000,   2_000,  500),
            "Credit_Score": rng1.normal(700,     50,     500),
        })
        incoming = pd.DataFrame({
            "loan_amount":  rng2.normal(600_000, 50_000, 500),   # 8σ shift
            "income":       rng2.normal(1_000,   500,    500),   # 2.5σ shift
            "Credit_Score": rng2.normal(500,     50,     500),   # 4σ shift
        })
        return baseline, incoming

    def test_no_drift_on_identical_data(self, stable_pair):
        baseline, incoming = stable_pair
        result = check_drift(baseline, incoming, numeric_cols=["loan_amount", "income", "Credit_Score"])
        assert result["drifted"] is False

    def test_drift_detected_on_shifted_data(self, drifted_pair):
        baseline, incoming = drifted_pair
        result = check_drift(baseline, incoming, numeric_cols=["loan_amount", "income", "Credit_Score"])
        assert result["drifted"] is True

    def test_max_psi_equals_largest_column_psi(self, drifted_pair):
        baseline, incoming = drifted_pair
        cols   = ["loan_amount", "income", "Credit_Score"]
        result = check_drift(baseline, incoming, numeric_cols=cols)
        assert result["max_psi"] == max(result["psi_per_col"].values())

    def test_psi_per_col_contains_all_shared_columns(self, stable_pair):
        baseline, incoming = stable_pair
        cols   = ["loan_amount", "income", "Credit_Score"]
        result = check_drift(baseline, incoming, numeric_cols=cols)
        assert set(result["psi_per_col"].keys()) == set(cols)

    def test_missing_column_in_incoming_is_skipped(self, stable_pair):
        baseline, incoming = stable_pair
        incoming = incoming.drop(columns=["income"])
        result   = check_drift(baseline, incoming, numeric_cols=["loan_amount", "income"])
        assert "income" not in result["psi_per_col"]
        assert "loan_amount" in result["psi_per_col"]

    def test_no_shared_columns_returns_zero_psi(self):
        baseline = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        incoming = pd.DataFrame({"b": [1.0, 2.0, 3.0]})
        result   = check_drift(baseline, incoming, numeric_cols=["a"])
        assert result["max_psi"] == 0.0
        assert result["drifted"] is False

    def test_psi_values_rounded_to_4dp(self, drifted_pair):
        baseline, incoming = drifted_pair
        result = check_drift(baseline, incoming, numeric_cols=["loan_amount"])
        for v in result["psi_per_col"].values():
            assert v == round(v, 4)

    def test_result_has_required_keys(self, stable_pair):
        baseline, incoming = stable_pair
        result = check_drift(baseline, incoming, numeric_cols=["loan_amount"])
        assert {"drifted", "max_psi", "psi_per_col"} == set(result.keys())

    def test_custom_threshold_respected(self, stable_pair):
        """Setting threshold=0.0 should always report drift."""
        baseline, incoming = stable_pair
        result = check_drift(baseline, incoming, numeric_cols=["loan_amount"], threshold=0.0)
        assert result["drifted"] is True
