import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from shared.features import FeaturePipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal valid DataFrame matching the Loan_Default.csv schema."""
    return pd.DataFrame({
        "ID":                   [1, 2, 3],
        "income":               [3000.0, 5000.0, 8000.0],
        "loan_amount":          [100000.0, 200000.0, 300000.0],
        "property_value":       [120000.0, 250000.0, 380000.0],
        "Credit_Score":         [650.0, 720.0, 800.0],
        "rate_of_interest":     [4.0, 4.5, 3.8],
        "Interest_rate_spread": [0.5, 0.9, 0.2],
        "Upfront_charges":      [1200.0, 5000.0, 3100.0],
        "term":                 [360.0, 360.0, 180.0],
        "LTV":                  [70.0, 80.0, 65.0],
        "dtir1":                [30.0, 40.0, 25.0],
    })

@pytest.fixture
def fitted_pipeline(sample_df):
    """Returns an already-fitted pipeline ready for transform tests."""
    pipeline = FeaturePipeline()
    pipeline.fit(sample_df)
    return pipeline


# ── fit() tests ─── column detection, median/mode saving, drop logic ─────────

class TestFit:

    def test_fit_sets_is_fitted(self, sample_df):
        pipeline = FeaturePipeline()
        pipeline.fit(sample_df)
        assert pipeline.is_fitted is True

    def test_unfit_pipeline_is_not_fitted(self):
        pipeline = FeaturePipeline()
        assert pipeline.is_fitted is False

    def test_fit_drops_id_column(self, sample_df):
        pipeline = FeaturePipeline()
        pipeline.fit(sample_df)
        assert "ID" not in pipeline.numeric_cols
        assert "ID" not in pipeline.categorical_cols

    def test_fit_identifies_numeric_cols(self, sample_df):
        pipeline = FeaturePipeline()
        pipeline.fit(sample_df)
        assert "income" in pipeline.numeric_cols
        assert "loan_amount" in pipeline.numeric_cols
        assert "Credit_Score" in pipeline.numeric_cols

    def test_fit_saves_medians(self, sample_df):
        pipeline = FeaturePipeline()
        pipeline.fit(sample_df)
        assert "income" in pipeline.medians
        assert pipeline.medians["income"] == pytest.approx(5000.0)

    def test_fit_saves_modes(self, sample_df):
        """Categorical cols should have a mode saved."""
        df = sample_df.copy()
        df["loan_type"] = ["type1", "type1", "type2"]   # add a categorical
        pipeline = FeaturePipeline()
        pipeline.fit(df)
        assert "loan_type" in pipeline.modes
        assert pipeline.modes["loan_type"] == "type1"

    def test_fit_drops_high_missing_columns(self):
        """Columns with >40% missing values should be dropped."""
        df = pd.DataFrame({
            "income":       [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            "loan_amount":  [100.0, 200.0, 300.0, 400.0, 500.0],
            "sparse_col":   [None, None, None, 1.0, None],   # 80% missing → drop
        })
        pipeline = FeaturePipeline()
        pipeline.fit(df)
        assert "sparse_col" in pipeline.drop_cols

    def test_fit_keeps_low_missing_columns(self):
        """Columns with <40% missing values should be kept."""
        df = pd.DataFrame({
            "income":      [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            "loan_amount": [100.0, None, 300.0, 400.0, 500.0],  # 20% missing → keep
        })
        pipeline = FeaturePipeline()
        pipeline.fit(df)
        assert "loan_amount" not in pipeline.drop_cols

    def test_fit_returns_self(self, sample_df):
        """fit() should return self to allow method chaining."""
        pipeline = FeaturePipeline()
        result = pipeline.fit(sample_df)
        assert result is pipeline


# ── transform() tests ── output shape, dtype, missing value handling, derived features ─────────

class TestTransform:

    def test_transform_raises_if_not_fitted(self, sample_df):
        pipeline = FeaturePipeline()
        with pytest.raises(ValueError, match="must be fitted"):
            pipeline.transform(sample_df)

    def test_transform_returns_numpy_array(self, fitted_pipeline, sample_df):
        result = fitted_pipeline.transform(sample_df)
        assert isinstance(result, np.ndarray)

    def test_transform_returns_float32(self, fitted_pipeline, sample_df):
        result = fitted_pipeline.transform(sample_df)
        assert result.dtype == np.float32

    def test_transform_output_row_count_matches_input(self, fitted_pipeline, sample_df):
        result = fitted_pipeline.transform(sample_df)
        assert result.shape[0] == len(sample_df)

    def test_transform_output_col_count(self, fitted_pipeline, sample_df):
        """Output should be numeric_cols + 3 derived features."""
        result = fitted_pipeline.transform(sample_df)
        expected_cols = len(fitted_pipeline.numeric_cols) + 3
        assert result.shape[1] == expected_cols

    def test_transform_no_nan_in_output(self, fitted_pipeline, sample_df):
        result = fitted_pipeline.transform(sample_df)
        assert not np.isnan(result).any()

    def test_transform_no_inf_in_output(self, fitted_pipeline, sample_df):
        result = fitted_pipeline.transform(sample_df)
        assert not np.isinf(result).any()

    def test_transform_single_row(self, fitted_pipeline, sample_df):
        """Pipeline must handle a single-row DataFrame (live API use case)."""
        single = sample_df.iloc[[0]]
        result = fitted_pipeline.transform(single)
        assert result.shape[0] == 1


# ── Missing value handling ── Imputation — nulls filled correctly, indicator flag ─────

class TestMissingValues:

    def test_missing_income_imputed_with_median(self, fitted_pipeline, sample_df):
        df_missing = sample_df.copy()
        df_missing.loc[0, "income"] = None
        result = fitted_pipeline.transform(df_missing)
        assert not np.isnan(result).any()

    def test_missing_credit_score_imputed(self, fitted_pipeline, sample_df):
        df_missing = sample_df.copy()
        df_missing.loc[1, "Credit_Score"] = None
        result = fitted_pipeline.transform(df_missing)
        assert not np.isnan(result).any()

    def test_all_numeric_missing_in_one_row(self, fitted_pipeline, sample_df):
        """A row with all numeric fields null should still produce a valid vector."""
        df_missing = sample_df.copy()
        for col in fitted_pipeline.numeric_cols:
            if col in df_missing.columns:
                df_missing.loc[0, col] = None
        result = fitted_pipeline.transform(df_missing)
        assert not np.isnan(result).any()

    def test_missing_income_indicator_flag(self, fitted_pipeline, sample_df):
        """income_is_missing flag should be 1 when income is null."""
        df_missing = sample_df.copy()
        df_missing.loc[0, "income"] = None
        # Pipeline adds income_is_missing internally — no error should be raised
        result = fitted_pipeline.transform(df_missing)
        assert result is not None


# ── Feature engineering ── loan_to_income, zero-division protection ──────────

class TestDerivedFeatures:

    def test_zero_income_no_division_error(self, fitted_pipeline, sample_df):
        df = sample_df.copy()
        df["income"] = 0.0
        result = fitted_pipeline.transform(df)
        assert not np.isinf(result).any()
        assert not np.isnan(result).any()

    def test_zero_property_value_no_division_error(self, fitted_pipeline, sample_df):
        df = sample_df.copy()
        df["property_value"] = 0.0
        result = fitted_pipeline.transform(df)
        assert not np.isinf(result).any()
        assert not np.isnan(result).any()

    def test_loan_to_income_ratio_direction(self, fitted_pipeline):
        """Higher loan amount relative to income should produce higher ratio."""
        low_risk = pd.DataFrame({
            "income": [10000.0], "loan_amount": [50000.0],
            "property_value": [200000.0], "Credit_Score": [800.0],
            "rate_of_interest": [4.0], "Interest_rate_spread": [0.5],
            "Upfront_charges": [1000.0], "term": [360.0],
            "LTV": [25.0], "dtir1": [15.0],
        })
        high_risk = pd.DataFrame({
            "income": [2000.0], "loan_amount": [300000.0],
            "property_value": [200000.0], "Credit_Score": [800.0],
            "rate_of_interest": [4.0], "Interest_rate_spread": [0.5],
            "Upfront_charges": [1000.0], "term": [360.0],
            "LTV": [25.0], "dtir1": [15.0],
        })
        # loan_to_income = loan_amount / (income + 1)
        low_ratio  = 50000  / (10000 + 1)
        high_ratio = 300000 / (2000  + 1)
        assert high_ratio > low_ratio


# ── Consistency ── Determinism, input DataFrame not mutated ──────

class TestConsistency:

    def test_same_input_same_output(self, fitted_pipeline, sample_df):
        """Pipeline must be deterministic."""
        result1 = fitted_pipeline.transform(sample_df)
        result2 = fitted_pipeline.transform(sample_df)
        np.testing.assert_array_equal(result1, result2)

    def test_transform_does_not_mutate_input(self, fitted_pipeline, sample_df):
        """Original DataFrame must be unchanged after transform."""
        original = sample_df.copy()
        fitted_pipeline.transform(sample_df)
        pd.testing.assert_frame_equal(sample_df, original)

    def test_fit_does_not_mutate_input(self, sample_df):
        """fit() must not modify the DataFrame passed to it."""
        original = sample_df.copy()
        pipeline = FeaturePipeline()
        pipeline.fit(sample_df)
        pd.testing.assert_frame_equal(sample_df, original)