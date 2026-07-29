import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_DERIVED_FEATURES = ['loan_to_income', 'loan_to_property', 'credit_to_income']


class FeaturePipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_cols = []      # imputation columns (excludes derived — they are recomputed)
        self.categorical_cols = []
        self.drop_cols = []
        self.medians = {}
        self.modes = {}
        self.label_encoders = {}    # col -> {category_str: code}, learned at fit time
        self.clip_bounds = {}       # col -> (q1, q99), learned at fit time; covers numeric + derived cols
        self.feature_names = []     # authoritative column order matching the training matrix
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = 'Status'):
        df_copy = df.copy()

        if 'ID' in df_copy.columns:
            df_copy = df_copy.drop(['ID'], axis=1)
        if target_col in df_copy.columns:
            df_copy = df_copy.drop([target_col], axis=1)

        # Columns to drop (>40% missing)
        missing_pct = (df_copy.isnull().sum() / len(df_copy) * 100)
        self.drop_cols = missing_pct[missing_pct > 40].index.tolist()
        df_copy = df_copy.drop(columns=self.drop_cols, errors='ignore')

        # Authoritative column order — whatever order the training matrix had
        self.feature_names = df_copy.columns.tolist()

        # Imputation columns: numeric cols except derived features (they are recomputed
        # in transform(), so excluding them here prevents double-appending)
        all_numeric = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols = [c for c in all_numeric if c not in _DERIVED_FEATURES]
        self.categorical_cols = df_copy.select_dtypes(include=['object']).columns.tolist()

        for col in self.numeric_cols:
            self.medians[col] = float(df_copy[col].median()) if not df_copy[col].isnull().all() else 0.0
        for col in self.categorical_cols:
            self.modes[col] = df_copy[col].mode()[0] if not df_copy[col].isnull().all() else "missing"

        # Impute before encoding/clipping so both are learned on complete data —
        # mirrors the impute -> encode -> clip order used in notebooks/retrain.py
        for col in self.numeric_cols:
            df_copy[col] = df_copy[col].fillna(self.medians[col])
        for col in self.categorical_cols:
            df_copy[col] = df_copy[col].fillna(self.modes[col])

        # Label-encode categoricals with the same scheme as retrain.py's sklearn
        # LabelEncoder (codes assigned in sorted order of the observed categories),
        # stored as a plain dict so transform() can fall back cleanly on unseen values.
        self.label_encoders = {}
        for col in self.categorical_cols:
            classes = sorted(df_copy[col].astype(str).unique())
            self.label_encoders[col] = {cls: code for code, cls in enumerate(classes)}
            df_copy[col] = df_copy[col].astype(str).map(self.label_encoders[col])

        # Clip numeric columns to their 1st/99th percentile — matches retrain.py's
        # outlier handling before scaling. Derived features are already present in
        # the training frame at this point (see class docstring) so they get their
        # own learned bounds too, applied after they are recomputed in transform().
        clip_cols = self.numeric_cols + [c for c in _DERIVED_FEATURES if c in df_copy.columns]
        self.clip_bounds = {}
        for col in clip_cols:
            q1, q99 = df_copy[col].quantile(0.01), df_copy[col].quantile(0.99)
            self.clip_bounds[col] = (float(q1), float(q99))
            df_copy[col] = df_copy[col].clip(q1, q99)

        # Fit scaler on real training data in the authoritative column order
        scaler_input = df_copy[self.feature_names].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        self.scaler.fit(scaler_input.to_numpy(dtype=np.float64))

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("FeaturePipeline must be fitted before running transform.")

        df_processed = df.copy()

        if 'ID' in df_processed.columns:
            df_processed = df_processed.drop(['ID'], axis=1)
        df_processed = df_processed.drop(columns=self.drop_cols, errors='ignore')

        # Impute numeric columns.
        for col in self.numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col] = df_processed[col].fillna(self.medians.get(col, 0.0))
            else:
                df_processed[col] = self.medians.get(col, 0.0)

        # Impute categorical columns
        for col in self.categorical_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(self.modes.get(col, "missing"))
            else:
                df_processed[col] = self.modes.get(col, "missing")

        # Label-encode categoricals with the mapping learned at fit time. A category
        # never seen during training falls back to the trained mode's code — the
        # same "fall back to the trained-on value" rule the numeric median uses above —
        # instead of being silently coerced away.
        for col in self.categorical_cols:
            mapping = self.label_encoders.get(col, {})
            fallback_code = mapping.get(str(self.modes.get(col)), 0)
            df_processed[col] = df_processed[col].astype(str).map(mapping).fillna(fallback_code)

        # Recompute derived features
        loan_amt   = df_processed['loan_amount']    if 'loan_amount'    in df_processed.columns else 0.0
        income_val = df_processed['income']         if 'income'         in df_processed.columns else 0.0
        prop_val   = df_processed['property_value'] if 'property_value' in df_processed.columns else 0.0
        credit_val = df_processed['Credit_Score']   if 'Credit_Score'   in df_processed.columns else 0.0
        df_processed['loan_to_income']   = loan_amt   / (income_val + 1.0)
        df_processed['loan_to_property'] = loan_amt   / (prop_val   + 1.0)
        df_processed['credit_to_income'] = credit_val / (income_val + 1.0)

        # Clip outliers to the bounds learned at fit time (numeric cols + derived features)
        for col, (lo, hi) in self.clip_bounds.items():
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').clip(lo, hi)

        # Select columns in the exact order the scaler and model were trained on
        X = df_processed.reindex(columns=self.feature_names, fill_value=0.0)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        return self.scaler.transform(X.to_numpy(dtype=np.float64))

