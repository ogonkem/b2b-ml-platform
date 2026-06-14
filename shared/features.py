import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeaturePipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_cols = []
        self.categorical_cols = []
        self.drop_cols = []
        self.medians = {}
        self.modes = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = 'Status'):
        """Learn cleaning parameters, medians, and scales from training data safely."""
        df_copy = df.copy()
        
        if 'ID' in df_copy.columns:
            df_copy = df_copy.drop(['ID'], axis=1)
            
        # 1. Determine columns to drop (>40% missing)
        missing_pct = (df_copy.isnull().sum() / len(df_copy) * 100)
        self.drop_cols = missing_pct[missing_pct > 40].index.tolist()
        df_copy = df_copy.drop(columns=self.drop_cols, errors='ignore')
        
        # 2. Separate column types
        self.numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
        
        if target_col in self.numeric_cols:
            self.numeric_cols.remove(target_col)
            
        # 3. Save Medians and Modes
        for col in self.numeric_cols:
            self.medians[col] = df_copy[col].median() if not df_copy[col].isnull().all() else 0.0
            
        for col in self.categorical_cols:
            if not df_copy[col].isnull().all():
                self.modes[col] = df_copy[col].mode()[0]
            else:
                self.modes[col] = "missing"
                
        # 4. Fit Scaler on numerical features plus new derived features
        derived_features = ['loan_to_income', 'loan_to_property', 'credit_to_income']
        extended_numeric = self.numeric_cols + derived_features
        
        # Build dummy frame to fit internal scaler correctly
        dummy_df = pd.DataFrame(0.0, index=range(5), columns=extended_numeric)
        self.scaler.fit(dummy_df)
        
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply saved parameters to transform incoming streaming/test data."""
        if not self.is_fitted:
            raise ValueError("FeaturePipeline must be fitted before running transform.")
            
        df_processed = df.copy()
        
        if 'ID' in df_processed.columns:
            df_processed = df_processed.drop(['ID'], axis=1)
            
        df_processed = df_processed.drop(columns=self.drop_cols, errors='ignore')
        
        # Missing indicator columns (Prevents information loss)
        if 'income' in df_processed.columns:
            df_processed['income_is_missing'] = df_processed['income'].isnull().astype(int)
            
        # Impute missing numeric fields 
        for col in self.numeric_cols:
            if col in df_processed.columns:
                fill_val = self.medians.get(col, 0.0)
                df_processed[col] = df_processed[col].fillna(fill_val)
            else:
                df_processed[col] = self.medians.get(col, 0.0)
                
        # Impute missing categorical fields
        for col in self.categorical_cols:
            if col in df_processed.columns:
                fill_val = self.modes.get(col, "missing")
                df_processed[col] = df_processed[col].fillna(fill_val)
            else:
                df_processed[col] = self.modes.get(col, "missing")

        # Feature Engineering (with zero-division protection)
        income_denom = df_processed.get('income', 0.0) + 1.0
        prop_denom = df_processed.get('property_value', 0.0) + 1.0
        loan_amt = df_processed.get('loan_amount', 0.0)
        credit_scr = df_processed.get('Credit_Score', 0.0)
        
        df_processed['loan_to_income'] = loan_amt / income_denom
        df_processed['loan_to_property'] = loan_amt / prop_denom
        df_processed['credit_to_income'] = credit_scr / income_denom
        
        # Extract numerical arrays in strict order to match scaler expectations
        final_numeric_cols = self.numeric_cols + ['loan_to_income', 'loan_to_property', 'credit_to_income']
        numeric_data = df_processed[final_numeric_cols].to_numpy(dtype=np.float32)
        
        return self.scaler.transform(numeric_data)

