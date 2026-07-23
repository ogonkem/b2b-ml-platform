import numpy as np
import pandas as pd
from typing import List, Optional

PSI_THRESHOLD = 0.2   # >= 0.2 = significant drift, retraining recommended

KEY_FEATURES = [
    "loan_amount", "property_value", "income",
    "Credit_Score", "LTV", "dtir1", "term",
]


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two numeric distributions.
      PSI < 0.10  — stable
      0.10–0.20   — moderate shift, monitor
      >= 0.20     — significant drift, trigger retraining
    """
    eps     = 1e-6
    min_val = min(float(expected.min()), float(actual.min()))
    max_val = max(float(expected.max()), float(actual.max()))

    if min_val == max_val:
        return 0.0

    edges = np.linspace(min_val, max_val, bins + 1)
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual,   bins=edges)

    exp_pct = (exp_counts + eps) / (len(expected) + eps * bins)
    act_pct = (act_counts + eps) / (len(actual)   + eps * bins)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def check_drift(
    baseline_df:  pd.DataFrame,
    incoming_df:  pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    threshold:    float = PSI_THRESHOLD,
) -> dict:
    """
    Compute per-column PSI between baseline training data and incoming labeled data.
    Returns {"drifted": bool, "max_psi": float, "psi_per_col": dict}.
    """
    if numeric_cols is None:
        numeric_cols = KEY_FEATURES

    shared_cols = [
        c for c in numeric_cols
        if c in baseline_df.columns and c in incoming_df.columns
    ]

    psi_scores = {
        col: compute_psi(
            baseline_df[col].dropna().to_numpy(dtype=float),
            incoming_df[col].dropna().to_numpy(dtype=float),
        )
        for col in shared_cols
    }

    max_psi = max(psi_scores.values()) if psi_scores else 0.0

    return {
        "drifted":     max_psi >= threshold,
        "max_psi":     round(max_psi, 4),
        "psi_per_col": {
            k: round(v, 4)
            for k, v in sorted(psi_scores.items(), key=lambda x: x[1], reverse=True)
        },
    }
