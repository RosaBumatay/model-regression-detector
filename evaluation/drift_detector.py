"""Data drift detection using PSI and the two-sample KS test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def population_stability_index(expected, actual, bins: int = 10) -> float:
    """PSI between two numeric samples. <0.1 stable, 0.1-0.25 moderate, >0.25 major."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(expected, edges)[0] / len(expected)
    a_pct = np.histogram(actual, edges)[0] / len(actual)
    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def detect_drift(
    baseline_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    psi_threshold: float = 0.25,
    ks_alpha: float = 0.01,
) -> pd.DataFrame:
    """Per-feature drift report. A feature drifts if PSI > threshold or KS p < alpha."""
    rows = []
    for col in features:
        psi = population_stability_index(baseline_df[col], test_df[col])
        ks_stat, p_value = ks_2samp(baseline_df[col], test_df[col])
        rows.append(
            {
                "feature": col,
                "psi": round(psi, 4),
                "ks_stat": round(float(ks_stat), 4),
                "ks_p_value": float(p_value),
                "baseline_mean": round(float(baseline_df[col].mean()), 4),
                "test_mean": round(float(test_df[col].mean()), 4),
                "drifted": bool(psi > psi_threshold or p_value < ks_alpha),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
