"""Recency-weighted (decay) rolling helper shared by the T-1 feature builders.

A NaN-robust weighted average: it drops NaN inside the window first -- so divide-by-zero-flagged rows
and the leading shift(1) NaN are skipped rather than fillna-poisoned (see the leakage discipline in the
feature builders) -- and weights the most-recent observation highest. Mirrors
features.weighted_recent_average, but NaN-robust.
"""
from __future__ import annotations

import numpy as np

from src.data_tool.features import DWA_WEIGHTS


def weighted_recent_average_nan(window_values) -> float:
    """DWA weighted average that drops NaN first (recency-weighted, most-recent = 0.35).
    Empty / all-NaN window -> NaN."""
    values = np.asarray(window_values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    active_weights = DWA_WEIGHTS[: values.size]
    return float(np.dot(values[::-1], active_weights) / active_weights.sum())
