"""B1 application-table business ratio features."""

from __future__ import annotations

import numpy as np
import pandas as pd


ID_COLUMN = "SK_ID_CURR"
EXT_SOURCE_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
BUSINESS_FEATURE_NAMES = [
    "annuity_income_percentage",
    "credit_to_income_ratio",
    "credit_to_annuity_ratio",
    "external_sources_mean",
    "external_sources_min",
    "external_sources_max",
    "external_sources_std",
]


def build_b1_business_features(application: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return the constrained B1 business features plus feature-name metadata."""
    features = pd.DataFrame({ID_COLUMN: application[ID_COLUMN].values}, index=application.index)
    features["annuity_income_percentage"] = _safe_divide(application["AMT_ANNUITY"], application["AMT_INCOME_TOTAL"])
    features["credit_to_income_ratio"] = _safe_divide(application["AMT_CREDIT"], application["AMT_INCOME_TOTAL"])
    features["credit_to_annuity_ratio"] = _safe_divide(application["AMT_CREDIT"], application["AMT_ANNUITY"])

    ext_sources = application[EXT_SOURCE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    features["external_sources_mean"] = ext_sources.mean(axis=1, skipna=True)
    features["external_sources_min"] = ext_sources.min(axis=1, skipna=True)
    features["external_sources_max"] = ext_sources.max(axis=1, skipna=True)
    features["external_sources_std"] = ext_sources.std(axis=1, skipna=True, ddof=0)
    features[BUSINESS_FEATURE_NAMES] = features[BUSINESS_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)

    return features[[ID_COLUMN] + BUSINESS_FEATURE_NAMES], list(BUSINESS_FEATURE_NAMES)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_numeric = pd.to_numeric(numerator, errors="coerce")
    denominator_numeric = pd.to_numeric(denominator, errors="coerce")
    result = numerator_numeric / denominator_numeric
    result = result.where(denominator_numeric.notna() & (denominator_numeric != 0))
    return result.where(np.isfinite(result), np.nan)
