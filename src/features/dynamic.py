"""S5 dynamic ratio and trend features.

Plan reference: reproduction_plan.md §4.7

Features (7 total):
  Dynamic ratios (4):
    - pos_overdue_ratio_10_50   = recent10 overdue / (recent50 overdue + eps)
    - pos_sk_dpd_mean_ratio_10_50 = recent10 SK_DPD mean / (recent50 SK_DPD mean + eps)
    - installments_overdue_ratio_10_50 = recent10 overdue / (recent50 overdue + eps)
    - installments_overdue_days_ratio_10_50 = recent10 overdue days / (recent50 overdue days + eps)

  POS_CASH SK_DPD trends (3):
    - pos_sk_dpd_trend_12m  = linear slope of SK_DPD over last 12 months
    - pos_sk_dpd_trend_30m  = linear slope of SK_DPD over last 30 months
    - pos_sk_dpd_trend_60m  = linear slope of SK_DPD over last 60 months
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "SK_ID_CURR"
EPS = 1e-6

# ── feature name lists ─────────────────────────────────────────────────

DYNAMIC_RATIO_NAMES = [
    "pos_overdue_ratio_10_50",
    "pos_sk_dpd_mean_ratio_10_50",
    "installments_overdue_ratio_10_50",
    "installments_overdue_days_ratio_10_50",
]

DYNAMIC_TREND_NAMES = [
    "pos_sk_dpd_trend_12m",
    "pos_sk_dpd_trend_30m",
    "pos_sk_dpd_trend_60m",
]

DYNAMIC_FEATURE_NAMES = DYNAMIC_RATIO_NAMES + DYNAMIC_TREND_NAMES


# ── safe divide ────────────────────────────────────────────────────────

def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    n = pd.to_numeric(numerator, errors="coerce")
    d = pd.to_numeric(denominator, errors="coerce")
    result = n / (d + EPS)
    result = result.where(d.notna() & n.notna())
    return result.where(np.isfinite(result), np.nan)


# ── trend computation ──────────────────────────────────────────────────

def _trend_slope(y: np.ndarray) -> float:
    """Linear slope via polyfit.  Returns NaN when fewer than 2 points."""
    if len(y) < 2:
        return np.nan
    return float(np.polyfit(np.arange(len(y)), y, 1)[0])


def _compute_pos_cash_trends(data_dir: str | Path) -> pd.DataFrame:
    """Compute SK_DPD linear trends over 12/30/60-month windows.

    Uses a single groupby pass (sorted once, one apply call per SK_ID_CURR).
    A positive slope means SK_DPD is worsening (higher toward the present).
    """
    data_path = Path(data_dir)
    pos = pd.read_csv(
        data_path / "POS_CASH_balance.csv",
        usecols=[ID_COLUMN, "MONTHS_BALANCE", "SK_DPD"],
    )
    pos = pos.sort_values([ID_COLUMN, "MONTHS_BALANCE"])

    def _trends_for_group(grp: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "pos_sk_dpd_trend_12m": _trend_slope(
                    grp[grp["MONTHS_BALANCE"] >= -12]["SK_DPD"].values
                ),
                "pos_sk_dpd_trend_30m": _trend_slope(
                    grp[grp["MONTHS_BALANCE"] >= -30]["SK_DPD"].values
                ),
                "pos_sk_dpd_trend_60m": _trend_slope(
                    grp[grp["MONTHS_BALANCE"] >= -60]["SK_DPD"].values
                ),
            }
        )

    trends = pos.groupby(ID_COLUMN, sort=False).apply(_trends_for_group).reset_index()
    trends[DYNAMIC_TREND_NAMES] = trends[DYNAMIC_TREND_NAMES].replace(
        [np.inf, -np.inf], np.nan
    )
    return trends[[ID_COLUMN] + DYNAMIC_TREND_NAMES]


# ── ratio computation ──────────────────────────────────────────────────

def _compute_dynamic_ratios(features: pd.DataFrame) -> pd.DataFrame:
    """Compute the four short/long-term ratios from precomputed recent-window columns.

    Expects the following columns to already exist in *features*:
        pos_recent10_overdue_ratio   pos_recent50_overdue_ratio
        pos_recent10_sk_dpd_mean     pos_recent50_sk_dpd_mean
        installments_recent10_overdue_ratio   installments_recent50_overdue_ratio
        installments_recent10_overdue_days_mean  installments_recent50_overdue_days_mean
    """
    out = pd.DataFrame({ID_COLUMN: features[ID_COLUMN].values})
    out["pos_overdue_ratio_10_50"] = _safe_divide(
        features["pos_recent10_overdue_ratio"],
        features["pos_recent50_overdue_ratio"],
    )
    out["pos_sk_dpd_mean_ratio_10_50"] = _safe_divide(
        features["pos_recent10_sk_dpd_mean"],
        features["pos_recent50_sk_dpd_mean"],
    )
    out["installments_overdue_ratio_10_50"] = _safe_divide(
        features["installments_recent10_overdue_ratio"],
        features["installments_recent50_overdue_ratio"],
    )
    out["installments_overdue_days_ratio_10_50"] = _safe_divide(
        features["installments_recent10_overdue_days_mean"],
        features["installments_recent50_overdue_days_mean"],
    )
    out[DYNAMIC_RATIO_NAMES] = out[DYNAMIC_RATIO_NAMES].replace(
        [np.inf, -np.inf], np.nan
    )
    return out[[ID_COLUMN] + DYNAMIC_RATIO_NAMES]


# ── public API ─────────────────────────────────────────────────────────

def build_s5_dynamic_features(
    data_dir: str | Path,
    b2_precomputed: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Return S5 dynamic features at SK_ID_CURR grain.

    Parameters
    ----------
    data_dir : path to data/original/
    b2_precomputed : B2 precomputed feature DataFrame (needed for ratio computation)
    """
    data_path = Path(data_dir)

    trends = _compute_pos_cash_trends(data_path)
    ratios = _compute_dynamic_ratios(b2_precomputed)

    features = trends.merge(ratios, on=ID_COLUMN, how="outer", validate="one_to_one")
    features[DYNAMIC_FEATURE_NAMES] = features[DYNAMIC_FEATURE_NAMES].replace(
        [np.inf, -np.inf], np.nan
    )
    return features[[ID_COLUMN] + DYNAMIC_FEATURE_NAMES], list(DYNAMIC_FEATURE_NAMES)
