"""S4 / B2  group-relative position and recent-window behavior features.

This module provides the feature blocks shared by S4 and B2:
- Fold-safe group-relative position (diff and abs_diff from group mean)
- Recent-window aggregations on cleaned historical tables
- S3-style full-history aggregations on cleaned historical tables

For B2, all historical-table features are computed on cleaned raw data
(using cleaning_v2).  S4 would re-use the same feature builders without
the cleaning step.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.cleaning_v2 import (
    clean_bureau,
    clean_credit_card,
    clean_previous_application,
)
from src.features.history_basic import HISTORY_FEATURE_NAMES, build_s3_history_features

ID_COLUMN = "SK_ID_CURR"
MISSING_CATEGORY = "__MISSING__"

# ═══════════════════════════════════════════════════════════════════════
# Group-relative position feature specification
# ═══════════════════════════════════════════════════════════════════════

RELATIVE_VALUES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

RELATIVE_GROUP_SPECS: list[list[str]] = [
    ["OCCUPATION_TYPE"],
    ["NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"],
    ["CODE_GENDER", "NAME_EDUCATION_TYPE"],
]


def _build_relative_feature_names() -> list[str]:
    """Generate the fixed ordered list of group-relative feature names."""
    names: list[str] = []
    for group_cols in RELATIVE_GROUP_SPECS:
        group_key = "__".join(group_cols)
        for value_col in RELATIVE_VALUES:
            names.append(f"relative_diff__{group_key}__{value_col}")
            names.append(f"relative_abs_diff__{group_key}__{value_col}")
    return names


RELATIVE_FEATURE_NAMES = _build_relative_feature_names()


class FoldSafeGroupRelativePosition:
    """Fold-safe group-relative position features.

    For each (value_col, group_cols) pair, computes:
        diff     = value_col - group_mean(value_col)
        abs_diff = |value_col - group_mean(value_col)|

    Group means are fitted on the training fold only.  Unseen groups in
    validation / test are filled with the training-fold global mean of
    *value_col*.
    """

    def __init__(
        self,
        value_columns: list[str] | None = None,
        group_specs: list[list[str]] | None = None,
    ):
        self.value_columns = value_columns or RELATIVE_VALUES
        self.group_specs = group_specs or RELATIVE_GROUP_SPECS
        self._means: list[tuple[list[str], str, pd.DataFrame, float]] = []
        self.feature_names_: list[str] = []

    # -- fit / transform -------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "FoldSafeGroupRelativePosition":
        self._means = []
        self.feature_names_ = []
        prepared = self._prepare_group_keys(train_df)

        for group_cols in self.group_specs:
            available = [c for c in group_cols if c in prepared.columns]
            if len(available) != len(group_cols):
                continue
            for value_col in self.value_columns:
                if value_col not in prepared.columns:
                    continue
                gm = (
                    prepared.groupby(available, dropna=False)[value_col]
                    .mean()
                    .reset_index()
                    .rename(columns={value_col: "__gm__"})
                )
                global_mean = float(prepared[value_col].mean())
                self._means.append((available, value_col, gm, global_mean))

                gk = "__".join(available)
                self.feature_names_.append(f"relative_diff__{gk}__{value_col}")
                self.feature_names_.append(f"relative_abs_diff__{gk}__{value_col}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df[[ID_COLUMN]].copy()
        work = self._prepare_group_keys(df)

        for group_cols, value_col, gm_table, global_mean in self._means:
            merged = work[[ID_COLUMN] + group_cols + [value_col]].merge(
                gm_table, on=group_cols, how="left"
            )
            merged["__gm__"] = merged["__gm__"].fillna(global_mean)

            gk = "__".join(group_cols)
            diff_name = f"relative_diff__{gk}__{value_col}"
            abs_name = f"relative_abs_diff__{gk}__{value_col}"

            merged[diff_name] = merged[value_col] - merged["__gm__"]
            merged[abs_name] = (merged[value_col] - merged["__gm__"]).abs()

            result = result.merge(
                merged[[ID_COLUMN, diff_name, abs_name]], on=ID_COLUMN, how="left"
            )

        return result[[ID_COLUMN] + self.feature_names_]

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _prepare_group_keys(df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        all_cols = {c for spec in RELATIVE_GROUP_SPECS for c in spec}
        for col in all_cols:
            if col in prepared.columns:
                prepared[col] = (
                    prepared[col]
                    .astype("object")
                    .where(prepared[col].notna(), MISSING_CATEGORY)
                )
        return prepared


# ═══════════════════════════════════════════════════════════════════════
# S3-style full-history features computed on CLEANED data
# ═══════════════════════════════════════════════════════════════════════

BUREAU_FEATURE_NAMES_C = [
    "bureau_loan_count",
    "bureau_active_loan_ratio",
    "bureau_amt_credit_sum_mean",
    "bureau_amt_credit_sum_max",
    "bureau_amt_credit_sum_sum",
    "bureau_amt_credit_sum_debt_mean",
    "bureau_amt_credit_sum_debt_max",
    "bureau_amt_credit_sum_debt_sum",
    "bureau_amt_credit_sum_overdue_mean",
    "bureau_amt_credit_sum_overdue_max",
    "bureau_days_credit_mean",
    "bureau_days_credit_min",
]

PREVIOUS_FEATURE_NAMES_C = [
    "previous_application_count",
    "previous_approved_ratio",
    "previous_amt_application_mean",
    "previous_amt_application_max",
    "previous_amt_credit_mean",
    "previous_amt_credit_max",
    "previous_days_decision_mean",
    "previous_days_decision_min",
    "previous_cnt_payment_mean",
    "previous_cnt_payment_max",
]

INSTALLMENTS_FEATURE_NAMES_C = [
    "installments_payment_count",
    "installments_overdue_days_mean",
    "installments_overdue_ratio",
    "installments_underpayment_mean",
    "installments_overpayment_mean",
]

POS_CASH_FEATURE_NAMES_C = [
    "pos_cash_count",
    "pos_sk_dpd_mean",
    "pos_sk_dpd_max",
    "pos_sk_dpd_def_mean",
    "pos_sk_dpd_def_max",
    "pos_overdue_ratio",
]

CREDIT_CARD_FEATURE_NAMES_C = [
    "credit_card_count",
    "credit_card_amt_balance_mean",
    "credit_card_amt_balance_max",
    "credit_card_amt_credit_limit_actual_mean",
    "credit_card_balance_limit_ratio_mean",
    "credit_card_balance_limit_ratio_max",
    "credit_card_sk_dpd_mean",
    "credit_card_sk_dpd_max",
]

HISTORY_CLEANED_FEATURE_NAMES = (
    BUREAU_FEATURE_NAMES_C
    + PREVIOUS_FEATURE_NAMES_C
    + INSTALLMENTS_FEATURE_NAMES_C
    + POS_CASH_FEATURE_NAMES_C
    + CREDIT_CARD_FEATURE_NAMES_C
)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    n = pd.to_numeric(numerator, errors="coerce")
    d = pd.to_numeric(denominator, errors="coerce")
    result = n / d
    result = result.where(d.notna() & (d != 0))
    return result.where(np.isfinite(result), np.nan)


# ── bureau (cleaned) ───────────────────────────────────────────────────

def _build_bureau_features_cleaned(path: Path) -> pd.DataFrame:
    bureau = pd.read_csv(
        path,
        usecols=[
            ID_COLUMN,
            "SK_ID_BUREAU",
            "CREDIT_ACTIVE",
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_OVERDUE",
            "DAYS_CREDIT",
            "DAYS_CREDIT_ENDDATE",
            "DAYS_CREDIT_UPDATE",
            "DAYS_ENDDATE_FACT",
        ],
    )
    bureau = clean_bureau(bureau)
    bureau["_active_flag"] = bureau["CREDIT_ACTIVE"].ne("Closed").astype("float64")
    features = (
        bureau.groupby(ID_COLUMN, sort=False)
        .agg(
            bureau_loan_count=("SK_ID_BUREAU", "count"),
            bureau_active_loan_ratio=("_active_flag", "mean"),
            bureau_amt_credit_sum_mean=("AMT_CREDIT_SUM", "mean"),
            bureau_amt_credit_sum_max=("AMT_CREDIT_SUM", "max"),
            bureau_amt_credit_sum_sum=("AMT_CREDIT_SUM", "sum"),
            bureau_amt_credit_sum_debt_mean=("AMT_CREDIT_SUM_DEBT", "mean"),
            bureau_amt_credit_sum_debt_max=("AMT_CREDIT_SUM_DEBT", "max"),
            bureau_amt_credit_sum_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
            bureau_amt_credit_sum_overdue_mean=("AMT_CREDIT_SUM_OVERDUE", "mean"),
            bureau_amt_credit_sum_overdue_max=("AMT_CREDIT_SUM_OVERDUE", "max"),
            bureau_days_credit_mean=("DAYS_CREDIT", "mean"),
            bureau_days_credit_min=("DAYS_CREDIT", "min"),
        )
        .reset_index()
    )
    return features[[ID_COLUMN] + BUREAU_FEATURE_NAMES_C]


# ── previous_application (cleaned) ─────────────────────────────────────

def _build_previous_features_cleaned(path: Path) -> pd.DataFrame:
    previous = pd.read_csv(
        path,
        usecols=[
            ID_COLUMN,
            "SK_ID_PREV",
            "NAME_CONTRACT_STATUS",
            "AMT_APPLICATION",
            "AMT_CREDIT",
            "DAYS_DECISION",
            "CNT_PAYMENT",
            "DAYS_FIRST_DRAWING",
            "DAYS_FIRST_DUE",
            "DAYS_LAST_DUE_1ST_VERSION",
            "DAYS_LAST_DUE",
            "DAYS_TERMINATION",
        ],
    )
    previous = clean_previous_application(previous)
    previous["_approved_flag"] = (
        previous["NAME_CONTRACT_STATUS"].eq("Approved").astype("float64")
    )
    features = (
        previous.groupby(ID_COLUMN, sort=False)
        .agg(
            previous_application_count=("SK_ID_PREV", "count"),
            previous_approved_ratio=("_approved_flag", "mean"),
            previous_amt_application_mean=("AMT_APPLICATION", "mean"),
            previous_amt_application_max=("AMT_APPLICATION", "max"),
            previous_amt_credit_mean=("AMT_CREDIT", "mean"),
            previous_amt_credit_max=("AMT_CREDIT", "max"),
            previous_days_decision_mean=("DAYS_DECISION", "mean"),
            previous_days_decision_min=("DAYS_DECISION", "min"),
            previous_cnt_payment_mean=("CNT_PAYMENT", "mean"),
            previous_cnt_payment_max=("CNT_PAYMENT", "max"),
        )
        .reset_index()
    )
    return features[[ID_COLUMN] + PREVIOUS_FEATURE_NAMES_C]


# ── installments_payments (cleaned) ────────────────────────────────────

def _build_installments_features_cleaned(path: Path) -> pd.DataFrame:
    installments = pd.read_csv(
        path,
        usecols=[
            ID_COLUMN,
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ],
    )
    overdue_days = installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    installments["_overdue_days"] = overdue_days.clip(lower=0)
    installments["_overdue_flag"] = (
        installments["_overdue_days"].gt(0).astype("float64")
    )
    installments["_underpayment"] = (
        installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]
    ).clip(lower=0)
    installments["_overpayment"] = (
        installments["AMT_PAYMENT"] - installments["AMT_INSTALMENT"]
    ).clip(lower=0)
    features = (
        installments.groupby(ID_COLUMN, sort=False)
        .agg(
            installments_payment_count=(ID_COLUMN, "size"),
            installments_overdue_days_mean=("_overdue_days", "mean"),
            installments_overdue_ratio=("_overdue_flag", "mean"),
            installments_underpayment_mean=("_underpayment", "mean"),
            installments_overpayment_mean=("_overpayment", "mean"),
        )
        .reset_index()
    )
    return features[[ID_COLUMN] + INSTALLMENTS_FEATURE_NAMES_C]


# ── POS_CASH_balance (cleaned) ─────────────────────────────────────────

def _build_pos_cash_features_cleaned(path: Path) -> pd.DataFrame:
    pos_cash = pd.read_csv(path, usecols=[ID_COLUMN, "SK_DPD", "SK_DPD_DEF"])
    pos_cash["_overdue_flag"] = pos_cash["SK_DPD"].gt(0).astype("float64")
    features = (
        pos_cash.groupby(ID_COLUMN, sort=False)
        .agg(
            pos_cash_count=(ID_COLUMN, "size"),
            pos_sk_dpd_mean=("SK_DPD", "mean"),
            pos_sk_dpd_max=("SK_DPD", "max"),
            pos_sk_dpd_def_mean=("SK_DPD_DEF", "mean"),
            pos_sk_dpd_def_max=("SK_DPD_DEF", "max"),
            pos_overdue_ratio=("_overdue_flag", "mean"),
        )
        .reset_index()
    )
    return features[[ID_COLUMN] + POS_CASH_FEATURE_NAMES_C]


# ── credit_card_balance (cleaned) ──────────────────────────────────────

def _build_credit_card_features_cleaned(path: Path) -> pd.DataFrame:
    credit_card = pd.read_csv(
        path,
        usecols=[
            ID_COLUMN,
            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "SK_DPD",
            "AMT_DRAWINGS_ATM_CURRENT",
            "AMT_DRAWINGS_CURRENT",
        ],
    )
    credit_card = clean_credit_card(credit_card)
    credit_card["_balance_limit_ratio"] = _safe_divide(
        credit_card["AMT_BALANCE"], credit_card["AMT_CREDIT_LIMIT_ACTUAL"]
    )
    features = (
        credit_card.groupby(ID_COLUMN, sort=False)
        .agg(
            credit_card_count=(ID_COLUMN, "size"),
            credit_card_amt_balance_mean=("AMT_BALANCE", "mean"),
            credit_card_amt_balance_max=("AMT_BALANCE", "max"),
            credit_card_amt_credit_limit_actual_mean=(
                "AMT_CREDIT_LIMIT_ACTUAL",
                "mean",
            ),
            credit_card_balance_limit_ratio_mean=("_balance_limit_ratio", "mean"),
            credit_card_balance_limit_ratio_max=("_balance_limit_ratio", "max"),
            credit_card_sk_dpd_mean=("SK_DPD", "mean"),
            credit_card_sk_dpd_max=("SK_DPD", "max"),
        )
        .reset_index()
    )
    return features[[ID_COLUMN] + CREDIT_CARD_FEATURE_NAMES_C]


# ═══════════════════════════════════════════════════════════════════════
# Recent-window features (computed on cleaned data)
# ═══════════════════════════════════════════════════════════════════════

RECENT_PREVIOUS_FEATURE_NAMES = [
    "prev_recent1_approved_ratio",
    "prev_recent1_amt_application_mean",
    "prev_recent1_amt_credit_mean",
    "prev_recent1_cnt_payment_mean",
    "prev_recent3_approved_ratio",
    "prev_recent3_amt_application_mean",
    "prev_recent3_amt_credit_mean",
    "prev_recent3_cnt_payment_mean",
    "prev_recent5_approved_ratio",
    "prev_recent5_amt_application_mean",
    "prev_recent5_amt_credit_mean",
    "prev_recent5_cnt_payment_mean",
]

RECENT_INSTALLMENTS_FEATURE_NAMES = [
    "installments_recent10_overdue_days_mean",
    "installments_recent10_overdue_ratio",
    "installments_recent10_underpayment_mean",
    "installments_recent10_overpayment_mean",
    "installments_recent50_overdue_days_mean",
    "installments_recent50_overdue_ratio",
    "installments_recent50_underpayment_mean",
    "installments_recent50_overpayment_mean",
]

RECENT_POS_CASH_FEATURE_NAMES = [
    "pos_recent10_sk_dpd_mean",
    "pos_recent10_sk_dpd_max",
    "pos_recent10_sk_dpd_def_mean",
    "pos_recent10_overdue_ratio",
    "pos_recent50_sk_dpd_mean",
    "pos_recent50_sk_dpd_max",
    "pos_recent50_sk_dpd_def_mean",
    "pos_recent50_overdue_ratio",
]

RECENT_WINDOW_FEATURE_NAMES = (
    RECENT_PREVIOUS_FEATURE_NAMES
    + RECENT_INSTALLMENTS_FEATURE_NAMES
    + RECENT_POS_CASH_FEATURE_NAMES
)


def _build_recent_previous_features_uncleaned(path: Path) -> pd.DataFrame:
    """Recent-window previous_application features WITHOUT pre-aggregation cleaning.

    This is the S4 variant — identical aggregation logic to B2 but without
    calling clean_previous_application(), so that the B2−S4 comparison
    isolates the effect of cleaning.
    """
    previous = pd.read_csv(
        path,
        usecols=[
            ID_COLUMN,
            "SK_ID_PREV",
            "DAYS_DECISION",
            "NAME_CONTRACT_STATUS",
            "AMT_APPLICATION",
            "AMT_CREDIT",
            "CNT_PAYMENT",
        ],
    )
    previous["_approved_flag"] = (
        previous["NAME_CONTRACT_STATUS"].eq("Approved").astype("float64")
    )
    previous = previous.sort_values([ID_COLUMN, "DAYS_DECISION"])

    blocks = []
    for k in [1, 3, 5]:
        recent = previous.groupby(ID_COLUMN, sort=False).tail(k)
        agg = (
            recent.groupby(ID_COLUMN, sort=False)
            .agg(
                **{
                    f"prev_recent{k}_approved_ratio": ("_approved_flag", "mean"),
                    f"prev_recent{k}_amt_application_mean": ("AMT_APPLICATION", "mean"),
                    f"prev_recent{k}_amt_credit_mean": ("AMT_CREDIT", "mean"),
                    f"prev_recent{k}_cnt_payment_mean": ("CNT_PAYMENT", "mean"),
                }
            )
            .reset_index()
        )
        blocks.append(agg)

    result = reduce(
        lambda left, right: left.merge(right, on=ID_COLUMN, how="outer"), blocks
    )
    return result[[ID_COLUMN] + RECENT_PREVIOUS_FEATURE_NAMES]


def _build_recent_previous_features_cleaned(path: Path) -> pd.DataFrame:
    previous = pd.read_csv(
        path,
        usecols=[
            ID_COLUMN,
            "SK_ID_PREV",
            "DAYS_DECISION",
            "NAME_CONTRACT_STATUS",
            "AMT_APPLICATION",
            "AMT_CREDIT",
            "CNT_PAYMENT",
            "DAYS_FIRST_DRAWING",
            "DAYS_FIRST_DUE",
            "DAYS_LAST_DUE_1ST_VERSION",
            "DAYS_LAST_DUE",
            "DAYS_TERMINATION",
        ],
    )
    previous = clean_previous_application(previous)
    previous["_approved_flag"] = (
        previous["NAME_CONTRACT_STATUS"].eq("Approved").astype("float64")
    )
    previous = previous.sort_values([ID_COLUMN, "DAYS_DECISION"])

    blocks = []
    for k in [1, 3, 5]:
        recent = previous.groupby(ID_COLUMN, sort=False).tail(k)
        agg = (
            recent.groupby(ID_COLUMN, sort=False)
            .agg(
                **{
                    f"prev_recent{k}_approved_ratio": ("_approved_flag", "mean"),
                    f"prev_recent{k}_amt_application_mean": ("AMT_APPLICATION", "mean"),
                    f"prev_recent{k}_amt_credit_mean": ("AMT_CREDIT", "mean"),
                    f"prev_recent{k}_cnt_payment_mean": ("CNT_PAYMENT", "mean"),
                }
            )
            .reset_index()
        )
        blocks.append(agg)

    result = reduce(
        lambda left, right: left.merge(right, on=ID_COLUMN, how="outer"), blocks
    )
    return result[[ID_COLUMN] + RECENT_PREVIOUS_FEATURE_NAMES]


def _build_recent_installments_features_cleaned(path: Path) -> pd.DataFrame:
    installments = pd.read_csv(
        path,
        usecols=[
            ID_COLUMN,
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ],
    )
    overdue_days = installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    installments["_overdue_days"] = overdue_days.clip(lower=0)
    installments["_overdue_flag"] = (
        installments["_overdue_days"].gt(0).astype("float64")
    )
    installments["_underpayment"] = (
        installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]
    ).clip(lower=0)
    installments["_overpayment"] = (
        installments["AMT_PAYMENT"] - installments["AMT_INSTALMENT"]
    ).clip(lower=0)
    installments = installments.sort_values([ID_COLUMN, "DAYS_INSTALMENT"])

    blocks = []
    for k in [10, 50]:
        recent = installments.groupby(ID_COLUMN, sort=False).tail(k)
        agg = (
            recent.groupby(ID_COLUMN, sort=False)
            .agg(
                **{
                    f"installments_recent{k}_overdue_days_mean": (
                        "_overdue_days",
                        "mean",
                    ),
                    f"installments_recent{k}_overdue_ratio": ("_overdue_flag", "mean"),
                    f"installments_recent{k}_underpayment_mean": (
                        "_underpayment",
                        "mean",
                    ),
                    f"installments_recent{k}_overpayment_mean": (
                        "_overpayment",
                        "mean",
                    ),
                }
            )
            .reset_index()
        )
        blocks.append(agg)

    result = reduce(
        lambda left, right: left.merge(right, on=ID_COLUMN, how="outer"), blocks
    )
    return result[[ID_COLUMN] + RECENT_INSTALLMENTS_FEATURE_NAMES]


def _build_recent_pos_cash_features_cleaned(path: Path) -> pd.DataFrame:
    pos_cash = pd.read_csv(
        path, usecols=[ID_COLUMN, "MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF"]
    )
    pos_cash["_overdue_flag"] = pos_cash["SK_DPD"].gt(0).astype("float64")

    blocks = []
    for k in [10, 50]:
        recent = pos_cash[pos_cash["MONTHS_BALANCE"] >= -k]
        agg = (
            recent.groupby(ID_COLUMN, sort=False)
            .agg(
                **{
                    f"pos_recent{k}_sk_dpd_mean": ("SK_DPD", "mean"),
                    f"pos_recent{k}_sk_dpd_max": ("SK_DPD", "max"),
                    f"pos_recent{k}_sk_dpd_def_mean": ("SK_DPD_DEF", "mean"),
                    f"pos_recent{k}_overdue_ratio": ("_overdue_flag", "mean"),
                }
            )
            .reset_index()
        )
        blocks.append(agg)

    result = reduce(
        lambda left, right: left.merge(right, on=ID_COLUMN, how="outer"), blocks
    )
    return result[[ID_COLUMN] + RECENT_POS_CASH_FEATURE_NAMES]


# ═══════════════════════════════════════════════════════════════════════
# B2 precomputed feature assembly
# ═══════════════════════════════════════════════════════════════════════

B2_PRECOMPUTED_FEATURE_NAMES = HISTORY_CLEANED_FEATURE_NAMES + RECENT_WINDOW_FEATURE_NAMES

# ═══════════════════════════════════════════════════════════════════════
# S4 precomputed feature assembly (UNCLEANED)
# ═══════════════════════════════════════════════════════════════════════

S4_PRECOMPUTED_FEATURE_NAMES = list(HISTORY_FEATURE_NAMES) + list(RECENT_WINDOW_FEATURE_NAMES)


def build_s4_precomputed_features(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Return all S4 precomputed features at SK_ID_CURR grain.

    Includes:
      - S3-style full-history aggregations on UNCLEANED data
        (reuses build_s3_history_features from history_basic)
      - Recent-window aggregations on UNCLEANED data
        (previous_application: uncleaned; installments & POS_CASH:
         same builders as B2 since those tables have no B2 cleaning)
    """
    data_path = Path(data_dir)

    # Uncleaned S3 history features
    history_features, _ = build_s3_history_features(data_path)

    # Recent-window features on uncleaned data
    recent_blocks = [
        _build_recent_previous_features_uncleaned(data_path / "previous_application.csv"),
        _build_recent_installments_features_cleaned(data_path / "installments_payments.csv"),
        _build_recent_pos_cash_features_cleaned(data_path / "POS_CASH_balance.csv"),
    ]
    recent_features = reduce(
        lambda left, right: left.merge(right, on=ID_COLUMN, how="outer", validate="one_to_one"),
        recent_blocks,
    )

    features = history_features.merge(recent_features, on=ID_COLUMN, how="outer", validate="one_to_one")
    features[S4_PRECOMPUTED_FEATURE_NAMES] = features[S4_PRECOMPUTED_FEATURE_NAMES].replace(
        [np.inf, -np.inf], np.nan
    )
    return features[[ID_COLUMN] + S4_PRECOMPUTED_FEATURE_NAMES], list(S4_PRECOMPUTED_FEATURE_NAMES)


def build_b2_precomputed_features(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Return all B2 precomputed features at SK_ID_CURR grain.

    Includes:
      - S3-style full-history aggregations on CLEANED data
      - Recent-window aggregations on CLEANED data

    These features do not use TARGET and are computed from raw history
    only, so they are safe to compute once outside the fold loop.
    Group-relative position features are NOT included here because they
    must be fold-safe (fit on each training fold separately).
    """
    data_path = Path(data_dir)

    history_blocks = [
        _build_bureau_features_cleaned(data_path / "bureau.csv"),
        _build_previous_features_cleaned(data_path / "previous_application.csv"),
        _build_installments_features_cleaned(data_path / "installments_payments.csv"),
        _build_pos_cash_features_cleaned(data_path / "POS_CASH_balance.csv"),
        _build_credit_card_features_cleaned(data_path / "credit_card_balance.csv"),
    ]
    history_features = reduce(
        lambda left, right: left.merge(right, on=ID_COLUMN, how="outer", validate="one_to_one"),
        history_blocks,
    )

    recent_blocks = [
        _build_recent_previous_features_cleaned(data_path / "previous_application.csv"),
        _build_recent_installments_features_cleaned(data_path / "installments_payments.csv"),
        _build_recent_pos_cash_features_cleaned(data_path / "POS_CASH_balance.csv"),
    ]
    recent_features = reduce(
        lambda left, right: left.merge(right, on=ID_COLUMN, how="outer", validate="one_to_one"),
        recent_blocks,
    )

    features = history_features.merge(recent_features, on=ID_COLUMN, how="outer", validate="one_to_one")
    features[B2_PRECOMPUTED_FEATURE_NAMES] = features[B2_PRECOMPUTED_FEATURE_NAMES].replace(
        [np.inf, -np.inf], np.nan
    )
    return features[[ID_COLUMN] + B2_PRECOMPUTED_FEATURE_NAMES], list(B2_PRECOMPUTED_FEATURE_NAMES)
