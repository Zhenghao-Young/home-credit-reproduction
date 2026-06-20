"""S3 applicant-level historical table aggregations."""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMN = "SK_ID_CURR"

BUREAU_FEATURE_NAMES = [
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
PREVIOUS_FEATURE_NAMES = [
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
INSTALLMENTS_FEATURE_NAMES = [
    "installments_payment_count",
    "installments_overdue_days_mean",
    "installments_overdue_ratio",
    "installments_underpayment_mean",
    "installments_overpayment_mean",
]
POS_CASH_FEATURE_NAMES = [
    "pos_cash_count",
    "pos_sk_dpd_mean",
    "pos_sk_dpd_max",
    "pos_sk_dpd_def_mean",
    "pos_sk_dpd_def_max",
    "pos_overdue_ratio",
]
CREDIT_CARD_FEATURE_NAMES = [
    "credit_card_count",
    "credit_card_amt_balance_mean",
    "credit_card_amt_balance_max",
    "credit_card_amt_credit_limit_actual_mean",
    "credit_card_balance_limit_ratio_mean",
    "credit_card_balance_limit_ratio_max",
    "credit_card_sk_dpd_mean",
    "credit_card_sk_dpd_max",
]
HISTORY_FEATURE_NAMES = (
    BUREAU_FEATURE_NAMES
    + PREVIOUS_FEATURE_NAMES
    + INSTALLMENTS_FEATURE_NAMES
    + POS_CASH_FEATURE_NAMES
    + CREDIT_CARD_FEATURE_NAMES
)


def build_s3_history_features(data_dir: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Return S3 history aggregations at SK_ID_CURR grain."""
    data_path = Path(data_dir)
    blocks = [
        _build_bureau_features(data_path / "bureau.csv"),
        _build_previous_features(data_path / "previous_application.csv"),
        _build_installments_features(data_path / "installments_payments.csv"),
        _build_pos_cash_features(data_path / "POS_CASH_balance.csv"),
        _build_credit_card_features(data_path / "credit_card_balance.csv"),
    ]
    features = reduce(
        lambda left, right: left.merge(right, on=ID_COLUMN, how="outer", validate="one_to_one"),
        blocks,
    )
    features[HISTORY_FEATURE_NAMES] = features[HISTORY_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    return features[[ID_COLUMN] + HISTORY_FEATURE_NAMES], list(HISTORY_FEATURE_NAMES)


def _build_bureau_features(path: Path) -> pd.DataFrame:
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
        ],
    )
    bureau["bureau_active_flag"] = bureau["CREDIT_ACTIVE"].ne("Closed").astype("float64")
    features = (
        bureau.groupby(ID_COLUMN, sort=False)
        .agg(
            bureau_loan_count=("SK_ID_BUREAU", "count"),
            bureau_active_loan_ratio=("bureau_active_flag", "mean"),
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
    return features[[ID_COLUMN] + BUREAU_FEATURE_NAMES]


def _build_previous_features(path: Path) -> pd.DataFrame:
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
        ],
    )
    previous["previous_approved_flag"] = previous["NAME_CONTRACT_STATUS"].eq("Approved").astype("float64")
    features = (
        previous.groupby(ID_COLUMN, sort=False)
        .agg(
            previous_application_count=("SK_ID_PREV", "count"),
            previous_approved_ratio=("previous_approved_flag", "mean"),
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
    return features[[ID_COLUMN] + PREVIOUS_FEATURE_NAMES]


def _build_installments_features(path: Path) -> pd.DataFrame:
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
    installments["installments_overdue_days"] = overdue_days.clip(lower=0)
    installments["installments_overdue_flag"] = installments["installments_overdue_days"].gt(0).astype("float64")
    installments["installments_underpayment"] = (installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]).clip(lower=0)
    installments["installments_overpayment"] = (installments["AMT_PAYMENT"] - installments["AMT_INSTALMENT"]).clip(lower=0)
    features = (
        installments.groupby(ID_COLUMN, sort=False)
        .agg(
            installments_payment_count=(ID_COLUMN, "size"),
            installments_overdue_days_mean=("installments_overdue_days", "mean"),
            installments_overdue_ratio=("installments_overdue_flag", "mean"),
            installments_underpayment_mean=("installments_underpayment", "mean"),
            installments_overpayment_mean=("installments_overpayment", "mean"),
        )
        .reset_index()
    )
    return features[[ID_COLUMN] + INSTALLMENTS_FEATURE_NAMES]


def _build_pos_cash_features(path: Path) -> pd.DataFrame:
    pos_cash = pd.read_csv(path, usecols=[ID_COLUMN, "SK_DPD", "SK_DPD_DEF"])
    pos_cash["pos_overdue_flag"] = pos_cash["SK_DPD"].gt(0).astype("float64")
    features = (
        pos_cash.groupby(ID_COLUMN, sort=False)
        .agg(
            pos_cash_count=(ID_COLUMN, "size"),
            pos_sk_dpd_mean=("SK_DPD", "mean"),
            pos_sk_dpd_max=("SK_DPD", "max"),
            pos_sk_dpd_def_mean=("SK_DPD_DEF", "mean"),
            pos_sk_dpd_def_max=("SK_DPD_DEF", "max"),
            pos_overdue_ratio=("pos_overdue_flag", "mean"),
        )
        .reset_index()
    )
    return features[[ID_COLUMN] + POS_CASH_FEATURE_NAMES]


def _build_credit_card_features(path: Path) -> pd.DataFrame:
    credit_card = pd.read_csv(
        path,
        usecols=[ID_COLUMN, "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"],
    )
    credit_card["credit_card_balance_limit_ratio"] = _safe_divide(
        credit_card["AMT_BALANCE"],
        credit_card["AMT_CREDIT_LIMIT_ACTUAL"],
    )
    features = (
        credit_card.groupby(ID_COLUMN, sort=False)
        .agg(
            credit_card_count=(ID_COLUMN, "size"),
            credit_card_amt_balance_mean=("AMT_BALANCE", "mean"),
            credit_card_amt_balance_max=("AMT_BALANCE", "max"),
            credit_card_amt_credit_limit_actual_mean=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
            credit_card_balance_limit_ratio_mean=("credit_card_balance_limit_ratio", "mean"),
            credit_card_balance_limit_ratio_max=("credit_card_balance_limit_ratio", "max"),
            credit_card_sk_dpd_mean=("SK_DPD", "mean"),
            credit_card_sk_dpd_max=("SK_DPD", "max"),
        )
        .reset_index()
    )
    return features[[ID_COLUMN] + CREDIT_CARD_FEATURE_NAMES]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_numeric = pd.to_numeric(numerator, errors="coerce")
    denominator_numeric = pd.to_numeric(denominator, errors="coerce")
    result = numerator_numeric / denominator_numeric
    result = result.where(denominator_numeric.notna() & (denominator_numeric != 0))
    return result.where(np.isfinite(result), np.nan)
