"""B2 pre-aggregation cleaning for historical tables.

These cleaning rules are applied to raw historical tables BEFORE any
aggregation, fixing the bug where upstream Tulip (solution-4) only
cleaned handcrafted features but not the raw tables fed to aggregators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_previous_application(df: pd.DataFrame) -> pd.DataFrame:
    """Apply B2 cleaning contract to previous_application.

    Rules (from reproduction_plan.md §4.6):
        DAYS_FIRST_DRAWING       = 365243  -> NaN
        DAYS_FIRST_DUE           = 365243  -> NaN
        DAYS_LAST_DUE_1ST_VERSION = 365243 -> NaN
        DAYS_LAST_DUE            = 365243  -> NaN
        DAYS_TERMINATION         = 365243  -> NaN
    """
    cleaned = df.copy()
    sentinel_cols = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    for col in sentinel_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].replace(365243, np.nan)
    return cleaned


def clean_bureau(df: pd.DataFrame) -> pd.DataFrame:
    """Apply B2 cleaning contract to bureau.

    Rules (from reproduction_plan.md §4.6):
        DAYS_CREDIT_ENDDATE  < -40000  -> NaN
        DAYS_CREDIT_UPDATE   < -40000  -> NaN
        DAYS_ENDDATE_FACT    < -40000  -> NaN
    """
    cleaned = df.copy()
    extreme_date_cols = [
        "DAYS_CREDIT_ENDDATE",
        "DAYS_CREDIT_UPDATE",
        "DAYS_ENDDATE_FACT",
    ]
    for col in extreme_date_cols:
        if col in cleaned.columns:
            cleaned.loc[cleaned[col] < -40000, col] = np.nan
    return cleaned


def clean_credit_card(df: pd.DataFrame) -> pd.DataFrame:
    """Apply B2 cleaning contract to credit_card_balance.

    Rules (from reproduction_plan.md §4.6):
        AMT_DRAWINGS_ATM_CURRENT  < 0  -> NaN
        AMT_DRAWINGS_CURRENT      < 0  -> NaN
    """
    cleaned = df.copy()
    negative_amt_cols = [
        "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_CURRENT",
    ]
    for col in negative_amt_cols:
        if col in cleaned.columns:
            cleaned.loc[cleaned[col] < 0, col] = np.nan
    return cleaned
