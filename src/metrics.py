"""Metrics and output validation for controlled CV experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
FOLD_COLUMN = "fold_id"
PREDICTION_COLUMN = "prediction"


def auc_score(y_true, y_pred) -> float:
    """Compute ROC-AUC as a plain float."""
    return float(roc_auc_score(y_true, y_pred))


def build_fold_metrics(
    stage: str,
    model: str,
    fold_id: int,
    auc: float,
    n_train: int,
    n_valid: int,
    n_features: int,
) -> dict:
    """Return one fold metrics record with the shared schema."""
    return {
        "stage": stage,
        "model": model,
        "fold_id": int(fold_id),
        "auc": float(auc),
        "n_train": int(n_train),
        "n_valid": int(n_valid),
        "n_features": int(n_features),
    }


def validate_oof(oof_df: pd.DataFrame, folds_df: pd.DataFrame) -> None:
    """Validate that OOF predictions match the fixed folds exactly."""
    required = {ID_COLUMN, TARGET_COLUMN, FOLD_COLUMN, PREDICTION_COLUMN}
    missing = required - set(oof_df.columns)
    if missing:
        raise ValueError(f"OOF missing columns: {sorted(missing)}")

    if len(oof_df) != len(folds_df):
        raise ValueError(f"OOF row count {len(oof_df)} != folds row count {len(folds_df)}")
    if oof_df[ID_COLUMN].duplicated().any():
        raise ValueError("OOF contains duplicate SK_ID_CURR values")
    if oof_df[PREDICTION_COLUMN].isna().any():
        raise ValueError("OOF contains missing predictions")
    if not np.isfinite(oof_df[PREDICTION_COLUMN]).all():
        raise ValueError("OOF contains non-finite predictions")
    if not oof_df[PREDICTION_COLUMN].between(0, 1).all():
        raise ValueError("OOF predictions must be in [0, 1]")

    fold_cols = [ID_COLUMN, TARGET_COLUMN, FOLD_COLUMN]
    expected = folds_df[fold_cols].sort_values(ID_COLUMN).reset_index(drop=True)
    actual = oof_df[fold_cols].sort_values(ID_COLUMN).reset_index(drop=True)
    if not expected.equals(actual):
        raise ValueError("OOF SK_ID_CURR/TARGET/fold_id does not match data/folds.csv")


def summarize_oof_auc(oof_df: pd.DataFrame) -> float:
    """Compute the global OOF AUC."""
    return auc_score(oof_df[TARGET_COLUMN], oof_df[PREDICTION_COLUMN])
