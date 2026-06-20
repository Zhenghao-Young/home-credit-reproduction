"""S6 stacking: simple average and L2-Logistic meta-model from first-level OOF predictions.

Reads four existing OOF parquet files (s2_logistic, s3, s4, s5), builds a
prediction matrix, computes simple average and L2-Logistic stacking with
2-layer CV, and writes results/s6/ outputs.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.metrics import (
    ID_COLUMN,
    PREDICTION_COLUMN,
    TARGET_COLUMN,
    auc_score,
    build_fold_metrics,
    summarize_oof_auc,
    validate_oof,
)
from src.split import FOLD_COLUMN, load_folds

# The four first-level models whose OOF predictions feed into stacking.
BASE_STAGES = [
    ("s2_logistic", "logistic"),
    ("s3", "lightgbm"),
    ("s4", "lightgbm"),
    ("s5", "lightgbm"),
]

SIMPLE_AVG_STAGE = "s6_avg"
LOGISTIC_STACK_STAGE = "s6_stack"


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    folds = load_folds(args.folds_file)

    # 1. Load prediction matrix Z from existing OOF files
    z, target, fold_ids = _build_prediction_matrix(results_dir, folds)

    # 2. Simple average
    simple_avg_oof = _simple_average_oof(z, target, fold_ids)
    simple_test = _simple_average_test(results_dir)

    # 3. L2-Logistic stacking with 2-layer CV
    logistic_params = {
        "penalty": "l2",
        "solver": "newton-cholesky",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 2026,
    }
    stack_oof, meta_model = _logistic_stacking_oof(z, target, fold_ids, logistic_params)
    stack_test = _logistic_stacking_test(results_dir, meta_model)

    # 4. Write outputs
    output_dir = results_dir / "s6"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_oof(output_dir, simple_avg_oof, "simple_average")
    _write_oof(output_dir, stack_oof, "logistic_stack")

    _write_fold_metrics(results_dir, output_dir, simple_avg_oof, folds, "simple_average", SIMPLE_AVG_STAGE)
    _write_fold_metrics(results_dir, output_dir, stack_oof, folds, "logistic_stack", LOGISTIC_STACK_STAGE)
    _write_feature_names(output_dir)
    _copy_config(output_dir)

    if args.predict_test:
        _write_submission(results_dir, output_dir, simple_test, "simple_average")
        _write_submission(results_dir, output_dir, stack_test, "logistic_stack")

    # Print final results
    print(f"Simple average OOF AUC: {summarize_oof_auc(simple_avg_oof):.6f}")
    print(f"Logistic stacking OOF AUC: {summarize_oof_auc(stack_oof):.6f}")
    print(f"Saved outputs to {output_dir}")


def _build_prediction_matrix(
    results_dir: Path, folds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load OOF predictions from base stages and join into a matrix Z."""
    prediction_columns: dict[str, pd.Series] = {}
    for stage, model in BASE_STAGES:
        oof_path = results_dir / stage / "oof.parquet"
        if not oof_path.exists():
            raise FileNotFoundError(f"missing OOF file: {oof_path}")
        oof = pd.read_parquet(oof_path)
        validate_oof(oof, folds)
        col_name = f"p_{model}_{stage}" if stage != "s2_logistic" else "p_lr_s2"
        prediction_columns[col_name] = oof.set_index(ID_COLUMN)[PREDICTION_COLUMN]

    z = pd.DataFrame(prediction_columns)
    z = z.loc[folds[ID_COLUMN]]  # ensure same order as folds
    target = folds.set_index(ID_COLUMN).loc[z.index, TARGET_COLUMN]
    fold_ids = folds.set_index(ID_COLUMN).loc[z.index, FOLD_COLUMN]
    return z, target, fold_ids


def _simple_average_oof(
    z: pd.DataFrame, target: pd.Series, fold_ids: pd.Series
) -> pd.DataFrame:
    """Compute simple row-wise average of Z columns."""
    avg_pred = z.mean(axis=1).values
    oof = pd.DataFrame(
        {
            ID_COLUMN: z.index.values,
            TARGET_COLUMN: target.values,
            FOLD_COLUMN: fold_ids.values,
            PREDICTION_COLUMN: avg_pred,
        }
    )
    return oof


def _simple_average_test(results_dir: Path) -> pd.Series | None:
    """Compute simple row-wise average of base model test submissions."""
    submissions = []
    for stage, _model in BASE_STAGES:
        sub_path = results_dir / stage / "submission.csv"
        if not sub_path.exists():
            return None
        sub = pd.read_csv(sub_path).set_index(ID_COLUMN)[TARGET_COLUMN]
        submissions.append(sub)
    if not submissions:
        return None
    avg = pd.concat(submissions, axis=1).mean(axis=1)
    avg.name = TARGET_COLUMN
    return avg


def _logistic_stacking_oof(
    z: pd.DataFrame,
    target: pd.Series,
    fold_ids: pd.Series,
    params: dict,
) -> tuple[pd.DataFrame, LogisticRegression]:
    """Run 2-layer CV with L2-Logistic stacking on Z."""
    unique_folds = sorted(fold_ids.unique())
    stack_preds = pd.Series(np.nan, index=z.index, dtype="float64")
    feature_cols = list(z.columns)

    for fold_id in unique_folds:
        train_mask = fold_ids != fold_id
        valid_mask = fold_ids == fold_id

        x_train = z.loc[train_mask, feature_cols].values
        y_train = target.loc[train_mask].values
        x_valid = z.loc[valid_mask, feature_cols].values

        meta = LogisticRegression(**params)
        meta.fit(x_train, y_train)
        stack_preds.loc[valid_mask] = meta.predict_proba(x_valid)[:, 1]

    # Fit final meta-model on all data (for test prediction and coefficients)
    final_meta = LogisticRegression(**params)
    final_meta.fit(z[feature_cols].values, target.values)

    oof = pd.DataFrame(
        {
            ID_COLUMN: z.index.values,
            TARGET_COLUMN: target.values,
            FOLD_COLUMN: fold_ids.values,
            PREDICTION_COLUMN: stack_preds.values,
        }
    )
    return oof, final_meta


def _logistic_stacking_test(
    results_dir: Path, meta_model: LogisticRegression
) -> pd.Series | None:
    """Apply fitted meta-model to base model test submissions."""
    test_parts: list[pd.Series] = []
    index = None
    for stage, model in BASE_STAGES:
        sub_path = results_dir / stage / "submission.csv"
        if not sub_path.exists():
            return None
        sub = pd.read_csv(sub_path).set_index(ID_COLUMN)[TARGET_COLUMN]
        if index is None:
            index = sub.index
        col_name = f"p_{model}_{stage}" if stage != "s2_logistic" else "p_lr_s2"
        test_parts.append(pd.Series(sub.values, index=sub.index, name=col_name))

    if not test_parts:
        return None
    x_test = pd.concat(test_parts, axis=1)
    # feature_cols must match _build_prediction_matrix naming
    feature_cols = ["p_lr_s2", "p_lightgbm_s3", "p_lightgbm_s4", "p_lightgbm_s5"]
    x_test = x_test[feature_cols]
    preds = meta_model.predict_proba(x_test.values)[:, 1]
    return pd.Series(preds, index=x_test.index, name=TARGET_COLUMN)


def _write_oof(output_dir: Path, oof: pd.DataFrame, variant: str) -> None:
    """Write OOF parquet (variant: 'simple_average' or 'logistic_stack')."""
    path = output_dir / f"oof_{variant}.parquet"
    oof.to_parquet(path, index=False)


def _write_fold_metrics(
    results_dir: Path,
    output_dir: Path,
    oof: pd.DataFrame,
    folds: pd.DataFrame,
    variant: str,
    stage_name: str,
) -> None:
    """Write fold_metrics.csv with per-fold AUC for one stacking variant."""
    rows = []
    for fold_id in sorted(folds[FOLD_COLUMN].unique()):
        mask = oof[FOLD_COLUMN] == fold_id
        fold_oof = oof.loc[mask]
        fold_auc = auc_score(fold_oof[TARGET_COLUMN], fold_oof[PREDICTION_COLUMN])
        rows.append(
            build_fold_metrics(
                stage=stage_name,
                model="stacking",
                fold_id=int(fold_id),
                auc=fold_auc,
                n_train=len(folds[folds[FOLD_COLUMN] != fold_id]),
                n_valid=len(fold_oof),
                n_features=4,
            )
        )

    # Append to existing fold_metrics.csv if it exists
    metrics_path = output_dir / "fold_metrics.csv"
    new_rows = pd.DataFrame(rows)
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        # Remove old rows for this stage
        existing = existing[existing["stage"] != stage_name]
        new_rows = pd.concat([existing, new_rows], axis=0, ignore_index=True)
    new_rows.to_csv(metrics_path, index=False)
    _update_summary(results_dir, output_dir, oof, folds, stage_name)


def _update_summary(
    results_dir: Path,
    output_dir: Path,
    oof: pd.DataFrame,
    folds: pd.DataFrame,
    stage_name: str,
) -> None:
    """Upsert summary.csv with one stacking stage row."""
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        return

    summary = pd.read_csv(summary_path)
    kaggle_cols = [
        "kaggle_public_auc", "kaggle_private_auc",
        "kaggle_submission_date", "kaggle_submission_status",
        "kaggle_submission_description", "kaggle_file_name",
    ]
    for col in kaggle_cols:
        if col not in summary.columns:
            summary[col] = np.nan

    oof_auc = summarize_oof_auc(oof)
    per_fold = []
    for fold_id in sorted(folds[FOLD_COLUMN].unique()):
        mask = oof[FOLD_COLUMN] == fold_id
        per_fold.append(auc_score(oof.loc[mask, TARGET_COLUMN], oof.loc[mask, PREDICTION_COLUMN]))
    fold_auc_mean = float(np.mean(per_fold))
    fold_auc_std = float(np.std(per_fold, ddof=0))

    row = pd.DataFrame([{
        "stage": stage_name,
        "model": "stacking",
        "oof_auc": float(oof_auc),
        "fold_auc_mean": fold_auc_mean,
        "fold_auc_std": fold_auc_std,
        "n_features": 4,
        "output_dir": str(output_dir),
    }])
    for col in kaggle_cols:
        row[col] = np.nan

    matched = (summary["stage"] == stage_name) & (summary["model"] == "stacking")
    summary = summary[~matched]
    summary = pd.concat([summary, row], axis=0, ignore_index=True)
    summary.to_csv(summary_path, index=False)


def _write_feature_names(output_dir: Path) -> None:
    """Write feature_names.txt listing the four base model prediction columns."""
    names = "p_lr_s2\np_lightgbm_s3\np_lightgbm_s4\np_lightgbm_s5\n"
    (output_dir / "feature_names.txt").write_text(names, encoding="utf-8")


def _copy_config(output_dir: Path) -> None:
    """Copy the relevant logistic config for provenance."""
    config_path = Path("configs/base_lgbm.yaml")
    if config_path.exists():
        shutil.copyfile(config_path, output_dir / "config.yaml")


def _write_submission(
    results_dir: Path,
    output_dir: Path,
    test_preds: pd.Series | None,
    variant: str,
) -> None:
    """Write submission CSV reordered to match sample_submission."""
    if test_preds is None:
        print(f"No test predictions for {variant} — skipping submission")
        return
    sample_path = results_dir.parent / "data" / "original" / "sample_submission.csv"
    if not sample_path.exists():
        print(f"sample_submission.csv not found — skipping {variant} submission")
        return
    sample = pd.read_csv(sample_path)
    sub_df = pd.DataFrame({ID_COLUMN: test_preds.index, TARGET_COLUMN: test_preds.values})
    sub = sample[[ID_COLUMN]].merge(sub_df, on=ID_COLUMN, how="left", validate="one_to_one")
    if sub[TARGET_COLUMN].isna().any():
        raise ValueError(f"{variant} submission contains missing predictions")
    if not sub[TARGET_COLUMN].between(0, 1).all():
        raise ValueError(f"{variant} submission predictions must be in [0, 1]")
    path = output_dir / f"submission_{variant}.csv"
    sub.to_csv(path, index=False)
    print(f"Saved {variant} submission to {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S6 stacking from first-level OOF predictions.")
    parser.add_argument("--results-dir", default="results", help="Directory with stage outputs")
    parser.add_argument("--folds-file", default="data/folds.csv", help="Path to folds.csv")
    parser.add_argument("--predict-test", action="store_true", help="Also produce test submissions")
    return parser.parse_args()


if __name__ == "__main__":
    main()
