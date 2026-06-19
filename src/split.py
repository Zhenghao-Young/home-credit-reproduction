"""Create and load the fixed CV folds used by all controlled experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
FOLD_COLUMN = "fold_id"


def make_folds(
    train_path: str | Path,
    output_path: str | Path,
    n_splits: int = 5,
    seed: int = 2026,
) -> pd.DataFrame:
    """Create stratified folds and persist them as CSV."""
    train_path = Path(train_path)
    output_path = Path(output_path)

    train = pd.read_csv(train_path, usecols=[ID_COLUMN, TARGET_COLUMN])
    if train[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be unique in {train_path}")

    folds = train.copy()
    folds[FOLD_COLUMN] = -1

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_id, (_, valid_idx) in enumerate(cv.split(folds[[ID_COLUMN]], folds[TARGET_COLUMN])):
        folds.loc[valid_idx, FOLD_COLUMN] = fold_id

    _validate_folds(folds, n_splits=n_splits, expected_rows=len(train))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_path, index=False)
    return folds


def load_folds(path: str | Path) -> pd.DataFrame:
    """Load and validate a fixed folds file."""
    folds = pd.read_csv(path)
    _validate_folds(folds)
    return folds


def _validate_folds(
    folds: pd.DataFrame,
    n_splits: int | None = None,
    expected_rows: int | None = None,
) -> None:
    required = {ID_COLUMN, TARGET_COLUMN, FOLD_COLUMN}
    missing = required - set(folds.columns)
    if missing:
        raise ValueError(f"folds file missing columns: {sorted(missing)}")
    if expected_rows is not None and len(folds) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(folds)}")
    if folds[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be unique in folds")
    if folds[FOLD_COLUMN].isna().any():
        raise ValueError(f"{FOLD_COLUMN} contains missing values")

    fold_values = sorted(folds[FOLD_COLUMN].unique().tolist())
    if n_splits is None:
        n_splits = len(fold_values)
    expected_fold_values = list(range(n_splits))
    if fold_values != expected_fold_values:
        raise ValueError(f"fold ids must be {expected_fold_values}, got {fold_values}")

    global_rate = folds[TARGET_COLUMN].mean()
    fold_rates = folds.groupby(FOLD_COLUMN)[TARGET_COLUMN].mean()
    max_deviation = (fold_rates - global_rate).abs().max()
    if max_deviation > 0.01:
        raise ValueError(
            "fold target rates deviate too much from global target rate: "
            f"global={global_rate:.6f}, max_deviation={max_deviation:.6f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed Home Credit CV folds.")
    parser.add_argument("--train", required=True, help="Path to application_train.csv")
    parser.add_argument("--out", required=True, help="Output path for folds.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    folds = make_folds(args.train, args.out, n_splits=args.n_splits, seed=args.seed)
    summary = folds.groupby(FOLD_COLUMN)[TARGET_COLUMN].agg(["count", "mean"])
    print(summary.to_string())
    print(f"Saved folds to {args.out}")


if __name__ == "__main__":
    main()
