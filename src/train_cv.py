"""Unified CV runner for Member A controlled experiments."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.application_base import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    build_s1_features,
    clean_application,
    load_application_train,
)
from src.features.application_business import build_b1_business_features
from src.features.application_groupby import (
    FULL_GROUPBY_SPECS,
    FoldSafeGroupbyAggregateDiffs,
)
from src.features.history_basic import build_s3_history_features
from src.features.relative_recent import (
    FoldSafeGroupRelativePosition,
    build_b2_precomputed_features,
    build_s4_precomputed_features,
)
from src.metrics import auc_score, build_fold_metrics, summarize_oof_auc, validate_oof
from src.split import FOLD_COLUMN, load_folds


SUPPORTED_STAGES = {"s1", "b1", "s2", "s2_full", "s2_logistic", "s3", "s4", "b2"}
SUPPORTED_MODELS = {"lightgbm", "logistic"}
MISSING_CATEGORY = "__MISSING__"
UNKNOWN_CATEGORY = "__UNKNOWN__"


class OrdinalCategoryEncoder:
    """Small fold-local ordinal encoder with explicit missing/unknown handling."""

    def __init__(self, categorical_columns: list[str]):
        self.categorical_columns = categorical_columns
        self.mappings_: dict[str, dict[object, int]] = {}

    def fit(self, df: pd.DataFrame) -> "OrdinalCategoryEncoder":
        for col in self.categorical_columns:
            values = self._normalize(df[col])
            categories = list(dict.fromkeys(values.tolist()))
            for sentinel in [MISSING_CATEGORY, UNKNOWN_CATEGORY]:
                if sentinel not in categories:
                    categories.append(sentinel)
            self.mappings_[col] = {value: idx for idx, value in enumerate(categories)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy()
        for col in self.categorical_columns:
            values = self._normalize(transformed[col])
            mapping = self.mappings_[col]
            transformed[col] = values.where(values.isin(mapping), UNKNOWN_CATEGORY).map(mapping).astype("int32")
        return transformed

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        return series.astype("object").where(series.notna(), MISSING_CATEGORY)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cv(stage: str, model_name: str, config_path: str | Path, predict_test: bool = False) -> None:
    if stage not in SUPPORTED_STAGES:
        raise ValueError(f"unsupported stage {stage!r}; expected one of {sorted(SUPPORTED_STAGES)}")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model {model_name!r}; expected one of {sorted(SUPPORTED_MODELS)}")
    if stage in {"s1", "b1", "s3", "s4", "b2"} and model_name != "lightgbm":
        raise ValueError("S1, B1, S3, S4, and B2 are only defined for LightGBM")
    if stage == "s2_logistic" and model_name != "logistic":
        raise ValueError("s2_logistic must use --model logistic")
    if stage in {"s2", "s2_full"} and model_name != "lightgbm":
        raise ValueError("S2 main experiments must use --model lightgbm")

    config = load_config(config_path)
    data_dir = Path(config["data_dir"])
    train_path = data_dir / config["train_file"]
    test_path = data_dir / config["test_file"]
    sample_submission_path = data_dir / "sample_submission.csv"
    folds = load_folds(config.get("folds_file", "data/folds.csv"))
    application = load_application_train(train_path)
    base_features, base_feature_names, base_categorical = build_s1_features(application)
    application_support = clean_application(application)
    history_features = None
    history_feature_names: list[str] = []
    if stage == "s3":
        history_features, history_feature_names = build_s3_history_features(data_dir)
    if stage == "s4":
        history_features, history_feature_names = build_s4_precomputed_features(data_dir)
    if stage == "b2":
        history_features, history_feature_names = build_b2_precomputed_features(data_dir)
    test_base_features = None
    test_support = None
    if predict_test:
        test_application = pd.read_csv(test_path)
        test_base_features, _, _ = build_s1_features(test_application)
        test_support = clean_application(test_application)

    data = folds.merge(base_features, on=[ID_COLUMN, TARGET_COLUMN], how="left", validate="one_to_one")
    data = _merge_support_columns(data, application_support)
    if len(data) != len(folds):
        raise ValueError("feature table row count does not match folds")

    output_stage = "s2_logistic" if stage == "s2_logistic" else stage
    output_dir = Path(config.get("results_dir", "results")) / output_stage
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_parts = []
    fold_metric_rows = []
    test_prediction_parts = []
    final_feature_names: list[str] | None = None

    for fold_id in sorted(folds[FOLD_COLUMN].unique()):
        train_mask = data[FOLD_COLUMN] != fold_id
        valid_mask = data[FOLD_COLUMN] == fold_id
        train_df = data.loc[train_mask].copy()
        valid_df = data.loc[valid_mask].copy()

        train_features, valid_features, test_features, feature_names, categorical_columns = _build_fold_features(
            stage=stage,
            train_df=train_df,
            valid_df=valid_df,
            test_df=None if test_base_features is None else _merge_support_columns(test_base_features, test_support),
            base_feature_names=base_feature_names,
            base_categorical=base_categorical,
            history_features=history_features,
            history_feature_names=history_feature_names,
        )
        final_feature_names = feature_names

        if model_name == "lightgbm":
            y_valid_pred, y_test_pred = _fit_predict_lightgbm(
                train_features,
                train_df[TARGET_COLUMN],
                valid_features,
                valid_df[TARGET_COLUMN],
                test_features,
                categorical_columns,
                config["lightgbm"],
            )
        else:
            y_valid_pred, y_test_pred = _fit_predict_logistic(
                train_features,
                train_df[TARGET_COLUMN],
                valid_features,
                test_features,
                categorical_columns,
                config["logistic"],
            )

        fold_auc = auc_score(valid_df[TARGET_COLUMN], y_valid_pred)
        print(f"{output_stage}/{model_name} fold {fold_id}: AUC={fold_auc:.6f}")

        oof_parts.append(
            pd.DataFrame(
                {
                    ID_COLUMN: valid_df[ID_COLUMN].values,
                    TARGET_COLUMN: valid_df[TARGET_COLUMN].values,
                    FOLD_COLUMN: valid_df[FOLD_COLUMN].values,
                    "prediction": y_valid_pred,
                }
            )
        )
        fold_metric_rows.append(
            build_fold_metrics(
                stage=output_stage,
                model=model_name,
                fold_id=int(fold_id),
                auc=fold_auc,
                n_train=len(train_df),
                n_valid=len(valid_df),
                n_features=len(feature_names),
            )
        )
        if y_test_pred is not None:
            test_prediction_parts.append(y_test_pred)

    oof = pd.concat(oof_parts, axis=0, ignore_index=True)
    validate_oof(oof, folds)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    oof_auc = summarize_oof_auc(oof)
    print(f"{output_stage}/{model_name} OOF AUC={oof_auc:.6f}")

    oof.to_parquet(output_dir / "oof.parquet", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    (output_dir / "feature_names.txt").write_text("\n".join(final_feature_names or []) + "\n", encoding="utf-8")
    shutil.copyfile(config_path, output_dir / "config.yaml")
    _update_summary(config, output_stage, model_name, fold_metrics, oof_auc, len(final_feature_names or []), output_dir)
    if predict_test:
        submission = _build_submission(test_base_features, test_prediction_parts, sample_submission_path)
        submission_path = output_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Saved submission to {submission_path}")


def _build_fold_features(
    stage: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    base_feature_names: list[str],
    base_categorical: list[str],
    history_features: pd.DataFrame | None = None,
    history_feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, list[str], list[str]]:
    train_features = train_df[[ID_COLUMN] + base_feature_names].copy()
    valid_features = valid_df[[ID_COLUMN] + base_feature_names].copy()
    test_features = None
    if test_df is not None:
        test_features = test_df[[ID_COLUMN] + base_feature_names].copy()
    feature_names = list(base_feature_names)
    categorical_columns = [col for col in base_categorical if col in feature_names]

    if stage in {"b1", "s3"}:
        train_business, business_feature_names = build_b1_business_features(train_df)
        valid_business, _ = build_b1_business_features(valid_df)
        train_features = train_features.merge(train_business, on=ID_COLUMN, how="left", validate="one_to_one")
        valid_features = valid_features.merge(valid_business, on=ID_COLUMN, how="left", validate="one_to_one")
        if test_features is not None:
            test_business, _ = build_b1_business_features(test_df)
            test_features = test_features.merge(test_business, on=ID_COLUMN, how="left", validate="one_to_one")
        feature_names = feature_names + business_feature_names
        if stage == "s3":
            if history_features is None or history_feature_names is None:
                raise ValueError("S3 requires precomputed history features")
            train_features = train_features.merge(history_features, on=ID_COLUMN, how="left", validate="many_to_one")
            valid_features = valid_features.merge(history_features, on=ID_COLUMN, how="left", validate="many_to_one")
            if test_features is not None:
                test_features = test_features.merge(history_features, on=ID_COLUMN, how="left", validate="many_to_one")
            feature_names = feature_names + list(history_feature_names)
    elif stage == "s4":
        # B1 business features
        train_business, business_feature_names = build_b1_business_features(train_df)
        valid_business, _ = build_b1_business_features(valid_df)
        train_features = train_features.merge(train_business, on=ID_COLUMN, how="left", validate="one_to_one")
        valid_features = valid_features.merge(valid_business, on=ID_COLUMN, how="left", validate="one_to_one")
        if test_features is not None:
            test_business, _ = build_b1_business_features(test_df)
            test_features = test_features.merge(test_business, on=ID_COLUMN, how="left", validate="one_to_one")
        feature_names = feature_names + business_feature_names
        # precomputed features: S3-style history + recent windows (both on UNCLEANED data)
        if history_features is None or history_feature_names is None:
            raise ValueError("S4 requires precomputed history features")
        train_features = train_features.merge(history_features, on=ID_COLUMN, how="left", validate="many_to_one")
        valid_features = valid_features.merge(history_features, on=ID_COLUMN, how="left", validate="many_to_one")
        if test_features is not None:
            test_features = test_features.merge(history_features, on=ID_COLUMN, how="left", validate="many_to_one")
        feature_names = feature_names + list(history_feature_names)
        # fold-safe group-relative position
        relative = FoldSafeGroupRelativePosition().fit(train_df)
        train_relative = relative.transform(train_df)
        valid_relative = relative.transform(valid_df)
        train_features = train_features.merge(train_relative, on=ID_COLUMN, how="left", validate="one_to_one")
        valid_features = valid_features.merge(valid_relative, on=ID_COLUMN, how="left", validate="one_to_one")
        if test_features is not None:
            test_relative = relative.transform(test_df)
            test_features = test_features.merge(test_relative, on=ID_COLUMN, how="left", validate="one_to_one")
        feature_names = feature_names + relative.feature_names_
    elif stage == "b2":
        # B1 business features (same as s3)
        train_business, business_feature_names = build_b1_business_features(train_df)
        valid_business, _ = build_b1_business_features(valid_df)
        train_features = train_features.merge(train_business, on=ID_COLUMN, how="left", validate="one_to_one")
        valid_features = valid_features.merge(valid_business, on=ID_COLUMN, how="left", validate="one_to_one")
        if test_features is not None:
            test_business, _ = build_b1_business_features(test_df)
            test_features = test_features.merge(test_business, on=ID_COLUMN, how="left", validate="one_to_one")
        feature_names = feature_names + business_feature_names
        # precomputed features: S3-style history + recent windows  (both on cleaned data)
        if history_features is None or history_feature_names is None:
            raise ValueError("B2 requires precomputed history features")
        train_features = train_features.merge(history_features, on=ID_COLUMN, how="left", validate="many_to_one")
        valid_features = valid_features.merge(history_features, on=ID_COLUMN, how="left", validate="many_to_one")
        if test_features is not None:
            test_features = test_features.merge(history_features, on=ID_COLUMN, how="left", validate="many_to_one")
        feature_names = feature_names + list(history_feature_names)
        # fold-safe group-relative position
        relative = FoldSafeGroupRelativePosition().fit(train_df)
        train_relative = relative.transform(train_df)
        valid_relative = relative.transform(valid_df)
        train_features = train_features.merge(train_relative, on=ID_COLUMN, how="left", validate="one_to_one")
        valid_features = valid_features.merge(valid_relative, on=ID_COLUMN, how="left", validate="one_to_one")
        if test_features is not None:
            test_relative = relative.transform(test_df)
            test_features = test_features.merge(test_relative, on=ID_COLUMN, how="left", validate="one_to_one")
        feature_names = feature_names + relative.feature_names_
    elif stage in {"s2", "s2_logistic"}:
        groupby = FoldSafeGroupbyAggregateDiffs(
            specs=FULL_GROUPBY_SPECS,
            include_group_values=True,
            include_diffs=False,
        ).fit(train_df)
        train_groupby = groupby.transform(train_df)
        valid_groupby = groupby.transform(valid_df)
        train_features = train_features.merge(train_groupby, on=ID_COLUMN, how="left", validate="one_to_one")
        valid_features = valid_features.merge(valid_groupby, on=ID_COLUMN, how="left", validate="one_to_one")
        if test_features is not None:
            test_groupby = groupby.transform(test_df)
            test_features = test_features.merge(test_groupby, on=ID_COLUMN, how="left", validate="one_to_one")
        feature_names = feature_names + groupby.feature_names_
    elif stage == "s2_full":
        groupby = FoldSafeGroupbyAggregateDiffs(include_group_values=True, include_diffs=True).fit(train_df)
        train_groupby = groupby.transform(train_df)
        valid_groupby = groupby.transform(valid_df)
        train_features = train_features.merge(train_groupby, on=ID_COLUMN, how="left", validate="one_to_one")
        valid_features = valid_features.merge(valid_groupby, on=ID_COLUMN, how="left", validate="one_to_one")
        if test_features is not None:
            test_groupby = groupby.transform(test_df)
            test_features = test_features.merge(test_groupby, on=ID_COLUMN, how="left", validate="one_to_one")
        feature_names = feature_names + groupby.feature_names_

    return (
        train_features[feature_names],
        valid_features[feature_names],
        None if test_features is None else test_features[feature_names],
        feature_names,
        categorical_columns,
    )


def _merge_support_columns(features: pd.DataFrame, application: pd.DataFrame) -> pd.DataFrame:
    """Attach non-model application columns needed by full groupby recipes."""
    support_columns = [col for col in application.columns if col not in features.columns]
    if not support_columns:
        return features
    support = application[[ID_COLUMN] + support_columns].copy()
    return features.merge(support, on=ID_COLUMN, how="left", validate="one_to_one")


def _fit_predict_lightgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    x_test: pd.DataFrame | None,
    categorical_columns: list[str],
    params: dict,
) -> tuple[np.ndarray, np.ndarray | None]:
    encoder = OrdinalCategoryEncoder(categorical_columns).fit(x_train)
    x_train_encoded = encoder.transform(x_train)
    x_valid_encoded = encoder.transform(x_valid)
    x_test_encoded = encoder.transform(x_test) if x_test is not None else None

    lgb_params = dict(params)
    early_stopping_rounds = int(lgb_params.pop("early_stopping_rounds"))
    metric = lgb_params.get("metric", "auc")
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        x_train_encoded,
        y_train,
        eval_set=[(x_valid_encoded, y_valid)],
        eval_metric=metric,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    valid_pred = model.predict_proba(x_valid_encoded)[:, 1]
    test_pred = model.predict_proba(x_test_encoded)[:, 1] if x_test_encoded is not None else None
    return valid_pred, test_pred


def _fit_predict_logistic(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame | None,
    categorical_columns: list[str],
    params: dict,
) -> tuple[np.ndarray, np.ndarray | None]:
    numeric_columns = [col for col in x_train.columns if col not in categorical_columns]
    categorical_columns = [col for col in categorical_columns if col in x_train.columns]

    train_prepared = x_train.copy()
    valid_prepared = x_valid.copy()
    test_prepared = x_test.copy() if x_test is not None else None
    for col in categorical_columns:
        train_prepared[col] = train_prepared[col].astype("object").where(train_prepared[col].notna(), MISSING_CATEGORY)
        valid_prepared[col] = valid_prepared[col].astype("object").where(valid_prepared[col].notna(), MISSING_CATEGORY)
        if test_prepared is not None:
            test_prepared[col] = test_prepared[col].astype("object").where(test_prepared[col].notna(), MISSING_CATEGORY)

    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=True)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_columns,
            ),
            ("cat", one_hot, categorical_columns),
        ],
        sparse_threshold=0.0,
    )
    model = LogisticRegression(**params)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(train_prepared, y_train)
    valid_pred = pipeline.predict_proba(valid_prepared)[:, 1]
    test_pred = pipeline.predict_proba(test_prepared)[:, 1] if test_prepared is not None else None
    return valid_pred, test_pred


def _build_submission(
    test_features: pd.DataFrame,
    test_prediction_parts: list[np.ndarray],
    sample_submission_path: Path,
) -> pd.DataFrame:
    if not test_prediction_parts:
        raise ValueError("predict_test=True but no test predictions were collected")
    prediction_matrix = np.vstack(test_prediction_parts)
    predictions = prediction_matrix.mean(axis=0)
    submission = pd.DataFrame({ID_COLUMN: test_features[ID_COLUMN].values, TARGET_COLUMN: predictions})

    if sample_submission_path.exists():
        sample_submission = pd.read_csv(sample_submission_path)
        if list(sample_submission.columns) != [ID_COLUMN, TARGET_COLUMN]:
            raise ValueError(f"unexpected sample submission columns: {list(sample_submission.columns)}")
        if len(submission) != len(sample_submission):
            raise ValueError(f"submission row count {len(submission)} != sample row count {len(sample_submission)}")
        submission = sample_submission[[ID_COLUMN]].merge(submission, on=ID_COLUMN, how="left", validate="one_to_one")

    if submission[TARGET_COLUMN].isna().any():
        raise ValueError("submission contains missing predictions")
    if not submission[TARGET_COLUMN].between(0, 1).all():
        raise ValueError("submission predictions must be in [0, 1]")
    return submission


def _update_summary(
    config: dict,
    stage: str,
    model_name: str,
    fold_metrics: pd.DataFrame,
    oof_auc: float,
    n_features: int,
    output_dir: Path,
) -> None:
    summary_path = Path(config.get("results_dir", "results")) / "summary.csv"
    row = pd.DataFrame(
        [
            {
                "stage": stage,
                "model": model_name,
                "oof_auc": float(oof_auc),
                "fold_auc_mean": float(fold_metrics["auc"].mean()),
                "fold_auc_std": float(fold_metrics["auc"].std(ddof=0)),
                "n_features": int(n_features),
                "output_dir": str(output_dir),
            }
        ]
    )
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary[~((summary["stage"] == stage) & (summary["model"] == model_name))]
        summary = pd.concat([summary, row], axis=0, ignore_index=True)
    else:
        summary = row
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled CV experiment.")
    parser.add_argument("--stage", required=True, choices=sorted(SUPPORTED_STAGES))
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--config", default="configs/base_lgbm.yaml")
    parser.add_argument("--predict-test", action="store_true", help="Also save averaged fold predictions for Kaggle test")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_cv(stage=args.stage, model_name=args.model, config_path=args.config, predict_test=args.predict_test)


if __name__ == "__main__":
    main()
