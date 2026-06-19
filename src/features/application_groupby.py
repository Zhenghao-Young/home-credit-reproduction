"""Fold-safe S2 application groupby features."""

from __future__ import annotations

import pandas as pd


ID_COLUMN = "SK_ID_CURR"
MISSING_CATEGORY = "__MISSING__"

GROUPBY_SPECS = [
    (["OCCUPATION_TYPE"], ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_INCOME_TOTAL", "AMT_CREDIT"]),
    (["NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"], ["EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_INCOME_TOTAL"]),
]

COLS_TO_AGG = [
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_INCOME_TOTAL",
    "AMT_GOODS_PRICE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "OWN_CAR_AGE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_REGISTRATION",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "DAYS_ID_PUBLISH",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
]
AGGS = ["min", "mean", "max", "sum", "var", "median"]
AGGREGATION_PAIRS = [(col, agg) for col in COLS_TO_AGG for agg in AGGS]

FULL_GROUPBY_SPECS = [
    (["NAME_EDUCATION_TYPE", "CODE_GENDER"], AGGREGATION_PAIRS),
    (["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"], AGGREGATION_PAIRS),
    (["NAME_FAMILY_STATUS", "CODE_GENDER"], AGGREGATION_PAIRS),
    (
        ["CODE_GENDER", "ORGANIZATION_TYPE"],
        [
            ("AMT_ANNUITY", "mean"),
            ("AMT_INCOME_TOTAL", "mean"),
            ("DAYS_REGISTRATION", "mean"),
            ("EXT_SOURCE_1", "mean"),
        ],
    ),
    (
        ["CODE_GENDER", "REG_CITY_NOT_WORK_CITY"],
        [("AMT_ANNUITY", "mean"), ("CNT_CHILDREN", "mean"), ("DAYS_ID_PUBLISH", "mean")],
    ),
    (
        ["CODE_GENDER", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE", "REG_CITY_NOT_WORK_CITY"],
        [("EXT_SOURCE_1", "mean"), ("EXT_SOURCE_2", "mean")],
    ),
    (
        ["NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"],
        [
            ("AMT_CREDIT", "mean"),
            ("AMT_REQ_CREDIT_BUREAU_YEAR", "mean"),
            ("APARTMENTS_AVG", "mean"),
            ("BASEMENTAREA_AVG", "mean"),
            ("EXT_SOURCE_1", "mean"),
            ("EXT_SOURCE_2", "mean"),
            ("EXT_SOURCE_3", "mean"),
            ("NONLIVINGAREA_AVG", "mean"),
            ("OWN_CAR_AGE", "mean"),
            ("YEARS_BUILD_AVG", "mean"),
        ],
    ),
    (
        ["NAME_EDUCATION_TYPE", "OCCUPATION_TYPE", "REG_CITY_NOT_WORK_CITY"],
        [("ELEVATORS_AVG", "mean"), ("EXT_SOURCE_1", "mean")],
    ),
    (
        ["OCCUPATION_TYPE"],
        [
            ("AMT_ANNUITY", "mean"),
            ("CNT_CHILDREN", "mean"),
            ("CNT_FAM_MEMBERS", "mean"),
            ("DAYS_BIRTH", "mean"),
            ("DAYS_EMPLOYED", "mean"),
            ("DAYS_ID_PUBLISH", "mean"),
            ("DAYS_REGISTRATION", "mean"),
            ("EXT_SOURCE_1", "mean"),
            ("EXT_SOURCE_2", "mean"),
            ("EXT_SOURCE_3", "mean"),
        ],
    ),
]


class FoldSafeGroupbyMean:
    """Fit group means on one training fold and map them to train/valid data."""

    def __init__(self, specs=None):
        self.specs = specs or GROUPBY_SPECS
        self.group_tables_: list[tuple[list[str], pd.DataFrame]] = []
        self.global_means_: dict[str, float] = {}
        self.feature_names_: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> "FoldSafeGroupbyMean":
        self.group_tables_ = []
        self.global_means_ = {}
        self.feature_names_ = []
        train_prepared = self._prepare_group_keys(train_df)

        for group_cols, value_cols in self.specs:
            available_values = [col for col in value_cols if col in train_prepared.columns]
            if not available_values:
                continue

            agg = train_prepared.groupby(group_cols, dropna=False)[available_values].mean().reset_index()
            rename_map = {
                value_col: self._feature_name(group_cols, value_col)
                for value_col in available_values
            }
            agg = agg.rename(columns=rename_map)
            self.group_tables_.append((group_cols, agg))

            for value_col in available_values:
                feature_name = rename_map[value_col]
                self.global_means_[feature_name] = float(train_prepared[value_col].mean())
                self.feature_names_.append(feature_name)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.group_tables_:
            return pd.DataFrame({ID_COLUMN: df[ID_COLUMN].values})

        transformed = self._prepare_group_keys(df)[[ID_COLUMN]].copy()
        work = self._prepare_group_keys(df)
        for group_cols, group_table in self.group_tables_:
            mapped = work[[ID_COLUMN] + group_cols].merge(group_table, on=group_cols, how="left")
            mapped = mapped[[ID_COLUMN] + [col for col in group_table.columns if col not in group_cols]]
            transformed = transformed.merge(mapped, on=ID_COLUMN, how="left")

        for feature_name in self.feature_names_:
            transformed[feature_name] = transformed[feature_name].fillna(self.global_means_[feature_name])

        return transformed[[ID_COLUMN] + self.feature_names_]

    @staticmethod
    def _feature_name(group_cols: list[str], value_col: str) -> str:
        return f"group_mean__{'__'.join(group_cols)}__{value_col}"

    def _prepare_group_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        group_columns = {col for group_cols, _ in self.specs for col in group_cols}
        for col in group_columns:
            if col in prepared.columns:
                prepared[col] = prepared[col].astype("object").where(prepared[col].notna(), MISSING_CATEGORY)
        return prepared


class FoldSafeGroupbyAggregateDiffs:
    """Fold-safe version of the old project's application aggregation recipes."""

    def __init__(self, specs=None, include_group_values: bool = True, include_diffs: bool = True):
        self.specs = specs or FULL_GROUPBY_SPECS
        self.include_group_values = include_group_values
        self.include_diffs = include_diffs
        self.group_tables_: list[tuple[list[str], pd.DataFrame]] = []
        self.groupby_feature_names_: list[str] = []
        self.diff_feature_names_: list[str] = []
        self.feature_names_: list[str] = []
        self.global_means_: dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "FoldSafeGroupbyAggregateDiffs":
        self.group_tables_ = []
        self.groupby_feature_names_ = []
        self.diff_feature_names_ = []
        self.feature_names_ = []
        self.global_means_ = {}
        train_prepared = self._prepare_group_keys(train_df)

        for group_cols, specs in self.specs:
            available_specs = [(col, agg) for col, agg in specs if col in train_prepared.columns]
            if not available_specs:
                continue
            named_aggs = {
                self._groupby_feature_name(group_cols, col, agg): (col, agg)
                for col, agg in available_specs
            }
            group_table = train_prepared.groupby(group_cols, dropna=False).agg(**named_aggs).reset_index()
            self.group_tables_.append((group_cols, group_table))
            for feature_name in named_aggs:
                self.groupby_feature_names_.append(feature_name)
                self.global_means_[feature_name] = float(group_table[feature_name].mean())

        self.feature_names_ = []
        if self.include_group_values:
            self.feature_names_.extend(self.groupby_feature_names_)
        if self.include_diffs:
            for group_cols, specs in self.specs:
                for value_col, agg in specs:
                    if value_col not in train_prepared.columns or agg not in {"mean", "median", "max", "min"}:
                        continue
                    group_feature = self._groupby_feature_name(group_cols, value_col, agg)
                    if group_feature not in self.groupby_feature_names_:
                        continue
                    self.diff_feature_names_.extend(
                        [
                            f"{group_feature}_diff",
                            f"{group_feature}_abs_diff",
                        ]
                    )
            self.feature_names_.extend(self.diff_feature_names_)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        work = self._prepare_group_keys(df)
        merged = work.copy()
        for group_cols, group_table in self.group_tables_:
            new_columns = [col for col in group_table.columns if col not in group_cols]
            group_values = work[[ID_COLUMN] + group_cols].merge(group_table, on=group_cols, how="left")
            group_values = group_values[[ID_COLUMN] + new_columns]
            merged = merged.merge(group_values, on=ID_COLUMN, how="left")

        feature_data = {ID_COLUMN: merged[ID_COLUMN].values}
        for feature_name in self.groupby_feature_names_:
            merged[feature_name] = merged[feature_name].fillna(self.global_means_[feature_name])
            if self.include_group_values:
                feature_data[feature_name] = merged[feature_name].values

        if self.include_diffs:
            for group_cols, specs in self.specs:
                for value_col, agg in specs:
                    if value_col not in merged.columns or agg not in {"mean", "median", "max", "min"}:
                        continue
                    group_feature = self._groupby_feature_name(group_cols, value_col, agg)
                    if group_feature not in merged.columns:
                        continue
                    diff_feature = f"{group_feature}_diff"
                    abs_diff_feature = f"{group_feature}_abs_diff"
                    diff_values = merged[value_col] - merged[group_feature]
                    feature_data[diff_feature] = diff_values.values
                    feature_data[abs_diff_feature] = diff_values.abs().values

        return pd.DataFrame(feature_data, index=merged.index)[[ID_COLUMN] + self.feature_names_]

    @staticmethod
    def _groupby_feature_name(group_cols: list[str], value_col: str, agg: str) -> str:
        return f"group_full__{'__'.join(group_cols)}__{value_col}__{agg}"

    def _prepare_group_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        group_columns = {col for group_cols, _ in self.specs for col in group_cols}
        for col in group_columns:
            if col in prepared.columns:
                prepared[col] = prepared[col].astype("object").where(prepared[col].notna(), MISSING_CATEGORY)
        return prepared
