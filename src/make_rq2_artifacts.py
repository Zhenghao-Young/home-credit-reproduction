"""Build RQ2 summary table from completed stage outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RQ2_STAGE_ROWS = [
    ("s3", "lightgbm", None, "S3: Blossom baseline (historical tables)"),
    ("s4", "lightgbm", "s3", "S4: group-relative position + recent windows"),
    ("b2", "lightgbm", "s4", "B2: pre-aggregation cleaning (bridge)"),
    ("s5", "lightgbm", "b2", "S5: short/long-term ratios + SK_DPD trends"),
]
RQ2_COMPARISONS = [
    ("s4", "lightgbm", "s3", "lightgbm", "S4 - S3", "group-relative position + recent-window behavior"),
    ("b2", "lightgbm", "s4", "lightgbm", "B2 - S4", "pre-aggregation cleaning fix"),
    ("s5", "lightgbm", "b2", "lightgbm", "S5 - B2", "short/long-term ratios + trend features"),
]


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "rq2"
    rq2 = build_rq2_results(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rq2_path = output_dir / "rq2_results.csv"
    rq2.to_csv(rq2_path, index=False)
    print(f"Saved {rq2_path}")


def build_rq2_results(results_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(results_dir / "summary.csv")
    rows = [
        _stage_row(summary, stage, model, parent, label)
        for stage, model, parent, label in RQ2_STAGE_ROWS
    ]
    rows.extend(
        _comparison_row(results_dir, summary, comparison)
        for comparison in RQ2_COMPARISONS
    )
    df = pd.DataFrame(rows)
    # Round numeric columns to reasonable precision
    for col in df.select_dtypes(include=["float64"]).columns:
        if col != "positive_fold_deltas":
            df[col] = df[col].round(15)
    return df


def _stage_row(
    summary: pd.DataFrame,
    stage: str,
    model: str,
    parent: str | None,
    label: str,
) -> dict:
    row = summary[(summary["stage"] == stage) & (summary["model"] == model)]
    if row.empty:
        raise ValueError(f"No summary row for stage={stage} model={model}")
    s = row.iloc[0]
    return {
        "row_type": "stage",
        "stage": stage,
        "model": model,
        "parent_stage": parent or "",
        "comparison": "",
        "label": label,
        "interpretation": "",
        "oof_auc": float(s["oof_auc"]),
        "fold_auc_mean": float(s["fold_auc_mean"]),
        "fold_auc_std": float(s["fold_auc_std"]),
        "n_features": int(s["n_features"]),
        "delta_oof_auc": "",
        "fold_delta_mean": "",
        "fold_delta_std": "",
        "positive_fold_deltas": "",
        "stability": "",
    }


def _comparison_row(
    results_dir: Path,
    summary: pd.DataFrame,
    comparison: tuple,
) -> dict:
    stage_a, model_a, stage_b, model_b, comp_label, interpretation = comparison

    row_a = summary[(summary["stage"] == stage_a) & (summary["model"] == model_a)]
    row_b = summary[(summary["stage"] == stage_b) & (summary["model"] == model_b)]
    if row_a.empty:
        raise ValueError(f"No summary row for {stage_a}/{model_a}")
    if row_b.empty:
        raise ValueError(f"No summary row for {stage_b}/{model_b}")

    # Load fold-level metrics for per-fold delta computation
    metrics_a = pd.read_csv(results_dir / stage_a / "fold_metrics.csv")
    metrics_b = pd.read_csv(results_dir / stage_b / "fold_metrics.csv")
    merged = metrics_a[["fold_id", "auc"]].merge(
        metrics_b[["fold_id", "auc"]],
        on="fold_id",
        suffixes=("_a", "_b"),
    )
    merged["delta"] = merged["auc_a"] - merged["auc_b"]
    delta_mean = float(merged["delta"].mean())
    delta_std = float(merged["delta"].std(ddof=0))
    positive_folds = int((merged["delta"] > 0).sum())

    # Stability rule: >= 4 positive and mean > 0 → stable
    if delta_mean > 0 and positive_folds >= 4:
        stability = "stable gain"
    else:
        stability = "unstable or insufficient evidence"

    s_a = row_a.iloc[0]
    return {
        "row_type": "comparison",
        "stage": stage_a,
        "model": model_a,
        "parent_stage": stage_b,
        "comparison": comp_label,
        "label": comp_label,
        "interpretation": interpretation,
        "oof_auc": float(s_a["oof_auc"]),
        "fold_auc_mean": float(s_a["fold_auc_mean"]),
        "fold_auc_std": float(s_a["fold_auc_std"]),
        "n_features": int(s_a["n_features"]),
        "delta_oof_auc": float(row_a.iloc[0]["oof_auc"]) - float(row_b.iloc[0]["oof_auc"]),
        "fold_delta_mean": delta_mean,
        "fold_delta_std": delta_std,
        "positive_fold_deltas": float(positive_folds),
        "stability": stability,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RQ2 results table.")
    parser.add_argument("--results-dir", default="results", help="Directory with stage outputs")
    parser.add_argument("--output-dir", default=None, help="Output directory for rq2 artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    main()
