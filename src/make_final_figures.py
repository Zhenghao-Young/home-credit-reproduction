"""Generate RQ3 artifacts: stacking comparison, meta coefficients, correlation matrix,
and final evidence chain figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import ID_COLUMN, PREDICTION_COLUMN, TARGET_COLUMN
from src.split import FOLD_COLUMN


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "rq3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build stacking comparison table
    stacking_csv = build_stacking_results(results_dir)
    stacking_path = output_dir / "stacking_results.csv"
    stacking_csv.to_csv(stacking_path, index=False)
    print(f"Saved {stacking_path}")

    # 2. Build meta coefficients
    meta_csv = build_meta_coefficients(results_dir)
    meta_path = output_dir / "meta_coefficients.csv"
    meta_csv.to_csv(meta_path, index=False)
    print(f"Saved {meta_path}")

    # 3. Build prediction correlation matrix
    corr_csv, corr_figs = build_prediction_correlation(results_dir, output_dir)
    print(f"Saved {corr_csv}")
    for p in corr_figs:
        print(f"Saved {p}")

    # 4. Build final evidence chain figure
    chain_figs = build_evidence_chain_figure(results_dir, output_dir)
    for p in chain_figs:
        print(f"Saved {p}")


def build_stacking_results(results_dir: Path) -> pd.DataFrame:
    """Build S5 vs simple avg vs logistic stacking comparison table."""
    summary = pd.read_csv(results_dir / "summary.csv")

    def _get_summary_row(stage: str, model: str) -> pd.Series:
        row = summary[(summary["stage"] == stage) & (summary["model"] == model)]
        if row.empty:
            raise ValueError(f"No summary row for {stage}/{model}")
        return row.iloc[0]

    s5_row = _get_summary_row("s5", "lightgbm")
    s6_avg_row = _get_summary_row("s6_avg", "stacking")
    s6_stack_row = _get_summary_row("s6_stack", "stacking")
    s5_auc = float(s5_row["oof_auc"])
    s6_avg_auc = float(s6_avg_row["oof_auc"])
    s6_stack_auc = float(s6_stack_row["oof_auc"])

    def _official_score(row: pd.Series, column: str) -> float:
        return float(row[column]) if column in row and pd.notna(row[column]) else np.nan

    s5_folds = pd.read_csv(results_dir / "s5" / "fold_metrics.csv")
    s6_folds = pd.read_csv(results_dir / "s6" / "fold_metrics.csv")
    s6_avg_folds = s6_folds[s6_folds["stage"] == "s6_avg"]
    s6_stack_folds = s6_folds[s6_folds["stage"] == "s6_stack"]

    def _fold_delta(child_folds, parent_folds):
        merged = child_folds[["fold_id", "auc"]].merge(
            parent_folds[["fold_id", "auc"]], on="fold_id", suffixes=("_c", "_p")
        )
        merged["delta"] = merged["auc_c"] - merged["auc_p"]
        return (
            float(merged["delta"].mean()),
            float(merged["delta"].std(ddof=0)),
            int((merged["delta"] > 0).sum()),
        )

    avg_delta_mean, avg_delta_std, avg_pos = _fold_delta(s6_avg_folds, s5_folds)
    stack_delta_mean, stack_delta_std, stack_pos = _fold_delta(s6_stack_folds, s5_folds)

    rows = [
        {
            "method": "S5: best single model (LightGBM)",
            "oof_auc": s5_auc,
            "kaggle_public_auc": _official_score(s5_row, "kaggle_public_auc"),
            "kaggle_private_auc": _official_score(s5_row, "kaggle_private_auc"),
            "delta_vs_s5": 0.0,
            "fold_delta_mean": 0.0,
            "fold_delta_std": 0.0,
            "positive_folds": 5,
            "stability": "baseline",
        },
        {
            "method": "Simple average (s2_lr + s3 + s4 + s5)",
            "oof_auc": s6_avg_auc,
            "kaggle_public_auc": _official_score(s6_avg_row, "kaggle_public_auc"),
            "kaggle_private_auc": _official_score(s6_avg_row, "kaggle_private_auc"),
            "delta_vs_s5": s6_avg_auc - s5_auc,
            "fold_delta_mean": avg_delta_mean,
            "fold_delta_std": avg_delta_std,
            "positive_folds": avg_pos,
            "stability": "stable gain" if (s6_avg_auc > s5_auc and avg_pos >= 4) else "unstable",
        },
        {
            "method": "L2-Logistic stacking (s2_lr + s3 + s4 + s5)",
            "oof_auc": s6_stack_auc,
            "kaggle_public_auc": _official_score(s6_stack_row, "kaggle_public_auc"),
            "kaggle_private_auc": _official_score(s6_stack_row, "kaggle_private_auc"),
            "delta_vs_s5": s6_stack_auc - s5_auc,
            "fold_delta_mean": stack_delta_mean,
            "fold_delta_std": stack_delta_std,
            "positive_folds": stack_pos,
            "stability": "stable gain" if (s6_stack_auc > s5_auc and stack_pos >= 4) else "unstable",
        },
    ]
    return pd.DataFrame(rows)


def build_meta_coefficients(results_dir: Path) -> pd.DataFrame:
    """Extract logistic stacking coefficients from s6 stacked OOF."""
    folds = pd.read_csv("data/folds.csv")

    # Rebuild Z from base OOFs
    base_stages = [
        ("s2_logistic", "p_lr_s2"),
        ("s3", "p_lightgbm_s3"),
        ("s4", "p_lightgbm_s4"),
        ("s5", "p_lightgbm_s5"),
    ]
    preds = {}
    for stage, col_name in base_stages:
        oof = pd.read_parquet(results_dir / stage / "oof.parquet")
        preds[col_name] = oof.set_index(ID_COLUMN)[PREDICTION_COLUMN]
    z = pd.DataFrame(preds).loc[folds[ID_COLUMN]]
    target = folds.set_index(ID_COLUMN).loc[z.index, TARGET_COLUMN]

    # Fit logistic on all data (same as final_meta in stacking.py)
    from sklearn.linear_model import LogisticRegression

    meta = LogisticRegression(
        penalty="l2",
        solver="newton-cholesky",
        max_iter=1000,
        class_weight="balanced",
        random_state=2026,
    )
    meta.fit(z.values, target.values)

    rows = []
    for i, name in enumerate(z.columns):
        rows.append(
            {
                "feature": name,
                "coefficient": float(meta.coef_[0, i]),
                "intercept": False,
            }
        )
    rows.append(
        {
            "feature": "intercept",
            "coefficient": float(meta.intercept_[0]),
            "intercept": True,
        }
    )
    return pd.DataFrame(rows)


def build_prediction_correlation(results_dir: Path, output_dir: Path) -> tuple[Path, list[Path]]:
    """Build correlation matrix of base model OOF predictions."""
    folds = pd.read_csv("data/folds.csv")

    stage_labels = [
        ("s2_logistic", "S2-LR"),
        ("s3", "S3"),
        ("s4", "S4"),
        ("s5", "S5"),
    ]

    preds = {}
    for stage, label in stage_labels:
        oof = pd.read_parquet(results_dir / stage / "oof.parquet")
        preds[label] = oof.set_index(ID_COLUMN)[PREDICTION_COLUMN]

    z = pd.DataFrame(preds)
    corr = z.corr()

    # Save CSV
    csv_path = output_dir / "prediction_correlation.csv"
    corr.to_csv(csv_path)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, vmin=0.5, vmax=1.0, cmap="YlOrRd")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.4f}", ha="center", va="center", fontsize=10)
    ax.set_title("First-Level OOF Prediction Correlation")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    fig_paths = []
    for fmt in ("pdf", "png"):
        p = output_dir / f"prediction_correlation.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        fig_paths.append(p)
    plt.close(fig)

    return csv_path, fig_paths


def build_evidence_chain_figure(results_dir: Path, output_dir: Path) -> list[Path]:
    """Build the full S1 -> S6 evidence chain gain tree."""
    summary = pd.read_csv(results_dir / "summary.csv")

    def _auc(stage, model):
        row = summary[(summary["stage"] == stage) & (summary["model"] == model)]
        return float(row.iloc[0]["oof_auc"])

    aucs = {
        "s1": _auc("s1", "lightgbm"),
        "b1": _auc("b1", "lightgbm"),
        "s3": _auc("s3", "lightgbm"),
        "s4": _auc("s4", "lightgbm"),
        "b2": _auc("b2", "lightgbm"),
        "s5": _auc("s5", "lightgbm"),
        "s6_avg": _auc("s6_avg", "stacking"),
        "s6_stack": _auc("s6_stack", "stacking"),
    }

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    nodes = {
        "s1": (0.06, 0.55),
        "b1": (0.26, 0.80),
        "s3": (0.26, 0.30),
        "s4": (0.46, 0.80),
        "b2": (0.46, 0.30),
        "s5": (0.66, 0.80),
        "s6_avg": (0.88, 0.80),
        "s6_stack": (0.88, 0.30),
    }

    colors = {
        "s1": "#F3F4F6",
        "b1": "#FEF3C7",
        "s3": "#DCFCE7",
        "s4": "#E0F2FE",
        "b2": "#FCE7F3",
        "s5": "#DDD6FE",
        "s6_avg": "#FED7AA",
        "s6_stack": "#FED7AA",
    }

    for node, (x, y) in nodes.items():
        ax.text(
            x, y, f"{node.upper()}\n{aucs[node]:.6f}",
            ha="center", va="center", fontweight="bold", fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": colors.get(node, "#F3F4F6"),
                "edgecolor": "#374151",
                "linewidth": 0.8,
            },
        )

    edges = [
        ("s1", "b1"),
        ("b1", "s3"),
        ("b1", "s4"),
        ("b1", "b2"),
        ("b2", "s5"),
        ("s4", "s5"),
        ("s5", "s6_avg"),
        ("s5", "s6_stack"),
    ]

    for src, dst in edges:
        delta = aucs[dst] - aucs[src]
        sign = "+" if delta > 0 else ""
        _draw_arrow(ax, nodes[src], nodes[dst], f"{sign}{delta:.6f}")

    ax.set_title("Full Evidence Chain: OOF AUC Gains (S1 -> S6)", fontweight="bold", fontsize=13)
    ax.text(0.5, 0.01, "Each arrow shows OOF AUC delta between stages", ha="center", fontsize=9)

    fig.tight_layout()
    fig_paths = []
    for fmt in ("pdf", "png"):
        p = output_dir / f"final_evidence_chain.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        fig_paths.append(p)
    plt.close(fig)

    return fig_paths


def _draw_arrow(ax, src, dst, label):
    """Draw a simple arrow with label from src to dst."""
    ax.annotate(
        "",
        xy=dst, xytext=src,
        arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1.2),
    )
    mid_x = (src[0] + dst[0]) / 2
    mid_y = (src[1] + dst[1]) / 2
    ax.text(mid_x, mid_y + 0.025, label, ha="center", va="bottom", fontsize=7, color="#374151")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RQ3 analysis artifacts.")
    parser.add_argument("--results-dir", default="results", help="Directory with stage outputs")
    parser.add_argument("--output-dir", default=None, help="Output directory for rq3 artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    main()
