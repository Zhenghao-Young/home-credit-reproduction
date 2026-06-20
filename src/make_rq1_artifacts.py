"""Build RQ1 summary table and gain tree figure from completed stage outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RQ1_STAGE_ROWS = [
    ("s1", "lightgbm", None, "S1: application baseline"),
    ("s2", "lightgbm", "s1", "S2: application group aggregations"),
    ("s2_logistic", "logistic", "s2", "S2-Logistic: linear model bridge"),
    ("b1", "lightgbm", "s1", "B1: business ratios"),
    ("s3", "lightgbm", "b1", "S3: historical tables"),
]
RQ1_COMPARISONS = [
    ("s2", "lightgbm", "s1", "lightgbm", "S2 - S1", "ordinary application group aggregations"),
    ("b1", "lightgbm", "s1", "lightgbm", "B1 - S1", "business ratios and EXT_SOURCE summaries"),
    ("s3", "lightgbm", "b1", "lightgbm", "S3 - B1", "multi-table historical aggregations"),
    ("s2", "lightgbm", "s2_logistic", "logistic", "S2-LGBM - S2-Logistic", "tree nonlinearity on S2 features"),
]


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "rq1"
    rq1 = build_rq1_results(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rq1_path = output_dir / "rq1_results.csv"
    rq1.to_csv(rq1_path, index=False)
    figure_paths = build_gain_tree_figure(rq1, output_dir / "rq1_gain_tree")
    print(f"Saved {rq1_path}")
    for path in figure_paths:
        print(f"Saved {path}")


def build_rq1_results(results_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(results_dir / "summary.csv")
    rows = [_stage_row(summary, stage, model, parent, label) for stage, model, parent, label in RQ1_STAGE_ROWS]
    rows.extend(_comparison_row(results_dir, summary, comparison) for comparison in RQ1_COMPARISONS)
    return pd.DataFrame(rows)


def build_gain_tree_figure(rq1: pd.DataFrame, output_stem: Path) -> list[Path]:
    stage_rows = rq1[rq1["row_type"].eq("stage")].set_index("stage")
    comparison_rows = rq1[rq1["row_type"].eq("comparison")].set_index("comparison")
    deltas = {
        "S2 - S1": float(comparison_rows.loc["S2 - S1", "delta_oof_auc"]),
        "B1 - S1": float(comparison_rows.loc["B1 - S1", "delta_oof_auc"]),
        "S3 - B1": float(comparison_rows.loc["S3 - B1", "delta_oof_auc"]),
    }
    aucs = {
        stage: float(stage_rows.loc[stage, "oof_auc"])
        for stage in ["s1", "s2", "b1", "s3"]
    }

    with _style_context():
        fig, ax = _subplots()
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        nodes = {
            "s1": (0.12, 0.63),
            "s2": (0.55, 0.82),
            "b1": (0.55, 0.48),
            "s3": (0.88, 0.48),
        }
        labels = {
            "s1": f"S1\nAUC {aucs['s1']:.6f}",
            "s2": f"S2\nAUC {aucs['s2']:.6f}",
            "b1": f"B1\nAUC {aucs['b1']:.6f}",
            "s3": f"S3\nAUC {aucs['s3']:.6f}",
        }
        colors = {
            "s1": "#F3F4F6",
            "s2": "#E0F2FE",
            "b1": "#FEF3C7",
            "s3": "#DCFCE7",
        }
        for node, (x, y) in nodes.items():
            ax.text(
                x,
                y,
                labels[node],
                ha="center",
                va="center",
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": colors[node],
                    "edgecolor": "#374151",
                    "linewidth": 0.9,
                },
            )

        _arrow(ax, nodes["s1"], nodes["s2"], _format_delta(deltas["S2 - S1"]), label_offset=0.045)
        _arrow(ax, nodes["s1"], nodes["b1"], _format_delta(deltas["B1 - S1"]), label_offset=0.045)
        _arrow(ax, nodes["b1"], nodes["s3"], _format_delta(deltas["S3 - B1"]), label_offset=0.095)

        ax.text(0.5, 0.95, "RQ1 OOF AUC Gain Tree", ha="center", va="center", fontweight="bold")
        ax.text(
            0.5,
            0.1,
            "Largest stable gain: historical table aggregations (S3 - B1)",
            ha="center",
            va="center",
        )
        return _savefig(fig, output_stem, formats=("pdf", "png"))


def _stage_row(summary: pd.DataFrame, stage: str, model: str, parent: str | None, label: str) -> dict:
    matched = summary[(summary["stage"].eq(stage)) & (summary["model"].eq(model))]
    if matched.empty:
        raise FileNotFoundError(f"missing summary row for {stage}/{model}")
    row = matched.iloc[-1]
    return {
        "row_type": "stage",
        "stage": stage,
        "model": model,
        "parent_stage": parent,
        "comparison": "",
        "label": label,
        "interpretation": "",
        "oof_auc": row["oof_auc"],
        "fold_auc_mean": row["fold_auc_mean"],
        "fold_auc_std": row["fold_auc_std"],
        "n_features": int(row["n_features"]),
        "delta_oof_auc": np.nan,
        "fold_delta_mean": np.nan,
        "fold_delta_std": np.nan,
        "positive_fold_deltas": np.nan,
        "stability": "",
    }


def _comparison_row(results_dir: Path, summary: pd.DataFrame, comparison: tuple[str, str, str, str, str, str]) -> dict:
    child_stage, child_model, parent_stage, parent_model, comparison_label, interpretation = comparison
    child_summary = _summary_row(summary, child_stage, child_model)
    parent_summary = _summary_row(summary, parent_stage, parent_model)
    child_folds = _fold_metrics(results_dir, child_stage, child_model)
    parent_folds = _fold_metrics(results_dir, parent_stage, parent_model)
    fold_deltas = child_folds.merge(parent_folds, on="fold_id", suffixes=("_child", "_parent"))
    fold_deltas["delta"] = fold_deltas["auc_child"] - fold_deltas["auc_parent"]
    positive = int((fold_deltas["delta"] > 0).sum())
    delta_oof = float(child_summary["oof_auc"] - parent_summary["oof_auc"])
    stable = delta_oof > 0 and positive >= 4
    return {
        "row_type": "comparison",
        "stage": child_stage,
        "model": child_model,
        "parent_stage": parent_stage,
        "comparison": comparison_label,
        "label": comparison_label,
        "interpretation": interpretation,
        "oof_auc": child_summary["oof_auc"],
        "fold_auc_mean": child_summary["fold_auc_mean"],
        "fold_auc_std": child_summary["fold_auc_std"],
        "n_features": int(child_summary["n_features"]),
        "delta_oof_auc": delta_oof,
        "fold_delta_mean": float(fold_deltas["delta"].mean()),
        "fold_delta_std": float(fold_deltas["delta"].std(ddof=0)),
        "positive_fold_deltas": positive,
        "stability": "stable gain" if stable else "unstable or insufficient evidence",
    }


def _summary_row(summary: pd.DataFrame, stage: str, model: str) -> pd.Series:
    matched = summary[(summary["stage"].eq(stage)) & (summary["model"].eq(model))]
    if matched.empty:
        raise FileNotFoundError(f"missing summary row for {stage}/{model}")
    return matched.iloc[-1]


def _fold_metrics(results_dir: Path, stage: str, model: str) -> pd.DataFrame:
    metrics = pd.read_csv(results_dir / stage / "fold_metrics.csv")
    metrics = metrics[metrics["model"].eq(model)]
    if metrics.empty:
        raise FileNotFoundError(f"missing fold metrics for {stage}/{model}")
    return metrics[["fold_id", "auc"]]


def _style_context():
    try:
        from paperplot import style_context

        return style_context(
            "manuscript_double",
            usetex=True,
            plot_width_cm=15.5,
            plot_height_cm=8.0,
            margin_left_cm=0.5,
            margin_right_cm=0.5,
            margin_top_cm=0.5,
            margin_bottom_cm=0.5,
        )
    except ImportError:
        return plt.rc_context(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.size": 9,
                "figure.dpi": 160,
                "savefig.dpi": 300,
            }
        )


def _subplots():
    try:
        from paperplot import subplots

        return subplots()
    except ImportError:
        fig, ax = plt.subplots(figsize=(6.1, 3.15), dpi=160)
        fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.08)
        return fig, ax


def _savefig(fig, path: Path, formats: tuple[str, ...]) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from paperplot import savefig

        written = savefig(fig, path, formats=formats)
    except ImportError:
        written = []
        for fmt in formats:
            output_path = path.with_suffix(f".{fmt}")
            fig.savefig(output_path)
            written.append(output_path)
    plt.close(fig)
    return [Path(p) for p in written]


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], label: str, label_offset: float) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "#111827", "shrinkA": 32, "shrinkB": 34},
    )
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    ax.text(
        mid_x,
        mid_y + label_offset,
        rf"$\Delta$ AUC {label}",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )


def _format_delta(value: float) -> str:
    return f"{value:+.6f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RQ1 table and gain tree artifacts.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default=None, help="Defaults to <results-dir>/rq1")
    return parser.parse_args()


if __name__ == "__main__":
    main()
