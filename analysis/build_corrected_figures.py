from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METRIC_LABELS = {
    "r2": "R²",
    "mape_pct": "MAPE (%)",
    "rmse": "Overall RMSE (USD)",
    "rmse_q3": "RMSE ≥ training Q3 (USD)",
    "rmse_q3_iqr2": "RMSE ≥ training Q3 + IQR/2 (USD)",
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.png", facecolor="white")
    figure.savefig(output_dir / f"{stem}.pdf", facecolor="white")
    plt.close(figure)


def performance_figure(metrics: pd.DataFrame, output_dir: Path) -> None:
    focus = metrics[
        (metrics["feature_set"] == "A_all_21")
        & (
            metrics["model_family"].eq("single")
            | metrics["model_family"].eq("two_stage_tl_nested_selected")
        )
    ].copy()
    label_order = [
        "CatBoost",
        "XGBoost",
        "Monotonic GBM",
        "ANN",
        "Two-stage TL\n(training-selected)",
    ]
    focus["plot_label"] = np.where(
        focus["model_family"].eq("two_stage_tl_nested_selected"),
        "Two-stage TL\n(training-selected)",
        focus["model"],
    )
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#B279A2", "#2F855A"]
    metrics_to_plot = ["r2", "mape_pct", "rmse", "rmse_q3", "rmse_q3_iqr2"]
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 6.7))
    axes = axes.ravel()
    for index, metric in enumerate(metrics_to_plot):
        axis = axes[index]
        summary = (
            focus.groupby("plot_label")[metric]
            .agg(["mean", "std"])
            .reindex(label_order)
        )
        x = np.arange(len(summary))
        axis.bar(
            x,
            summary["mean"],
            yerr=summary["std"],
            capsize=3,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
        )
        axis.set_title(f"({chr(97 + index)}) {METRIC_LABELS[metric]}")
        axis.set_xticks(x)
        axis.set_xticklabels(label_order, rotation=24, ha="right")
        if metric == "r2":
            lower = min(0.0, float(summary["mean"].min() - summary["std"].max() - 0.05))
            axis.set_ylim(lower, 1.0)
        axis.grid(axis="x", visible=False)
    axes[-1].axis("off")
    figure.suptitle(
        "Leakage-controlled outer-test performance across 10 seeds",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    figure.tight_layout()
    save_figure(figure, output_dir, "figure_corrected_model_performance")


def ablation_figure(metrics: pd.DataFrame, output_dir: Path) -> None:
    focus = metrics[
        metrics["model_family"].eq("two_stage_tl_nested_selected")
    ].copy()
    feature_order = ["A_all_21", "B_no_component_costs", "C_physical_quantity"]
    feature_labels = [
        "A: all 21 inputs",
        "B: exclude 4 component costs",
        "C: physical/categorical/quantity",
    ]
    colors = ["#2F855A", "#D97706", "#9C4221"]
    metrics_to_plot = ["r2", "rmse", "rmse_q3", "rmse_q3_iqr2"]
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))
    for index, (axis, metric) in enumerate(zip(axes.ravel(), metrics_to_plot)):
        summary = (
            focus.groupby("feature_set")[metric]
            .agg(["mean", "std"])
            .reindex(feature_order)
        )
        x = np.arange(len(summary))
        axis.bar(
            x,
            summary["mean"],
            yerr=summary["std"],
            capsize=4,
            color=colors,
            edgecolor="white",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(feature_labels, rotation=18, ha="right")
        axis.set_title(f"({chr(97 + index)}) {METRIC_LABELS[metric]}")
        axis.grid(axis="x", visible=False)
        if metric == "r2":
            lower = min(0.0, float(summary["mean"].min() - summary["std"].max() - 0.05))
            axis.set_ylim(lower, 1.0)
    figure.suptitle(
        "Component-cost ablation for the training-selected two-stage procedure",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    figure.tight_layout()
    save_figure(figure, output_dir, "figure_component_cost_ablation")


def svm_figure(svm: pd.DataFrame, output_dir: Path) -> None:
    focus = svm[svm["feature_set"] == "A_all_21"].copy()
    cutoff_order = ["Mean", "Q3", "Q3+IQR/2"]
    score_columns = [
        "classifier_precision",
        "classifier_recall",
        "classifier_f1",
    ]
    labels = ["Precision", "Recall", "F1"]
    colors = ["#4C78A8", "#F58518", "#2F855A"]
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(cutoff_order))
    width = 0.24
    for offset, (score, label, color) in enumerate(
        zip(score_columns, labels, colors)
    ):
        summary = focus.groupby("cutoff_rule")[score].agg(["mean", "std"]).reindex(
            cutoff_order
        )
        axis.bar(
            x + (offset - 1) * width,
            summary["mean"],
            width,
            yerr=summary["std"],
            capsize=3,
            label=label,
            color=color,
            edgecolor="white",
        )
    axis.set_xticks(x)
    axis.set_xticklabels(cutoff_order)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Outer-test classification score")
    axis.set_title("SVM classification under training-derived cutoffs")
    axis.legend(frameon=False, ncol=3, loc="upper right")
    axis.grid(axis="x", visible=False)
    figure.tight_layout()
    save_figure(figure, output_dir, "figure_svm_cutoff_performance")


def selection_figure(selected: pd.DataFrame, output_dir: Path) -> None:
    feature_order = ["A_all_21", "B_no_component_costs", "C_physical_quantity"]
    feature_titles = [
        "A: all inputs",
        "B: no component costs",
        "C: physical/quantity",
    ]
    cutoff_order = ["Mean", "Q3", "Q3+IQR/2"]
    strategy_order = ["Conservative", "Balanced", "Aggressive"]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for axis, feature_set, title in zip(axes, feature_order, feature_titles):
        subset = selected[selected["feature_set"] == feature_set]
        counts = (
            subset.groupby(["cutoff_rule", "transfer_strategy"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=cutoff_order, columns=strategy_order, fill_value=0)
        )
        sns.heatmap(
            counts,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            vmin=0,
            vmax=10,
            cbar=axis is axes[-1],
            ax=axis,
            linewidths=0.5,
            linecolor="white",
        )
        axis.set_title(title)
        axis.set_xlabel("Transfer strategy")
        axis.set_ylabel("Cutoff" if axis is axes[0] else "")
        axis.tick_params(axis="x", rotation=25)
        axis.tick_params(axis="y", rotation=0)
    figure.suptitle(
        "Training-only cutoff/strategy selections across 10 outer seeds",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    figure.tight_layout()
    save_figure(figure, output_dir, "figure_training_selection_frequency")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build corrected publication figures")
    parser.add_argument(
        "--compiled-dir",
        type=Path,
        default=Path("results")
        / "corrected_analysis_20260717"
        / "compiled",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results")
        / "corrected_analysis_20260717"
        / "figures",
    )
    args = parser.parse_args()
    setup_style()
    metrics = pd.read_csv(args.compiled_dir / "model_metrics_seed_level.csv")
    svm = pd.read_csv(args.compiled_dir / "svm_results_seed_level.csv")
    selected = pd.read_csv(args.compiled_dir / "selected_configurations.csv")
    performance_figure(metrics, args.output_dir)
    ablation_figure(metrics, args.output_dir)
    svm_figure(svm, args.output_dir)
    selection_figure(selected, args.output_dir)
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
