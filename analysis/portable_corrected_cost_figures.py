"""Portable JPG figure generator for the corrected cost analysis.

This script is intentionally separate from model fitting.  It reads the
``tasks`` directory produced by ``portable_corrected_cost_analysis.py`` and
creates the nine manuscript figures plus compact summary CSV files.  No
project-local module or precompiled result table is required.

Google Colab example::

    !pip -q install pandas numpy matplotlib pillow
    !python portable_corrected_cost_figures.py \
        --results-dir corrected_analysis_output \
        --output-dir corrected_analysis_figures

The default descriptive outer split is seed 42.  Figures 6--7 are descriptive
only; confirmatory interpretation must use all available seed-level metrics.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import warnings

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "portable_cost_mplconfig")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


OUT = Path("corrected_analysis_figures")

BLUE = "#2F5D8C"
LIGHT_BLUE = "#DCEAF5"
GREEN = "#2E8B57"
LIGHT_GREEN = "#DCEFE4"
ORANGE = "#D97706"
LIGHT_ORANGE = "#FCE8CC"
RED = "#B33A3A"
GRAY = "#4B5563"
LIGHT_GRAY = "#EEF1F4"


def save_jpg(fig: plt.Figure, filename: str) -> None:
    path = OUT / filename
    fig.savefig(
        path,
        dpi=300,
        format="jpg",
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(fig)


def read_task_table(results_dir: Path, filename: str) -> pd.DataFrame:
    """Read and concatenate one artifact from every completed task."""
    task_root = results_dir / "tasks"
    paths = sorted(task_root.glob(f"*/{filename}"))
    if not paths:
        task_root = results_dir / "smoke_tasks"
        paths = sorted(task_root.glob(f"*/{filename}"))
    if not paths:
        raise FileNotFoundError(
            f"No {filename!r} files were found under {results_dir / 'tasks'} "
            f"or {results_dir / 'smoke_tasks'}. "
            "Run the analysis script first."
        )
    frames: list[pd.DataFrame] = []
    for path in paths:
        marker = path.parent / "complete.json"
        if not marker.exists():
            warnings.warn(f"Skipping incomplete task directory: {path.parent}")
            continue
        frame = pd.read_csv(path)
        frame["source_task"] = path.parent.name
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"All task directories containing {filename} were incomplete")
    return pd.concat(frames, ignore_index=True)


def mean_sd_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    """Return tidy mean/SD/N summaries without requiring a separate compiler."""
    grouped = frame.groupby(group_columns, dropna=False)[value_columns].agg(["mean", "std", "count"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    return grouped.reset_index()


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = LIGHT_BLUE,
    edgecolor: str = BLUE,
    fontsize: float = 10,
    weight: str = "normal",
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.5,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        weight=weight,
        wrap=True,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    style: str = "-|>",
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=1.5,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


METRIC_LABELS = {
    "r2": r"$R^2$",
    "mape_pct": "MAPE (%)",
    "rmse": "Overall RMSE (USD)",
    "rmse_q3": "RMSE >= training Q3 (USD)",
    "rmse_q3_iqr2": "RMSE >= training Q3 + IQR/2 (USD)",
}


def _ordered_metric_summary(
    frame: pd.DataFrame,
    group_column: str,
    metric: str,
    order: list[str],
) -> pd.DataFrame:
    summary = frame.groupby(group_column)[metric].agg(["mean", "std", "count"]).reindex(order)
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def build_performance_figure(metrics: pd.DataFrame, feature_set: str) -> None:
    focus = metrics[
        metrics["feature_set"].eq(feature_set)
        & (
            metrics["model_family"].eq("single")
            | metrics["model_family"].eq("two_stage_tl_nested_selected")
        )
    ].copy()
    focus["plot_label"] = np.where(
        focus["model_family"].eq("two_stage_tl_nested_selected"),
        "Two-stage TL\n(training-selected)",
        focus["model"],
    )
    preferred = [
        "CatBoost",
        "XGBoost",
        "Monotonic GBM",
        "ANN",
        "Two-stage TL\n(training-selected)",
    ]
    label_order = [label for label in preferred if label in set(focus["plot_label"])]
    if not label_order:
        raise RuntimeError(f"No performance rows are available for feature set {feature_set}")
    color_map = {
        "CatBoost": "#4C78A8",
        "XGBoost": "#72B7B2",
        "Monotonic GBM": "#F58518",
        "ANN": "#B279A2",
        "Two-stage TL\n(training-selected)": "#2F855A",
    }
    metrics_to_plot = ["r2", "mape_pct", "rmse", "rmse_q3", "rmse_q3_iqr2"]
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 6.7))
    axes = axes.ravel()
    for index, metric in enumerate(metrics_to_plot):
        axis = axes[index]
        summary = _ordered_metric_summary(focus, "plot_label", metric, label_order)
        x = np.arange(len(summary))
        axis.bar(
            x,
            summary["mean"].to_numpy(),
            yerr=summary["std"].to_numpy(),
            capsize=3,
            color=[color_map[label] for label in label_order],
            edgecolor="white",
            linewidth=0.6,
        )
        axis.set_title(f"({chr(97 + index)}) {METRIC_LABELS[metric]}")
        axis.set_xticks(x, label_order, rotation=24, ha="right")
        if metric == "r2":
            lower = min(0.0, float((summary["mean"] - summary["std"]).min() - 0.05))
            axis.set_ylim(lower, 1.0)
        axis.grid(axis="y", alpha=0.28)
    axes[-1].axis("off")
    n_seeds = int(focus["seed"].nunique())
    seed_word = "seed" if n_seeds == 1 else "seeds"
    figure.suptitle(
        f"Leakage-controlled outer-test performance across {n_seeds} {seed_word}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    figure.tight_layout()
    save_jpg(figure, "figure5_corrected_model_performance.jpg")


def build_svm_figure(svm: pd.DataFrame, feature_set: str) -> None:
    focus = svm[svm["feature_set"].eq(feature_set)].copy()
    cutoff_order = [cutoff for cutoff in ["Mean", "Q3", "Q3+IQR/2"] if cutoff in set(focus["cutoff_rule"])]
    if not cutoff_order:
        raise RuntimeError(f"No SVM rows are available for feature set {feature_set}")
    score_columns = ["classifier_precision", "classifier_recall", "classifier_f1"]
    labels = ["Precision", "Recall", "F1"]
    colors = ["#4C78A8", "#F58518", "#2F855A"]
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(cutoff_order))
    width = 0.24
    for offset, (score, label, color) in enumerate(zip(score_columns, labels, colors)):
        summary = _ordered_metric_summary(focus, "cutoff_rule", score, cutoff_order)
        axis.bar(
            x + (offset - 1) * width,
            summary["mean"].to_numpy(),
            width,
            yerr=summary["std"].to_numpy(),
            capsize=3,
            label=label,
            color=color,
            edgecolor="white",
        )
    axis.set_xticks(x, cutoff_order)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Outer-test classification score")
    axis.set_title("SVM classification under training-derived cutoffs")
    axis.legend(frameon=False, ncol=3, loc="upper right")
    axis.grid(axis="y", alpha=0.28)
    figure.tight_layout()
    save_jpg(figure, "figure8_training_derived_svm_performance.jpg")


def build_ablation_figure(metrics: pd.DataFrame) -> None:
    focus = metrics[metrics["model_family"].eq("two_stage_tl_nested_selected")].copy()
    preferred = ["A_all_21", "B_no_component_costs", "C_physical_quantity"]
    feature_order = [feature for feature in preferred if feature in set(focus["feature_set"])]
    if not feature_order:
        raise RuntimeError("No training-selected two-stage rows are available for ablation")
    label_map = {
        "A_all_21": "A: all 21 inputs",
        "B_no_component_costs": "B: exclude 4 component costs",
        "C_physical_quantity": "C: physical/categorical/quantity",
    }
    color_map = {
        "A_all_21": "#2F855A",
        "B_no_component_costs": "#D97706",
        "C_physical_quantity": "#9C4221",
    }
    metrics_to_plot = ["r2", "rmse", "rmse_q3", "rmse_q3_iqr2"]
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))
    for index, (axis, metric) in enumerate(zip(axes.ravel(), metrics_to_plot)):
        summary = _ordered_metric_summary(focus, "feature_set", metric, feature_order)
        x = np.arange(len(summary))
        axis.bar(
            x,
            summary["mean"].to_numpy(),
            yerr=summary["std"].to_numpy(),
            capsize=4,
            color=[color_map[feature] for feature in feature_order],
            edgecolor="white",
        )
        axis.set_xticks(
            x,
            [label_map[feature] for feature in feature_order],
            rotation=18,
            ha="right",
        )
        axis.set_title(f"({chr(97 + index)}) {METRIC_LABELS[metric]}")
        axis.grid(axis="y", alpha=0.28)
        if metric == "r2":
            lower = min(0.0, float((summary["mean"] - summary["std"]).min() - 0.05))
            axis.set_ylim(lower, 1.0)
    figure.suptitle(
        "Component-cost ablation for the training-selected two-stage procedure",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    figure.tight_layout()
    save_jpg(figure, "figure_new_component_cost_ablation.jpg")


def build_selection_figure(selected: pd.DataFrame) -> None:
    preferred = ["A_all_21", "B_no_component_costs", "C_physical_quantity"]
    feature_order = [feature for feature in preferred if feature in set(selected["feature_set"])]
    if not feature_order:
        raise RuntimeError("No training-only selected configurations are available")
    title_map = {
        "A_all_21": "A: all inputs",
        "B_no_component_costs": "B: no component costs",
        "C_physical_quantity": "C: physical/quantity",
    }
    cutoff_order = ["Mean", "Q3", "Q3+IQR/2"]
    strategy_order = ["Conservative", "Balanced", "Aggressive"]
    figure, axes = plt.subplots(1, len(feature_order), figsize=(3.55 * len(feature_order), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    max_count = max(1, int(selected.groupby("feature_set")["seed"].nunique().max()))
    image_artist = None
    for axis, feature_set in zip(axes, feature_order):
        subset = selected[selected["feature_set"].eq(feature_set)]
        counts = (
            subset.groupby(["cutoff_rule", "transfer_strategy"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=cutoff_order, columns=strategy_order, fill_value=0)
        )
        image_artist = axis.imshow(counts.to_numpy(), cmap="YlGnBu", vmin=0, vmax=max_count, aspect="auto")
        for row in range(counts.shape[0]):
            for column in range(counts.shape[1]):
                value = int(counts.iat[row, column])
                color = "white" if value > max_count * 0.55 else "#1F2937"
                axis.text(column, row, str(value), ha="center", va="center", color=color, fontsize=10)
        axis.set_title(title_map[feature_set])
        axis.set_xticks(range(len(strategy_order)), strategy_order, rotation=25, ha="right")
        axis.set_yticks(range(len(cutoff_order)), cutoff_order)
        axis.set_xlabel("Transfer strategy")
        axis.set_ylabel("Cutoff" if axis is axes[0] else "")
        axis.set_xticks(np.arange(-0.5, len(strategy_order), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(cutoff_order), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.2)
        axis.tick_params(which="minor", bottom=False, left=False)
    n_seeds = int(selected["seed"].nunique())
    seed_word = "seed" if n_seeds == 1 else "seeds"
    figure.suptitle(
        f"Training-only cutoff/strategy selections across {n_seeds} outer {seed_word}",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    figure.subplots_adjust(left=0.08, right=0.88, bottom=0.25, top=0.82, wspace=0.18)
    if image_artist is not None:
        color_axis = figure.add_axes([0.91, 0.25, 0.018, 0.57])
        figure.colorbar(image_artist, cax=color_axis, label="Selections")
    save_jpg(figure, "figure_new_training_selection_frequency.jpg")


def build_figure1() -> None:
    fig, ax = plt.subplots(figsize=(15.2, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle(
        "Leakage-controlled nested evaluation and component-cost ablation",
        fontsize=20,
        weight="bold",
        y=0.98,
    )

    box(ax, (0.035, 0.72), 0.12, 0.13, "Raw data\n419 projects", weight="bold")
    box(
        ax,
        (0.195, 0.69),
        0.15,
        0.19,
        "Outer split first\n70% train / 30% test\nSeeds 42-51",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
        weight="bold",
    )
    arrow(ax, (0.155, 0.785), (0.195, 0.785))

    box(
        ax,
        (0.39, 0.79),
        0.18,
        0.12,
        "Three feature sets\nA: all 21 inputs\nB: exclude N10/N11/N14/N15\nC: physical/quantity",
        fontsize=9.2,
    )
    box(
        ax,
        (0.39, 0.59),
        0.18,
        0.14,
        "Training-derived cutoffs\nMean, Q3, Q3 + IQR/2\nfrom y_train only",
        facecolor=LIGHT_GREEN,
        edgecolor=GREEN,
        weight="bold",
    )
    arrow(ax, (0.345, 0.785), (0.39, 0.85))
    arrow(ax, (0.345, 0.75), (0.39, 0.66))

    box(
        ax,
        (0.615, 0.76),
        0.17,
        0.15,
        "Inner 3-fold CV\nencoding and scaling inside folds\nmodel and SVM tuning",
        fontsize=9.3,
    )
    box(
        ax,
        (0.615, 0.56),
        0.17,
        0.14,
        "Training-only selection\ncutoff x transfer strategy\nprimary: RMSE >= fold Q3",
        facecolor=LIGHT_GREEN,
        edgecolor=GREEN,
        fontsize=9.3,
        weight="bold",
    )
    arrow(ax, (0.57, 0.85), (0.615, 0.84))
    arrow(ax, (0.57, 0.66), (0.615, 0.63))
    arrow(ax, (0.70, 0.76), (0.70, 0.70))

    box(
        ax,
        (0.83, 0.66),
        0.135,
        0.17,
        "Refit selected model\non all outer training data\nlock preprocessing and cutoff",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
        fontsize=9.2,
        weight="bold",
    )
    arrow(ax, (0.785, 0.63), (0.83, 0.72))

    box(
        ax,
        (0.39, 0.25),
        0.18,
        0.14,
        "Untouched outer test\n126 projects per seed\nno fitting or selection",
        facecolor="#FDE2E2",
        edgecolor=RED,
        weight="bold",
    )
    arrow(ax, (0.27, 0.69), (0.39, 0.32), color=RED, connectionstyle="arc3,rad=0.08")

    box(
        ax,
        (0.64, 0.24),
        0.17,
        0.16,
        "Evaluate once\nR2, MAPE, overall RMSE\nRMSE >= training Q3\nRMSE >= training Q3 + IQR/2",
        facecolor=LIGHT_GRAY,
        edgecolor=GRAY,
        fontsize=9.2,
        weight="bold",
    )
    arrow(ax, (0.57, 0.32), (0.64, 0.32), color=RED)
    arrow(ax, (0.897, 0.66), (0.81, 0.38), connectionstyle="arc3,rad=-0.10")

    box(
        ax,
        (0.845, 0.24),
        0.12,
        0.16,
        "Seed-blocked inference\npaired bootstrap CI\nHolm adjustment\nrepeated measures",
        facecolor=LIGHT_GRAY,
        edgecolor=GRAY,
        fontsize=9.0,
    )
    arrow(ax, (0.81, 0.32), (0.845, 0.32))

    ax.text(
        0.49,
        0.50,
        "TRAINING PARTITION ONLY",
        ha="center",
        va="center",
        fontsize=11,
        color=GREEN,
        weight="bold",
    )
    ax.plot([0.365, 0.98], [0.47, 0.47], color=GREEN, linewidth=1.2, linestyle="--")
    ax.text(
        0.49,
        0.455,
        "Outer test remains sealed until the final evaluation",
        ha="center",
        va="top",
        fontsize=10,
        color=RED,
        weight="bold",
    )
    save_jpg(fig, "figure1_leakage_controlled_workflow.jpg")


def build_residual_figure(pred: pd.DataFrame, feature_set: str, seed_value: int) -> None:
    seed = pred[(pred.feature_set == feature_set) & (pred.seed == seed_value)].copy()
    ann = seed[(seed.model_family == "single") & (seed.model == "ANN")].copy()
    tl = seed[seed.model_family == "two_stage_tl_nested_selected"].copy()
    if ann.empty or tl.empty:
        raise RuntimeError(
            f"Seed {seed_value} does not contain both the single ANN and selected two-stage predictions"
        )
    panels = [(ann, "Single ANN"), (tl, "Training-selected two-stage TL")]
    all_residuals = pd.concat([ann.residual, tl.residual]).to_numpy() / 1000
    limit = max(50.0, float(np.nanpercentile(np.abs(all_residuals), 99)) * 1.15)
    bins = np.linspace(-limit, limit, 30)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True, sharey=True)
    for ax, (data, title) in zip(axes, panels):
        high = data.at_or_above_train_q3.astype(str).str.lower().eq("true")
        ax.hist(
            data.loc[~high, "residual"] / 1000,
            bins=bins,
            alpha=0.72,
            color=BLUE,
            label="Below training Q3",
        )
        ax.hist(
            data.loc[high, "residual"] / 1000,
            bins=bins,
            alpha=0.72,
            color=ORANGE,
            label="At/above training Q3",
        )
        ax.axvline(0, color="#111827", linewidth=1.3, linestyle="--")
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xlabel("Residual = predicted - observed (thousand USD)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Outer-test projects")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle(
        f"Residual distributions on the prespecified seed-{seed_value} outer test",
        fontsize=17,
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.01,
        "Descriptive visualization only; confirmatory comparisons use all 10 seed-level metrics.",
        ha="center",
        fontsize=10,
        color=GRAY,
    )
    fig.tight_layout()
    save_jpg(fig, f"figure6_corrected_residuals_seed{seed_value}.jpg")


def build_prediction_figure(pred: pd.DataFrame, feature_set: str, seed_value: int) -> None:
    seed = pred[(pred.feature_set == feature_set) & (pred.seed == seed_value)].copy()
    ann = seed[(seed.model_family == "single") & (seed.model == "ANN")].copy()
    tl = seed[seed.model_family == "two_stage_tl_nested_selected"].copy()
    if ann.empty or tl.empty:
        raise RuntimeError(
            f"Seed {seed_value} does not contain both the single ANN and selected two-stage predictions"
        )
    panels = [(ann, "Single ANN"), (tl, "Training-selected two-stage TL")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharex=True, sharey=True)
    for ax, (data, title) in zip(axes, panels):
        data = data.sort_values("y_true").reset_index(drop=True)
        x = np.arange(1, len(data) + 1)
        ax.plot(x, data.y_true / 1000, color="#111827", linewidth=1.8, label="Observed")
        ax.plot(x, data.y_pred / 1000, color=GREEN, linewidth=1.4, label="Predicted")
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xlabel("Outer-test projects ranked by observed cost")
        ax.grid(alpha=0.24)
    axes[0].set_ylabel("Direct construction cost (thousand USD)")
    axes[1].legend(frameon=False, loc="upper left")
    config = f"{tl.cutoff_rule.iloc[0]} cutoff / {tl.transfer_strategy.iloc[0]} transfer"
    fig.suptitle(
        f"Observed and predicted costs on the prespecified seed-{seed_value} outer test\nSelected configuration: {config}",
        fontsize=16.5,
        weight="bold",
        y=1.04,
    )
    fig.text(
        0.5,
        -0.01,
        "Descriptive visualization only; the model was selected using outer-training data only.",
        ha="center",
        fontsize=10,
        color=GRAY,
    )
    fig.tight_layout()
    save_jpg(fig, f"figure7_corrected_predictions_seed{seed_value}.jpg")


def build_class_count_figure(thresholds: pd.DataFrame, feature_set: str) -> None:
    data = thresholds[thresholds.feature_set == feature_set].copy()
    if data.empty:
        raise RuntimeError(f"No threshold rows are available for feature set {feature_set}")
    cutoff_order = ["Mean", "Q3", "Q3+IQR/2"]
    train_n = int(round((data["train_low_count"] + data["train_high_count"]).median()))
    test_n = int(round((data["test_true_low_count"] + data["test_true_high_count"]).median()))
    specs = [
        (f"Outer training partition (n={train_n})", "train_low_count", "train_high_count"),
        (f"Outer test partition (n={test_n})", "test_true_low_count", "test_true_high_count"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), sharey=False)
    x = np.arange(len(cutoff_order))
    width = 0.34
    for ax, (title, low_col, high_col) in zip(axes, specs):
        grouped = data.groupby("cutoff_rule")[[low_col, high_col]].agg(["mean", "std"])
        low_mean = np.array([grouped.loc[c, (low_col, "mean")] for c in cutoff_order])
        low_sd = np.nan_to_num(np.array([grouped.loc[c, (low_col, "std")] for c in cutoff_order]))
        high_mean = np.array([grouped.loc[c, (high_col, "mean")] for c in cutoff_order])
        high_sd = np.nan_to_num(np.array([grouped.loc[c, (high_col, "std")] for c in cutoff_order]))
        b1 = ax.bar(x - width / 2, low_mean, width, yerr=low_sd, capsize=4, color="#77BBDD", label="Low cost")
        b2 = ax.bar(x + width / 2, high_mean, width, yerr=high_sd, capsize=4, color="#EE8866", label="High cost")
        for bars, sds in ((b1, low_sd), (b2, high_sd)):
            for bar, sd in zip(bars, sds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + sd + max(2.5, bar.get_height() * 0.015),
                    f"{bar.get_height():.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xticks(x, cutoff_order)
        n_seeds = int(data["seed"].nunique())
        seed_word = "seed" if n_seeds == 1 else "seeds"
        ax.set_ylabel(f"Projects, mean +/- SD across {n_seeds} {seed_word}")
        ax.grid(axis="y", alpha=0.25)
        ymax = max(np.max(low_mean + low_sd), np.max(high_mean + high_sd))
        ax.set_ylim(0, ymax * 1.18)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(
        "Split-specific class counts under training-derived cutoffs",
        fontsize=18,
        weight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_jpg(fig, "figure9_split_specific_class_counts.jpg")


def build_figure12() -> None:
    fig, ax = plt.subplots(figsize=(14.4, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle(
        "Evidence-gated implementation workflow",
        fontsize=20,
        weight="bold",
        y=0.98,
    )
    box(ax, (0.04, 0.69), 0.16, 0.16, "Define decision point\nand intended users", weight="bold")
    box(
        ax,
        (0.25, 0.66),
        0.19,
        0.22,
        "Verify predictor availability\nN10/N11/N14/N15 available?\nPreliminary or realized costs?",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
        weight="bold",
    )
    arrow(ax, (0.20, 0.77), (0.25, 0.77))

    box(
        ax,
        (0.51, 0.76),
        0.18,
        0.15,
        "If documented and available\napply full-input model",
        facecolor=LIGHT_GREEN,
        edgecolor=GREEN,
        weight="bold",
    )
    box(
        ax,
        (0.51, 0.51),
        0.18,
        0.16,
        "If unavailable\nuse physical/quantity model\nand disclose lower accuracy",
        facecolor="#FDE2E2",
        edgecolor=RED,
        weight="bold",
    )
    arrow(ax, (0.44, 0.80), (0.51, 0.83), color=GREEN)
    arrow(ax, (0.44, 0.71), (0.51, 0.59), color=RED)
    ax.text(0.475, 0.84, "YES", fontsize=9, color=GREEN, weight="bold")
    ax.text(0.475, 0.62, "NO", fontsize=9, color=RED, weight="bold")

    box(
        ax,
        (0.76, 0.64),
        0.19,
        0.21,
        "Apply frozen pipeline\npreprocessing + classifier + regressors\nreport model scope and uncertainty",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
        weight="bold",
    )
    arrow(ax, (0.69, 0.83), (0.76, 0.78))
    arrow(ax, (0.69, 0.59), (0.76, 0.70))

    box(
        ax,
        (0.18, 0.23),
        0.20,
        0.16,
        "Monitor and recalibrate\nnew region/building type\ndata drift and error audit",
        facecolor=LIGHT_GRAY,
        edgecolor=GRAY,
        weight="bold",
    )
    box(
        ax,
        (0.47, 0.23),
        0.20,
        0.16,
        "Human review\ncheck scope, unusual projects,\nand component-cost provenance",
        facecolor=LIGHT_GRAY,
        edgecolor=GRAY,
        weight="bold",
    )
    box(
        ax,
        (0.76, 0.23),
        0.19,
        0.16,
        "Decision support output\ncost estimate + high-cost flag\nmodel/version recorded",
        facecolor=LIGHT_GRAY,
        edgecolor=GRAY,
        weight="bold",
    )
    arrow(ax, (0.855, 0.64), (0.855, 0.39))
    arrow(ax, (0.76, 0.31), (0.67, 0.31))
    arrow(ax, (0.47, 0.31), (0.38, 0.31))
    arrow(ax, (0.28, 0.39), (0.34, 0.66), color=ORANGE)
    ax.text(0.285, 0.51, "recalibrate", fontsize=9, color=ORANGE, weight="bold", rotation=77)
    ax.text(
        0.58,
        0.08,
        "Do not claim early-stage use until the timing and provenance of component-cost inputs are documented.",
        ha="center",
        va="center",
        fontsize=11,
        color=RED,
        weight="bold",
    )
    save_jpg(fig, "figure12_evidence_gated_implementation_workflow.jpg")


def write_compact_summaries(
    metrics: pd.DataFrame,
    svm: pd.DataFrame,
    thresholds: pd.DataFrame,
    selected: pd.DataFrame,
    feature_set: str,
) -> None:
    metric_columns = ["r2", "mape_pct", "rmse", "rmse_q3", "rmse_q3_iqr2"]
    performance = metrics[
        metrics["feature_set"].eq(feature_set)
        & (
            metrics["model_family"].eq("single")
            | metrics["model_family"].eq("two_stage_tl_nested_selected")
        )
    ]
    mean_sd_summary(
        performance,
        ["feature_set", "model_family", "model"],
        metric_columns,
    ).to_csv(OUT / "figure5_model_performance_summary.csv", index=False)

    ablation = metrics[metrics["model_family"].eq("two_stage_tl_nested_selected")]
    mean_sd_summary(ablation, ["feature_set"], metric_columns).to_csv(
        OUT / "figure11_component_cost_ablation_summary.csv", index=False
    )

    svm_columns = ["classifier_precision", "classifier_recall", "classifier_f1"]
    mean_sd_summary(
        svm[svm["feature_set"].eq(feature_set)],
        ["feature_set", "cutoff_rule"],
        svm_columns,
    ).to_csv(OUT / "figure8_svm_summary.csv", index=False)

    threshold_columns = [
        "training_derived_cutoff_value",
        "train_low_count",
        "train_high_count",
        "test_true_low_count",
        "test_true_high_count",
        "test_n_at_or_above_train_q3",
        "test_n_at_or_above_train_q3_iqr2",
    ]
    mean_sd_summary(
        thresholds[thresholds["feature_set"].eq(feature_set)],
        ["feature_set", "cutoff_rule"],
        threshold_columns,
    ).to_csv(OUT / "figure9_split_count_summary.csv", index=False)

    (
        selected.groupby(["feature_set", "cutoff_rule", "transfer_strategy"])
        .size()
        .rename("selection_count")
        .reset_index()
        .to_csv(OUT / "figure10_selection_frequency.csv", index=False)
    )


def write_manifest(seed_value: int, n_seeds: int) -> None:
    seed_word = "seed" if n_seeds == 1 else "seeds"
    rows = [
        ("figure1_leakage_controlled_workflow.jpg", "Replace Figure 1", "Leakage-controlled nested evaluation and component-cost ablation."),
        ("figure5_corrected_model_performance.jpg", "Replace Figure 5", f"Leakage-controlled outer-test performance across {n_seeds} {seed_word}."),
        (f"figure6_corrected_residuals_seed{seed_value}.jpg", "Optional replacement for Figure 6", f"Residuals on the prespecified seed-{seed_value} outer test; descriptive only."),
        (f"figure7_corrected_predictions_seed{seed_value}.jpg", "Optional replacement for Figure 7", f"Observed and predicted costs on the prespecified seed-{seed_value} outer test; descriptive only."),
        ("figure8_training_derived_svm_performance.jpg", "Replace Figure 8", "SVM classification under training-derived cutoffs."),
        ("figure9_split_specific_class_counts.jpg", "Replace Figure 9", "Split-specific class counts under training-derived cutoffs."),
        ("figure12_evidence_gated_implementation_workflow.jpg", "Replace Figure 12", "Evidence-gated implementation workflow."),
        ("figure_new_component_cost_ablation.jpg", "Use as new Figure 11 after corrected Table 4", "Component-cost ablation for the training-selected procedure."),
        ("figure_new_training_selection_frequency.jpg", "Use as new Figure 10 near configuration selection results", "Training-only cutoff and transfer-strategy selections."),
    ]
    pd.DataFrame(rows, columns=["file", "manuscript_action", "recommended_caption"]).to_csv(
        OUT / "figure_manifest.csv", index=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create manuscript JPGs directly from portable corrected-analysis task outputs"
    )
    parser.add_argument("--results-dir", type=Path, default=Path("corrected_analysis_output"))
    parser.add_argument("--output-dir", type=Path, default=Path("corrected_analysis_figures"))
    parser.add_argument("--feature-set", default="A_all_21")
    parser.add_argument("--descriptive-seed", type=int, default=42)
    args = parser.parse_args()

    global OUT
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)

    metrics = read_task_table(args.results_dir, "model_metrics.csv")
    pred = read_task_table(args.results_dir, "test_predictions.csv")
    svm = read_task_table(args.results_dir, "svm_results.csv")
    thresholds = read_task_table(args.results_dir, "training_thresholds.csv")
    selected = read_task_table(args.results_dir, "selected_configuration.csv")

    numeric_columns = {
        "metrics": (metrics, ["seed", "r2", "mape_pct", "rmse", "rmse_q3", "rmse_q3_iqr2"]),
        "predictions": (pred, ["seed", "y_true", "y_pred", "residual"]),
        "svm": (svm, ["seed", "classifier_precision", "classifier_recall", "classifier_f1"]),
        "thresholds": (
            thresholds,
            [
                "seed",
                "training_derived_cutoff_value",
                "train_low_count",
                "train_high_count",
                "test_true_low_count",
                "test_true_high_count",
                "test_n_at_or_above_train_q3",
                "test_n_at_or_above_train_q3_iqr2",
            ],
        ),
        "selected": (selected, ["seed"]),
    }
    for table_name, (frame, columns) in numeric_columns.items():
        for column in columns:
            if column not in frame.columns:
                raise KeyError(f"{table_name} output is missing required column {column!r}")
            frame[column] = pd.to_numeric(frame[column], errors="raise")

    ann_seeds = set(
        pred[
            pred["feature_set"].eq(args.feature_set)
            & pred["model_family"].eq("single")
            & pred["model"].eq("ANN")
        ]["seed"].astype(int)
    )
    tl_seeds = set(
        pred[
            pred["feature_set"].eq(args.feature_set)
            & pred["model_family"].eq("two_stage_tl_nested_selected")
        ]["seed"].astype(int)
    )
    available_descriptive_seeds = sorted(ann_seeds & tl_seeds)
    if not available_descriptive_seeds:
        raise RuntimeError(
            f"No seed has both ANN and selected two-stage predictions for {args.feature_set}"
        )
    seed_value = args.descriptive_seed
    if seed_value not in available_descriptive_seeds:
        seed_value = available_descriptive_seeds[0]
        warnings.warn(
            f"Requested seed {args.descriptive_seed} is unavailable; using seed {seed_value}"
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    build_figure1()
    build_performance_figure(metrics, args.feature_set)
    build_residual_figure(pred, args.feature_set, seed_value)
    build_prediction_figure(pred, args.feature_set, seed_value)
    build_svm_figure(svm, args.feature_set)
    build_class_count_figure(thresholds, args.feature_set)
    build_figure12()
    build_ablation_figure(metrics)
    build_selection_figure(selected)
    write_compact_summaries(metrics, svm, thresholds, selected, args.feature_set)
    write_manifest(seed_value, int(metrics["seed"].nunique()))
    print(f"Created portable figure package in: {OUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
