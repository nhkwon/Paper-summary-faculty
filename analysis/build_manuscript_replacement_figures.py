from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
COMPILED = ROOT / "results" / "corrected_analysis_20260717" / "compiled"
SOURCE_FIGURES = ROOT / "results" / "corrected_analysis_20260717" / "figures"
OUT = ROOT / "output" / "jpg" / "manuscript_260716_replacements"

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


def convert_png(source_name: str, target_name: str) -> None:
    with Image.open(SOURCE_FIGURES / source_name) as im:
        rgb = im.convert("RGB")
        rgb.save(
            OUT / target_name,
            format="JPEG",
            quality=95,
            subsampling=0,
            dpi=(300, 300),
        )


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


def build_residual_figure(pred: pd.DataFrame) -> None:
    seed = pred[(pred.feature_set == "A_all_21") & (pred.seed == 42)].copy()
    ann = seed[(seed.model_family == "single") & (seed.model == "ANN")].copy()
    tl = seed[seed.model_family == "two_stage_tl_nested_selected"].copy()
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
        "Residual distributions on the prespecified seed-42 outer test",
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
    save_jpg(fig, "figure6_corrected_residuals_seed42.jpg")


def build_prediction_figure(pred: pd.DataFrame) -> None:
    seed = pred[(pred.feature_set == "A_all_21") & (pred.seed == 42)].copy()
    ann = seed[(seed.model_family == "single") & (seed.model == "ANN")].copy()
    tl = seed[seed.model_family == "two_stage_tl_nested_selected"].copy()
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
        f"Observed and predicted costs on the prespecified seed-42 outer test\nSelected configuration: {config}",
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
    save_jpg(fig, "figure7_corrected_predictions_seed42.jpg")


def build_class_count_figure(thresholds: pd.DataFrame) -> None:
    data = thresholds[thresholds.feature_set == "A_all_21"].copy()
    cutoff_order = ["Mean", "Q3", "Q3+IQR/2"]
    specs = [
        ("Outer training partition (n=293)", "train_low_count", "train_high_count"),
        ("Outer test partition (n=126)", "test_true_low_count", "test_true_high_count"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), sharey=False)
    x = np.arange(len(cutoff_order))
    width = 0.34
    for ax, (title, low_col, high_col) in zip(axes, specs):
        grouped = data.groupby("cutoff_rule")[[low_col, high_col]].agg(["mean", "std"])
        low_mean = np.array([grouped.loc[c, (low_col, "mean")] for c in cutoff_order])
        low_sd = np.array([grouped.loc[c, (low_col, "std")] for c in cutoff_order])
        high_mean = np.array([grouped.loc[c, (high_col, "mean")] for c in cutoff_order])
        high_sd = np.array([grouped.loc[c, (high_col, "std")] for c in cutoff_order])
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
        ax.set_ylabel("Projects, mean +/- SD across 10 seeds")
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


def write_manifest() -> None:
    rows = [
        ("figure1_leakage_controlled_workflow.jpg", "Replace Figure 1", "Leakage-controlled nested evaluation and component-cost ablation."),
        ("figure5_corrected_model_performance.jpg", "Replace Figure 5", "Leakage-controlled outer-test performance across 10 seeds."),
        ("figure6_corrected_residuals_seed42.jpg", "Optional replacement for Figure 6", "Residuals on the prespecified seed-42 outer test; descriptive only."),
        ("figure7_corrected_predictions_seed42.jpg", "Optional replacement for Figure 7", "Observed and predicted costs on the prespecified seed-42 outer test; descriptive only."),
        ("figure8_training_derived_svm_performance.jpg", "Replace Figure 8", "SVM classification under training-derived cutoffs."),
        ("figure9_split_specific_class_counts.jpg", "Replace Figure 9", "Split-specific class counts under training-derived cutoffs."),
        ("figure12_evidence_gated_implementation_workflow.jpg", "Replace Figure 12", "Evidence-gated implementation workflow."),
        ("figure_new_component_cost_ablation.jpg", "Use as new Figure 11 after corrected Table 4", "Component-cost ablation for the training-selected procedure."),
        ("figure_new_training_selection_frequency.jpg", "Use as new Figure 10 near configuration selection results", "Training-only cutoff and transfer-strategy selections."),
    ]
    pd.DataFrame(rows, columns=["file", "manuscript_action", "recommended_caption"]).to_csv(
        OUT / "figure_manifest.csv", index=False
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(COMPILED / "test_predictions.csv")
    thresholds = pd.read_csv(COMPILED / "training_thresholds_seed_level.csv")

    build_figure1()
    convert_png("figure_corrected_model_performance.png", "figure5_corrected_model_performance.jpg")
    build_residual_figure(pred)
    build_prediction_figure(pred)
    convert_png("figure_svm_cutoff_performance.png", "figure8_training_derived_svm_performance.jpg")
    build_class_count_figure(thresholds)
    build_figure12()
    convert_png("figure_component_cost_ablation.png", "figure_new_component_cost_ablation.jpg")
    convert_png("figure_training_selection_frequency.png", "figure_new_training_selection_frequency.jpg")
    write_manifest()
    print(OUT)


if __name__ == "__main__":
    main()
