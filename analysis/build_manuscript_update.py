from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_LABELS = {
    "r2": "R²",
    "mape_pct": "MAPE (%)",
    "rmse": "overall RMSE (USD)",
    "rmse_q3": "RMSE at or above training-derived Q3 (USD)",
    "rmse_q3_iqr2": "RMSE at or above training-derived Q3 + IQR/2 (USD)",
}


def format_number(metric: str, value: float) -> str:
    if metric == "r2":
        return f"{value:.4f}"
    if metric == "mape_pct":
        return f"{value:.2f}"
    return f"{value:,.0f}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    subset = frame[columns].copy()
    for column in columns:
        subset[column] = subset[column].map(
            lambda value: "" if pd.isna(value) else str(value).replace("|", r"\|")
        )
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in subset.to_numpy(dtype=str)]
    return "\n".join([header, divider, *body])


def direction_sentence(metric: str, primary_mean: float, baseline_mean: float) -> str:
    if metric == "r2":
        difference = primary_mean - baseline_mean
        direction = "higher" if difference > 0 else "lower"
        return (
            f"{METRIC_LABELS[metric]} was {format_number(metric, primary_mean)} versus "
            f"{format_number(metric, baseline_mean)} for the single ANN ({direction} by "
            f"{abs(difference):.4f})."
        )
    percent = (baseline_mean - primary_mean) / baseline_mean * 100.0
    verb = "reduced" if percent > 0 else "increased"
    return (
        f"{METRIC_LABELS[metric]} was {format_number(metric, primary_mean)} versus "
        f"{format_number(metric, baseline_mean)} for the single ANN ({verb} by "
        f"{abs(percent):.1f}%)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build paste-ready manuscript updates from corrected outputs"
    )
    parser.add_argument(
        "--compiled-dir",
        type=Path,
        default=Path("results")
        / "corrected_analysis_20260717"
        / "compiled",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_notes")
        / "corrected_analysis_manuscript_updates.md",
    )
    args = parser.parse_args()

    table4 = pd.read_csv(args.compiled_dir / "table4_corrected_summary.csv")
    table3 = pd.read_csv(args.compiled_dir / "table3_split_specific.csv")
    primary = pd.read_csv(args.compiled_dir / "primary_model_comparisons.csv")
    ablation = pd.read_csv(args.compiled_dir / "ablation_summary.csv")
    ablation_contrasts = pd.read_csv(
        args.compiled_dir / "ablation_paired_contrasts.csv"
    )
    anova = pd.read_csv(args.compiled_dir / "repeated_measures_anova.csv")
    planned = pd.read_csv(args.compiled_dir / "planned_transfer_contrasts.csv")
    selection = pd.read_csv(args.compiled_dir / "selection_frequency.csv")

    primary_ann = primary[
        (primary["feature_set"] == "A_all_21") & (primary["model_b"] == "ANN")
    ].set_index("metric")
    performance_sentences = []
    for metric in ["r2", "mape_pct", "rmse", "rmse_q3", "rmse_q3_iqr2"]:
        row = primary_ann.loc[metric]
        performance_sentences.append(
            direction_sentence(metric, row["model_a_mean"], row["model_b_mean"])
        )

    table4_full = table4[table4["feature_set"] == "A_all_21"].copy()
    table4_view = table4_full[
        [
            "analysis_label",
            "r2_mean_sd",
            "mape_pct_mean_sd",
            "rmse_mean_sd",
            "rmse_q3_mean_sd",
            "rmse_q3_iqr2_mean_sd",
        ]
    ].rename(
        columns={
            "analysis_label": "Model/configuration",
            "r2_mean_sd": "R²",
            "mape_pct_mean_sd": "MAPE (%)",
            "rmse_mean_sd": "RMSE",
            "rmse_q3_mean_sd": "RMSE ≥ train Q3",
            "rmse_q3_iqr2_mean_sd": "RMSE ≥ train Q3+IQR/2",
        }
    )

    table3_view = table3.copy()
    for column in ["mean", "sd", "minimum", "maximum"]:
        table3_view[column] = table3_view[column].map(lambda value: f"{value:,.2f}")

    ablation_primary = ablation[
        ablation["analysis_label"] == "Two-stage TL | training-selected"
    ].copy()
    ablation_view = ablation_primary[
        [
            "feature_set",
            "r2_mean",
            "mape_pct_mean",
            "rmse_mean",
            "rmse_q3_mean",
            "rmse_q3_iqr2_mean",
        ]
    ].copy()
    ablation_view.columns = [
        "Feature set",
        "R²",
        "MAPE (%)",
        "RMSE",
        "RMSE ≥ train Q3",
        "RMSE ≥ train Q3+IQR/2",
    ]
    for column in ablation_view.columns[1:]:
        if column == "R²":
            ablation_view[column] = ablation_view[column].map(lambda value: f"{value:.4f}")
        elif column == "MAPE (%)":
            ablation_view[column] = ablation_view[column].map(lambda value: f"{value:.2f}")
        else:
            ablation_view[column] = ablation_view[column].map(
                lambda value: f"{value:,.0f}"
            )

    significant_ablation = ablation_contrasts[
        ablation_contrasts["reject_holm_0_05"] == True
    ]
    full_anova = anova[anova["feature_set"] == "A_all_21"].copy()
    interaction = full_anova[full_anova["effect"].astype(str).str.contains(":")]
    significant_planned = planned[
        (planned["feature_set"] == "A_all_21")
        & (planned["reject_holm_0_05"] == True)
    ].copy()

    best_overall = table4_full.sort_values("rmse_mean").iloc[0]
    best_high = table4_full.sort_values("rmse_q3_mean").iloc[0]
    selection_view = selection.copy()
    selection_view["percent"] = selection_view["percent"].map(lambda value: f"{value:.1f}%")

    text = f"""# Corrected-analysis manuscript update package

## Status

These replacements supersede the original Table 3 thresholds used for model fitting, the
original Table 4 performance values, and the pooled Wilcoxon Tables 5–6. The corrected
pipeline uses seed as the independent block and keeps every outer test partition untouched
until final evaluation.

## Methods replacement: split, preprocessing, and model selection

For each of 10 prespecified random seeds (42–51), the 419 observations were randomly split
without target stratification into an outer training partition (n=293; 70%) and an outer
test partition (n=126; 30%). For each outer split, the Mean, Q3, and Q3 + IQR/2 thresholds
were computed from the outer training outcomes only and then applied unchanged to the
corresponding test partition. Categorical encoding and Min-Max scaling were fitted inside
the training/validation folds. Model and SVM hyperparameters were selected by three-fold
cross-validation within the outer training partition using mean absolute error and F1,
respectively. Neural-network target scaling and validation-loss early stopping used
training outcomes only. The Keras segment models used at most 200 epochs with patience 20;
the validation-best epoch count was retained and the model was refitted on all available
training observations in that segment. The primary two-stage estimate was defined by a
training-only selection procedure: among the three cutoff rules and three transfer
strategies, the configuration with the lowest validation RMSE at or above the
training-fold Q3 was selected, with overall RMSE and MAPE as tie-breakers. It was then
fitted to the complete outer training partition and evaluated once on the untouched outer
test partition. The nine fixed configurations were retained as exploratory comparisons.

## Corrected Table 3: split-specific thresholds and counts

The original full-dataset thresholds may be retained only as descriptive values labelled
“not used for model fitting.” Use the following split-specific evidence for the fitted
models.

{markdown_table(table3_view, ['cutoff_rule', 'variable', 'n_splits', 'mean', 'sd', 'minimum', 'maximum'])}

## Corrected Table 4

{markdown_table(table4_view, list(table4_view.columns))}

## Results replacement: primary comparison

The primary estimate is the performance of the training-selected two-stage procedure, not
the best fixed configuration identified after inspecting test results. Across the 10 outer
test splits, {' '.join(performance_sentences)} Paired seed-level effect estimates,
bootstrap 95% confidence intervals, and Holm-adjusted tests are reported in
`primary_model_comparisons.csv`.

Descriptively, the lowest mean overall RMSE in the corrected full-feature analysis was
observed for **{best_overall['analysis_label']}** ({best_overall['rmse_mean']:,.0f} USD),
whereas the lowest mean Q3-segment RMSE was observed for
**{best_high['analysis_label']}** ({best_high['rmse_q3_mean']:,.0f} USD). These descriptive
leaders must not be relabelled as independently selected “best models” unless they coincide
with the frozen training-only selection rule.

## Training-only selection frequency

{markdown_table(selection_view, ['feature_set', 'cutoff_rule', 'transfer_strategy', 'count', 'percent'])}

## Component-cost ablation

{markdown_table(ablation_view, list(ablation_view.columns))}

There were {len(significant_ablation)} Holm-significant paired feature-set contrasts across
the reported model/metric families. Exact effect estimates and bootstrap intervals are in
`ablation_paired_contrasts.csv`. The workbook does not reveal whether N10, N11, N14, and
N15 are preliminary estimates available at the claimed decision point or realized
components of final direct construction cost. Add author-supplied provenance and timing.
If early availability cannot be documented, replace “early-stage cost estimation” with
“conditional retrofit cost estimation using partial component-cost information.”

## Repeated-measures replacement for Tables 5–6

The 3-cutoff × 3-strategy analysis used seed as the repeated-measures subject. The
full-feature interaction results are:

{markdown_table(interaction, ['metric', 'effect', 'f_value', 'num_df', 'den_df', 'p_value', 'status'])}

After Holm adjustment within the planned contrast families, {len(significant_planned)}
full-feature paired contrasts remained significant. Do not report the original pooled
n=30 Wilcoxon p-values. The complete interaction and contrast tables are
`repeated_measures_anova.csv` and `planned_transfer_contrasts.csv`.

## Abstract and conclusion gate

Delete the original R²=0.9566 and 46%/25% improvement claims unless the corrected primary
rows reproduce them; use the primary comparison values above instead. Describe fixed
cutoff/strategy leaders as exploratory. State explicitly that the independent replication
unit is the seed (n=10) and that component-cost input timing remains a limitation.

## Availability statement template

Data and code availability: The deidentified input data or governed-access procedure,
complete preprocessing and modeling code, pinned package versions, seeds 42–51,
hyperparameter grids, split indices and hashes, training-derived thresholds, classifier
metrics, seed-level regression metrics, row-level test predictions, and ablation outputs
are archived at [permanent repository and DOI]. Do not insert this statement until the
archive and permanent identifier actually exist.
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
