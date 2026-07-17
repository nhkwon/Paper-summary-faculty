from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


METRICS = ["r2", "mape_pct", "rmse", "rmse_q3", "rmse_q3_iqr2"]
FEATURE_SETS = ["A_all_21", "B_no_component_costs", "C_physical_quantity"]
SEEDS = list(range(42, 52))


def atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_task_table(task_dirs: list[Path], filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for task_dir in task_dirs:
        path = task_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {filename} in {task_dir}")
        frame = pd.read_csv(path)
        frame["source_task"] = task_dir.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def model_label(row: pd.Series) -> str:
    family = row["model_family"]
    if family == "single":
        return str(row["model"])
    if family == "two_stage_no_tl":
        return f"Two-stage no TL | {row['cutoff_rule']}"
    if family == "two_stage_tl_exploratory":
        return (
            f"Two-stage TL | {row['cutoff_rule']} | {row['transfer_strategy']}"
        )
    if family == "two_stage_tl_nested_selected":
        return "Two-stage TL | training-selected"
    return str(row["model"])


def bootstrap_mean_ci(
    values: Iterable[float], resamples: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, len(array), size=(resamples, len(array)))
    sampled_means = array[sampled_indices].mean(axis=1)
    return tuple(np.quantile(sampled_means, [0.025, 0.975]).tolist())


def paired_statistics(
    values_a: np.ndarray,
    values_b: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    differences = values_a[valid] - values_b[valid]
    if len(differences) == 0:
        return {
            "n_pairs": 0,
            "mean_difference_a_minus_b": np.nan,
            "median_difference_a_minus_b": np.nan,
            "sd_difference": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "wilcoxon_statistic": np.nan,
            "wilcoxon_method": "not available",
            "p_unadjusted": np.nan,
            "negative_difference_count": 0,
            "positive_difference_count": 0,
            "ties": 0,
        }
    ci_low, ci_high = bootstrap_mean_ci(differences, resamples, seed)
    if np.allclose(differences, 0):
        statistic, p_value, wilcoxon_method = 0.0, 1.0, "all differences zero"
    else:
        wilcoxon_method = "exact" if not np.any(differences == 0) else "approx"
        result = wilcoxon(
            differences,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method=wilcoxon_method,
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "n_pairs": len(differences),
        "mean_difference_a_minus_b": float(np.mean(differences)),
        "median_difference_a_minus_b": float(np.median(differences)),
        "sd_difference": float(np.std(differences, ddof=1))
        if len(differences) > 1
        else 0.0,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "wilcoxon_statistic": statistic,
        "wilcoxon_method": wilcoxon_method,
        "p_unadjusted": p_value,
        "negative_difference_count": int((differences < 0).sum()),
        "positive_difference_count": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
    }


def apply_holm(
    frame: pd.DataFrame, group_columns: list[str], p_column: str = "p_unadjusted"
) -> pd.DataFrame:
    frame = frame.copy()
    frame["p_holm"] = np.nan
    frame["reject_holm_0_05"] = False
    for _, indices in frame.groupby(group_columns, dropna=False).groups.items():
        indices = list(indices)
        p_values = frame.loc[indices, p_column].to_numpy(dtype=float)
        valid = np.isfinite(p_values)
        if valid.any():
            rejected, adjusted, _, _ = multipletests(
                p_values[valid], alpha=0.05, method="holm"
            )
            valid_indices = np.asarray(indices)[valid]
            frame.loc[valid_indices, "p_holm"] = adjusted
            frame.loc[valid_indices, "reject_holm_0_05"] = rejected
    return frame


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    selected_mask = metrics["model_family"].eq("two_stage_tl_nested_selected")
    metrics.loc[selected_mask, "cutoff_rule"] = "training-selected"
    metrics.loc[selected_mask, "transfer_strategy"] = "training-selected"
    grouping = [
        "feature_set",
        "model_family",
        "analysis_label",
        "model",
        "cutoff_rule",
        "transfer_strategy",
    ]
    summary = (
        metrics.groupby(grouping, dropna=False)[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    for metric in METRICS:
        decimals = 4 if metric == "r2" else 2
        summary[f"{metric}_mean_sd"] = summary.apply(
            lambda row: (
                f"{row[f'{metric}_mean']:.{decimals}f} ± "
                f"{row[f'{metric}_std']:.{decimals}f}"
            ),
            axis=1,
        )
    return summary


def make_table3(thresholds: pd.DataFrame) -> pd.DataFrame:
    preferred = thresholds[thresholds["feature_set"] == "A_all_21"].copy()
    if preferred.empty:
        preferred = thresholds.copy()
    preferred = preferred.drop_duplicates(["seed", "cutoff_rule"])
    value_columns = [
        "training_derived_cutoff_value",
        "train_low_count",
        "train_high_count",
        "test_true_low_count",
        "test_true_high_count",
        "test_n_at_or_above_train_q3",
        "test_n_at_or_above_train_q3_iqr2",
    ]
    rows: list[dict[str, Any]] = []
    for cutoff_rule, group in preferred.groupby("cutoff_rule"):
        for variable in value_columns:
            values = group[variable].to_numpy(dtype=float)
            rows.append(
                {
                    "cutoff_rule": cutoff_rule,
                    "variable": variable,
                    "n_splits": len(values),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def make_ablation_outputs(
    metrics: pd.DataFrame, resamples: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus = metrics[
        metrics["model_family"].isin(
            ["single", "two_stage_tl_nested_selected"]
        )
    ].copy()
    ablation_summary = (
        focus.groupby(["analysis_label", "feature_set"])[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    ablation_summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in ablation_summary.columns
    ]

    contrast_rows: list[dict[str, Any]] = []
    feature_pairs = list(combinations(FEATURE_SETS, 2))
    for label, label_frame in focus.groupby("analysis_label"):
        for metric_index, metric in enumerate(METRICS):
            pivot = label_frame.pivot_table(
                index="seed", columns="feature_set", values=metric, aggfunc="first"
            )
            for pair_index, (feature_a, feature_b) in enumerate(feature_pairs):
                if feature_a not in pivot or feature_b not in pivot:
                    continue
                statistics = paired_statistics(
                    pivot[feature_a].to_numpy(),
                    pivot[feature_b].to_numpy(),
                    resamples,
                    20260717 + metric_index * 100 + pair_index,
                )
                contrast_rows.append(
                    {
                        "analysis_label": label,
                        "metric": metric,
                        "feature_set_a": feature_a,
                        "feature_set_b": feature_b,
                        **statistics,
                    }
                )
    contrasts = pd.DataFrame(contrast_rows)
    if not contrasts.empty:
        contrasts = apply_holm(contrasts, ["analysis_label", "metric"])
    return ablation_summary, contrasts


def make_primary_comparisons(
    metrics: pd.DataFrame, resamples: int
) -> pd.DataFrame:
    selected = metrics[
        metrics["model_family"] == "two_stage_tl_nested_selected"
    ].copy()
    baselines = metrics[metrics["model_family"] == "single"].copy()
    rows: list[dict[str, Any]] = []
    for feature_set in sorted(metrics["feature_set"].unique()):
        selected_feature = selected[selected["feature_set"] == feature_set]
        for baseline_label, baseline_frame in baselines[
            baselines["feature_set"] == feature_set
        ].groupby("analysis_label"):
            for metric_index, metric in enumerate(METRICS):
                paired = selected_feature[["seed", metric]].merge(
                    baseline_frame[["seed", metric]],
                    on="seed",
                    suffixes=("_selected", "_baseline"),
                )
                statistics = paired_statistics(
                    paired[f"{metric}_selected"].to_numpy(),
                    paired[f"{metric}_baseline"].to_numpy(),
                    resamples,
                    20260717 + metric_index,
                )
                baseline_mean = float(paired[f"{metric}_baseline"].mean())
                selected_mean = float(paired[f"{metric}_selected"].mean())
                percent_change = (
                    (selected_mean - baseline_mean) / abs(baseline_mean) * 100.0
                    if baseline_mean != 0
                    else np.nan
                )
                rows.append(
                    {
                        "feature_set": feature_set,
                        "metric": metric,
                        "model_a": "Two-stage TL | training-selected",
                        "model_b": baseline_label,
                        "model_a_mean": selected_mean,
                        "model_b_mean": baseline_mean,
                        "percent_change_a_vs_b": percent_change,
                        **statistics,
                    }
                )
    comparisons = pd.DataFrame(rows)
    if not comparisons.empty:
        comparisons = apply_holm(comparisons, ["feature_set", "metric"])
    return comparisons


def repeated_measures_anova(metrics: pd.DataFrame) -> pd.DataFrame:
    exploratory = metrics[
        metrics["model_family"] == "two_stage_tl_exploratory"
    ].copy()
    rows: list[dict[str, Any]] = []
    for feature_set, feature_frame in exploratory.groupby("feature_set"):
        for metric in METRICS:
            analysis_data = feature_frame[
                ["seed", "cutoff_rule", "transfer_strategy", metric]
            ].dropna()
            try:
                result = AnovaRM(
                    analysis_data,
                    depvar=metric,
                    subject="seed",
                    within=["cutoff_rule", "transfer_strategy"],
                    aggregate_func="mean",
                ).fit()
                table = result.anova_table.reset_index()
                effect_column = table.columns[0]
                for _, record in table.iterrows():
                    rows.append(
                        {
                            "feature_set": feature_set,
                            "metric": metric,
                            "effect": str(record[effect_column]),
                            "f_value": float(record["F Value"]),
                            "num_df": float(record["Num DF"]),
                            "den_df": float(record["Den DF"]),
                            "p_value": float(record["Pr > F"]),
                            "status": "complete",
                            "error": "",
                        }
                    )
            except Exception as error:
                rows.append(
                    {
                        "feature_set": feature_set,
                        "metric": metric,
                        "effect": "",
                        "f_value": np.nan,
                        "num_df": np.nan,
                        "den_df": np.nan,
                        "p_value": np.nan,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    return pd.DataFrame(rows)


def planned_transfer_contrasts(
    metrics: pd.DataFrame, resamples: int
) -> pd.DataFrame:
    exploratory = metrics[
        metrics["model_family"] == "two_stage_tl_exploratory"
    ].copy()
    rows: list[dict[str, Any]] = []
    strategy_levels = ["Conservative", "Balanced", "Aggressive"]
    cutoff_levels = ["Mean", "Q3", "Q3+IQR/2"]
    for feature_set, feature_frame in exploratory.groupby("feature_set"):
        for metric_index, metric in enumerate(METRICS):
            for cutoff_index, cutoff_rule in enumerate(cutoff_levels):
                subset = feature_frame[feature_frame["cutoff_rule"] == cutoff_rule]
                pivot = subset.pivot_table(
                    index="seed",
                    columns="transfer_strategy",
                    values=metric,
                    aggfunc="first",
                )
                for pair_index, (level_a, level_b) in enumerate(
                    combinations(strategy_levels, 2)
                ):
                    if level_a not in pivot or level_b not in pivot:
                        continue
                    statistics = paired_statistics(
                        pivot[level_a].to_numpy(),
                        pivot[level_b].to_numpy(),
                        resamples,
                        20260717
                        + metric_index * 1000
                        + cutoff_index * 100
                        + pair_index,
                    )
                    rows.append(
                        {
                            "feature_set": feature_set,
                            "metric": metric,
                            "contrast_family": "strategy_within_cutoff",
                            "conditioning_level": cutoff_rule,
                            "level_a": level_a,
                            "level_b": level_b,
                            **statistics,
                        }
                    )
            for strategy_index, strategy in enumerate(strategy_levels):
                subset = feature_frame[
                    feature_frame["transfer_strategy"] == strategy
                ]
                pivot = subset.pivot_table(
                    index="seed",
                    columns="cutoff_rule",
                    values=metric,
                    aggfunc="first",
                )
                for pair_index, (level_a, level_b) in enumerate(
                    combinations(cutoff_levels, 2)
                ):
                    if level_a not in pivot or level_b not in pivot:
                        continue
                    statistics = paired_statistics(
                        pivot[level_a].to_numpy(),
                        pivot[level_b].to_numpy(),
                        resamples,
                        20260717
                        + metric_index * 1000
                        + strategy_index * 100
                        + pair_index
                        + 50000,
                    )
                    rows.append(
                        {
                            "feature_set": feature_set,
                            "metric": metric,
                            "contrast_family": "cutoff_within_strategy",
                            "conditioning_level": strategy,
                            "level_a": level_a,
                            "level_b": level_b,
                            **statistics,
                        }
                    )
    contrasts = pd.DataFrame(rows)
    if not contrasts.empty:
        contrasts = apply_holm(
            contrasts, ["feature_set", "metric", "contrast_family"]
        )
    return contrasts


def make_svm_summary(svm: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cv_f1",
        "classifier_precision",
        "classifier_recall",
        "classifier_f1",
        "test_predicted_high_count",
    ]
    summary = (
        svm.groupby(["feature_set", "cutoff_rule"])[columns]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows available._"
    subset = frame[columns].head(max_rows).copy()
    for column in subset.columns:
        subset[column] = subset[column].map(
            lambda value: (
                ""
                if pd.isna(value)
                else f"{value:.4g}"
                if isinstance(value, (float, np.floating))
                else str(value)
            )
        )
        subset[column] = subset[column].str.replace("|", r"\|", regex=False)
    header = "| " + " | ".join(subset.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(subset.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in subset.to_numpy(dtype=str)]
    return "\n".join([header, divider, *body])


def build_report(
    verification: dict[str, Any],
    table4: pd.DataFrame,
    selection_frequency: pd.DataFrame,
    primary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    anova: pd.DataFrame,
) -> str:
    lines = [
        "# Corrected construction-cost analysis report",
        "",
        "## Completion and protocol",
        "",
        f"- Completed tasks: {verification['completed_task_count']} / {verification['expected_task_count']}.",
        f"- Seeds: {', '.join(map(str, verification['expected_seeds']))}.",
        "- Every cutoff and high-cost evaluation threshold was computed from the corresponding outer training partition only.",
        "- The outer test sets were not used for preprocessing, hyperparameter tuning, early stopping, or cutoff/transfer-strategy selection.",
        "- The nine fixed cutoff-by-strategy configurations are exploratory. The primary two-stage estimate is the training-selected procedure.",
        "",
        "## Data audit",
        "",
        "The source contains 419 complete, nonduplicate observations, 21 inputs, and direct construction cost `O`. The monetary inputs N10, N11, N14, and N15 average about 30.25% of the mean target. Their availability timing cannot be established from the workbook, so the early-stage claim remains conditional on author-supplied provenance.",
        "",
        "## Full-feature performance summary",
        "",
        markdown_table(
            table4[table4["feature_set"] == "A_all_21"],
            [
                "analysis_label",
                "r2_mean_sd",
                "mape_pct_mean_sd",
                "rmse_mean_sd",
                "rmse_q3_mean_sd",
                "rmse_q3_iqr2_mean_sd",
            ],
            max_rows=30,
        ),
        "",
        "## Training-only configuration selection frequency",
        "",
        markdown_table(
            selection_frequency,
            ["feature_set", "cutoff_rule", "transfer_strategy", "count", "percent"],
            max_rows=30,
        ),
        "",
        "## Primary paired comparisons",
        "",
        "Negative A-minus-B differences favor the training-selected two-stage model for error metrics; positive differences favor it for R2.",
        "",
        markdown_table(
            primary[
                (primary["feature_set"] == "A_all_21")
                & (primary["model_b"] == "ANN")
            ],
            [
                "metric",
                "model_a_mean",
                "model_b_mean",
                "mean_difference_a_minus_b",
                "ci95_low",
                "ci95_high",
                "p_holm",
            ],
            max_rows=10,
        ),
        "",
        "## Component-cost ablation",
        "",
        markdown_table(
            ablation_summary[
                ablation_summary["analysis_label"]
                == "Two-stage TL | training-selected"
            ],
            [
                "feature_set",
                "r2_mean",
                "mape_pct_mean",
                "rmse_mean",
                "rmse_q3_mean",
                "rmse_q3_iqr2_mean",
            ],
            max_rows=10,
        ),
        "",
        "## Repeated-measures analysis",
        "",
        "The 3x3 exploratory design is analyzed with seed as the subject. Interaction results should be read before the Holm-adjusted paired contrasts in `planned_transfer_contrasts.csv`.",
        "",
        markdown_table(
            anova[
                (anova["feature_set"] == "A_all_21")
                & anova["effect"].astype(str).str.contains(":", regex=False)
            ],
            ["metric", "effect", "f_value", "num_df", "den_df", "p_value", "status"],
            max_rows=10,
        ),
        "",
        "## Interpretation gate",
        "",
        "Do not carry the original Table 4 values, pooled n=30 Wilcoxon p-values, or the single-best-model wording into the manuscript. Update claims from the corrected seed-level files. The input-timing question for N10, N11, N14, and N15 still requires author documentation even after the ablation is complete.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and analyze corrected task outputs")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "corrected_analysis_20260717",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--feature-sets", default=",".join(FEATURE_SETS))
    args = parser.parse_args()

    expected_seeds = [int(value) for value in args.seeds.split(",") if value]
    expected_feature_sets = [
        value.strip() for value in args.feature_sets.split(",") if value.strip()
    ]
    expected_tasks = {
        f"{feature_set}_seed{seed}"
        for feature_set in expected_feature_sets
        for seed in expected_seeds
    }
    tasks_root = args.results_dir / "tasks"
    completed_task_dirs = sorted(
        task_dir
        for task_dir in tasks_root.glob("*")
        if (task_dir / "complete.json").exists()
    )
    completed_tasks = {task_dir.name for task_dir in completed_task_dirs}
    missing_tasks = sorted(expected_tasks - completed_tasks)
    unexpected_tasks = sorted(completed_tasks - expected_tasks)
    if missing_tasks and not args.allow_incomplete:
        raise RuntimeError(
            f"Analysis is incomplete. Missing {len(missing_tasks)} tasks: {missing_tasks}"
        )
    if not completed_task_dirs:
        raise RuntimeError("No completed task directories were found")

    compiled_dir = args.results_dir / "compiled"
    compiled_dir.mkdir(parents=True, exist_ok=True)
    completion_records = [
        json.loads((task_dir / "complete.json").read_text(encoding="utf-8"))
        for task_dir in completed_task_dirs
    ]
    run_metadata_path = args.results_dir / "run_metadata.json"
    run_metadata = (
        json.loads(run_metadata_path.read_text(encoding="utf-8"))
        if run_metadata_path.exists()
        else {}
    )
    feature_definitions: dict[str, list[str]] = {}
    for record in completion_records:
        feature_definitions.setdefault(record["task"].rsplit("_seed", 1)[0], record["features"])
    run_metadata.update(
        {
            "compiled_at": datetime.now(timezone.utc).isoformat(),
            "seeds": expected_seeds,
            "feature_sets": {
                name: feature_definitions.get(name, []) for name in expected_feature_sets
            },
            "completed_task_count": len(completed_task_dirs),
            "task_records": [
                {
                    "task": record["task"],
                    "seed": record["seed"],
                    "feature_count": record["feature_count"],
                    "elapsed_seconds": record["elapsed_seconds"],
                    "data_sha256": record["data_sha256"],
                    "package_versions": record["package_versions"],
                    "status": record["status"],
                }
                for record in completion_records
            ],
        }
    )
    atomic_write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False),
        compiled_dir / "compiled_run_metadata.json",
    )
    tables = {
        "model_metrics_seed_level.csv": "model_metrics.csv",
        "test_predictions.csv": "test_predictions.csv",
        "training_thresholds_seed_level.csv": "training_thresholds.csv",
        "split_indices.csv": "split_indices.csv",
        "hyperparameter_search.csv": "hyperparameter_search.csv",
        "svm_results_seed_level.csv": "svm_results.csv",
        "configuration_selection_cv.csv": "configuration_selection_cv.csv",
        "selected_configurations.csv": "selected_configuration.csv",
    }
    compiled: dict[str, pd.DataFrame] = {}
    for output_name, task_name in tables.items():
        frame = read_task_table(completed_task_dirs, task_name)
        compiled[output_name] = frame
        atomic_write_dataframe(frame, compiled_dir / output_name)

    metrics = compiled["model_metrics_seed_level.csv"].copy()
    metrics["analysis_label"] = metrics.apply(model_label, axis=1)
    atomic_write_dataframe(metrics, compiled_dir / "model_metrics_seed_level.csv")
    thresholds = compiled["training_thresholds_seed_level.csv"]
    svm = compiled["svm_results_seed_level.csv"]
    selected = compiled["selected_configurations.csv"]

    table4 = summary_table(metrics)
    atomic_write_dataframe(table4, compiled_dir / "table4_corrected_summary.csv")
    table3 = make_table3(thresholds)
    atomic_write_dataframe(table3, compiled_dir / "table3_split_specific.csv")
    selection_frequency = (
        selected.groupby(["feature_set", "cutoff_rule", "transfer_strategy"])
        .size()
        .rename("count")
        .reset_index()
    )
    selection_frequency["percent"] = selection_frequency.groupby("feature_set")[
        "count"
    ].transform(lambda values: values / values.sum() * 100.0)
    atomic_write_dataframe(
        selection_frequency, compiled_dir / "selection_frequency.csv"
    )
    svm_summary = make_svm_summary(svm)
    atomic_write_dataframe(svm_summary, compiled_dir / "svm_summary.csv")

    ablation_summary, ablation_contrasts = make_ablation_outputs(
        metrics, args.bootstrap_resamples
    )
    atomic_write_dataframe(ablation_summary, compiled_dir / "ablation_summary.csv")
    atomic_write_dataframe(
        ablation_contrasts, compiled_dir / "ablation_paired_contrasts.csv"
    )
    primary = make_primary_comparisons(metrics, args.bootstrap_resamples)
    atomic_write_dataframe(primary, compiled_dir / "primary_model_comparisons.csv")
    anova = repeated_measures_anova(metrics)
    atomic_write_dataframe(anova, compiled_dir / "repeated_measures_anova.csv")
    planned = planned_transfer_contrasts(metrics, args.bootstrap_resamples)
    atomic_write_dataframe(planned, compiled_dir / "planned_transfer_contrasts.csv")

    split_indices = compiled["split_indices.csv"]
    predictions = compiled["test_predictions.csv"]
    hyperparameters = compiled["hyperparameter_search.csv"]
    thresholds_compiled = compiled["training_thresholds_seed_level.csv"]
    svm_compiled = compiled["svm_results_seed_level.csv"]
    prediction_group_sizes = predictions.groupby(
        [
            "source_task",
            "model_family",
            "model",
            "cutoff_rule",
            "transfer_strategy",
        ],
        dropna=False,
    ).size()
    split_partition_counts = split_indices.groupby(["source_task", "partition"]).size()
    cross_feature_partition_consistent = (
        split_indices.groupby(["seed", "row_index"])["partition"].nunique().max() == 1
    )
    expected_metric_rows_per_task = 17
    verification = {
        "expected_task_count": len(expected_tasks),
        "completed_task_count": len(completed_tasks & expected_tasks),
        "missing_tasks": missing_tasks,
        "unexpected_tasks": unexpected_tasks,
        "expected_seeds": expected_seeds,
        "expected_feature_sets": expected_feature_sets,
        "model_metric_rows": len(metrics),
        "expected_model_metric_rows": len(completed_task_dirs)
        * expected_metric_rows_per_task,
        "metric_duplicate_key_count": int(
            metrics.duplicated(
                [
                    "seed",
                    "feature_set",
                    "model_family",
                    "model",
                    "cutoff_rule",
                    "transfer_strategy",
                ]
            ).sum()
        ),
        "nonfinite_core_metric_count": int(
            (~np.isfinite(metrics[METRICS].to_numpy(dtype=float))).sum()
        ),
        "split_rows_per_task_min": int(
            split_indices.groupby("source_task").size().min()
        ),
        "split_rows_per_task_max": int(
            split_indices.groupby("source_task").size().max()
        ),
        "all_tasks_have_419_split_rows": bool(
            (split_indices.groupby("source_task").size() == 419).all()
        ),
        "all_tasks_have_293_train_rows": bool(
            (
                split_partition_counts.xs("train", level="partition", drop_level=False)
                == 293
            ).all()
        ),
        "all_tasks_have_126_test_rows": bool(
            (
                split_partition_counts.xs("test", level="partition", drop_level=False)
                == 126
            ).all()
        ),
        "cross_feature_split_partitions_match": bool(
            cross_feature_partition_consistent
        ),
        "all_tasks_have_17_metric_rows": bool(
            (metrics.groupby("source_task").size() == expected_metric_rows_per_task).all()
        ),
        "all_model_prediction_groups_have_126_rows": bool(
            (prediction_group_sizes == 126).all()
        ),
        "all_tasks_have_90_hyperparameter_rows": bool(
            (hyperparameters.groupby("source_task").size() == 90).all()
        ),
        "all_tasks_have_3_threshold_rows": bool(
            (thresholds_compiled.groupby("source_task").size() == 3).all()
        ),
        "all_tasks_have_3_svm_rows": bool(
            (svm_compiled.groupby("source_task").size() == 3).all()
        ),
        "all_tasks_have_1_selected_configuration": bool(
            (selected.groupby("source_task").size() == 1).all()
        ),
        "unique_data_sha256_count": len(
            {record["data_sha256"] for record in completion_records}
        ),
        "all_tasks_use_same_data_sha256": len(
            {record["data_sha256"] for record in completion_records}
        )
        == 1,
        "status": "complete" if not missing_tasks else "incomplete",
    }
    atomic_write_text(
        json.dumps(verification, indent=2, ensure_ascii=False),
        compiled_dir / "verification.json",
    )
    report = build_report(
        verification,
        table4,
        selection_frequency,
        primary,
        ablation_summary,
        anova,
    )
    atomic_write_text(report, compiled_dir / "corrected_analysis_report.md")
    print(json.dumps(verification, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
