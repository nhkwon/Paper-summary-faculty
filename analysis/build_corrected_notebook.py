from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "construction_cost_models_corrected.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Leakage-controlled construction-cost reanalysis

This notebook is the review-oriented entry point for the corrected analysis. The model
implementation lives in `analysis/corrected_cost_pipeline.py`, and the frozen analysis
rules are documented in `analysis/PROTOCOL.md`.

Key changes from the supplied notebook:

- split before target-derived cutoffs, encoding, and scaling;
- calculate all cost thresholds from the outer training partition only;
- tune preprocessing and model hyperparameters inside training-only cross-validation;
- choose cutoff and transfer strategy without inspecting the outer test set;
- run the full, no-component-cost, and physical/quantity feature sets on identical splits;
- retain seed-level metrics, row-level test predictions, split indices, classifier scores,
  hyperparameters, epochs, and run metadata;
- treat seed—not seed x cutoff or seed x strategy—as the independent statistical block.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
DATA = ROOT / "Data_ML2.xlsx"
RESULTS = ROOT / "results" / "corrected_analysis_20260717"
PYTHON = ROOT / "tmp" / "python311" / "python.exe"

assert DATA.exists(), DATA
print(DATA)
print(RESULTS)"""
        ),
        nbf.v4.new_markdown_cell("## Source-data audit"),
        nbf.v4.new_code_cell(
            """df = pd.read_excel(DATA)
component_costs = ["N10", "N11", "N14", "N15"]
component_sum = df[component_costs].sum(axis=1)

audit = {
    "shape": df.shape,
    "missing_cells": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "target_mean_usd": float(df["O"].mean()),
    "component_cost_mean_usd": float(component_sum.mean()),
    "component_share_of_target_mean": float(component_sum.mean() / df["O"].mean()),
    "N10_target_correlation": float(df[["N10", "O"]].corr().iloc[0, 1]),
}
audit"""
        ),
        nbf.v4.new_markdown_cell(
            """The workbook cannot establish when the four component-cost inputs become
available. Their provenance and decision-stage timing must be documented by the study
authors even after the ablation results are available."""
        ),
        nbf.v4.new_markdown_cell("## Frozen protocol and configuration"),
        nbf.v4.new_code_cell(
            """from analysis.corrected_cost_pipeline import (
    ANN_GRID,
    CUTOFF_RULES,
    FEATURE_SETS,
    SEEDS,
    SVM_GRID,
    TL_BASE_CONFIG,
    TRANSFER_STRATEGIES,
    TREE_GRIDS,
)

display(pd.Series({name: len(features) for name, features in FEATURE_SETS.items()}, name="feature_count"))
print("Seeds:", SEEDS)
print("Cutoffs:", CUTOFF_RULES)
print("Primary training-only selection endpoint: RMSE at or above the training-fold Q3")
print("Transfer base:", TL_BASE_CONFIG)
print("Strategies:", TRANSFER_STRATEGIES)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Execute the corrected analysis

The pipeline is resumable: each feature-set/seed task receives a completion marker and is
skipped on rerun unless `--force` is supplied. A full run evaluates all 30 tasks."""
        ),
        nbf.v4.new_code_cell(
            """# One-task smoke test (uncomment to run):
# !\"{PYTHON}\" analysis/corrected_cost_pipeline.py --data \"{DATA}\" --output-dir \"{RESULTS}\" --seeds 42 --feature-sets A_all_21 --smoke"""
        ),
        nbf.v4.new_code_cell(
            """# Full 10-seed x 3-feature-set run (uncomment to run):
# !\"{PYTHON}\" analysis/corrected_cost_pipeline.py --data \"{DATA}\" --output-dir \"{RESULTS}\""""
        ),
        nbf.v4.new_markdown_cell("## Compile and statistically analyze completed tasks"),
        nbf.v4.new_code_cell(
            """# Compile only after all 30 task markers are present (uncomment to run):
# !\"{PYTHON}\" analysis/summarize_corrected_results.py --results-dir \"{RESULTS}\""""
        ),
        nbf.v4.new_markdown_cell("## Inspect corrected outputs"),
        nbf.v4.new_code_cell(
            """compiled = RESULTS / "compiled"
if compiled.exists():
    table4 = pd.read_csv(compiled / "table4_corrected_summary.csv")
    table3 = pd.read_csv(compiled / "table3_split_specific.csv")
    selection = pd.read_csv(compiled / "selection_frequency.csv")
    primary = pd.read_csv(compiled / "primary_model_comparisons.csv")
    ablation = pd.read_csv(compiled / "ablation_summary.csv")
    display(table4[table4["feature_set"] == "A_all_21"])
    display(selection)
    display(primary[(primary["feature_set"] == "A_all_21") & (primary["model_b"] == "ANN")])
    display(ablation[ablation["analysis_label"] == "Two-stage TL | training-selected"])
else:
    print("Compiled outputs are not available yet.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Statistical interpretation

Read `repeated_measures_anova.csv` before interpreting the paired contrasts. The contrast
file uses one paired observation per seed, bootstrap 95% confidence intervals, and Holm
adjustment. The nine fixed transfer configurations are exploratory; manuscript claims
should use the training-selected procedure for primary performance reporting."""
        ),
    ]
    notebook["nbformat"] = 4
    notebook["nbformat_minor"] = 5
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
