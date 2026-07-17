# Corrected analysis workflow

The supplied notebook is preserved as evidence. Use the corrected files below for the
rerun.

- `PROTOCOL.md`: frozen post-hoc corrective analysis rules.
- `CODE_AUDIT.md`: verified defects in the supplied notebook and their corrections.
- `corrected_cost_pipeline.py`: resumable model-fitting pipeline; one task is one feature
  set x outer seed.
- `run_full_analysis.ps1`: four-process local orchestrator for all 30 tasks.
- `summarize_corrected_results.py`: compiles task outputs, verifies completeness, runs
  seed-blocked statistics, and creates manuscript-ready CSV tables and a Markdown report.
- `build_corrected_figures.py`: creates publication-ready performance, ablation, SVM, and
  training-selection figures from the verified compiled CSV files.
- `build_manuscript_update.py`: converts the verified tables and paired statistics into a
  paste-ready manuscript revision package without carrying forward provisional numbers.
- `verify_pipeline.py`: fast deterministic checks for data shape, published descriptive
  thresholds, split hashes, transformed dimensions, monotonic constraints, and segment
  metric behavior.
- `build_corrected_notebook.py`: regenerates the review-oriented corrected notebook at the
  repository root.
- `requirements-analysis.txt`: pinned analysis dependencies.

## Reproduce

With Python 3.11 and the pinned packages installed:

```powershell
python analysis/corrected_cost_pipeline.py --data Data_ML2.xlsx --output-dir results/corrected_analysis_20260717
python analysis/summarize_corrected_results.py --results-dir results/corrected_analysis_20260717
python analysis/build_corrected_figures.py --compiled-dir results/corrected_analysis_20260717/compiled
python analysis/build_manuscript_update.py --compiled-dir results/corrected_analysis_20260717/compiled
python analysis/verify_pipeline.py --data Data_ML2.xlsx
```

After all 30 completion markers exist, the PowerShell finalizer runs compilation,
statistics, figures, manuscript updates, notebook regeneration, and deterministic checks:

```powershell
powershell -ExecutionPolicy Bypass -File analysis/finalize_corrected_analysis.ps1
```

For a fast integration check:

```powershell
python analysis/corrected_cost_pipeline.py --data Data_ML2.xlsx --output-dir results/corrected_analysis_20260717 --seeds 42 --feature-sets A_all_21 --smoke
```

Every completed task has its own `complete.json`. Re-running the pipeline skips completed
tasks unless `--force` is passed.

## Primary and exploratory outputs

The primary two-stage result is `two_stage_tl_nested_selected`: for each seed and feature
set, the cutoff and transfer strategy are selected using only the outer training data, with
RMSE at or above the training-fold Q3 as the primary endpoint. The nine fixed
cutoff-by-strategy rows are retained as `two_stage_tl_exploratory` and must not be used to
redefine the primary model after outer-test inspection.

The compiled directory contains:

- seed-level metrics and row-level predictions;
- a compiled run manifest with the common data SHA-256, package versions, task timings,
  complete seed list, and all feature definitions;
- split indices/hashes and split-specific thresholds/counts;
- complete hyperparameter and selection records;
- corrected Table 3 and Table 4 summaries;
- SVM summaries and selection frequencies;
- component-cost ablation estimates and paired confidence intervals;
- repeated-measures results and Holm-adjusted planned contrasts;
- primary paired comparisons and a verification manifest.
