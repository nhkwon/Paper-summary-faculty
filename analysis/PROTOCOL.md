# Corrected construction-cost analysis protocol

This is a post-hoc corrective reanalysis, not a prospective preregistration. The rule below
was frozen before inspecting any results from the corrected outer test evaluations. The
previous manuscript results were already known, so the nine individual transfer-learning
configurations remain exploratory. The confirmatory target is the performance of the
training-only configuration-selection procedure, not any configuration chosen from outer
test results.

## Data and outcome

- Source: `Data_ML2.xlsx`, first worksheet, 419 complete observations.
- Outcome: `O`, direct construction cost in USD.
- Row identity: zero-based row number in the imported data table; split indices and hashes
  are retained.

## Feature sets

- `A_all_21`: `C1`, `N1` through `N20`.
- `B_no_component_costs`: all inputs except monetary component-cost variables `N10`,
  `N11`, `N14`, and `N15`.
- `C_physical_quantity`: `C1`, `N2` through `N9`, `N12`, `N13`, and `N16` through `N20`.
  This operationalizes the audit request as categorical, physical, and quantity variables
  only; it additionally excludes temporal variable `N1`. The exact timing/provenance of the
  four monetary variables cannot be inferred from the workbook and remains a study-author
  documentation requirement.

## Outer evaluation

- Ten prespecified seeds: 42 through 51.
- Random 70:30 train/test split, with no target stratification, matching the original
  repeated-holdout design.
- The outer test partition is not used for thresholds, preprocessing, hyperparameter
  selection, transfer configuration selection, early stopping, or model fitting.
- For every outer split, Mean, Q3, and Q3 + IQR/2 are calculated from `y_train` only and
  applied unchanged to the corresponding outer test partition.
- RMSE above Q3 and above Q3 + IQR/2 use the same training-derived thresholds.

## Preprocessing and model selection

- `C1` is one-hot encoded with unknown-category handling; numeric inputs are Min-Max
  scaled. Preprocessing is fit inside each training/validation fold.
- Neural-network targets are standardized using outer-training or fold-training outcomes
  only; inverse-transformed predictions are used for every reported metric.
- Keras segment models use at most 200 epochs with validation-loss early stopping
  (patience 20). The validation-best epoch count is recorded, and the selected model is
  refitted on the complete available training segment for that many epochs.
- The scikit-learn single ANN uses at most 500 iterations with a 20% internal validation
  fraction and `n_iter_no_change=40`; both input and target transformations remain inside
  cross-validation.
- CatBoost, XGBoost, monotonic LightGBM, the single ANN, and SVM hyperparameters are
  selected using training-only cross-validation. The complete grids and selected values are
  saved.
- The monotonic constraints are non-decreasing for available members of
  `{N3, N4, N6, N8, N10, N16}`.

## Transfer-learning configuration selection

- All 3 cutoff rules x 3 transfer strategies are evaluated within the outer training data.
- Primary selection endpoint: training-validation RMSE for observations at or above the
  training-fold Q3.
- Tie-breakers, in order: overall RMSE, MAPE, then lexical configuration name.
- The selected cutoff/strategy is then fitted on the complete outer training partition and
  evaluated once on the untouched outer test partition.
- Outer-test results for the nine individual configurations are exploratory and cannot be
  used to redefine the selected configuration.

## Statistical unit and inference

- The independent block is the outer random seed (`n = 10`), not seed x cutoff or seed x
  strategy.
- The 3 x 3 transfer design is assessed with repeated-measures analysis using seed as the
  subject/block.
- Planned paired contrasts use one observation per seed, paired effect estimates, bootstrap
  95% confidence intervals, and Holm-adjusted p-values.
- Feature-set ablations use the identical split for each seed and paired seed-level
  differences.

## Reproducibility outputs

The run records package versions, data SHA-256, split membership, thresholds, counts,
classifier metrics, hyperparameters, early-stopping epochs, seed-level regression metrics,
test predictions, selection scores, run status, and timing.
