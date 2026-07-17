# Corrected-analysis manuscript update package

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

| cutoff_rule | variable | n_splits | mean | sd | minimum | maximum |
| --- | --- | --- | --- | --- | --- | --- |
| Mean | training_derived_cutoff_value | 10 | 147,584.42 | 3,777.47 | 140,511.27 | 154,018.67 |
| Mean | train_low_count | 10 | 168.60 | 4.70 | 162.00 | 178.00 |
| Mean | train_high_count | 10 | 124.40 | 4.70 | 115.00 | 131.00 |
| Mean | test_true_low_count | 10 | 71.90 | 7.06 | 59.00 | 78.00 |
| Mean | test_true_high_count | 10 | 54.10 | 7.06 | 48.00 | 67.00 |
| Mean | test_n_at_or_above_train_q3 | 10 | 34.20 | 6.30 | 24.00 | 43.00 |
| Mean | test_n_at_or_above_train_q3_iqr2 | 10 | 22.20 | 4.64 | 14.00 | 32.00 |
| Q3 | training_derived_cutoff_value | 10 | 179,321.11 | 5,983.40 | 171,813.56 | 190,358.76 |
| Q3 | train_low_count | 10 | 219.00 | 0.00 | 219.00 | 219.00 |
| Q3 | train_high_count | 10 | 74.00 | 0.00 | 74.00 | 74.00 |
| Q3 | test_true_low_count | 10 | 91.80 | 6.30 | 83.00 | 102.00 |
| Q3 | test_true_high_count | 10 | 34.20 | 6.30 | 24.00 | 43.00 |
| Q3 | test_n_at_or_above_train_q3 | 10 | 34.20 | 6.30 | 24.00 | 43.00 |
| Q3 | test_n_at_or_above_train_q3_iqr2 | 10 | 22.20 | 4.64 | 14.00 | 32.00 |
| Q3+IQR/2 | training_derived_cutoff_value | 10 | 226,413.71 | 8,466.76 | 215,802.76 | 242,028.25 |
| Q3+IQR/2 | train_low_count | 10 | 247.60 | 3.17 | 242.00 | 253.00 |
| Q3+IQR/2 | train_high_count | 10 | 45.40 | 3.17 | 40.00 | 51.00 |
| Q3+IQR/2 | test_true_low_count | 10 | 103.80 | 4.64 | 94.00 | 112.00 |
| Q3+IQR/2 | test_true_high_count | 10 | 22.20 | 4.64 | 14.00 | 32.00 |
| Q3+IQR/2 | test_n_at_or_above_train_q3 | 10 | 34.20 | 6.30 | 24.00 | 43.00 |
| Q3+IQR/2 | test_n_at_or_above_train_q3_iqr2 | 10 | 22.20 | 4.64 | 14.00 | 32.00 |

## Corrected Table 4

| Model/configuration | R² | MAPE (%) | RMSE | RMSE ≥ train Q3 | RMSE ≥ train Q3+IQR/2 |
| --- | --- | --- | --- | --- | --- |
| ANN | 0.9511 ± 0.0142 | 12.41 ± 1.90 | 22535.18 ± 2792.72 | 36683.01 ± 6181.18 | 44127.66 ± 6870.66 |
| CatBoost | 0.8835 ± 0.0620 | 14.75 ± 1.70 | 35246.83 ± 12613.84 | 60927.49 ± 26657.81 | 73876.56 ± 32606.84 |
| Monotonic GBM | 0.8040 ± 0.0634 | 20.96 ± 4.87 | 45792.03 ± 11048.12 | 74961.06 ± 23159.15 | 89520.37 ± 28677.38 |
| XGBoost | 0.8577 ± 0.0453 | 16.25 ± 1.89 | 39241.73 ± 9887.26 | 68316.23 ± 19819.38 | 81946.40 ± 24353.10 |
| Two-stage no TL \| Mean | 0.9340 ± 0.0307 | 14.86 ± 4.60 | 26106.32 ± 5940.62 | 40102.45 ± 12126.73 | 48188.06 ± 13733.01 |
| Two-stage no TL \| Q3 | 0.8825 ± 0.0617 | 17.01 ± 4.98 | 33697.62 ± 6835.80 | 51312.15 ± 14923.15 | 58528.61 ± 15907.87 |
| Two-stage no TL \| Q3+IQR/2 | 0.8527 ± 0.0618 | 14.36 ± 2.19 | 39835.67 ± 13232.83 | 70192.24 ± 23260.74 | 85475.24 ± 30956.37 |
| Two-stage TL \| Mean \| Aggressive | 0.9248 ± 0.0342 | 15.09 ± 4.30 | 27440.26 ± 4801.07 | 44652.15 ± 10269.21 | 53071.34 ± 11856.77 |
| Two-stage TL \| Mean \| Balanced | 0.9230 ± 0.0351 | 15.07 ± 4.43 | 27755.61 ± 5082.88 | 45307.54 ± 10783.12 | 54549.16 ± 13003.88 |
| Two-stage TL \| Mean \| Conservative | 0.8471 ± 0.0508 | 17.77 ± 4.48 | 40327.41 ± 9431.56 | 67613.42 ± 20981.90 | 80243.16 ± 26428.33 |
| Two-stage TL \| Q3 \| Aggressive | 0.7339 ± 0.2000 | 18.67 ± 6.04 | 48545.39 ± 21362.45 | 81296.87 ± 47142.98 | 96383.03 ± 60500.86 |
| Two-stage TL \| Q3 \| Balanced | 0.9290 ± 0.0244 | 14.92 ± 3.83 | 27158.33 ± 4901.74 | 43266.34 ± 10189.77 | 50653.83 ± 11808.91 |
| Two-stage TL \| Q3 \| Conservative | 0.8762 ± 0.0343 | 15.85 ± 3.93 | 36265.01 ± 7459.61 | 62191.74 ± 16237.71 | 71819.90 ± 21550.49 |
| Two-stage TL \| Q3+IQR/2 \| Aggressive | 0.8043 ± 0.1208 | 15.11 ± 2.40 | 44120.51 ± 15581.11 | 79279.28 ± 31126.36 | 96335.70 ± 40437.01 |
| Two-stage TL \| Q3+IQR/2 \| Balanced | 0.9132 ± 0.0391 | 13.77 ± 2.12 | 29566.87 ± 6469.00 | 51829.15 ± 14003.42 | 61676.29 ± 15622.29 |
| Two-stage TL \| Q3+IQR/2 \| Conservative | 0.9078 ± 0.0233 | 13.83 ± 2.42 | 31531.12 ± 7011.12 | 55745.09 ± 12668.79 | 66822.80 ± 17322.19 |
| Two-stage TL \| training-selected | 0.9201 ± 0.0370 | 14.81 ± 4.13 | 28235.22 ± 6185.86 | 46984.24 ± 13033.08 | 55808.02 ± 14667.85 |

## Results replacement: primary comparison

The primary estimate is the performance of the training-selected two-stage procedure, not
the best fixed configuration identified after inspecting test results. Across the 10 outer
test splits, R² was 0.9201 versus 0.9511 for the single ANN (lower by 0.0310). MAPE (%) was 14.81 versus 12.41 for the single ANN (increased by 19.4%). overall RMSE (USD) was 28,235 versus 22,535 for the single ANN (increased by 25.3%). RMSE at or above training-derived Q3 (USD) was 46,984 versus 36,683 for the single ANN (increased by 28.1%). RMSE at or above training-derived Q3 + IQR/2 (USD) was 55,808 versus 44,128 for the single ANN (increased by 26.5%). Paired seed-level effect estimates,
bootstrap 95% confidence intervals, and Holm-adjusted tests are reported in
`primary_model_comparisons.csv`.

Descriptively, the lowest mean overall RMSE in the corrected full-feature analysis was
observed for **ANN** (22,535 USD),
whereas the lowest mean Q3-segment RMSE was observed for
**ANN** (36,683 USD). These descriptive
leaders must not be relabelled as independently selected “best models” unless they coincide
with the frozen training-only selection rule.

## Training-only selection frequency

| feature_set | cutoff_rule | transfer_strategy | count | percent |
| --- | --- | --- | --- | --- |
| A_all_21 | Mean | Aggressive | 2 | 20.0% |
| A_all_21 | Mean | Balanced | 4 | 40.0% |
| A_all_21 | Q3 | Balanced | 2 | 20.0% |
| A_all_21 | Q3+IQR/2 | Balanced | 1 | 10.0% |
| A_all_21 | Q3+IQR/2 | Conservative | 1 | 10.0% |
| B_no_component_costs | Mean | Aggressive | 2 | 20.0% |
| B_no_component_costs | Mean | Balanced | 2 | 20.0% |
| B_no_component_costs | Q3 | Aggressive | 1 | 10.0% |
| B_no_component_costs | Q3 | Balanced | 2 | 20.0% |
| B_no_component_costs | Q3+IQR/2 | Balanced | 1 | 10.0% |
| B_no_component_costs | Q3+IQR/2 | Conservative | 2 | 20.0% |
| C_physical_quantity | Mean | Balanced | 1 | 10.0% |
| C_physical_quantity | Q3 | Balanced | 2 | 20.0% |
| C_physical_quantity | Q3 | Conservative | 3 | 30.0% |
| C_physical_quantity | Q3+IQR/2 | Balanced | 3 | 30.0% |
| C_physical_quantity | Q3+IQR/2 | Conservative | 1 | 10.0% |

## Component-cost ablation

| Feature set | R² | MAPE (%) | RMSE | RMSE ≥ train Q3 | RMSE ≥ train Q3+IQR/2 |
| --- | --- | --- | --- | --- | --- |
| A_all_21 | 0.9201 | 14.81 | 28,235 | 46,984 | 55,808 |
| B_no_component_costs | 0.7390 | 32.58 | 51,186 | 76,541 | 90,027 |
| C_physical_quantity | 0.7183 | 31.14 | 52,854 | 82,918 | 97,296 |

There were 44 Holm-significant paired feature-set contrasts across
the reported model/metric families. Exact effect estimates and bootstrap intervals are in
`ablation_paired_contrasts.csv`. The workbook does not reveal whether N10, N11, N14, and
N15 are preliminary estimates available at the claimed decision point or realized
components of final direct construction cost. Add author-supplied provenance and timing.
If early availability cannot be documented, replace “early-stage cost estimation” with
“conditional retrofit cost estimation using partial component-cost information.”

## Repeated-measures replacement for Tables 5–6

The 3-cutoff × 3-strategy analysis used seed as the repeated-measures subject. The
full-feature interaction results are:

| metric | effect | f_value | num_df | den_df | p_value | status |
| --- | --- | --- | --- | --- | --- | --- |
| r2 | cutoff_rule:transfer_strategy | 6.600369073135122 | 4.0 | 36.0 | 0.0004317422102408 | complete |
| mape_pct | cutoff_rule:transfer_strategy | 11.537813594423175 | 4.0 | 36.0 | 3.946262668603706e-06 | complete |
| rmse | cutoff_rule:transfer_strategy | 6.80889265368406 | 4.0 | 36.0 | 0.0003453633819548 | complete |
| rmse_q3 | cutoff_rule:transfer_strategy | 4.405469338105567 | 4.0 | 36.0 | 0.0053101257688912 | complete |
| rmse_q3_iqr2 | cutoff_rule:transfer_strategy | 4.156537137691831 | 4.0 | 36.0 | 0.0071949900946503 | complete |

After Holm adjustment within the planned contrast families, 14
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
