# Corrected construction-cost analysis report

## Completion and protocol

- Completed tasks: 30 / 30.
- Seeds: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51.
- Every cutoff and high-cost evaluation threshold was computed from the corresponding outer training partition only.
- The outer test sets were not used for preprocessing, hyperparameter tuning, early stopping, or cutoff/transfer-strategy selection.
- The nine fixed cutoff-by-strategy configurations are exploratory. The primary two-stage estimate is the training-selected procedure.

## Data audit

The source contains 419 complete, nonduplicate observations, 21 inputs, and direct construction cost `O`. The monetary inputs N10, N11, N14, and N15 average about 30.25% of the mean target. Their availability timing cannot be established from the workbook, so the early-stage claim remains conditional on author-supplied provenance.

## Full-feature performance summary

| analysis_label | r2_mean_sd | mape_pct_mean_sd | rmse_mean_sd | rmse_q3_mean_sd | rmse_q3_iqr2_mean_sd |
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

## Training-only configuration selection frequency

| feature_set | cutoff_rule | transfer_strategy | count | percent |
| --- | --- | --- | --- | --- |
| A_all_21 | Mean | Aggressive | 2 | 20 |
| A_all_21 | Mean | Balanced | 4 | 40 |
| A_all_21 | Q3 | Balanced | 2 | 20 |
| A_all_21 | Q3+IQR/2 | Balanced | 1 | 10 |
| A_all_21 | Q3+IQR/2 | Conservative | 1 | 10 |
| B_no_component_costs | Mean | Aggressive | 2 | 20 |
| B_no_component_costs | Mean | Balanced | 2 | 20 |
| B_no_component_costs | Q3 | Aggressive | 1 | 10 |
| B_no_component_costs | Q3 | Balanced | 2 | 20 |
| B_no_component_costs | Q3+IQR/2 | Balanced | 1 | 10 |
| B_no_component_costs | Q3+IQR/2 | Conservative | 2 | 20 |
| C_physical_quantity | Mean | Balanced | 1 | 10 |
| C_physical_quantity | Q3 | Balanced | 2 | 20 |
| C_physical_quantity | Q3 | Conservative | 3 | 30 |
| C_physical_quantity | Q3+IQR/2 | Balanced | 3 | 30 |
| C_physical_quantity | Q3+IQR/2 | Conservative | 1 | 10 |

## Primary paired comparisons

Negative A-minus-B differences favor the training-selected two-stage model for error metrics; positive differences favor it for R2.

| metric | model_a_mean | model_b_mean | mean_difference_a_minus_b | ci95_low | ci95_high | p_holm |
| --- | --- | --- | --- | --- | --- | --- |
| r2 | 0.9201 | 0.9511 | -0.03096 | -0.04684 | -0.01622 | 0.007812 |
| mape_pct | 14.81 | 12.41 | 2.404 | 0.5832 | 4.248 | 0.1465 |
| rmse | 2.824e+04 | 2.254e+04 | 5700 | 3179 | 8429 | 0.007812 |
| rmse_q3 | 4.698e+04 | 3.668e+04 | 1.03e+04 | 5318 | 1.584e+04 | 0.01172 |
| rmse_q3_iqr2 | 5.581e+04 | 4.413e+04 | 1.168e+04 | 5540 | 1.818e+04 | 0.01562 |

## Component-cost ablation

| feature_set | r2_mean | mape_pct_mean | rmse_mean | rmse_q3_mean | rmse_q3_iqr2_mean |
| --- | --- | --- | --- | --- | --- |
| A_all_21 | 0.9201 | 14.81 | 2.824e+04 | 4.698e+04 | 5.581e+04 |
| B_no_component_costs | 0.739 | 32.58 | 5.119e+04 | 7.654e+04 | 9.003e+04 |
| C_physical_quantity | 0.7183 | 31.14 | 5.285e+04 | 8.292e+04 | 9.73e+04 |

## Repeated-measures analysis

The 3x3 exploratory design is analyzed with seed as the subject. Interaction results should be read before the Holm-adjusted paired contrasts in `planned_transfer_contrasts.csv`.

| metric | effect | f_value | num_df | den_df | p_value | status |
| --- | --- | --- | --- | --- | --- | --- |
| r2 | cutoff_rule:transfer_strategy | 6.6 | 4 | 36 | 0.0004317 | complete |
| mape_pct | cutoff_rule:transfer_strategy | 11.54 | 4 | 36 | 3.946e-06 | complete |
| rmse | cutoff_rule:transfer_strategy | 6.809 | 4 | 36 | 0.0003454 | complete |
| rmse_q3 | cutoff_rule:transfer_strategy | 4.405 | 4 | 36 | 0.00531 | complete |
| rmse_q3_iqr2 | cutoff_rule:transfer_strategy | 4.157 | 4 | 36 | 0.007195 | complete |

## Interpretation gate

Do not carry the original Table 4 values, pooled n=30 Wilcoxon p-values, or the single-best-model wording into the manuscript. Update claims from the corrected seed-level files. The input-timing question for N10, N11, N14, and N15 still requires author documentation even after the ablation is complete.
