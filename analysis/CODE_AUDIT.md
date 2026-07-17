# Audit of `construction_cost_models (1).ipynb`

This audit is based on the supplied notebook source and `Data_ML2.xlsx`. The notebook has
no executed cells or saved outputs, so its numerical claims cannot be verified from the
file alone.

## Verified defects in the supplied notebook

1. **The data path is not portable.** Cell 4 points to another user's Google Drive path
   instead of the supplied workbook.
2. **Target-derived cutoffs use all 419 outcomes before splitting.** Cell 7 calculates the
   Mean, Q3, and Q3 + IQR/2 from the complete `y`. Those values are then used for SVM labels
   and high-cost test metrics in every seed.
3. **The reported high-cost evaluation subsets also use test outcomes to define their own
   thresholds.** `evaluate_metrics` in cell 9 references the full-dataset constants.
4. **Tree and single-ANN hyperparameters are selected only on the seed-42 split.** Cell 14
   reuses those values for seeds 43 through 51. This is not the per-outer-split model
   selection described in the revised manuscript.
5. **Input scaling occurs before inner cross-validation.** Cell 14 fits `MinMaxScaler` on
   the complete outer training partition and passes the already-scaled array to
   `GridSearchCV`; inner validation folds therefore influence the scaling parameters.
6. **The same inner-fold preprocessing problem occurs in the SVM search.** Cell 18 receives
   an array scaled on the complete outer training partition.
7. **Categorical encoding is performed before the outer split.** Cell 4 one-hot encodes
   `C1` on the full dataset. The supplied `C1` happens to be binary and present throughout,
   but the workflow still contradicts the stated split-first protocol.
8. **The Keras ANN is not tuned.** Cell 11 fixes `[128, 64]`, ReLU, learning rate 0.001,
   and batch size 32 despite the manuscript's broad GridSearchCV statement. The
   scikit-learn single ANN is tuned separately and is not the model used for transfer.
9. **ANN search and final-fit settings are inconsistent.** Cell 14 searches with
   `max_iter=50` and no early stopping, while cell 15 fits for 100 iterations with early
   stopping and a 50% validation fraction.
10. **Keras early stopping does not use validation data.** Cells 19 and 22 pass no
    `validation_data` or `validation_split`, so Keras monitors training loss by default.
11. **TensorFlow randomness is not controlled per seed.** The outer split and some
    estimators receive a seed, but ANN initialization, minibatch order, and validation
    behavior do not.
12. **The preferred transfer configuration is chosen after comparing the same outer test
    results.** The notebook evaluates all nine cutoff-by-strategy combinations but contains
    no training-only configuration-selection procedure.
13. **There is no component-cost ablation.** N10, N11, N14, and N15 are retained in every
    run. Their mean sum is USD 44,851.88, or 30.25% of the mean target, and N10 has a
    Pearson correlation of about 0.806 with `O`.
14. **The export is insufficient for audit.** It omits split indices, index hashes,
    training-derived thresholds, test predictions, classifier precision/recall/F1,
    per-fold selection scores, early-stopping epochs, package versions, and run status.
15. **A logging field is missing.** The no-transfer SVM log in cell 19 omits the selected
    kernel even though the grid searches both RBF and linear kernels.

## Corrective implementation

`corrected_cost_pipeline.py` implements the following controls:

- outer split first for each seed;
- training-only thresholds and high-cost evaluation cutoffs;
- fold-contained encoding and scaling;
- per-outer-split hyperparameter selection;
- explicit target scaling for ANN stability, fitted on training outcomes only;
- validation-loss early stopping followed by refitting for the selected epoch count;
- deterministic seeds for Python, NumPy, scikit-learn, and TensorFlow;
- training-only selection of cutoff and transfer strategy using high-cost RMSE as the
  frozen primary selection endpoint;
- all three feature sets required for the component-cost ablation;
- one-row-per-seed metrics plus row-level test predictions, split membership, hashes,
  thresholds, counts, classifier scores, grids, selected parameters, layer-freezing
  details, epochs, versions, and completion markers.

`summarize_corrected_results.py` treats seed as the independent block, performs the 3x3
repeated-measures analysis, applies Holm correction to planned paired contrasts, and
reports paired bootstrap confidence intervals for model and feature-set comparisons.

## Remaining non-computational evidence requirement

The workbook cannot establish whether N10, N11, N14, and N15 were preliminary estimates
available at the claimed decision point or realized components of final construction cost.
The study authors must document provenance and timing. Until then, even a successful
ablation cannot by itself validate the phrase “early-stage cost estimation.”

