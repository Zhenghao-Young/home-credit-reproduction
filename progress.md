# Home Credit Controlled Reproduction Progress

This document tracks the implementation and report progress for `plan.md`.

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

## Global Milestones

- [x] M1: S1 baseline runs end to end with fixed folds and OOF AUC.
- [ ] M2: RQ1 complete with S1, S2, B1, and S3 comparable results.
- [ ] M3: RQ2 complete with S4, B2, and S5 comparable results.
- [ ] M4: RQ3 complete with average and logistic stacking.
- [~] M5: Final report compiled in `report/`.

## Member A: Infrastructure, S1, S2

- [x] Create fixed folds: `data/folds.csv`.
- [x] Implement `src/split.py`.
- [x] Implement `src/metrics.py`.
- [x] Implement `src/train_cv.py`.
- [x] Create `configs/base_lgbm.yaml`.
- [x] Run S1 LightGBM.
- [x] Run S2 LightGBM.
- [x] Run S2 Logistic Regression.
- [x] Verify each OOF row appears exactly once.
- [x] Verify all stages use the same `fold_id`.
- [x] Verify fold-safe groupby features do not use validation-fold statistics.
- [x] Draft report Chapter 1: problem, open solution history, and RQs.
- [x] Draft report Chapter 2: controlled reproduction design.

## Member B: RQ1

- [ ] Implement application business features.
- [ ] Implement basic historical-table aggregations.
- [ ] Create `configs/b1.yaml`.
- [ ] Create `configs/s3.yaml`.
- [ ] Run B1.
- [ ] Run S3.
- [ ] Produce `rq1_results.csv`.
- [ ] Produce RQ1 gain tree figure.
- [ ] Draft report Chapter 3.

## Member C: RQ2

- [ ] Implement group-relative features.
- [ ] Implement recent-window behavior features.
- [ ] Implement cleaning-before-aggregation variant.
- [ ] Implement dynamic ratios and trend features.
- [ ] Create `configs/s4.yaml`.
- [ ] Create `configs/b2.yaml`.
- [ ] Create `configs/s5.yaml`.
- [ ] Run S4.
- [ ] Run B2.
- [ ] Run S5.
- [ ] Produce `rq2_results.csv`.
- [ ] Produce dynamic-feature decile risk figure.
- [ ] Draft report Chapter 4.

## Member D: RQ3 and Integration

- [ ] Implement `src/stacking.py`.
- [ ] Implement `src/make_final_figures.py`.
- [ ] Build first-level OOF prediction matrix.
- [ ] Compute OOF prediction correlation matrix.
- [ ] Run simple average.
- [ ] Run logistic stacking with second-level CV.
- [ ] Produce `stacking_results.csv`.
- [ ] Produce `meta_coefficients.csv`.
- [ ] Produce final evidence-chain figure.
- [ ] Draft report Chapters 5 and 6.
- [ ] Integrate and compile final report in `report/`.

## Output Contract

Every experiment stage should write:

```text
results/<stage>/oof.parquet
results/<stage>/fold_metrics.csv
results/<stage>/feature_names.txt
results/<stage>/config.yaml
```

## Report Directory

Use `report/` only for final course-report sources and compiled outputs, such as:

```text
report/draft_report.tex
report/references.bib
report/figures/
report/draft_report.pdf
```

Keep intermediate experiment outputs under `results/`, not under `report/`.
