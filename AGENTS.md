# Agent Operating Guide

## Project Mission

This repository is a controlled reproduction of the Kaggle Home Credit Default
Risk project. The goal is not leaderboard chasing; it is to build a verifiable
evidence chain for one research question:

> As the upstream `minerva-ml/open-solution-home-credit` solution improved from
> about 0.74 to about 0.80 AUC, how much of the gain came from feature
> representation, dynamic behavior modeling, and model ensembling?

Treat every code or documentation change as part of that controlled experiment.
Prefer reproducibility, fold-safety, and clear attribution over broad refactors
or opportunistic tuning.

## Repository Orientation

- Active code is in `src/`; feature builders live in `src/features/`.
- The single experiment entry point is `python -m src.train_cv`.
- Stage definitions and research framing live in `docs/reproduction_plan.md`.
- Current task status and measured AUC values live in `docs/progress.md`.
- Cross-stage metric rollups live in `results/summary.csv` when experiment
  outputs are available.
- Per-question artifacts should live under `results/rq*/`.
- `open-solution/` is a read-only upstream submodule. Do not edit it.

Before changing experiment logic, read the relevant parts of:

- `README.md` for the project overview and quickstart.
- `docs/workflow.md` for commands, environment setup, validation, and team
  workflow.
- `docs/progress.md` for current task status and latest measured results.
- `docs/reproduction_plan.md` for stable stage definitions and research framing.
- `docs/open_solution_guide.md` when studying upstream tags.

## Environment And Commands

Use Python 3.9. Do not try to run the pinned `pandas`, `scikit-learn`, and
`lightgbm` stack on Python 3.14.

Supported runners:

```bash
conda run -n credit python -m src.train_cv --help
```

```powershell
.\.venv\Scripts\python.exe -m src.train_cv --help
```

In command examples, `PY` means either `conda run -n credit python` or
`.\.venv\Scripts\python.exe`.

Run a stage:

```bash
PY -m src.train_cv --stage <stage> --model <model> --config configs/<config>.yaml --predict-test
```

Use `PY -m src.train_cv --help` and `docs/workflow.md` for the currently
supported stage/model combinations.

Regenerate research-question artifacts with the available scripts after relevant
outputs exist, for example:

```bash
PY -m src.make_rq1_artifacts
```

Validate an OOF file:

```bash
PY -c "import pandas as pd; from src.split import load_folds; from src.metrics import validate_oof, summarize_oof_auc; folds=load_folds('data/folds.csv'); oof=pd.read_parquet('results/<stage>/oof.parquet'); validate_oof(oof, folds); print(summarize_oof_auc(oof))"
```

Compile the report from `report/`:

```bash
xelatex -interaction=nonstopmode draft_report.tex
xelatex -interaction=nonstopmode draft_report.tex
```

The report is written in Chinese and uses the `ctexrep` class, so build it with
XeLaTeX rather than `pdflatex`; on Windows it relies on system CJK fonts via
`fontset=windows`.

## Architecture Notes

- `src/train_cv.py` orchestrates experiments. `run_cv()` loads data and folds,
  loops over fixed folds, fits models, validates OOF, writes stage outputs, and
  upserts `results/summary.csv`.
- `_build_fold_features(...)` in `src/train_cv.py` is the stage dispatch point.
  Add stage wiring there only after feature logic is placed in a focused module.
- `src/split.py` owns `data/folds.csv` creation and validation.
- `src/metrics.py` owns the OOF contract and AUC helpers.
- `src/features/application_base.py` implements S1 cleaning and application
  features.
- `src/features/application_groupby.py` implements fold-safe S2 groupby
  features.
- `src/features/application_business.py` implements B1 business ratios and
  `EXT_SOURCE` summaries.
- `src/features/history_basic.py` implements S3 applicant-level aggregations
  from historical tables.

This file should avoid listing which stages are currently complete or in
progress. Use `docs/progress.md`, `results/summary.csv`, and the actual
`SUPPORTED_STAGES` value in `src/train_cv.py` as the source of truth.

## Fold-Safety Rules

These are hard requirements.

- `data/folds.csv` is the only fold source. Do not create ad hoc random splits
  inside training code.
- Every OOF row must match `data/folds.csv`; use `validate_oof` after any run.
- Validation folds and Kaggle test data must never be used to fit fold-local
  statistics.
- For group means, encoders, target-like encodings, imputers learned from
  application subgroups, or similar statistics: fit on the current fold's
  `train_df`, then transform train, valid, and test separately.
- Keep train, valid, and test feature columns in identical order. The written
  `feature_names.txt` must match the model input columns.
- Historical table aggregations are allowed at `SK_ID_CURR` grain when they do
  not use `TARGET` and are computed from raw history only.

## Adding Or Modifying Stages

Keep new feature logic out of `train_cv.py` unless it is pure orchestration.
Suggested modules from the current plan:

- `src/features/relative_recent.py` for S4 group-relative and recent-window
  behavior features.
- `src/cleaning_v2.py` or a focused feature module for B2 cleaning variants.
- `src/features/dynamic.py` for S5 dynamic ratios and trends.
- `src/stacking.py` for S6 averaging, OOF correlation, and logistic stacking.

When adding a stage:

1. Implement feature logic in a focused module with small, explicit functions.
2. Add the stage to `SUPPORTED_STAGES`.
3. Wire it into `_build_fold_features(...)`.
4. Add or choose an appropriate `configs/*.yaml`.
5. Run the stage with fixed folds.
6. Validate OOF and update or generate the relevant `results/rq*/` artifacts.
7. Update `docs/progress.md` and report text when the experiment result is part
   of the evidence chain.

## Required Outputs

Each completed stage must write:

```text
results/<stage>/oof.parquet
results/<stage>/fold_metrics.csv
results/<stage>/feature_names.txt
results/<stage>/config.yaml
```

When run with `--predict-test`, it must also write:

```text
results/<stage>/submission.csv
```

Expected validation checks:

- `oof.parquet` has one row per training applicant.
- `SK_ID_CURR` is unique in OOF.
- `prediction` is finite and in `[0, 1]`.
- `fold_id` matches `data/folds.csv`.
- `feature_names.txt` matches trained feature columns.
- `results/summary.csv` contains the latest `(stage, model)` row.

## Data And Security

- Raw Kaggle files belong only in ignored `data/original/`.
- Never commit raw CSV/ZIP data, `.venv/`, caches, credentials, or local secrets.
- Kaggle credentials should stay outside the repo, usually
  `~/.kaggle/kaggle.json`.
- Do not hard-code local absolute paths. Use config values and repository
  relative paths.
- Keep `open-solution/` clean. Use read-only commands such as:

```bash
git -C open-solution show solution-4:src/feature_extraction.py
git -C open-solution diff --stat solution-4 solution-5
```

## Coding Style

- Python, 4-space indentation.
- Add type hints where practical.
- Prefer small functions with explicit data-flow boundaries.
- Use pandas operations deliberately and keep merge cardinality checks such as
  `validate="one_to_one"` or `validate="many_to_one"` where applicable.
- Preserve existing naming conventions: stage names are lowercase (`s1`, `b1`,
  `s3`), and feature names should be descriptive and stable.
- Avoid unrelated refactors while implementing experiments.

## Documentation And Reporting

- Keep design and methodology in `docs/reproduction_plan.md`.
- Keep task completion and measured OOF AUC values in `docs/progress.md`.
- Keep setup and runnable commands in `docs/workflow.md`.
- Keep final course-report source and compiled artifacts under `report/`.
- Do not write temporary or unverified Kaggle leaderboard scores into the plan.
- When reporting an experiment, include the stage, model, feature count, OOF AUC,
  and the comparison baseline needed by the research question.

## Git And PR Guidance

- Keep commits focused and use concise imperative summaries.
- Mention affected stages or docs in commit messages when relevant.
- PRs should describe the change, list produced outputs, include OOF AUC or
  validation results when relevant, and state data/environment assumptions.
- Do not revert unrelated user changes. Work with the current tree.
