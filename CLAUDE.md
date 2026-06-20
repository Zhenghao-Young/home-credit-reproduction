# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A controlled reproduction of the Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition by a 4-person team. The goal is **not** to chase a leaderboard score, but to answer one research question with a verifiable evidence chain:

> As the upstream `minerva-ml/open-solution-home-credit` solution improved from ~0.74 to ~0.80 AUC, how much of the gain came from feature representation vs. dynamic behavior modeling vs. model ensembling?

The work is organized into **stages** (`s1`, `s2`, `b1`, `s3`, `s4`, `s5`, `s6`, `b2`) grouped under three research questions (RQ1–RQ3). Each stage is one controlled experiment that adds a specific block of features (or a different model) on top of a prior stage, and every stage's AUC is compared against a baseline to attribute the gain. Stage definitions live in [docs/reproduction_plan.md](docs/reproduction_plan.md); member ownership and the full workflow are in [docs/workflow.md](docs/workflow.md).

The upstream solution is a **read-only git submodule** at `open-solution/`. Never edit it. Study its 6 tagged versions with read-only git (`git -C open-solution show <tag>:<path>`), and re-implement ideas inside `src/`.

## Environment

Python **3.9** is required. Do not install the pinned `pandas==1.5.3` / `scikit-learn==1.2.2` / `lightgbm==3.3.5` stack on Python 3.14 — wheel/resolution breaks. Two supported setups:

- **Conda** (env name `credit`): `conda env create -f environment.yml`, then prefix commands with `conda run -n credit python ...`
- **`.venv`** (Windows, via `uv`): see [docs/workflow.md](docs/workflow.md) §1 "方案 B". Then prefix commands with `.\.venv\Scripts\python.exe ...`

A `.venv` exists in this repo. In examples below, `PY` stands for either `conda run -n credit python` or `.\.venv\Scripts\python.exe`.

Verify: `PY -c "import pandas, sklearn, lightgbm, pyarrow, yaml; print('env ok')"`

## Data

Raw Kaggle CSVs live in `data/original/` and are **git-ignored** — never commit them, `.venv/`, or caches. The full file list is in [data/README.md](data/README.md). The fixed CV split `data/folds.csv` **is** tracked and is the single source of truth for folds.

## Common commands

Run a stage (writes to `results/<stage>/`):

```bash
PY -m src.train_cv --stage s1 --model lightgbm --config configs/base_lgbm.yaml --predict-test
PY -m src.train_cv --stage s2 --model lightgbm --config configs/base_lgbm.yaml --predict-test
PY -m src.train_cv --stage s2_logistic --model logistic --config configs/base_lgbm.yaml --predict-test
PY -m src.train_cv --stage b1 --model lightgbm --config configs/b1.yaml --predict-test
PY -m src.train_cv --stage s3 --model lightgbm --config configs/s3.yaml --predict-test
```

(Re)generate the fixed folds (rarely — only on a full team reset):

```bash
PY -m src.split --train data/original/application_train.csv --out data/folds.csv
```

Build RQ1 summary table + figure from completed stage outputs:

```bash
PY -m src.make_rq1_artifacts
```

Validate a stage's OOF against the folds (this is the project's de-facto test):

```bash
PY -c "import pandas as pd; from src.split import load_folds; from src.metrics import validate_oof, summarize_oof_auc; folds=load_folds('data/folds.csv'); oof=pd.read_parquet('results/s2/oof.parquet'); validate_oof(oof, folds); print(summarize_oof_auc(oof))"
```

Compile the report: `cd report && pdflatex -interaction=nonstopmode draft_report.tex` (twice).

## Architecture

The whole framework is a single CV runner plus pluggable per-stage feature builders.

- **`src/train_cv.py`** — the orchestrator and the only entry point for experiments. `run_cv()` loads folds + the application table once, then loops over the 5 folds. The critical function is **`_build_fold_features(stage, ...)`**: a big `if stage in {...}` dispatch that assembles `train/valid/test` feature frames *inside each fold* and returns them with an identical `feature_names` ordering. It also enforces model/stage compatibility rules (e.g. S1/B1/S3 are LightGBM-only; `s2_logistic` must use `logistic`). After the loop it concatenates OOF, validates it, writes outputs, and updates `results/summary.csv`.
- **`src/split.py`** — creates/loads/validates `data/folds.csv` (stratified 5-fold, seed 2026). `_validate_folds` rejects folds whose per-fold target rate drifts >0.01 from global.
- **`src/metrics.py`** — shared column-name constants (`SK_ID_CURR`, `TARGET`, `fold_id`, `prediction`), `auc_score`, the fold-metrics schema, and `validate_oof` (the contract every OOF file must satisfy).
- **`src/features/`** — one module per feature block: `application_base.py` (S1 single-table features + cleaning), `application_groupby.py` (`FoldSafeGroupbyAggregateDiffs` for S2/S2-full), `application_business.py` (B1 ratios + EXT_SOURCE), `history_basic.py` (S3 aggregations of the 5 historical tables to `SK_ID_CURR` grain).
- **`configs/*.yaml`** — fixed hyperparameters (LightGBM params are deliberately not tuned). `base_lgbm.yaml`, `b1.yaml`, and `s3.yaml` are currently identical; each stage copies its config into its results dir for provenance.
- **`src/make_rq1_artifacts.py`** — post-hoc analysis: reads completed `results/<stage>/` and produces the RQ1 comparison table and gain-tree figure.

### Fold-safety is the core invariant

This is a leakage-controlled study, so the rules below are not optional:

- `data/folds.csv` is the only fold source; training code must never re-split randomly.
- Every training row appears in OOF **exactly once**; `validate_oof` enforces row count, no duplicate IDs, finite predictions in `[0,1]`, and exact match of `SK_ID_CURR`/`TARGET`/`fold_id` against the folds file.
- Any fold-dependent statistic (group means, encoders, etc.) must be **`fit` on the current fold's `train_df` only**, then `transform`-ed onto the train/valid/test frames separately. See `FoldSafeGroupbyAggregateDiffs` and `OrdinalCategoryEncoder` for the pattern. Validation folds and the test set never participate in any in-fold fit.

### Every stage produces the same outputs

In `results/<stage>/`: `oof.parquet`, `fold_metrics.csv`, `feature_names.txt`, `config.yaml`, and (with `--predict-test`) `submission.csv` (mean of the 5 fold test predictions, reordered to match `sample_submission.csv`). `results/summary.csv` is the cross-stage rollup, upserted per `(stage, model)`.

## Adding a new stage

1. Put feature logic in a new `src/features/<name>.py` module — keep it out of `train_cv.py`. Suggested module names per the plan: `relative_recent.py` (S4), `dynamic.py` (S5), `src/cleaning_v2.py` (B2), `src/stacking.py` (S6).
2. In `src/train_cv.py`: add the stage to `SUPPORTED_STAGES`, wire it into `_build_fold_features`, and ensure `train/valid/test` features share the exact same `feature_names` order.
3. Keep fold-safe stats fold-local (fit on `train_df`, transform each frame).
4. After running, confirm: OOF row count == train rows, no duplicate `SK_ID_CURR`, predictions finite in `[0,1]`, `fold_id` matches `data/folds.csv`, and `feature_names.txt` matches the trained columns.

## Conventions

- 4-space indent, type hints where practical, small functions with explicit data-flow boundaries. No hard-coded absolute paths — read everything from config.
- Stage names must match the plan exactly: `s1`, `s2`, `b1`, `s3`, `s4`, `b2`, `s5`, `s6`.
- Don't write transient Kaggle scores into the plan/docs — those record experiment *design and methodology*, not leaderboard numbers.
- Commits: concise imperative summaries naming affected stages/docs. PRs should list outputs and include OOF AUC / validation results when relevant.
