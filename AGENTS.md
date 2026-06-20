# Repository Guidelines

## Project Structure & Module Organization

This repository is a controlled reproduction of the Kaggle Home Credit Default Risk project. Active code lives in `src/`, with feature builders under `src/features/`. Configuration is in `configs/`, fixed folds are in `data/folds.csv`, and raw Kaggle files belong in ignored `data/original/`. Outputs go to `results/<stage>/`, reports to `report/`, and documentation to `docs/`. The upstream reference solution is a read-only submodule at `open-solution/`; do not edit it for controlled experiments.

## Build, Test, and Development Commands

Use Python 3.9. Conda setup:

```bash
conda env create -f environment.yml
conda run -n credit python -m src.train_cv --help
```

Non-Conda setup is documented in `docs/workflow.md`; after creating `.venv`, use:

```powershell
.\.venv\Scripts\python.exe -m src.train_cv --help
```

Run a baseline stage:

```bash
conda run -n credit python -m src.train_cv --stage s1 --model lightgbm --config configs/base_lgbm.yaml --predict-test
```

With `.venv`, run the same stage as:

```powershell
.\.venv\Scripts\python.exe -m src.train_cv --stage s1 --model lightgbm --config configs/base_lgbm.yaml --predict-test
```

Validate an OOF file with `src.metrics.validate_oof`; see `docs/workflow.md` for the full one-line command.

## Coding Style & Naming Conventions

Use Python with 4-space indentation, type hints where practical, and small functions with explicit data-flow boundaries. Keep stage-specific feature code in `src/features/`, using descriptive module names such as `application_business.py` or `history_basic.py`. Stage names should match the plan: `s1`, `s2`, `b1`, `s3`, `s4`, `b2`, `s5`, `s6`. Do not hard-code local absolute paths; use config files.

## Testing Guidelines

There is no standalone test suite yet. Treat reproducibility checks as required tests: command help must import cleanly, each `oof.parquet` must match `data/folds.csv`, predictions must be finite and in `[0, 1]`, and `feature_names.txt` must match trained columns. New stages must produce `oof.parquet`, `fold_metrics.csv`, `feature_names.txt`, and `config.yaml`.

## Commit & Pull Request Guidelines

Recent commits use concise imperative summaries, for example `Organize project docs and reference data layout`. Keep commits focused and mention affected stages or docs. Pull requests should describe the change, list outputs, include OOF AUC or validation results when relevant, and note data or environment assumptions. Never commit raw Kaggle CSV/ZIP files, `.venv/`, caches, or local credentials.

## Security & Configuration Tips

Kaggle credentials should stay outside the repository, typically in `~/.kaggle/kaggle.json`. Keep `open-solution/` clean as a submodule. If you need to experiment with upstream code, copy ideas into the controlled `src/` framework instead of modifying the submodule.
