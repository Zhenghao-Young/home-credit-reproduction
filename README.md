# Home Credit Default Risk 受控复现

本仓库用于 4 人小组合作研究和复现 Kaggle
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
竞赛。我们参考
[`minerva-ml/open-solution-home-credit`](https://github.com/minerva-ml/open-solution-home-credit)
的六个历史版本，但不直接运行旧仓库主流程，而是在当前 `src/` 下实现一套轻量、可控、便于归因分析的复现框架。

核心问题：

> Open Solution 从约 0.74 提升到约 0.80 的过程，主要动力是特征表示、动态行为刻画，还是模型集成？

## 现在有什么

当前已完成公共实验骨架、S1/S2 基础阶段，以及 RQ1 中的 B1/S3 阶段：

| 模块 | 当前状态 |
| --- | --- |
| 数据 | 原始 Kaggle 数据放在 `data/original/`，该目录已被 Git 忽略；固定折划分为 `data/folds.csv`。 |
| 代码 | `src/` 下已有固定折、指标校验、训练入口、S1/S2 application 特征、B1 业务特征和 S3 历史表基础聚合。 |
| 结果 | 已有 `results/s1/`、`results/s2/`、`results/s2_full/`、`results/s2_logistic/`、`results/b1/`、`results/s3/` 和 `results/rq1/`，项目汇总见 `results/summary.csv`。 |
| 参考实现 | `open-solution/` 是上游解答 submodule，只读参考，不在其中改代码。 |
| 报告 | `report/draft_report.tex` 和 `report/draft_report.pdf` 已包含第 1--3 章草稿，其中第 3 章完成 RQ1 分析。 |

旧的 `home-credit-default-risk/` 数据占位目录已废弃；当前默认数据入口是 `data/original/`。

## 先读哪份文档

| 你想做什么 | 读这里 |
| --- | --- |
| 了解 Kaggle 数据文件和表关系 | [data/README.md](data/README.md) |
| 理解完整实验设计、RQ 和成员分工 | [docs/reproduction_plan.md](docs/reproduction_plan.md) |
| 配环境、准备数据、运行实验、校验输出 | [docs/workflow.md](docs/workflow.md) |
| 查看当前任务进度 | [docs/progress.md](docs/progress.md) |
| 研究上游 6 个 solution tag | [docs/open_solution_guide.md](docs/open_solution_guide.md) |
| 查看旧直接拷贝目录中的本地补丁记录 | [docs/reference/open_solution_local_patch.md](docs/reference/open_solution_local_patch.md) |

## 最短上手

### 1. 初始化 submodule

```bash
git submodule update --init --recursive
```

### 2. 确认数据

原始 CSV 和 ZIP 不提交到 Git。请确认本地有：

```text
data/original/application_train.csv
data/original/application_test.csv
data/original/sample_submission.csv
```

完整文件清单见 [data/README.md](data/README.md)。

### 3. 准备 Python 环境

推荐 Python 3.9。不要直接用 Python 3.14 安装本项目的旧版 `pandas`、`scikit-learn`
和 `lightgbm` 组合。

Conda 路线：

```bash
conda env create -f environment.yml
conda env update -n credit -f environment.yml --prune
```

不用 Conda 的路线：

```powershell
py -3.14 -m pip install --user uv
& "$env:APPDATA\Python\Python314\Scripts\uv.exe" python install 3.9
& "$env:APPDATA\Python\Python314\Scripts\uv.exe" venv --python 3.9 .venv
& "$env:APPDATA\Python\Python314\Scripts\uv.exe" pip install --python .venv\Scripts\python.exe -r requirements.txt
```

上面是能运行当前 `src/` 框架的核心依赖。若要补齐 `environment.yml` 中的 XGBoost、CatBoost、Plotly 等扩展包，按 [docs/workflow.md](docs/workflow.md) 的“方案 B”继续安装。

### 4. 验证环境

Conda：

```bash
conda run -n credit python -c "import pandas, sklearn, lightgbm, pyarrow, yaml; print('env ok')"
conda run -n credit python -m src.train_cv --help
```

`.venv`：

```powershell
.\.venv\Scripts\python.exe -c "import pandas, sklearn, lightgbm, pyarrow, yaml; print('env ok')"
.\.venv\Scripts\python.exe -m src.train_cv --help
```

### 5. 跑一个基线阶段

Conda：

```bash
conda run -n credit python -m src.train_cv --stage s1 --model lightgbm --config configs/base_lgbm.yaml --predict-test
```

`.venv`：

```powershell
.\.venv\Scripts\python.exe -m src.train_cv --stage s1 --model lightgbm --config configs/base_lgbm.yaml --predict-test
```

输出会写到：

```text
results/s1/oof.parquet
results/s1/fold_metrics.csv
results/s1/feature_names.txt
results/s1/config.yaml
results/s1/submission.csv
```

## 实验规则

- `data/folds.csv` 是唯一 fold 来源，不允许训练阶段重新随机划分。
- OOF 中每个训练样本必须恰好出现一次。
- 验证折和测试集不能参与任何 fold 内统计量的 fit。
- LightGBM 使用 `configs/base_lgbm.yaml` 中的固定参数，不做大规模调参。
- 新增 stage 的结果写入 `results/<stage>/`。
- 临时 Kaggle 分数不要写入复现计划；计划只记录实验设计和实现口径。

更完整的运行、校验和成员协作流程见 [docs/workflow.md](docs/workflow.md)。

## 目录结构

```text
configs/                 受控复现实验配置
data/
  original/              本地 Kaggle 原始数据，已被 Git 忽略
  folds.csv              固定五折划分，会被 Git 跟踪
docs/                    项目说明、复现计划、进度和参考资料
open-solution/           上游开源解答 Git submodule，只读参考
report/                  课程报告 tex/pdf
results/                 已完成阶段的 OOF、submission 和指标
src/                     当前项目自己的轻量复现代码
```
