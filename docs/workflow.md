# 项目工作流指南

本文档给小组成员提供统一的上手路线：先理解数据和复现计划，再运行已有阶段，最后按分工接入新的 feature block。

## 1. 环境准备

本项目推荐 Python 3.9。不要直接用 Python 3.14 安装旧版 `pandas`、`scikit-learn`
和 `lightgbm` 组合，依赖解析和 wheel 兼容性都更容易出问题。

### 方案 A：Conda

默认 Conda 环境名是 `credit`：

```bash
conda env create -f environment.yml
```

如果环境已经存在，用下面命令更新：

```bash
conda env update -n credit -f environment.yml --prune
```

`requirements.txt` 是轻量 pip 依赖清单，主要用于核对核心运行库。使用 Conda 时，一般不需要在环境外单独安装。

环境验证：

```bash
conda run -n credit python -c "import pandas, sklearn, lightgbm, pyarrow, yaml; print('env ok')"
```

### 方案 B：不用 Conda，使用 `.venv`

Windows PowerShell 推荐用 `uv` 管理 Python 版本和虚拟环境。如果本机还没有 `uv`：

```powershell
py -3.14 -m pip install --user uv
```

如果本机没有 Python 3.9，让 `uv` 下载一份项目专用 Python：

```powershell
& "$env:APPDATA\Python\Python314\Scripts\uv.exe" python install 3.9
```

创建 `.venv`：

```powershell
& "$env:APPDATA\Python\Python314\Scripts\uv.exe" venv --python 3.9 .venv
```

安装依赖：

```powershell
& "$env:APPDATA\Python\Python314\Scripts\uv.exe" pip install --python .venv\Scripts\python.exe -r requirements.txt
& "$env:APPDATA\Python\Python314\Scripts\uv.exe" pip install --python .venv\Scripts\python.exe xgboost==1.7.6 catboost==1.2.5 category_encoders==2.6.3 hyperopt==0.2.7 scikit-optimize==0.9.0 plotly==6.3.0 click==8.1.7 ipython==8.18.1 pydot-ng==2.0.0 graphviz "setuptools==80.9.0" pip
```

`setuptools==80.9.0` 是为了兼容 `hyperopt==0.2.7` 对 `pkg_resources` 的旧式导入。若用更新
`setuptools`，可能只在导入 `hyperopt` 时才暴露问题。

如果你本机已经有 Python 3.9，也可以不用 `uv venv`：

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install xgboost==1.7.6 catboost==1.2.5 category_encoders==2.6.3 hyperopt==0.2.7 scikit-optimize==0.9.0 plotly==6.3.0 click==8.1.7 ipython==8.18.1 pydot-ng==2.0.0 graphviz "setuptools==80.9.0"
```

验证 `.venv`：

```powershell
.\.venv\Scripts\python.exe -c "import pandas, sklearn, lightgbm, pyarrow, yaml; print('env ok')"
```

当前关键版本：

```text
python=3.9
pandas=1.5.3
scikit-learn=1.2.2
lightgbm=3.3.5
pyarrow=21.0.0
kaggle=1.7.4.5
```

## 2. 数据准备

原始 Kaggle 文件统一放在：

```text
data/original/
```

需要包含：

```text
application_train.csv
application_test.csv
sample_submission.csv
bureau.csv
bureau_balance.csv
previous_application.csv
installments_payments.csv
POS_CASH_balance.csv
credit_card_balance.csv
HomeCredit_columns_description.csv
```

这些 CSV 和 ZIP 已被 `.gitignore` 排除，不应提交到远程仓库。字段和表关系说明见 [../data/README.md](../data/README.md)。

固定折划分已经生成：

```text
data/folds.csv
```

后续实验默认复用它。除非小组共同决定重置全部实验，否则不要重新切分。

## 3. 开源解答的使用方式

上游开源解答通过 submodule 放在：

```text
open-solution/
```

它用于阅读和研究，不作为本项目主运行框架。查看 6 个历史版本时，优先使用只读 Git 命令：

```bash
git -C open-solution show solution-4:src/feature_extraction.py
git -C open-solution diff --stat solution-4 solution-5
git -C open-solution for-each-ref refs/tags --sort=creatordate --format="%(refname:short) %(objectname:short) %(subject)"
```

不要直接修改 `open-solution/` 内文件。上游代码的阅读路线见 [open_solution_guide.md](open_solution_guide.md)。

## 4. 运行已有阶段

查看命令入口：

```bash
conda run -n credit python -m src.train_cv --help
```

不用 Conda 时：

```powershell
.\.venv\Scripts\python.exe -m src.train_cv --help
```

S1：application 主表基础特征：

```bash
conda run -n credit python -m src.train_cv \
  --stage s1 \
  --model lightgbm \
  --config configs/base_lgbm.yaml \
  --predict-test
```

S2：application 内部 groupby 聚合特征：

```bash
conda run -n credit python -m src.train_cv \
  --stage s2 \
  --model lightgbm \
  --config configs/base_lgbm.yaml \
  --predict-test
```

S2-Logistic：同一套 S2 特征上的 Logistic Regression：

```bash
conda run -n credit python -m src.train_cv \
  --stage s2_logistic \
  --model logistic \
  --config configs/base_lgbm.yaml \
  --predict-test
```

B1：application 业务比例和 `EXT_SOURCE` 汇总：

```bash
conda run -n credit python -m src.train_cv \
  --stage b1 \
  --model lightgbm \
  --config configs/b1.yaml \
  --predict-test
```

S3：B1 加五张历史表基础聚合：

```bash
conda run -n credit python -m src.train_cv \
  --stage s3 \
  --model lightgbm \
  --config configs/s3.yaml \
  --predict-test
```

不用 Conda 时，把上述命令开头的 `conda run -n credit python` 替换为
`.\.venv\Scripts\python.exe`，其余参数保持不变。

每个阶段输出：

```text
results/<stage>/oof.parquet
results/<stage>/fold_metrics.csv
results/<stage>/feature_names.txt
results/<stage>/config.yaml
```

如果带 `--predict-test`，还会输出：

```text
results/<stage>/submission.csv
```

## 5. 新阶段接入规则

新增特征模块建议放在 `src/features/` 下，不要把大量逻辑塞进 `src/train_cv.py`。

推荐文件：

```text
src/features/application_business.py   # B1
src/features/history_basic.py          # S3
src/features/relative_recent.py        # S4
src/features/dynamic.py                # S5
src/cleaning_v2.py                     # B2
src/stacking.py                        # S6
```

新增 stage 时，在 `src/train_cv.py` 中做三件事：

1. 把 stage 名加入 `SUPPORTED_STAGES`。
2. 在 `_build_fold_features(...)` 中接入对应特征构建逻辑。
3. 保证 `train_features`、`valid_features`、`test_features` 使用完全相同的 `feature_names` 顺序。

如果某个 stage 需要 fold-safe 统计量，应在当前 fold 的 `train_df` 上 `fit`，再分别 `transform` 训练折、验证折和测试集。

## 6. 结果校验

快速查看已完成结果汇总：

```bash
cat results/summary.csv
```

校验某个 OOF 文件：

```bash
conda run -n credit python -c "import pandas as pd; from src.split import load_folds; from src.metrics import validate_oof, summarize_oof_auc; folds=load_folds('data/folds.csv'); oof=pd.read_parquet('results/s2/oof.parquet'); validate_oof(oof, folds); print(summarize_oof_auc(oof))"
```

不用 Conda 时：

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; from src.split import load_folds; from src.metrics import validate_oof, summarize_oof_auc; folds=load_folds('data/folds.csv'); oof=pd.read_parquet('results/s2/oof.parquet'); validate_oof(oof, folds); print(summarize_oof_auc(oof))"
```

每个新 stage 完成后都应检查：

- `oof.parquet` 行数等于训练集行数。
- `SK_ID_CURR` 没有重复。
- `prediction` 没有缺失，且在 `[0, 1]` 内。
- `fold_id` 与 `data/folds.csv` 一致。
- `feature_names.txt` 与代码中实际训练列一致。

## 7. 四人分工

| 成员 | 负责实验 | 负责章节 | 主要交付物 |
| --- | --- | --- | --- |
| A | S1、S2、S2-Logistic | 第 1、2 章 | `folds.csv`、训练框架、S1/S2 OOF、六阶段总表 |
| B | B1、S3 | 第 3 章 RQ1 | application 业务特征、历史表基础聚合、RQ1 图表 |
| C | S4、B2、S5 | 第 4 章 RQ2 | 群体差异、最近窗口、清洗修复、趋势特征、RQ2 图表 |
| D | S6 | 第 5、6 章 RQ3 与总结 | OOF 相关矩阵、平均与 stacking、总证据链图、全文整合 |

详细阶段定义和解释口径见 [reproduction_plan.md](reproduction_plan.md)。

## 8. 报告编译

当前草稿：

```text
report/draft_report.tex
report/draft_report.pdf
```

编译命令：

```bash
cd report
pdflatex -interaction=nonstopmode draft_report.tex
pdflatex -interaction=nonstopmode draft_report.tex
```

`report/` 目录只保留最终 tex、pdf 和必要图表。编译产生的 `.aux`、`.log`、`.out` 等辅助文件不要提交。
