# Home Credit Controlled Reproduction Handoff

本仓库用于课程项目的 Home Credit Open Solution 受控复现。当前 Member A 已完成公共实验骨架、固定折划分、S1、S2、S2-full 诊断实验、S2-Logistic，以及第 1、2 章 draft。

后续成员不要从旧仓库主流程继续改。`open-solution-home-credit/` 只作为字段、清洗和特征逻辑参考；正式实验应接入当前 `src/` 下的轻量受控复现框架。

## 当前状态

已完成的核心文件：

```text
configs/base_lgbm.yaml
data/folds.csv
src/split.py
src/train_cv.py
src/metrics.py
src/features/application_base.py
src/features/application_groupby.py
results/member_a_summary.csv
report/draft_report.tex
report/draft_report.pdf
```

当前结果目录：

```text
results/s1/
results/s2/
results/s2_full/
results/s2_logistic/
```

`results/` 需要上传远程仓库，用于同步 OOF、submission 和 fold metrics 给其他成员。原始数据 CSV 不应上传。

## 数据目录

仓库保留一个空的 `home-credit-default-risk/` 目录作为数据放置位置。运行实验前，需要把 Kaggle 原始 CSV 文件放到该目录下：

```text
home-credit-default-risk/application_train.csv
home-credit-default-risk/application_test.csv
home-credit-default-risk/sample_submission.csv
home-credit-default-risk/bureau.csv
home-credit-default-risk/bureau_balance.csv
home-credit-default-risk/previous_application.csv
home-credit-default-risk/installments_payments.csv
home-credit-default-risk/POS_CASH_balance.csv
home-credit-default-risk/credit_card_balance.csv
home-credit-default-risk/HomeCredit_columns_description.csv
```

这些 CSV 是本地数据文件，已被 `.gitignore` 排除，不会上传到远程仓库。只有 `home-credit-default-risk/.gitkeep` 会被提交，用来保留空目录结构。

## 环境配置

默认环境名是 `credit`。优先使用 `environment.yml` 创建环境：

```bash
conda env create -f environment.yml
```

如果环境已经存在，用下面命令更新：

```bash
conda env update -n credit -f environment.yml --prune
```

`requirements.txt` 是轻量 pip 依赖清单，主要用于核对核心运行库；正常情况下不需要在 conda 环境外单独安装。若确实需要补 pip 依赖：

```bash
conda run -n credit pip install -r requirements.txt
```

环境验证：

```bash
conda run -n credit python -c "import pandas, sklearn, lightgbm, pyarrow, yaml; print('env ok')"
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

Kaggle CLI 已加入环境。若成员需要提交 Kaggle，需要本机有认证文件：

```text
~/.kaggle/kaggle.json
```

权限应为：

```bash
chmod 600 ~/.kaggle/kaggle.json
```

验证 Kaggle CLI：

```bash
conda run -n credit kaggle competitions submissions -c home-credit-default-risk
```

如果只做本地 OOF 实验，不需要 Kaggle 认证。

## 基本命令

默认使用 `credit` 环境：

```bash
conda run -n credit python ...
```

固定折已经生成，后续不要重新切分：

```bash
data/folds.csv
```

如需重新生成固定折，只能使用同一命令和同一随机种子：

```bash
conda run -n credit python -m src.split \
  --train home-credit-default-risk/application_train.csv \
  --out data/folds.csv
```

已完成阶段的标准训练命令：

```bash
conda run -n credit python -m src.train_cv \
  --stage s1 \
  --model lightgbm \
  --config configs/base_lgbm.yaml \
  --predict-test

conda run -n credit python -m src.train_cv \
  --stage s2 \
  --model lightgbm \
  --config configs/base_lgbm.yaml \
  --predict-test

conda run -n credit python -m src.train_cv \
  --stage s2_logistic \
  --model logistic \
  --config configs/base_lgbm.yaml \
  --predict-test
```

每个阶段必须输出：

```text
results/<stage>/oof.parquet
results/<stage>/fold_metrics.csv
results/<stage>/feature_names.txt
results/<stage>/config.yaml
```

如果生成测试集预测，还应输出：

```text
results/<stage>/submission.csv
```

## 统一实验规则

所有成员都必须遵守这些约束：

- `data/folds.csv` 是唯一 fold 来源，不允许在训练阶段重新随机划分。
- OOF 中每个训练样本必须恰好出现一次。
- 验证折和测试集不能参与任何 fold 内统计量的 fit。
- LightGBM 使用 `configs/base_lgbm.yaml` 中的固定参数，不做大规模调参。
- 新增 stage 的结果必须写入 `results/<stage>/`。
- 新增 stage 需要更新 `results/member_a_summary.csv` 或生成对应成员的 summary，再交给 D 汇总。
- `plan.md` 只写实验设计和实现口径，不写临时测试分数。
- 报告结果写入 `report/draft_report.tex` 或后续正式 report tex。

## 进度维护

项目进度统一记录在：

```text
progress.md
```

状态标记：

```text
[ ] Not started
[~] In progress
[x] Done
[!] Blocked
```

每个成员在开始或完成自己的实验后，都应更新对应条目。推荐规则：

- 开始写代码或跑实验时，把对应任务从 `[ ]` 改为 `[~]`。
- OOF、fold metrics、feature names 和 config 全部生成并校验后，再改为 `[x]`。
- 如果缺数据、环境或上游结果导致无法继续，标为 `[!]`，并在条目后写清楚阻塞原因。
- 不要只更新 README；真正的进度状态以 `progress.md` 为准。

当前核对后的状态：

- M1 已完成。
- M2、M3、M4 尚未完成。
- M5 为进行中；`report/draft_report.tex` 和 `report/draft_report.pdf` 已包含 Chapter 1 和 Chapter 2 draft。
- Member A 的基础设施、S1、S2、S2-Logistic 已完成。

## 代码接入方式

新增特征模块建议放在 `src/features/` 下。不要把大量逻辑塞进 `train_cv.py`。

建议模式：

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
3. 保证返回的 `train_features`、`valid_features`、`test_features` 使用完全相同的 `feature_names` 顺序。

如果某个 stage 需要 fold-safe 统计量，应在当前 fold 的 `train_df` 上 `fit`，再分别 transform `train_df`、`valid_df` 和 `test_df`。

## Member B 工作流

Member B 负责 RQ1 的 B1 和 S3。

建议新增文件：

```text
src/features/application_business.py
src/features/history_basic.py
configs/b1.yaml
configs/s3.yaml
```

B1 目标：

- 在 S1 基础上加入 application 业务比例。
- 加入 `EXT_SOURCE_1/2/3` 的 mean、min、max、std。
- 不加入历史表。
- 不加入 group relative diff。

S3 目标：

- 在 B1 基础上加入 bureau、previous application、installments、POS CASH、credit card 的基础申请人级聚合。
- 所有历史表最终都聚合到 `SK_ID_CURR`。
- 不加入最近窗口、趋势、短期/长期比值。

推荐验证：

```bash
conda run -n credit python -m src.train_cv \
  --stage b1 \
  --model lightgbm \
  --config configs/base_lgbm.yaml \
  --predict-test

conda run -n credit python -m src.train_cv \
  --stage s3 \
  --model lightgbm \
  --config configs/base_lgbm.yaml \
  --predict-test
```

交付物：

```text
results/b1/
results/s3/
rq1_results.csv
rq1_gain_tree.png
```

RQ1 至少需要比较：

```text
S2 - S1
B1 - S1
S3 - B1
```

## Member C 工作流

Member C 负责 RQ2 的 S4、B2、S5。

建议新增文件：

```text
src/features/relative_recent.py
src/features/dynamic.py
src/cleaning_v2.py
configs/s4.yaml
configs/b2.yaml
configs/s5.yaml
```

S4 目标：

- 在 S3 基础上加入群体相对位置特征。
- 加入 previous application、installments、POS CASH 的最近窗口特征。
- 群体相对位置只生成 `x - group_mean(x)` 和 `abs(x - group_mean(x))`。
- 不额外保留新的 group mean 原值，避免和 S2 解释重叠。

B2 目标：

- 使用与 S4 完全相同的特征名。
- 唯一区别是历史表先做异常值清洗，再聚合。
- B2 与 S4 的 `feature_names.txt` 应保持一致。

S5 目标：

- 在 B2 基础上加入动态特征。
- 只加入短期/长期比值和 POS CASH 趋势。
- 不再加入新的静态聚合。

推荐验证：

```bash
conda run -n credit python -m src.train_cv \
  --stage s4 \
  --model lightgbm \
  --config configs/base_lgbm.yaml \
  --predict-test

conda run -n credit python -m src.train_cv \
  --stage b2 \
  --model lightgbm \
  --config configs/base_lgbm.yaml \
  --predict-test

conda run -n credit python -m src.train_cv \
  --stage s5 \
  --model lightgbm \
  --config configs/base_lgbm.yaml \
  --predict-test
```

交付物：

```text
results/s4/
results/b2/
results/s5/
rq2_results.csv
dynamic_feature_decile.png
```

RQ2 至少需要比较：

```text
S4 - S3
B2 - S4
S5 - B2
```

## Member D 工作流

Member D 负责 S6 stacking、最终图表和全文整合。

建议新增文件：

```text
src/stacking.py
src/make_final_figures.py
```

S6 输入：

```text
results/s2_logistic/oof.parquet
results/s3/oof.parquet
results/s4/oof.parquet
results/s5/oof.parquet
```

S6 规则：

- 只读取一级模型 OOF，不重新训练一级模型。
- 二层 Logistic stacking 也必须做 fold-safe 训练。
- 简单平均可直接对一级 OOF 预测做行均值。
- 需要输出 OOF 相关矩阵、stacking 结果和元模型系数。

交付物：

```text
prediction_correlation.png
stacking_results.csv
meta_coefficients.csv
final_evidence_chain.png
```

D 还需要统一：

- 表格小数位数。
- 图形字体与风格。
- stage 名称。
- AUC 计算方式。
- report tex 中的章节衔接。

## 结果读取与校验

读取当前结果总表：

```bash
cat results/member_a_summary.csv
```

快速校验某个 OOF 文件：

```bash
conda run -n credit python -c "import pandas as pd; from src.split import load_folds; from src.metrics import validate_oof, summarize_oof_auc; folds=load_folds('data/folds.csv'); oof=pd.read_parquet('results/s2/oof.parquet'); validate_oof(oof, folds); print(summarize_oof_auc(oof))"
```

每个新 stage 完成后都应检查：

- `oof.parquet` 行数等于训练集行数。
- `SK_ID_CURR` 没有重复。
- `prediction` 没有缺失，且在 `[0, 1]` 内。
- `fold_id` 与 `data/folds.csv` 一致。
- `feature_names.txt` 与代码中实际训练列一致。

## 报告文件

当前 draft：

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

`report/` 目录只保留最终 tex 和 pdf；编译产生的 `.aux`、`.log`、`.out` 等辅助文件不要提交。

## 当前注意事项

- S2 是唯一正式 group aggregation stage，使用 379 个特征。
- `s2_full` 是诊断实验，用于说明更接近旧方案的 group relative features 有额外信号，不替代 S2。
- `s2_logistic` 已对齐到同一套 S2 379 特征。
- Logistic 训练使用 `newton-cholesky`，运行时可能因病态 Hessian 回退到 `lbfgs` 并出现收敛警告；目前 OOF 和 submission 已正常生成。
- 后续实验不应把临时 Kaggle 分数写入 `plan.md`。
