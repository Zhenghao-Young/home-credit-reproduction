# Home Credit 受控复现进度

本文档跟踪 [复现计划](reproduction_plan.md) 的实现和报告进度。

## 状态标记

- [ ] 未开始
- [~] 进行中
- [x] 已完成
- [!] 阻塞

## 全局里程碑

- [x] M1：S1 基线使用固定 folds 完整跑通，并能计算 OOF AUC。
- [x] M2：RQ1 完成，S1、S2、B1、S3 结果可比较，RQ1 表图与报告第 3 章已完成。
- [x] M3：RQ2 完成，S4、B2、S5 结果可比较。
- [x] M4：RQ3 完成，简单平均和 Logistic stacking 结果已产出，S6 对比分析已生成。
- [x] M5：最终报告在 `report/` 中编译完成（26 页）。

## Member A：基础设施、S1、S2

- [x] 创建固定 folds：`data/folds.csv`。
- [x] 实现 `src/split.py`。
- [x] 实现 `src/metrics.py`。
- [x] 实现 `src/train_cv.py`。
- [x] 创建 `configs/base_lgbm.yaml`。
- [x] 运行 S1 LightGBM。
- [x] 运行 S2 LightGBM。
- [x] 运行 S2 Logistic Regression。
- [x] 校验每个 OOF 样本恰好出现一次。
- [x] 校验所有阶段使用同一个 `fold_id`。
- [x] 校验 fold-safe groupby 特征不使用验证折统计量。
- [x] 起草报告第 1 章：问题、Open Solution 历史和研究问题。
- [x] 起草报告第 2 章：受控复现设计。

## Member B：RQ1

- [x] 实现 application 业务特征。
- [x] 实现基础历史表聚合。
- [x] 创建 `configs/b1.yaml`。
- [x] 创建 `configs/s3.yaml`。
- [x] 运行 B1。
- [x] 运行 S3。
- [x] 生成 `results/rq1/rq1_results.csv`。
- [x] 生成 `results/rq1/rq1_gain_tree.png`。
- [x] 起草报告第 3 章。

当前已完成结果：

| stage | model | OOF AUC | Kaggle public AUC | Kaggle private AUC | n_features |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 | LightGBM | 0.757646 | 0.74770 | 0.74418 | 78 |
| S2 | LightGBM | 0.757405 | 0.74656 | 0.74471 | 379 |
| S2-full | LightGBM | 0.760329 | 0.74920 | 0.74859 | 801 |
| S2-Logistic | Logistic Regression | 0.743054 | 0.73544 | 0.73049 | 379 |
| B1 | LightGBM | 0.768159 | 0.76716 | 0.76307 | 85 |
| S3 | LightGBM | 0.785109 | 0.78900 | 0.78562 | 126 |
| S4 | LightGBM | 0.786169 | 0.79294 | 0.78666 | 184 |
| B2 | LightGBM | 0.786196 | 0.79164 | 0.78686 | 184 |
| S5 | LightGBM | 0.786342 | 0.79137 | 0.78696 | 191 |
| S6-Avg | Stacking | 0.777088 | — | — | 4 |
| S6-Stack | Stacking | 0.780383 | — | — | 4 |

RQ1 当前关键增量：

- B1 − S1 = +0.010513
- S3 − B1 = +0.016950
- S2 − S1 = -0.000241
- S2-full − S1 = +0.002683（诊断性结果，包含 group-difference 特征）

## Member C：RQ2

- [x] 实现群体相对位置特征。
- [x] 实现最近窗口行为特征。
- [x] 实现聚合前清洗实验变体。
- [x] 实现动态比值和趋势特征。
- [x] 创建 `configs/s4.yaml`。
- [x] 创建 `configs/b2.yaml`。
- [x] 创建 `configs/s5.yaml`。
- [x] 运行 S4。
- [x] 运行 B2。
- [x] 运行 S5。
- [x] 生成 `results/rq2/rq2_results.csv`。
- [x] 生成 `results/rq2/rq2_gain_tree.png`。
- [x] 起草报告第 4 章。

RQ2 当前关键增量：

- S4 − S3 = +0.001060（群体相对位置 + 最近窗口的稳定增益，5 折同向）
- B2 − S4 = +0.000027（聚合前清洗修复几乎零增益，5 折中仅 2 折正）
- S5 − B2 = +0.000146（动态比值与趋势几乎零增益，5 折中仅 3 折正）

结论：RQ2 阶段最大且唯一稳定的增益来自 S4 的群体相对位置和最近窗口特征；
清洗修复（B2）和动态特征（S5）在实际数据上增益微乎其微。

## 官方提交状态

S1、S2、S2-full、S2-Logistic、B1、S3、S4、B2 和 S5 均已提交到 Kaggle
`home-credit-default-risk`，状态均为 `SubmissionStatus.COMPLETE`。官方 public/private AUC
记录在 `results/summary.csv` 的 `kaggle_*` 列中。

## Member D：RQ3 与整合

- [x] 实现 `src/stacking.py`。
- [x] 实现 `src/make_final_figures.py`。
- [x] 构建一级模型 OOF 预测矩阵。
- [x] 计算 OOF 预测相关矩阵。
- [x] 运行简单平均。
- [x] 使用二层 CV 运行 Logistic stacking。
- [x] 生成 `results/rq3/stacking_results.csv`。
- [x] 生成 `results/rq3/meta_coefficients.csv`。
- [x] 生成 `results/rq3/prediction_correlation.png`。
- [x] 生成 `results/rq3/final_evidence_chain.png`。
- [x] 起草报告第 5、6 章。
- [x] 整合并编译 `report/` 中的最终报告（`draft_report.pdf`，26 页）。

RQ3 当前关键发现：

| 方法 | OOF AUC | Δ vs S5 |
| --- | ---: | ---: |
| S5 (最佳单模型) | 0.786342 | — |
| 简单平均 (S2-LR + S3 + S4 + S5) | 0.777088 | -0.009254 |
| L2-Logistic stacking | 0.780383 | -0.005959 |

- S3、S4、S5 的 OOF 预测高度相关（ρ = 0.978–0.991），几乎犯相同错误，互补性极弱。
- S2-LR 与 LightGBM 预测相关性较低（ρ ≈ 0.72），但自身 AUC 太低（0.743），拖低平均值。
- 元系数排序：S5 (3.52) > S4 (2.29) > S3 (1.80) > S2-LR (1.72)，Logistic stacking 已自动给
  最佳模型最高权重，但仍无法超越 S5。

结论：在特征高度同质化的 LightGBM 系列中，stacking 无法产生额外增益；S5 即最终最佳模型。

## 输出约定

每个实验阶段都应写入：

```text
results/<stage>/oof.parquet
results/<stage>/fold_metrics.csv
results/<stage>/feature_names.txt
results/<stage>/config.yaml
```

## 报告目录

`report/` 只放最终课程报告源码和编译产物，例如：

```text
report/draft_report.tex
report/references.bib
report/figures/
report/draft_report.pdf
```

中间实验输出放在 `results/`，不要放进 `report/`。
