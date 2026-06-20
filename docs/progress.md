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
- [ ] M3：RQ2 完成，S4、B2、S5 结果可比较。
- [ ] M4：RQ3 完成，包含简单平均和 Logistic stacking。
- [~] M5：最终报告在 `report/` 中编译完成。

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

| stage | model | OOF AUC | n_features |
| --- | --- | --- | --- |
| S1 | LightGBM | 0.757646 | 78 |
| S2 | LightGBM | 0.757405 | 379 |
| S2-full | LightGBM | 0.760329 | 801 |
| S2-Logistic | Logistic Regression | 0.743054 | 379 |
| B1 | LightGBM | 0.768159 | 85 |
| S3 | LightGBM | 0.785109 | 126 |

RQ1 当前关键增量：

- B1 − S1 = +0.010513
- S3 − B1 = +0.016950
- S2 − S1 = -0.000241
- S2-full − S1 = +0.002683（诊断性结果，包含 group-difference 特征）

## Member C：RQ2

- [ ] 实现群体相对位置特征。
- [ ] 实现最近窗口行为特征。
- [ ] 实现聚合前清洗实验变体。
- [ ] 实现动态比值和趋势特征。
- [ ] 创建 `configs/s4.yaml`。
- [ ] 创建 `configs/b2.yaml`。
- [ ] 创建 `configs/s5.yaml`。
- [ ] 运行 S4。
- [ ] 运行 B2。
- [ ] 运行 S5。
- [ ] 生成 `results/rq2/rq2_results.csv`。
- [ ] 生成 `results/rq2/dynamic_feature_decile.png`。
- [ ] 起草报告第 4 章。

## Member D：RQ3 与整合

- [ ] 实现 `src/stacking.py`。
- [ ] 实现 `src/make_final_figures.py`。
- [ ] 构建一级模型 OOF 预测矩阵。
- [ ] 计算 OOF 预测相关矩阵。
- [ ] 运行简单平均。
- [ ] 使用二层 CV 运行 Logistic stacking。
- [ ] 生成 `results/rq3/stacking_results.csv`。
- [ ] 生成 `results/rq3/meta_coefficients.csv`。
- [ ] 生成 `results/rq3/prediction_correlation.png`。
- [ ] 生成 `results/rq3/final_evidence_chain.png`。
- [ ] 起草报告第 5、6 章。
- [ ] 整合并编译 `report/` 中的最终报告。

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
