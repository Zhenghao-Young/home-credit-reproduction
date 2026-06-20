# Home Credit Open Solution 受控复现计划

本文档是本项目的完整实验设计与实现口径。根目录只保留入口说明，具体实验阶段、研究问题、成员分工和报告结构以本文档为准。

## 一、把项目主线收束成一句话

建议将报告题目定为：

## 《Home Credit Open Solution 的渐进复现：性能增益究竟来自特征表示、动态行为还是模型集成？》

报告不再横向比较一大堆模型，而是沿着开源解答的六个版本向前走：
$$
\text{单表原始特征}
\rightarrow
\text{群组聚合}
\rightarrow
\text{多表历史特征}
\rightarrow
\text{更聪明的相对/近期特征}
\rightarrow
\text{清洗一致性与动态特征}
\rightarrow
\text{Stacking}
$$
你们的任务不是简单报告“第六版比第一版高”，而是把这条开发历史整理成一条**可验证的证据链**：

> 每向前一步，究竟增加了什么信息或统计结构？对应的 AUC 增益有多大？这种解释是否能通过受控对比得到支持？

开源仓库依次给出了 Chestnut、Seedling、Blossom、Tulip、Sunflower 和 Four Leaf Clover 六套方案，公开记录的成绩从 LB 0.742 逐步提高到 0.806；其中第三至第五版仍以单个 LightGBM 为主，第六版才引入模型与特征多样性的 stacking。

------

# 二、最终只保留三个 RQ

## RQ1：从 Chestnut 到 Blossom 的大幅提升，主要来自哪里？

具体拆成三个候选来源：

1. 普通的 groupby 聚合；
2. 有业务含义的比例、交互特征；
3. bureau、previous application、installments 等历史表带来的额外信息。

对应原方案的前三级：

- Chestnut：application 表中的基础特征 + LightGBM；
- Seedling：增加 groupby 特征，并提供 XGBoost、随机森林、Logistic、SVC 等多种模型；
- Blossom：转向多张历史表上的手工特征和统计聚合，仍使用 LightGBM。

------

## RQ2：从 Blossom 到 Sunflower 的继续提升，来自什么更精细的信息组织？

候选来源是：

1. 相对同类人群的位置，例如“个人值减去所属群体均值”；
2. 最近几次贷款或还款行为；
3. 异常值是否在聚合之前被正确清洗；
4. 短期行为相对于长期行为的变化和趋势。

Tulip 加入了群体均值差异、最近若干期分期还款、最近历史申请等更细致特征；Sunflower 又修复了“手工特征清洗了但聚合原表没有清洗”的问题，并加入短期/长期比值、POS 逾期趋势等动态特征。

------

## RQ3：从 Sunflower 到 Four Leaf Clover，Stacking 为什么还能提高？提升是否值得？

这里不再研究更多特征，而研究预测之间的互补性：

- 不同特征版本的 LightGBM 是否犯不同的错误？
- Logistic 与 LightGBM 是否提供了不同的信息？
- 简单平均是否已经足够？
- stacking 的增益是否仅有很小一截？

原方案第六版使用 Logistic Regression、神经网络以及多个特征子集上的 LightGBM 生成 OOF 预测，再进行 stacking，CV 从约 0.7950 提升到 0.7975。

------

# 三、不要原样跑六套旧代码，而要做“受控复现”

原六个版本不只是特征不同，模型参数、清洗方式、特征数量也同时变化。直接比较六个历史分数，无法确定增益来自哪里。

因此建议在一个现代化代码框架中实现：

- S1 至 S5 使用**完全相同的五折划分**；
- S1 至 S5 使用**同一套 LightGBM 参数**；
- 不做大规模调参；
- 每个阶段只打开新的 feature block；
- 所有阶段保存 OOF 预测；
- S6 只使用 OOF 预测训练二层模型。

推荐固定配置：

```
StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

LightGBM:
objective = binary
metric = auc
learning_rate = 0.03
num_leaves = 31
min_child_samples = 70
feature_fraction = 0.20
reg_lambda = 100
n_estimators = 3000
early_stopping_rounds = 100
```

参数本身不是研究对象。固定住模型，才能让 AUC 变化主要反映新增特征的作用。

------

# 四、最终实验矩阵：六个主阶段，加两个桥接实验

你们实际只需要完成下面 **8 个特征版本 + 1 个轻量级 stacking**。

| 编号  | 对应版本             | 特征内容                                | 作用          |
| --- | ---------------- | ----------------------------------- | ----------- |
| S1  | Chestnut         | application 基础数值与类别特征               | 最低基线        |
| S2  | Seedling         | S1 + application 内部 groupby 聚合原值    | 测量普通聚合的价值   |
| B1  | 桥接实验             | S1 + 业务比例与 `EXT_SOURCE` 汇总          | 单独测量业务特征价值  |
| S3  | Blossom          | B1 + 五类历史表的基础聚合                     | 测量历史数据价值    |
| S4  | Tulip            | S3 + 群体差异、最近 $k$ 期行为                | 测量相对位置与近期行为 |
| B2  | 桥接实验             | S4 + 聚合前异常清洗，但不加动态比值                | 单独测量清洗修复价值  |
| S5  | Sunflower        | B2 + 短期/长期比值与趋势特征                   | 测量动态变化价值    |
| S6  | Four Leaf Clover | S2-Logistic、S3、S4、S5 的 OOF stacking | 测量预测互补价值    |

其中 B1、B2 是整份报告中最重要的设计。它们像两块楔子，把原本纠缠在一起的改动拆开。

------

## 4.0 统一实现口径

为了让“受控复现”真正可比，所有阶段遵守下面的实现规则。

### 数据与样本

- 训练样本以 `application_train.csv` 中的 `SK_ID_CURR` 为准；
- 测试样本仅用于生成预测，不参与 OOF AUC；
- 所有辅助表特征最终都聚合到 `SK_ID_CURR` 粒度；
- 同一阶段训练集与验证集必须使用完全相同的特征列顺序；
- 任何阶段新增特征后产生的 `inf`、`-inf` 统一转为缺失值。

### 折内处理

- `data/folds.csv` 是唯一折划分来源；
- 每个阶段只读取 `fold_id`，不得重新随机切分；
- 数值缺失填补、类别编码、标准化、groupby 统计都只能在当前训练折上 `fit`；
- 验证折和测试集只能调用训练折上已经 `fit` 好的转换器；
- 如果验证折出现训练折没有见过的类别或 group，映射为缺失值，再由该特征的训练折全局均值填补。

### 类别变量

- LightGBM 阶段使用 ordinal encoding；
- Logistic 阶段使用 one-hot encoding；
- 编码器只在当前训练折上拟合；
- 类别缺失统一填为字符串 `__MISSING__`；
- 验证折或测试集未见类别统一映射为 `__UNKNOWN__`；
- 不使用 target encoding。

### 数值变量

- LightGBM 保留缺失值，由模型处理；
- Logistic 使用训练折中位数填补，并做标准化；
- 安全除法统一定义为：分母为 0 或缺失时，结果为缺失；
- 比例类特征计算后统一把非有限值转为缺失。

### 输出

每个阶段保存：

```
results/<stage>/oof.parquet
results/<stage>/fold_metrics.csv
results/<stage>/feature_names.txt
results/<stage>/config.yaml
```

`oof.parquet` 至少包含：

```
SK_ID_CURR
TARGET
fold_id
prediction
```

`fold_metrics.csv` 至少包含：

```
stage
model
fold_id
auc
n_train
n_valid
n_features
```

------

## 4.1 S1：Chestnut

使用：

- application 表原始数值变量；
- application 表原始类别变量；
- 明显异常编码转为缺失，例如 `DAYS_EMPLOYED=365243`。

具体实现口径：

- 数值列使用原项目 `pipeline_config.py` 中的 `NUMERICAL_COLUMNS`；
- 类别列使用原项目 `pipeline_config.py` 中的 `CATEGORICAL_COLUMNS`；
- 不使用 `TARGET`、`SK_ID_CURR` 作为模型特征；
- `CODE_GENDER='XNA'` 转为缺失；
- `NAME_FAMILY_STATUS='Unknown'` 转为缺失；
- `ORGANIZATION_TYPE='XNA'` 转为缺失；
- `DAYS_EMPLOYED=365243` 转为缺失；
- `DAYS_LAST_PHONE_CHANGE=0` 转为缺失；
- S1 不生成任何手工比例、外部评分汇总、groupby 或历史表特征。

不加入：

- 比例特征；
- groupby；
- 历史表；
- 动态窗口。

输出：

```
results/s1/oof.parquet
results/s1/fold_metrics.csv
results/s1/feature_names.txt
results/s1/config.yaml
```

------

## 4.2 S2：Seedling

在 S1 基础上增加 application 表内部的 fold-safe groupby 聚合原值。

正式 S2 使用当前仓库中的 `FULL_GROUPBY_SPECS`，共新增 301 个 group 聚合特征；加上 S1 的 78 个基础特征，S2-LightGBM 总特征数为 379。

聚合列固定为：

```
AMT_CREDIT
AMT_ANNUITY
AMT_INCOME_TOTAL
AMT_GOODS_PRICE
EXT_SOURCE_1
EXT_SOURCE_2
EXT_SOURCE_3
OWN_CAR_AGE
REGION_POPULATION_RELATIVE
DAYS_REGISTRATION
CNT_CHILDREN
CNT_FAM_MEMBERS
DAYS_ID_PUBLISH
DAYS_BIRTH
DAYS_EMPLOYED
```

基础聚合函数固定为：

```
min
mean
max
sum
var
median
```

基础 group key 固定为：

```
NAME_EDUCATION_TYPE × CODE_GENDER
NAME_FAMILY_STATUS × NAME_EDUCATION_TYPE
NAME_FAMILY_STATUS × CODE_GENDER
```

同时保留旧 Seedling/Tulip 中少量更具体的 application group 聚合原值：

```
CODE_GENDER × ORGANIZATION_TYPE:
    AMT_ANNUITY mean
    AMT_INCOME_TOTAL mean
    DAYS_REGISTRATION mean
    EXT_SOURCE_1 mean

CODE_GENDER × REG_CITY_NOT_WORK_CITY:
    AMT_ANNUITY mean
    CNT_CHILDREN mean
    DAYS_ID_PUBLISH mean

CODE_GENDER × NAME_EDUCATION_TYPE × OCCUPATION_TYPE × REG_CITY_NOT_WORK_CITY:
    EXT_SOURCE_1 mean
    EXT_SOURCE_2 mean

NAME_EDUCATION_TYPE × OCCUPATION_TYPE:
    AMT_CREDIT mean
    AMT_REQ_CREDIT_BUREAU_YEAR mean
    APARTMENTS_AVG mean
    BASEMENTAREA_AVG mean
    EXT_SOURCE_1 mean
    EXT_SOURCE_2 mean
    EXT_SOURCE_3 mean
    NONLIVINGAREA_AVG mean
    OWN_CAR_AGE mean
    YEARS_BUILD_AVG mean

NAME_EDUCATION_TYPE × OCCUPATION_TYPE × REG_CITY_NOT_WORK_CITY:
    ELEVATORS_AVG mean
    EXT_SOURCE_1 mean

OCCUPATION_TYPE:
    AMT_ANNUITY mean
    CNT_CHILDREN mean
    CNT_FAM_MEMBERS mean
    DAYS_BIRTH mean
    DAYS_EMPLOYED mean
    DAYS_ID_PUBLISH mean
    DAYS_REGISTRATION mean
    EXT_SOURCE_1 mean
    EXT_SOURCE_2 mean
    EXT_SOURCE_3 mean
```

具体实现口径：

- S2 只保留 group 聚合原值，不生成个人值减聚合值的 diff 特征；
- S2 不生成 `abs diff` 特征；
- diff 和 abs diff 留到 S4 的群体相对位置实验；
- groupby 统计只基于 application 表；
- group key 中的缺失类别先填为 `__MISSING__` 再分组；
- 每一折内，只在训练折 `fit` groupby 聚合表，再对训练折、验证折和测试集分别 `transform`；
- 验证折或测试集未见 group 使用该聚合特征在训练折上的全局均值填补；
- groupby 不能使用 `TARGET`，也不能使用验证折或测试集样本参与统计。

同时在同一套 S2 特征上运行一次 $L_2$-Logistic Regression：

- 不作为主结果；
- 用来体现 Seedling 的“模型扩展”思想；
- 其 OOF 预测留给 S6 stacking。

Logistic 的实现口径：

- 使用与 S2-LightGBM 相同的原始特征和 groupby 特征；
- 数值列按训练折中位数填补；
- 类别列 one-hot 后，未见类别落入 `__UNKNOWN__`；
- 连续特征标准化只在训练折拟合；
- 正则使用 $L_2$，不做调参；
- sklearn 求解器固定为 `newton-cholesky`，`max_iter=1000`，`class_weight="balanced"`。

------

## 4.3 B1：application 业务特征

只使用 application 表，但加入 Blossom 中最核心的比例：
$$
\frac{\text{AMT\_ANNUITY}}{\text{AMT\_INCOME\_TOTAL}},
\quad
\frac{\text{AMT\_CREDIT}}{\text{AMT\_INCOME\_TOTAL}},
\quad
\frac{\text{AMT\_CREDIT}}{\text{AMT\_ANNUITY}},
$$
以及：

```
EXT_SOURCE_1/2/3 的 mean、min、max、std
```

这些特征正是 Blossom 文档中明确列出的主要 application 工程特征。

具体实现口径：

- B1 = S1 + 上述 3 个比例 + 4 个 `EXT_SOURCE` 汇总；
- 不加入旧项目 `ApplicationFeatures` 中的其他比例，例如 car、children、phone、ID renewal 等；
- `EXT_SOURCE` 汇总只在 `EXT_SOURCE_1`、`EXT_SOURCE_2`、`EXT_SOURCE_3` 三列上计算；
- `mean`、`min`、`max` 跳过缺失值；
- `std` 使用总体标准差 `ddof=0`，三列全缺失时结果为缺失；
- 分母为 0 或缺失时，比例结果为缺失。

------

## 4.4 S3：Blossom

在 B1 上增加五类历史信息，每张表只保留少量核心聚合，不复刻全部数百个特征。

### Bureau

```
贷款记录数
活跃贷款比例
AMT_CREDIT_SUM：mean、max、sum
AMT_CREDIT_SUM_DEBT：mean、max、sum
AMT_CREDIT_SUM_OVERDUE：mean、max
DAYS_CREDIT：mean、min
```

### Previous application

```
历史申请数
批准比例
AMT_APPLICATION：mean、max
AMT_CREDIT：mean、max
DAYS_DECISION：mean、min
CNT_PAYMENT：mean、max
```

### Installments

```
还款记录数
平均/最大逾期天数
逾期比例
平均少还金额
平均多还金额
```

### POS CASH

```
记录数
SK_DPD：mean、max
SK_DPD_DEF：mean、max
发生逾期的比例
```

### Credit card

```
记录数
AMT_BALANCE：mean、max
AMT_CREDIT_LIMIT_ACTUAL：mean
余额/额度比值的 mean、max
SK_DPD：mean、max
```

Blossom 本身就是以多表手工特征和申请人级统计聚合为核心，并获得了 CV 0.784、LB 0.790。

具体实现口径：

- S3 = B1 + 五张历史表的基础聚合；
- 每张历史表先在全量历史记录上按 `SK_ID_CURR` 聚合，再左连接回 application 主表；
- 不在 S3 中加入最近窗口、趋势、短期/长期比值；
- 记录数使用每个表的行数或唯一 `SK_ID_PREV` 数，需在 `feature_names.txt` 中明确；
- 聚合缺失保留为缺失，不把“没有历史记录”和数值 0 混为一类。

各表派生特征定义如下：

- Bureau：
    - 贷款记录数 = `SK_ID_BUREAU` 计数；
    - 活跃贷款比例 = `CREDIT_ACTIVE != 'Closed'` 的均值；
    - 其余金额和日期字段按表中原值做 `mean`、`max`、`sum`、`min`。
- Previous application：
    - 历史申请数 = `SK_ID_PREV` 计数；
    - 批准比例 = `NAME_CONTRACT_STATUS == 'Approved'` 的均值；
    - 其余字段按原值聚合。
- Installments：
    - 逾期天数 = `max(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)`；
    - 逾期比例 = 逾期天数大于 0 的均值；
    - 少还金额 = `max(AMT_INSTALMENT - AMT_PAYMENT, 0)`；
    - 多还金额 = `max(AMT_PAYMENT - AMT_INSTALMENT, 0)`；
    - 平均少还金额和平均多还金额按所有还款记录求均值。
- POS CASH：
    - 发生逾期的比例 = `SK_DPD > 0` 的均值；
    - `SK_DPD` 和 `SK_DPD_DEF` 按原值聚合。
- Credit card：
    - 余额/额度比值 = `AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL`；
    - 额度为 0 或缺失时该比值为缺失。

------

## 4.5 S4：Tulip

在 S3 上只加入两类“聪明特征”。

### A. 群体相对位置

对于收入、信用额和 `EXT_SOURCE`：
$$
x_i-\overline{x}_{g(i)},
\qquad
|x_i-\overline{x}_{g(i)}|,
$$
其中群体 $g(i)$ 可以使用：

```
OCCUPATION_TYPE
NAME_EDUCATION_TYPE × OCCUPATION_TYPE
CODE_GENDER × NAME_EDUCATION_TYPE
```

### B. 最近行为

只选择统一窗口：

```
最近 1、3、5 笔 previous application
最近 10、50 条 installments
最近 10、50 个月 POS CASH
```

计算：

```
平均逾期天数
逾期比例
平均少还金额
平均 SK_DPD
最大 SK_DPD
```

这样足以复现 Tulip 的核心思想，不必照搬所有列。

具体实现口径：

- S4 = S3 + 群体相对位置 + 最近行为；
- 群体相对位置的 group mean 与 S2 一样必须 fold-safe；
- 相对位置只生成：
    - `x - group_mean(x)`；
    - `abs(x - group_mean(x))`；
- 不额外保留新的 group mean 原值，避免和 S2 重复解释。

相对位置中的 `x` 固定为：

```
AMT_INCOME_TOTAL
AMT_CREDIT
EXT_SOURCE_1
EXT_SOURCE_2
EXT_SOURCE_3
```

最近行为的排序和窗口固定为：

- previous application：
    - 按 `DAYS_DECISION` 从旧到新排序；
    - 最近申请使用每个 `SK_ID_CURR` 的最后 1、3、5 笔；
    - 计算最近窗口中的批准比例、`AMT_APPLICATION` 均值、`AMT_CREDIT` 均值、`CNT_PAYMENT` 均值。
- installments：
    - 按 `DAYS_INSTALMENT` 从旧到新排序；
    - 最近记录使用每个 `SK_ID_CURR` 的最后 10、50 条；
    - 计算平均逾期天数、逾期比例、平均少还金额、平均多还金额。
- POS CASH：
    - 以 `MONTHS_BALANCE` 越接近 0 表示越近期；
    - 最近 10、50 个月分别定义为 `MONTHS_BALANCE >= -10` 和 `MONTHS_BALANCE >= -50`；
    - 计算 `SK_DPD` 均值、`SK_DPD` 最大值、`SK_DPD_DEF` 均值、逾期比例。

------

## 4.6 B2：清洗修复实验

B2 与 S4 使用相同特征，只改变处理顺序：

### S4 方式

```
原表
→ 直接聚合
```

### B2 方式

```
原表
→ 异常编码转缺失
→ 再做聚合
```

重点处理：

```
DAYS_* 中 365243
bureau 中极端负日期
credit card 中负的提款金额
```

这一步专门验证 Sunflower 文档提到的修复：之前清洗只作用于手工特征，没有作用于聚合所使用的原表。

具体实现口径：

- B2 = S4 的全部特征，但历史表聚合和最近窗口在清洗后的原表上计算；
- 除清洗顺序外，B2 不新增任何特征；
- B2 与 S4 的 `feature_names.txt` 应完全一致。

清洗规则固定为：

- application：
    - 沿用 S1 的 application 清洗；
- previous application：
    - `DAYS_FIRST_DRAWING=365243` 转缺失；
    - `DAYS_FIRST_DUE=365243` 转缺失；
    - `DAYS_LAST_DUE_1ST_VERSION=365243` 转缺失；
    - `DAYS_LAST_DUE=365243` 转缺失；
    - `DAYS_TERMINATION=365243` 转缺失；
- bureau：
    - `DAYS_CREDIT_ENDDATE < -40000` 转缺失；
    - `DAYS_CREDIT_UPDATE < -40000` 转缺失；
    - `DAYS_ENDDATE_FACT < -40000` 转缺失；
- credit card：
    - `AMT_DRAWINGS_ATM_CURRENT < 0` 转缺失；
    - `AMT_DRAWINGS_CURRENT < 0` 转缺失。

------

## 4.7 S5：Sunflower

在 B2 上加入动态特征。

### 短期/长期比值

$$
R_{10/50}
=
\frac{\text{最近 10 期逾期率}}
{\text{最近 50 期逾期率}+\varepsilon},
$$

### 趋势

对最近 12、30、60 个月的 `SK_DPD` 拟合简单线性趋势：
$$
\text{SK\_DPD}_t=\alpha+\beta t+\varepsilon_t,
$$
保存斜率 $\widehat\beta$。

正斜率意味着逾期程度随时间恶化，负斜率意味着近期改善。Sunflower 文档明确使用了短期/长期特征比值和多个窗口上的 DPD 趋势。

具体实现口径：

- S5 = B2 + 动态比值 + 趋势特征；
- 不再加入新的静态聚合；
- 比值分母加 $\varepsilon=10^{-6}$；
- 如果短期或长期窗口没有记录，该比值为缺失。

动态比值固定为：

- POS CASH：
    - 最近 10 个月逾期比例 / 最近 50 个月逾期比例；
    - 最近 10 个月 `SK_DPD` 均值 / 最近 50 个月 `SK_DPD` 均值。
- Installments：
    - 最近 10 条逾期比例 / 最近 50 条逾期比例；
    - 最近 10 条平均逾期天数 / 最近 50 条平均逾期天数。

趋势特征固定为：

- 只在 POS CASH 上计算 `SK_DPD` 趋势；
- 窗口为最近 12、30、60 个月；
- 每个窗口内按 `MONTHS_BALANCE` 从旧到新排序；
- 令 $t=0,1,\ldots,n-1$，对排序后的 `SK_DPD` 拟合一元线性回归；
- 少于 2 条记录时趋势为缺失；
- 在这个定义下，正斜率表示越近期 `SK_DPD` 越高，即风险恶化。

------

## 4.8 S6：Four Leaf Clover

使用四列一级模型 OOF 预测：

```
p_lr_s2
p_lgb_s3
p_lgb_s4
p_lgb_s5
```

比较三种结果：

1. 最佳单模型 S5；
2. 四个预测的简单平均；
3. $L_2$-Logistic stacking。

二层模型仍需做一次五折交叉验证：

```
OOF 预测矩阵 Z
→ 二层 5-fold CV
→ 生成 stack 的二层 OOF 预测
```

不能在全部 OOF 矩阵上训练元模型，再用同一矩阵评价，否则二层结果仍然乐观。

具体实现口径：

- S6 只读取已经保存的一级 OOF，不重新训练一级模型；
- 二层 CV 复用 `data/folds.csv` 的同一 `fold_id`；
- 二层训练折只使用该训练折内的一级 OOF 预测训练元模型；
- 二层验证折只用于生成 stack OOF；
- 简单平均直接对四列一级 OOF 做行均值；
- Logistic stacking 使用 $L_2$ 正则；
- 元模型输入只包含四列一级预测，不加入原始特征；
- 最终汇报 S5、简单平均、Logistic stacking 三个 OOF AUC。

------

# 五、三个 RQ 到底怎样回答

定义：
$$
\Delta_{A\to B}
=
\operatorname{AUC}_{\mathrm{OOF}}(B)
-
\operatorname{AUC}_{\mathrm{OOF}}(A).
$$
同时记录五个折上的配对差值。不要做复杂显著性检验，只使用一个预先规定的描述性标准：

> 平均增益为正，并且至少 4 个折方向一致，称为“稳定增益”；否则称为“不稳定或证据不足”。

------

## RQ1 的回答逻辑

| 对比                    | 解释                       |
| --------------------- | ------------------------ |
| S2 − S1               | 普通群组聚合的价值                |
| B1 − S1               | 业务比例和 `EXT_SOURCE` 汇总的价值 |
| S3 − B1               | 多张历史表的价值                 |
| S2-LGBM − S2-Logistic | 同一 S2 特征空间下非线性树模型的额外价值 |

当前 RQ1 已完成，结论按最大稳定增益写：

> S1 至 S3 的主要提升来自多表历史聚合，因为 S3 − B1 贡献了最大的稳定
> OOF AUC 增益（+0.016950，五折全部同向）；业务比例和 `EXT_SOURCE`
> 汇总也带来稳定且高效的增益（B1 − S1 = +0.010513），相比之下，普通
> application 群组聚合原值单独几乎没有贡献（S2 − S1 = -0.000241）。

RQ1 只需一张图：

```
S1
├── S2：application group aggregations
└── B1：business ratios
      └── S3：historical tables
```

图中在每条箭头上标注 $\Delta AUC$。

------

## RQ2 的回答逻辑

| 对比    | 解释                      |
| ------- | ------------------------- |
| S4 − S3 | 群体差异和近期行为的价值  |
| B2 − S4 | 聚合前正确清洗的价值      |
| S5 − B2 | 短期/长期比值及趋势的价值 |

最终结论模板：

> Blossom 之后的提升并非简单来自“特征更多”。S4 的增益说明 ______；B2 的结果说明数据清洗 ______；S5 的进一步增益表明模型能够从 ______ 中获得信息。

本章再加一张解释图即可：

- 将某个“最近 10 期 / 最近 50 期逾期率”按十分位分组；
- 画出每组的实际违约率；
- 观察动态恶化是否与风险单调相关。

------

## RQ3 的回答逻辑

先计算一级模型 OOF 预测相关矩阵：
$$
\rho_{jk}
=
\operatorname{Corr}
\left(
\widehat p_j,\widehat p_k
\right).
$$
再比较：

| 方法                | OOF AUC |
| ----------------- | ------- |
| S5 最佳单模型          |         |
| 简单平均              |         |
| Logistic stacking |         |

最后看元模型系数。

结论按实际结果二选一：

### stacking 明显提升

> 尽管部分模型单独表现较弱，但其 OOF 预测与 S5 相关性较低，元模型赋予其非零权重，说明互补误差提供了额外信息。

### stacking 提升很小或没有提升

> 各强模型的 OOF 预测高度相关，说明它们主要依赖相似的风险信号。复杂集成只能榨出很小的尾部收益，项目的主要增益仍来自 S1 至 S5 的特征表示演化。

第二种结论同样是优秀结果，不需要强行让 stacking 获胜。

------

# 六、报告章节结构

## 第 1 章：问题、开源方案与研究问题

### 需要做的工作

- 简述违约预测任务和 AUC；
- 画出数据关系图；
- 整理六个历史版本及原始成绩；
- 提出总问题和三个 RQ。

### 必须汇报

- 图 1：六阶段演化路线；
- 表 1：六个原方案的核心变化与历史成绩；
- 三个 RQ；
- 一句话假设：

> 我们预期主要性能增益来自多表关系型特征和动态行为表示，而 stacking 仅提供较小的边际提升。

------

## 第 2 章：受控复现设计

### 需要做的工作

- 固定五折；
- 固定 LightGBM；
- 定义 S1 至 S6、B1、B2；
- 说明 OOF；
- 说明 fold-safe groupby；
- 说明没有进行大规模调参。

### 必须汇报

- 表 2：八个特征版本的开关；
- 数据处理流程图；
- 评价指标：

$$
\text{OOF AUC},\quad
\text{五折 AUC 均值和标准差},\quad
\Delta AUC.
$$

不要在本章铺开大量 EDA。只保留类别比例、表结构和缺失情况。

------

## 第 3 章：RQ1，基础特征如何演化为关系型表示

### 需要运行

```
S1
S2-LightGBM
S2-Logistic
B1
S3
```

### 必须汇报

- S1、S2、B1、S3 的 OOF AUC；
- 三个增量；
- 一张分支式增益图；
- 2 至 3 个代表性特征的业务解释。

### 本章最后必须明确回答

> 前三级中最大的增益究竟来自 groupby、业务比例还是多表历史。

------

## 第 4 章：RQ2，动态行为为什么比简单聚合更有效

### 需要运行

```
S3
S4
B2
S5
```

### 必须汇报

- 三个配对增量；
- 各版本特征数量；
- 一个动态特征的十分位风险图；
- 清洗顺序是否真的改变结果。

### 本章最后必须明确回答

> 后期增益究竟来自近期行为、数据清洗还是动态变化。

------

## 第 5 章：RQ3，Stacking 的收益来自哪里

### 需要运行

```
简单平均
Logistic stacking
```

### 必须汇报

- 一级模型 OOF 相关矩阵；
- S5、平均、stacking 的 AUC；
- 元模型系数；
- stacking 的增益与额外复杂度。

### 本章最后必须明确回答

> stacking 是否利用了互补误差，以及其边际收益是否值得。

------

## 第 6 章：总证据链与课程理解

只放一张总图：
$$
S1
\rightarrow
S2
\rightarrow
S3
\rightarrow
S4
\rightarrow
S5
\rightarrow
S6
$$
每条箭头标出：

```
新增机制
ΔOOF AUC
是否稳定
```

然后总结三点课程认识：

1. **特征表示决定了模型能够看到什么。**
    把变长历史转化为申请人级聚合，是本题最重要的建模步骤。
2. **模型复杂度不仅是树的深度，也是特征空间的复杂度。**
    越来越细的近期、趋势和群体差异特征扩大了可学习的函数空间。
3. **集成的价值取决于误差相关性。**
    多个高分但高度相似的模型，不一定构成有效的 ensemble。

局限性只写三条：

- 这是核心机制复现，不是 1092 个特征的逐列复刻；
- 固定模型参数有利于归因，但不代表每个阶段都达到自身最优；
- CV 增益只能支持关联性解释，不能证明业务因果关系。

------

# 七、四人分工，直接对应章节和实验

| 成员 | 负责实验            | 负责章节              | 具体交付物                                          |
| ---- | ------------------- | --------------------- | --------------------------------------------------- |
| A    | S1、S2、S2-Logistic | 第 1、2 章            | `folds.csv`、训练框架、S1/S2 OOF、六阶段总表        |
| B    | B1、S3              | 第 3 章 RQ1           | application 业务特征、历史表基础聚合、RQ1 图表      |
| C    | S4、B2、S5          | 第 4 章 RQ2           | 群体差异、最近窗口、清洗修复、趋势特征、RQ2 图表    |
| D    | S6                  | 第 5、6 章 RQ3 与总结 | OOF 相关矩阵、平均与 stacking、总证据链图、全文整合 |

------

## 成员 A：公共基础设施

必须最先完成：

```
src/split.py
src/train_cv.py
src/metrics.py
configs/base_lgbm.yaml
data/folds.csv
```

统一规定每个实验输出：

```
results/<stage>/oof.parquet
results/<stage>/fold_metrics.csv
results/<stage>/feature_names.txt
results/<stage>/config.yaml
```

A 还负责检查：

- 所有阶段使用同一个 `fold_id`；
- OOF 每个申请人恰好出现一次；
- groupby 不读取验证折统计量；
- S6 不使用训练内预测。

------

## 成员 B：RQ1

负责文件：

```
src/features/application_business.py
src/features/history_basic.py
configs/b1.yaml
configs/s3.yaml
```

交付：

```
results/rq1/rq1_results.csv
results/rq1/rq1_gain_tree.png
```

B 的工作完成标准不是“特征全部写完”，而是能够计算：
$$
\Delta_{S1\to S2},
\quad
\Delta_{S1\to B1},
\quad
\Delta_{B1\to S3}.
$$

------

## 成员 C：RQ2

负责文件：

```
src/features/relative_recent.py
src/features/dynamic.py
src/cleaning_v2.py
configs/s4.yaml
configs/b2.yaml
configs/s5.yaml
```

交付：

```
results/rq2/rq2_results.csv
results/rq2/dynamic_feature_decile.png
```

C 必须保证 B2 和 S5 只有一个差异：

```
S5 = B2 + dynamic ratios/trends
```

否则无法解释 S5 的增益。

------

## 成员 D：RQ3 与全文整合

负责文件：

```
src/stacking.py
src/make_final_figures.py
```

输入：

```
results/s2_logistic/oof.parquet
results/s3/oof.parquet
results/s4/oof.parquet
results/s5/oof.parquet
```

交付：

```
results/rq3/prediction_correlation.png
results/rq3/stacking_results.csv
results/rq3/meta_coefficients.csv
results/rq3/final_evidence_chain.png
```

D 负责统一所有章节中的：

- 模型名称；
- 小数位数；
- AUC 计算方式；
- 图形字体；
- 表格格式。

------

# 八、推荐推进顺序

## 里程碑 1：先跑通 S1

完成条件：

- 数据成功读取；
- 固定折生成；
- OOF AUC 能计算；
- 结果能自动保存。

没有 S1，不允许四个人各自开荒。

## 里程碑 2：完成 RQ1

A 和 B 联合完成 S1、S2、B1、S3。

此时报告已经有一条完整主结论，即使后续时间紧，也能形成合格项目。

## 里程碑 3：完成 RQ2

C 接入相同 pipeline，完成 S4、B2、S5。

此时单模型主线结束，冻结特征，不再继续增加新花样。

## 里程碑 4：完成 RQ3

D 读取已经保存的 OOF 文件完成 stacking，不重新训练前面的模型。

------

# 九、明确删除的内容

为了确保做得完，正文不再包含：

- LDA、QDA、Naive Bayes；
- 完整模型动物园；
- 100 次随机超参数搜索；
- 1092 个特征的逐列精确复刻；
- 完整 SHAP 分析；
- 概率校准；
- 公平性分析；
- 多套 stacking 元模型；
- 频繁提交 Kaggle 榜单。

这些内容都可能有价值，但会让主线重新长成一片藤蔓林。

最终工作量控制为：
$$
7\text{ 个 LightGBM 特征版本}
+
1\text{ 个 Logistic}
+
1\text{ 个轻量 stacking}.
$$
整份报告最终只需要回答一句核心结论：

> **Open Solution 从 0.74 左右走向 0.80 左右的过程，最主要的动力究竟是数据表示的升级、动态行为的刻画，还是模型集成？**

这条主线既保留了六套开源方案的历史脉络，又能通过 B1、B2 两个桥接实验把“版本更新日志”变成真正的统计学习证据链。
