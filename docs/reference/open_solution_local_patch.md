# `open-solution-home-credit` 本地兼容补丁记录

当前仓库最初直接拷贝了一份 `open-solution-home-credit/`。这份拷贝并非完全等同于上游
`minerva-ml/open-solution-home-credit` 的 `master` 快照，而是包含少量本地兼容修改。

本项目后续采用干净的 Git submodule：

```text
open-solution -> https://github.com/minerva-ml/open-solution-home-credit.git
```

因此这些兼容修改不再应用到 `open-solution/` 里，而是作为研究记录保存在：

```text
docs/reference/open_solution_local_patch.diff
```

## 补丁内容概览

- 将旧代码里的 `sklearn.externals.joblib` 兼容为独立 `joblib`。
- 给缺失的 `keras` 依赖增加延迟报错，避免导入非神经网络代码时立即失败。
- 增加一个离线 `deepsense.neptune` stub，便于绕过旧 Neptune 实验跟踪依赖。
- 将 `yaml.load` 改为 `yaml.safe_load`。
- 将 `pipeline_config.py` 的 fallback 配置路径改为相对源码文件定位。
- `configs/neptune.yaml` 中曾写入本机绝对数据路径；该路径只作为历史记录保留，不应在当前项目中继续使用。

当前受控复现实验不依赖这些补丁；正式运行入口是父项目 `src/` 下的轻量复现框架。
