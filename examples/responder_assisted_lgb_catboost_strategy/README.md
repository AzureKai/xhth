# Responder-assisted LightGBM/CatBoost Strategy

这个目录用于逐步实现利用 `responder_*` 辅助训练的 LightGBM/CatBoost 策略。

当前第一阶段只分析 responder 与最终 `target` 的关系，不会训练模型，也不会把真实
responder 当作推理特征。正式测试不提供 `responder_*`，后续阶段将使用严格时间
OOF 的 responder 预测进行 stacking。

## 第一阶段：Responder 相关性分析

```bash
python examples/responder_assisted_lgb_catboost_strategy/analyze_responders.py \
  --data-root data \
  --output-dir examples/responder_assisted_lgb_catboost_strategy/analysis
```

Windows PowerShell 单行命令：

```powershell
python examples/responder_assisted_lgb_catboost_strategy/analyze_responders.py --data-root data --output-dir examples/responder_assisted_lgb_catboost_strategy/analysis
```

默认最多抽样 2,000,000 行。小规模冒烟测试：

```powershell
python examples/responder_assisted_lgb_catboost_strategy/analyze_responders.py --data-root data --output-dir examples/responder_assisted_lgb_catboost_strategy/analysis_smoke --max-rows 100000 --time-bins 5
```

输出文件：

- `responder_summary.csv`：综合相关性和稳定性排名；
- `responder_period_correlations.csv`：连续时间段内的加权相关性；
- `responder_time_ic.csv`：每个 `time_id` 的截面 Spearman IC；
- `analysis_report.json`：运行信息和前十名 responder。

`responder_summary.csv` 包含：

- 全局加权 Pearson、普通 Pearson 和 Spearman；
- 按 `time_id` 去均值后的相关性；
- 按 `asset_id` 去均值后的相关性；
- 逐时点截面 Spearman IC 的均值、标准差、ICIR 和方向稳定性；
- 连续时间段相关性的均值、波动、方向稳定性和最近一期相关性；
- 用于初筛的 `screening_score`。

`screening_score` 只用于缩小候选 responder 范围。最终是否采用某个 responder，
仍需在后续阶段通过严格时间 OOF responder 预测和 target 样本外增益判断。

## 第二阶段：Responder OOF Stacking + LightGBM

当前实现使用以下四个 responder：

```text
responder_03, responder_28, responder_29, responder_02
```

训练链路：

1. 按完整 `time_id` 流式读取 Parquet，避免截断资产截面；
2. 将原始 `feature_* + asset_id` 写成磁盘 NumPy shards；
3. 使用 expanding-window 时间折训练 responder LightGBM，生成无泄漏 OOF 预测；
4. 用 OOF `responder_hat` 和原始特征联合训练 target LightGBM；
5. 在验证截止点之前的全部数据上重训四个 responder 模型；
6. 推理时先预测 responder，再预测 target。

LightGBM 通过磁盘分片 `Sequence` 联合构造 Dataset。它不会一次性创建完整的
float32 特征矩阵；模型看到的是所选时间区间内的全部分片，而不是逐分片重复 `fit`。
LightGBM 内部仍会为量化后的 Dataset、梯度和树结构分配内存。

```powershell
python examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model
```

首次运行会建立磁盘缓存，后续运行默认复用。数据或特征定义变化后添加
`--rebuild-cache`。缓存可能占用数十 GB 磁盘空间。

断点恢复并跳过已有模型：

```powershell
python examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --skip-existing-models
```

启用该选项后，程序会复用 `work/oof_models` 中已有的 OOF 折模型以及 `model`
中的最终模型，仅训练缺失文件。如果五个最终模型和 `metadata.json` 全部存在，程序
会在启动时直接输出已有 metadata 并结束。训练过程中会输出缓存分片、OOF 折、
各 responder、分批预测和 target 模型的进度。

本地顺序推理验证：

```powershell
python timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/responder_assisted_lgb_catboost_strategy --output examples/responder_assisted_lgb_catboost_strategy/submission.csv
```
