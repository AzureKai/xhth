# V3 Entity Residual LightGBM

该目录现在只保留有效的个体项，不再训练或推理市场共同项：

```text
entity_target(t, i)
  = (target(t, i) - mean_i(target(t, i)))
  - (V3(t, i) - mean_i(V3(t, i)))

prediction
  = V3 + beta * centered_entity_prediction
```

entity prediction 在每个 `time_id` 内强制去均值，因此只调整资产之间的相对预测，不改变 V3 的市场共同水平。

## 特征

模型保留原实验中表现稳定的 16 个 entity-state feature：

- 当前 raw feature 和 V3 prediction。
- 使用历史状态形成的 entity z20。
- entity EMA20/EMA60 gap。
- 当前时刻的 cross-sectional z-score。
- entity 历史长度和 categorical `asset_id`。

另外只为 V3 排序中的下一组 8 个 feature 增加 cross-sectional z-score，默认是：

```text
feature_284 feature_039 feature_045 feature_148
feature_014 feature_096 feature_287 feature_263
```

这 8 个 feature 不生成 EMA 或 entity history，额外外存只有 8 列。

## 无泄漏验证

1. 复用 V3 的三种子严格 OOF、5 折 expanding walk-forward、30 时间步 purge 和 15% terminal holdout。
2. entity CV 的第 2～5 块只能使用更早 OOF 块训练，第 1 块只作 warmup。
3. `beta` 只从预注册 development OOF 网格选择。
4. terminal holdout 不参与权重选择。
5. 四个 entity CV 折和 holdout 必须全部正增益，否则部署权重自动归零并退化为原始 V3。

## 训练

该策略复用 `lgb_v3_regime_residual_strategy/work` 中的 `base_oof_prediction.npy` 和 `residual_features.npy`。一行训练命令：

```bash
python3 examples/lgb_v3_market_entity_strategy/train.py --data-root data --base-model-dir examples/lightgbm_baseline/model_forward_lowrisk_v3 --base-cache-dir examples/lightgbm_baseline/.low_memory_cache_forward_v2 --source-work-dir examples/lgb_v3_regime_residual_strategy/work --work-dir examples/lgb_v3_market_entity_strategy/work --model-dir examples/lgb_v3_market_entity_strategy/model --state-feature-count 16 --extra-cross-z-count 8 --entity-weights 0,0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.0 --entity-rounds 500 --threads 8 --skip-existing-models
```

训练成功后立即执行推理的一行命令：

```bash
python3 examples/lgb_v3_market_entity_strategy/train.py --data-root data --base-model-dir examples/lightgbm_baseline/model_forward_lowrisk_v3 --base-cache-dir examples/lightgbm_baseline/.low_memory_cache_forward_v2 --source-work-dir examples/lgb_v3_regime_residual_strategy/work --work-dir examples/lgb_v3_market_entity_strategy/work --model-dir examples/lgb_v3_market_entity_strategy/model --state-feature-count 16 --extra-cross-z-count 8 --entity-weights 0,0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.0 --entity-rounds 500 --threads 8 --skip-existing-models && python3 timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/lgb_v3_market_entity_strategy --output examples/lgb_v3_market_entity_strategy/submission.csv
```

旧版市场/个体模型的 metadata schema 不兼容，首次运行会自动重训；`--skip-existing-models` 仍会复用签名兼容的新增 cross-z 外存和 entity 折模型。

主要产物：

- `model/metadata.json`：权重搜索、逐折结果、holdout 和晋级门。
- `model/entity_feature_importance.csv`：开发期 entity 模型的重要性。
- `model/entity_seed*.txt`：晋级后保存的三种子部署模型。

## 推理

推理默认让 V3 和 entity LightGBM 对每个小型横截面使用单线程，以降低 OpenMP 调度开销：

```bash
python3 timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/lgb_v3_market_entity_strategy --output examples/lgb_v3_market_entity_strategy/submission.csv
```

可分别通过 `LIGHTGBM_BASELINE_PREDICT_THREADS` 和 `LIGHTGBM_ENTITY_PREDICT_THREADS` 覆盖线程数。

## Asset ID 时间安全消融

`run_entity_identity_ablation.py` 使用与生产模型完全相同的 purged walk-forward OOF 和 terminal holdout，对比三组：

- `full`：当前 categorical `asset_id` 与全部实体状态；直接复用生产报告，不重复训练。
- `no_asset_id`：仅删除 `asset_id`，保留 raw、横截面和因果实体 feature 状态。
- `frozen_prior`：删除 `asset_id`，加入收缩型历史实体残差先验。训练行的 prior 只能读取更早时间的标签；进入 validation/holdout 后状态被冻结，因此没有验证标签泄漏，也符合部署时只能使用训练期标签的约束。

完整的一行训练、消融和推理命令如下。消融只输出审计报告，不会自动替换生产模型，所以最后的推理仍使用当前通过晋级门的 `full` 模型：

```bash
python3 examples/lgb_v3_market_entity_strategy/train.py --data-root data --base-model-dir examples/lightgbm_baseline/model_forward_lowrisk_v3 --base-cache-dir examples/lightgbm_baseline/.low_memory_cache_forward_v2 --source-work-dir examples/lgb_v3_regime_residual_strategy/work --work-dir examples/lgb_v3_market_entity_strategy/work --model-dir examples/lgb_v3_market_entity_strategy/model --state-feature-count 16 --extra-cross-z-count 8 --entity-weights 0,0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.0 --entity-rounds 500 --threads 8 --skip-existing-models && python3 examples/lgb_v3_market_entity_strategy/run_entity_identity_ablation.py --data-root data --base-model-dir examples/lightgbm_baseline/model_forward_lowrisk_v3 --base-cache-dir examples/lightgbm_baseline/.low_memory_cache_forward_v2 --source-work-dir examples/lgb_v3_regime_residual_strategy/work --work-dir examples/lgb_v3_market_entity_strategy/work --model-dir examples/lgb_v3_market_entity_strategy/model --prior-shrinkage 100 --entity-rounds 500 --threads 8 --skip-existing-models && python3 timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/lgb_v3_market_entity_strategy --output examples/lgb_v3_market_entity_strategy/submission.csv
```

主要消融产物位于 `work/entity_identity_ablation/`：

- `summary.json` 和 `summary.csv`：三组 CV、逐折、holdout 以及相对 full 的差值。
- `<variant>/report.json`：单组完整权重搜索与诊断。
- `<variant>/feature_importance.csv`：该组 terminal holdout 模型的重要性。

`--skip-existing-models` 会复用签名完全一致的折模型和 prior 外存；修改 prior 收缩强度或 feature schema 后会自动失效重算。
