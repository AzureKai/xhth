# V3 市场项 / 个体项分解 LightGBM

该策略冻结 `lightgbm_baseline` low-risk V3，并把严格时间外 residual 拆成两个互斥部分：

```text
market_residual(t) = mean(target_t) - mean(V3_t)
entity_residual(t, i) = (target_t,i - mean(target_t)) - (V3_t,i - mean(V3_t))

prediction = V3
           + alpha * predicted_market_residual
           + beta  * centered_predicted_entity_residual
```

测试接口不会提供 `weight`，所以市场项使用当前 `time_id` 内的无权均值；个体修正也会在每个推理时刻强制去均值。这样市场模型只改变整组资产的共同水平，个体模型只改变资产之间的相对差异，二者不会重复解释同一部分。

## 输入

策略复用 `lgb_v3_regime_residual_strategy/work` 中已经生成的严格三种子 V3 OOF 和因果 entity-state 外存矩阵，不重复训练基础 V3：

- `base_oof_prediction.npy`
- `base_oof_stage.json`
- `residual_features.npy`
- `residual_feature_stage.json`

市场模型读取当前时刻可观测的横截面聚合量：V3 预测分布、原始 feature 的均值和离散度、entity z-score、EMA20/60 gap、历史长度和连续市场状态。个体模型读取 `asset_id`、当前原始 feature、V3 预测及因果 entity state，但不读取市场 regime 列。

## 无泄漏验证

1. 完全复用 V3 的 5 折 expanding walk-forward、30 时间步 purge 和 15% terminal holdout。
2. 第 2～5 个 OOF 块的市场/个体模型只能使用更早 OOF 块训练；第 1 块仅作 warmup。
3. `alpha`、`beta` 只从预注册网格中按 development OOF 均值选择。
4. terminal holdout 不参与权重选择。
5. 只有四个 component CV 折及 terminal holdout 全部正增益时才保存并部署分解模型；否则推理自动退化为原始 V3。

## 训练

先保证原 residual 策略的 `work/` 产物仍在，然后运行一行命令：

```bash
python3 examples/lgb_v3_market_entity_strategy/train.py --data-root data --base-model-dir examples/lightgbm_baseline/model_forward_lowrisk_v3 --base-cache-dir examples/lightgbm_baseline/.low_memory_cache_forward_v2 --source-work-dir examples/lgb_v3_regime_residual_strategy/work --work-dir examples/lgb_v3_market_entity_strategy/work --model-dir examples/lgb_v3_market_entity_strategy/model --state-feature-count 16 --component-weights 0,0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.0 --market-rounds 500 --entity-rounds 500 --threads 8 --skip-existing-models
```

训练会显示分解物化、市场/个体 component CV、terminal holdout 和最终多种子模型的进度。`--skip-existing-models` 会复用签名兼容的外存矩阵及折模型。

## 结果解释

`model/metadata.json` 包含：

- OOF 选出的 `selected_oof_market_weight` 和 `selected_oof_entity_weight`。
- 市场单独、个体单独及联合修正的消融结果。
- 每折增益、terminal holdout 分数和晋级门。
- 最终部署权重；若晋级失败，两者都为 `0`。

`model/component_feature_importance.csv` 分别记录市场模型和个体模型的重要性。`work/`、`model/` 与 submission 都是运行产物，不提交 Git。

## 推理

```bash
python3 timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/lgb_v3_market_entity_strategy --output examples/lgb_v3_market_entity_strategy/submission.csv
```
