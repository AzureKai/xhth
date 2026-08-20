# V3 Regime Residual LightGBM

该策略冻结 `lightgbm_baseline` low-risk V3 作为基础预测器，用严格时间外预测定义 residual，再使用轻量 LightGBM 学习 V3 未解释的部分：

```text
prediction = V3_prediction + beta * residual_prediction
```

不依赖真实 responder，也不使用推理时不可见的历史 target/residual。二层模型只读取当前原始特征、V3 预测以及由这些可观测量维护的状态。

## 输入状态

默认从 V3 已选择的历史 feature 中取前 16 个，为每个 `asset_id` 因果维护：

- 前一时刻状态形成的 `entity_z20`。
- `EMA20 - EMA60` 趋势状态。
- 当前横截面的 `cross_z`。
- entity 历史长度。
- V3 当前预测。

每个 `time_id` 还产生市场级 regime 指标：entity 冲击强度、趋势强度、横截面离散度、冲击比例以及 V3 预测的横截面强度。训练会比较：

- `ENTITY_STATE`：不使用市场 regime。
- `REGIME_ENTITY`：entity state 加连续 regime 指标和 categorical `regime_id`。

## 无泄漏协议

1. 完全复用 V3 的 5 折 expanding walk-forward、15% terminal holdout 和 30 时间步 purge。
2. 使用 V3 已选定的 `leaves63_smoothed` 参数和 2026/2027/2028 三种子重训每折模型；每折取种子均值，生成与最终 V3 ensemble 对齐的 OOF prediction。
3. residual CV 的第 2～5 折只能使用更早 OOF 块的 residual 训练；第 1 个 OOF 块只作为 residual warmup。
4. 每个 OOF 块和 terminal holdout 都从空 entity state 启动，与 Time-Series API 冷启动一致。
5. residual 权重只从 development OOF 的预设集合 `0,0.05,0.10,0.15,0.25` 中选择。
6. terminal holdout 不参与选权重。只有四个 residual CV 折和 holdout 都产生正增益时，最终 residual 权重才会部署；否则策略自动退化为原始 V3。

最终 residual 模型使用严格 OOF residual 加 terminal holdout 的时间外 residual 重训，不会使用 V3 对自身训练集的拟合误差。

## 训练

推荐复用 V3 的低内存缓存。命令保持一行：

```bash
python3 examples/lgb_v3_regime_residual_strategy/train.py --data-root data --base-model-dir examples/lightgbm_baseline/model_forward_lowrisk_v3 --base-cache-dir examples/lightgbm_baseline/.low_memory_cache_forward_v2 --work-dir examples/lgb_v3_regime_residual_strategy/work --model-dir examples/lgb_v3_regime_residual_strategy/model --state-feature-count 16 --base-oof-seeds 2026,2027,2028 --residual-weights 0,0.05,0.10,0.15,0.25 --threads 8 --skip-existing-models
```

首次不存在 V3 cache 时会自动构建；已有兼容 cache、V3 OOF 和 state matrix 时，`--skip-existing-models` 会复用它们。训练过程显示 V3 OOF、状态物化、residual CV、holdout 和最终多种子模型进度。

## 产物

- `model/metadata.json`：两种 residual 候选、逐折增益、权重搜索、holdout、shuffled-regime 诊断和晋级门槛。
- `model/residual_feature_importance.csv`：用于 terminal holdout 评估的开发期 residual 模型特征重要性。
- `model/residual_seed*.txt`：门槛通过后生成的三种子 residual 模型。
- `work/base_oof_prediction.npy`：V3 的严格 OOF 与 holdout 预测。
- `work/residual_features.npy`：磁盘支持的 entity/regime 特征矩阵。

`work/`、`model/` 和 submission 都不提交 Git。

## 推理

```bash
python3 timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/lgb_v3_regime_residual_strategy --output examples/lgb_v3_regime_residual_strategy/submission.csv
```

若晋级失败，`metadata.json` 中的 `residual_weight` 为 0，推理结果与原始 V3 一致。
