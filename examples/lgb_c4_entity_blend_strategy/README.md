# C4 × V3 Entity 严格 OOF 融合

该策略融合两个已经独立验证的预测器：

- `responder_assisted_lgb_catboost_strategy` 当前部署的 `LGB468_C4_STABLE`。
- `lgb_v3_market_entity_strategy` 当前通过晋级门的 V3 + centered entity residual。

预测公式为：

```text
prediction = (1 - entity_weight) * C4 + entity_weight * V3_entity
```

## 验证协议

两套源模型原始 OOF 折边界不同，禁止直接融合各自折分数。本策略要求源训练器导出逐行的 `development_oof_predictions.npz` 和 `validation_predictions.npz`，然后：

1. 以 `(time_id, asset_id)` 严格校验 target、weight 和行顺序。
2. development 只使用两套严格 OOF 的共同区间。
3. 使用两套源折边界的并集，将共同区间切成连续时间段。
4. 只在 development 时间段搜索预注册的固定权重。
5. terminal holdout 不参与调权，只用于冻结验收。

融合只有同时超过两个父模型的平均分段分数、整体 OOF、最新时间段和 terminal holdout，并在至少 70% 时间段超过较强父模型，才会部署内部权重。否则自动退回共同 OOF 上更强的父模型。

## 一行训练与推理

首次生成源 OOF、执行融合校准并推理：

```bash
cd ~/xhth && source .venv/bin/activate && python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode in-memory --temporal-plan examples/responder_assisted_lgb_catboost_strategy/long_horizon_468_feature_plan.json --experiment-suite next-step --target-param-candidates smoothed --skip-existing-models --threads 8 && python3 examples/lgb_v3_market_entity_strategy/train.py --data-root data --base-model-dir examples/lightgbm_baseline/model_forward_lowrisk_v3 --base-cache-dir examples/lightgbm_baseline/.low_memory_cache_forward_v2 --source-work-dir examples/lgb_v3_regime_residual_strategy/work --work-dir examples/lgb_v3_market_entity_strategy/work --model-dir examples/lgb_v3_market_entity_strategy/model --state-feature-count 16 --extra-cross-z-count 8 --entity-weights 0,0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.0 --entity-rounds 500 --threads 8 --skip-existing-models && python3 examples/lgb_c4_entity_blend_strategy/train.py --c4-model-dir examples/responder_assisted_lgb_catboost_strategy/model --entity-model-dir examples/lgb_v3_market_entity_strategy/model --work-dir examples/lgb_c4_entity_blend_strategy/work --model-dir examples/lgb_c4_entity_blend_strategy/model --entity-weights 0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0 --required-positive-segment-rate 0.70 --skip-existing-models && python3 timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/lgb_c4_entity_blend_strategy --output examples/lgb_c4_entity_blend_strategy/submission.csv
```

源 OOF 生成后，后续只需执行最后两个 `python3` 命令。融合训练本身不训练 LightGBM，只读取已对齐预测并搜索权重。

## 产物

- `model/metadata.json`：权重搜索、父模型对照、逐段结果、holdout 和晋级门。
- `work/fusion_report.json`：与 metadata 相同的分析副本。
- `submission.csv`：通过晋级门的融合预测；失败时是较强父模型的安全回退预测。

源模型、OOF、work、model 和 submission 都是本地/远程运行产物，不提交 Git。
