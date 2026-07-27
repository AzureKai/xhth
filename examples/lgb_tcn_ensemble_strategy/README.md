# LightGBM + TCN 时序集成

该策略使用现有 responder-assisted LightGBM 作为基模型，再训练一个按 `asset_id` 构造历史窗口的轻量因果 TCN。TCN 直接读取历史路径，不依赖预先压缩的 rolling/EMA 特征。

训练严格分为：

```text
历史训练区间：训练 TCN
最终验证前半段：早停并校准受限融合权重
最终验证后半段：只用于评估
```

同时在每个 `time_id` 内打乱 TCN 预测，生成 shuffled 对照。只有正常融合增量明显高于 shuffled 增量，才能认为 TCN 提供了真实的样本级时序信息。

## 1. 导出 LightGBM 验证预测

先更新并运行原策略。已有模型会被复用，只补充生成 `validation_predictions.npz`：

```powershell
python examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode in-memory --skip-existing-models --threads 8
```

## 2. 训练 TCN 并评估融合

```powershell
python examples/lgb_tcn_ensemble_strategy/train.py --data-root data --base-model-dir examples/responder_assisted_lgb_catboost_strategy/model --work-dir examples/lgb_tcn_ensemble_strategy/work --model-dir examples/lgb_tcn_ensemble_strategy/model --feature-count 48 --sequence-length 32 --hidden-size 64 --batch-size 1024 --device auto
```

显存或内存不足时，可先运行低成本验证：

```powershell
python examples/lgb_tcn_ensemble_strategy/train.py --data-root data --base-model-dir examples/responder_assisted_lgb_catboost_strategy/model --work-dir examples/lgb_tcn_ensemble_strategy/work --model-dir examples/lgb_tcn_ensemble_strategy/model --feature-count 32 --sequence-length 24 --hidden-size 32 --levels 3 --batch-size 512 --max-train-rows 2000000 --device auto
```

主要输出：

- `model/evaluation_report.json`：基模型、TCN、融合和 shuffled 对照分数。
- `model/metadata.json`：特征、标准化参数、窗口长度和融合权重。
- `model/tcn.pt`：TorchScript 推理模型。
- `work/validation_predictions.npz`：逐行验证预测，便于后续相关性和分段分析。

远程训练完成后回传：

```bat
examples\lgb_tcn_ensemble_strategy\pull_remote_results.bat
```

## 3. 推理

```powershell
python timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/lgb_tcn_ensemble_strategy --output examples/lgb_tcn_ensemble_strategy/submission.csv
```

推理必须按递增 `time_id` 调用。策略内部同时维护 LightGBM 时序状态和每个 asset 的 TCN 历史窗口。

## 判断是否继续投入

建议只有在以下条件大致满足时继续扩大 TCN：

- 融合相对基模型的增量为正；
- 增量明显高于 shuffled 对照；
- 多数连续时间段增量为正；
- 校准得到的融合权重没有贴近 0；
- 多随机种子方向一致。

第一版的目标是检测序列模型是否提供正交信息，而不是直接替换 LightGBM。
