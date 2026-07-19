# Period Weighted LightGBM Strategy / 时间片加权 LightGBM 策略

这个策略把训练集按 `time_id` 切成可变数量的连续时间段，在每个时间段内训练一个轻量 LightGBM probe model，然后对不同时间片的 feature importance 做时间加权平均。默认越靠近验证段的时间片权重越高，用来更贴近推理期的特征权重分布。

This strategy splits training data into configurable contiguous `time_id` periods, trains one lightweight LightGBM probe model per period, then computes a time-weighted average of feature importance across periods. By default, recent periods receive larger weights so the final feature set better reflects the expected inference-time feature distribution.

## Training / 训练

Inspect the `time_id` split plan before training:

训练前先检查 `time_id` 分段方案：

```bash
python examples/period_stable_lgb_strategy/train.py --data-root data --model-dir examples/period_stable_lgb_strategy/model --num-periods 5 --dry-run
```

Train with the default recent-weighted feature selection:

使用默认近期加权特征选择训练：

```bash
python examples/period_stable_lgb_strategy/train.py --data-root data --model-dir examples/period_stable_lgb_strategy/model
```

Common options / 常用参数：

```bash
python examples/period_stable_lgb_strategy/train.py --data-root data --model-dir examples/period_stable_lgb_strategy/model --num-periods 5 --weighted-feature-count 120 --period-weighting exp_recent --period-weight-decay 0.7 --ema-feature-count 50 --ema-halflives 5,20,60 --max-rows-per-period 200000 --max-valid-rows 300000
```

旧参数 `--stable-feature-count` 仍然兼容；如果没有显式传入 `--weighted-feature-count`，会沿用 `--stable-feature-count` 的值。

The legacy `--stable-feature-count` argument remains compatible. If `--weighted-feature-count` is not provided, the trainer uses the value from `--stable-feature-count`.

## Feature Weighting / 特征权重

The final selected features are ranked by `weighted_importance`, not by a stability score.

最终入选特征按 `weighted_importance` 排序，不再使用稳定性惩罚分数。

Supported period weighting modes / 支持的时间片权重：

- `exp_recent`: exponential recent weighting, default. / 近期指数加权，默认。
- `linear_recent`: linear recent weighting. / 近期线性加权。
- `equal`: simple average across periods. / 各时间片等权平均。

For `exp_recent`, `--period-weight-decay 0.7` means the latest period has weight `1.0`, the previous period has `0.7`, then `0.49`, and so on before normalization.

对 `exp_recent`，`--period-weight-decay 0.7` 表示最新时间片原始权重为 `1.0`，前一片为 `0.7`，再前一片为 `0.49`，之后统一归一化。

## EMA Features / EMA 时序特征

EMA features are added after weighted feature selection. By default, the strategy takes the first 50 selected features and creates low-cost sequential features:

EMA 特征在加权特征筛选之后添加。默认取加权排序前 50 个入选特征，生成轻量时序特征：

- `ema_gap_h*_feature_*`: current value minus previous EMA state. / 当前值减去此前 EMA 状态。
- `ema_spread_h*_h*_feature_*`: short-half-life EMA state minus long-half-life EMA state. / 短半衰期状态减去长半衰期状态。

Set `--ema-feature-count 0` to disable EMA features.

设置 `--ema-feature-count 0` 可以关闭 EMA 特征。

## Artifact Check / 产物检查

After training, run:

训练完成后运行：

```bash
python examples/period_stable_lgb_strategy/check_artifacts.py --model-dir examples/period_stable_lgb_strategy/model
```

This checks that `metadata.json`, `weighted_feature_importance.csv`, `period_feature_importance.csv`, `period_metrics.csv`, and `final_lightgbm.txt` are mutually consistent.

它会检查 `metadata.json`、加权特征表、分段 importance、分段指标和最终模型文件是否相互一致。

## Local Time-Series API Check / 本地顺序推理验证

```bash
python timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/period_stable_lgb_strategy --output examples/period_stable_lgb_strategy/submission.csv
```

## Outputs / 输出文件

- `final_lightgbm.txt`: final model trained with selected weighted features. / 使用加权入选特征训练出的最终模型。
- `metadata.json`: selected features, weighting settings, EMA settings, split settings, validation metrics, and prediction post-processing. / 记录入选特征、时间片权重、EMA 设置、切分参数、验证指标和预测后处理。
- `period_metrics.csv`: inner-period and future-validation score for each probe model. / 每个分段 probe model 的段内验证和未来验证指标。
- `period_feature_importance.csv`: raw feature importance from each period model. / 每个时间段模型的原始 importance。
- `weighted_feature_importance.csv`: time-weighted feature importance summary and selected features. / 时间片加权 importance 汇总和最终入选特征。

## Notes / 注意事项

- Probe models are training-time artifacts only; final inference loads only `final_lightgbm.txt` and `metadata.json`.
- Feature selection uses raw `feature_*` columns only; it does not use target, responder, or future test data.
- The future validation period is used for transfer evaluation, not for probe early stopping.
- EMA features are derived only from the selected features of the current training run.

- Probe 模型只是训练阶段产物；最终推理只加载 `final_lightgbm.txt` 和 `metadata.json`。
- 特征选择只使用原始 `feature_*`，不使用 target、responder 或未来测试数据。
- 未来验证段只用于迁移评估，不用于 probe early stopping。
- EMA 特征只来自本次训练选出的加权特征。
