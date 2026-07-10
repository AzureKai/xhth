# EMA LightGBM / CatBoost Strategy

这是一个 EMA 增强策略目录，在`examples/lgb_catboost_strategy/`基础上实现。

This is a standalone EMA-enhanced strategy directory. It does not overwrite the
existing `examples/lgb_catboost_strategy/`.

## Core Idea / 核心思路

训练脚本会读取旧模型的 `feature_importance.csv`，默认选取 top 50 个原始
`feature_*`，为它们生成多尺度 EMA 历史摘要特征：

The training script reads the previous model's `feature_importance.csv`, selects
the top 50 raw `feature_*` columns by default, and creates multi-scale EMA
features:

- `ema_gap_h{N}_{feature}` = current value - previous EMA with half-life `N`
- `ema_spread_h{short}_h{long}_{feature}` = short EMA - long EMA

默认 half-life:

Default half-lives:

```text
5,20,60
```

为了让 EMA 更接近真实时序，训练采样按连续 `time_id` 块进行，并保留每个
`time_id` 下完整 asset 截面。

To keep EMA closer to real time-series behavior, sampling uses contiguous
`time_id` blocks and preserves the full asset cross-section for each selected
`time_id`.

## Train / 训练

LightGBM only:

```bash
python examples/ema_lgb_catboost_strategy/train.py --data-root data --model-dir examples/ema_lgb_catboost_strategy/model --train-catboost 0
```

With CatBoost:

```bash
python examples/ema_lgb_catboost_strategy/train.py --data-root data --model-dir examples/ema_lgb_catboost_strategy/model --train-catboost 1
```

Custom EMA setup:

```bash
python examples/ema_lgb_catboost_strategy/train.py --data-root data --model-dir examples/ema_lgb_catboost_strategy/model --ema-feature-count 80 --ema-halflives 3,10,40 --train-catboost 0
```

## Local Time-Series API Check / 本地顺序推理验证

```bash
python timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/ema_lgb_catboost_strategy --output examples/ema_lgb_catboost_strategy/submission.csv
```

## Output Files / 输出文件

- `metadata.json`: feature columns, EMA feature columns, validation metrics, prediction scale, and clip bounds.
- `feature_importance.csv`: feature importance summary for the EMA model.
- `feature_importance_by_model.csv`: per-model feature importance.
- `lightgbm.txt`: LightGBM model file, if enabled.
- `catboost.cbm`: CatBoost model file, if enabled.

## Notes / 注意事项

- EMA features are generated from previous state, then the state is updated with the current row.
- Inference maintains EMA state per `asset_id`.
- The first observed row for each asset has zero EMA gaps because no previous state exists.
- 推理阶段不读取 parquet，不加载训练集。
