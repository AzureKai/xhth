# LightGBM / CatBoost Strategy Demo

这是一个面向 Time-Series API 的轻量强基线策略。训练阶段按照严格的
`time_id` 时间切分训练 LightGBM 和/或 CatBoost，保存紧凑的模型文件；
推理阶段只加载这些模型文件，不读取训练数据。

This is a lightweight baseline strategy for the Time-Series API. It trains
LightGBM and/or CatBoost with a strict `time_id` split, saves compact model
files, and loads only those files during inference.

默认特征集：

Default feature set:

- `feature_*`
- `asset_id`
- selected cross-sectional `demean` / `rank` features

推理阶段不会使用 `responder_*`、`target` 或 `weight`。

The inference path deliberately does not use `responder_*`, `target`, or
`weight`.

## Environment Check / 环境检查

```bash
python --version
python -c "import lightgbm, catboost, pandas, pyarrow, numpy; print('ok')"
```

CatBoost 是可选依赖。如果本地未安装，或推理速度太慢，可以使用
`--train-catboost 0` 关闭。

CatBoost is optional. If it is unavailable or too slow, disable it with
`--train-catboost 0`.

## Train / 训练

小样本冒烟测试：

Small smoke run:

```bash
python examples/lgb_catboost_strategy/train.py --data-root data --model-dir examples/lgb_catboost_strategy/model --sample-frac 0.05 --train-catboost 0
```

全量训练：

Full run:

```bash
python examples/lgb_catboost_strategy/train.py --data-root data --model-dir examples/lgb_catboost_strategy/model
```

默认训练会按 parquet batch 流式读取数据，并最多抽样 `500,000` 行训练集和
`150,000` 行验证集，以避免一次性加载千万级数据导致内存溢出。
训练结束后会在验证集上搜索 `prediction_scale`，用于收缩或放大最终预测。
如果 `model/feature_importance.csv` 已存在，训练会默认选取其中前 `50` 个原始
`feature_*` 生成截面 demean/rank 特征；否则使用前 `50` 个 `feature_*`。

By default, training streams parquet batches and samples at most `500,000`
training rows and `150,000` validation rows. This avoids loading tens of
millions of rows into memory at once.
After training, the script searches `prediction_scale` on the validation set to
shrink or amplify final predictions.
If `model/feature_importance.csv` already exists, training uses its top `50`
raw `feature_*` columns for cross-sectional demean/rank features. Otherwise, it
falls back to the first `50` `feature_*` columns.

只训练 LightGBM：

LightGBM only:

```bash
python examples/lgb_catboost_strategy/train.py --data-root data --model-dir examples/lgb_catboost_strategy/model --train-catboost 0
```

自定义预测缩放搜索网格：

Custom prediction-scale search grid:

```bash
python examples/lgb_catboost_strategy/train.py --data-root data --model-dir examples/lgb_catboost_strategy/model --alpha-grid 0.1,0.2,0.3,0.5,0.8,1.0
```

自定义截面特征数量或显式特征列表：

Custom cross-sectional feature count or explicit feature list:

```bash
python examples/lgb_catboost_strategy/train.py --data-root data --model-dir examples/lgb_catboost_strategy/model --xs-feature-count 80
python examples/lgb_catboost_strategy/train.py --data-root data --model-dir examples/lgb_catboost_strategy/model --xs-features feature_000,feature_010,feature_123
```

如果机器内存足够，可以关闭行数上限尝试更大规模训练：

If the machine has enough memory, disable row caps for a larger run:

```bash
python examples/lgb_catboost_strategy/train.py --data-root data --model-dir examples/lgb_catboost_strategy/model --max-train-rows 0 --max-valid-rows 0
```

## Local Time-Series API Check / 本地顺序推理验证

```bash
python timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/lgb_catboost_strategy --output examples/lgb_catboost_strategy/submission.csv
```

runner 会输出 JSON 运行报告，其中包含初始化耗时、单步推理耗时、超时次数等
诊断信息。建议根据这些 timing 结果决定最终策略包中是否保留 CatBoost。

The runner output includes timing diagnostics such as initialization time,
per-step inference time, and timeout counts. Use those numbers to decide whether
CatBoost should remain in the final strategy package.

## Output Files / 输出文件

训练完成后，`model/` 目录会包含：

After training, the `model/` directory contains:

- `metadata.json`：特征列、模型列表、验证 L2/R2、ensemble 权重、预测缩放和预测 clip 范围。
- `feature_importance.csv`：按模型平均后的特征重要性汇总。
- `feature_importance_by_model.csv`：每个模型各自输出的特征重要性。
- `lightgbm.txt`：LightGBM 模型文件，如果启用 LightGBM。
- `catboost.cbm`：CatBoost 模型文件，如果启用 CatBoost。

- `metadata.json`: feature columns, model list, validation L2/R2 scores,
  ensemble weights, prediction scale, and prediction clip bounds.
- `feature_importance.csv`: feature importance summary averaged across models.
- `feature_importance_by_model.csv`: per-model feature importance output.
- `lightgbm.txt`: LightGBM model file, if LightGBM is enabled.
- `catboost.cbm`: CatBoost model file, if CatBoost is enabled.

## Notes / 注意事项

- 训练使用 `weight` 作为样本权重；推理阶段不会访问 `weight`。
- 验证集默认取最后 `20%` 的 `time_id`，避免未来信息泄露。
- 训练脚本流式读取 parquet，避免一次性加载全量训练集。
- 截面特征只使用当前 `time_id` 的可见样本，不使用未来信息。
- 最终推理环境较弱时，优先使用 LightGBM 单模型或小规模 ensemble。
- `main.py` 不会读取 parquet，也不会加载训练集。

- Training uses `weight` as sample weights; inference never accesses `weight`.
- The validation split uses the last `20%` of `time_id` values by default to
  avoid future leakage.
- The training script streams parquet input instead of loading the full training
  set at once.
- Cross-sectional features use only visible samples from the current `time_id`.
- On weak final inference machines, prefer a single LightGBM model or a small
  ensemble.
- `main.py` does not read parquet files or load the training set.
