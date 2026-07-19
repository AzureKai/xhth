# TCN + Cross-Asset Attention Strategy

这是一个面向 Time-Series API 的独立 PyTorch 时序策略。它不会读取任何其他策略的
artifact，也不会使用其他模型产生的 feature importance、预测值、蒸馏标签、榜单统计
或测试集统计。

This is a standalone PyTorch time-series strategy for the Time-Series API. It
does not read artifacts from any other strategy and does not use feature
importance files, model predictions, distillation labels, or leaderboard/test
statistics.

## Core Idea / 核心思路

对每个 `time_id`，模型构造一个固定的 15 资产面板：

- 每个资产最近一段时间的全部 `feature_*` rolling history；
- 当前截面的 `demean`、`rank`、`zscore` 特征；
- 共享的因果 TCN，用于编码每个资产自己的历史状态；
- 小型 self-attention 模块，用于在当前 `time_id` 混合不同资产之间的信息；
- MLP head，为每个可见资产输出一个连续预测值。

For each `time_id`, the model builds a fixed 15-asset panel:

- per-asset rolling history of all `feature_*` columns;
- current cross-sectional `demean`, `rank`, and `zscore` features;
- a shared causal TCN that encodes each asset's own history;
- a small self-attention block that mixes information across assets at the
  current `time_id`;
- an MLP head that predicts one continuous value per visible asset.

训练阶段使用 `target` 和样本 `weight`。推理阶段只使用 `row_id`、`time_id`、
`asset_id` 和 `feature_*`。

Training uses `target` and sample `weight`. Inference only uses `row_id`,
`time_id`, `asset_id`, and `feature_*`.

## Train / 训练

小规模冒烟测试：

Small smoke run:

```bash
python examples/tcn_attention_strategy/train.py --data-root data --model-dir examples/tcn_attention_strategy/model --max-train-times 200 --max-valid-times 50 --epochs 1 --batch-size 16
```

较大的本地训练：

Larger local run:

```bash
python examples/tcn_attention_strategy/train.py --data-root data --model-dir examples/tcn_attention_strategy/model
```

常用参数：

Useful knobs:

```bash
python examples/tcn_attention_strategy/train.py \
  --data-root data \
  --model-dir examples/tcn_attention_strategy/model \
  --window-size 32 \
  --hidden-dim 64 \
  --domain-count 5 \
  --rex-lambda 1.0 \
  --epochs 20 \
  --batch-size 256
```

迁移学习 / domain generalization 设置：

Transfer learning / domain generalization knobs:

- `--domain-count`：将训练 `time_id` 按时间顺序切成多少个连续 domain。
- `--rex-lambda`：REx 惩罚强度；设为 `0` 时退化为普通 ERM 训练。
- 开启 REx 后，日志里的 `train_loss` 是 `train_erm_loss + rex_lambda * train_rex_penalty`。
- `valid_loss` 和 `valid_score` 仍然按普通 weighted MSE / weighted zero-mean R2 计算，不包含 REx 惩罚。

- `--domain-count`: number of contiguous time domains built from training `time_id` values.
- `--rex-lambda`: REx penalty strength; set it to `0` to recover plain ERM training.
- With REx enabled, logged `train_loss` is `train_erm_loss + rex_lambda * train_rex_penalty`.
- `valid_loss` and `valid_score` remain plain weighted MSE / weighted zero-mean R2 metrics without the REx penalty.

## Local Time-Series API Check / 本地 Time-Series API 验证

```bash
python timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/tcn_attention_strategy --output examples/tcn_attention_strategy/submission.csv --per-step-timeout-seconds 0.5
```

## Model Artifacts / 模型产物

推理只需要以下两个文件：

Inference requires only:

- `model/metadata.json`
- `model/model.pt`

`metadata.json` 保存本模型自己的特征列、归一化统计量、资产槽位映射、网络结构参数、
预测缩放、clip 边界和验证指标。推理路径不会读取 parquet 文件、训练集、responder
列、target 列、weight 列、其他策略目录或 importance 文件。

`metadata.json` stores this model's own feature list, normalization statistics,
asset slot mapping, architecture settings, prediction scale, clipping bounds,
and validation metrics. The inference path does not read parquet files, training
data, responder columns, target columns, weights, other strategy directories, or
importance files.
