# Random Strategy Demo

这是最小但完整的参赛策略示例，展示：

- 读取发布目录中的 `train/*.parquet`；
- 按 `time_id` 做时间序列 train/validation 切分；
- 用训练集估计一个简单预测尺度；
- 保存预训练模型到 `model/random_model.json`；
- 在 `main.py` 中加载预训练模型并通过 Time-Series API 推理。

训练：

```bash
python inference_api/random_strategy/train.py \
  --release-root /path/to/release \
  --model-dir inference_api/random_strategy/model
```

本地顺序推理：

```bash
python scripts/run_timeseries_api.py \
  --release-root /path/to/release \
  --strategy-dir inference_api/random_strategy \
  --output /tmp/random_submission.csv
```
