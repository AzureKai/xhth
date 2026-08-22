# 资产簇残差专家

该策略检验不同资产群体是否具有不同的 feature-target 映射。基础预测固定为当前 `LGB468_C4_STABLE`，专家只学习严格 OOF 残差：

```text
prediction = C4 + residual_scale * centered_cluster_expert
```

## 时间安全协议

1. 从 C4 当前稳定模型的重要性中选择 24 个原始 feature；不重复生成大批时序特征。
2. 将 C4 development OOF 按时间切成 5 块，第一块只训练，后四块进行 expanding walk-forward 验证。
3. 每一折只用训练前缀计算资产画像。画像由 12 个 feature 的均值、波动率、残差相关性和资产残差统计组成。
4. 资产固定分成 4 簇，每簇至少需要 3 个资产才能晋级。`asset_id` 只负责路由，不进入 LightGBM 特征，降低身份捷径风险。
5. 每个簇训练一个强正则化 LightGBM 残差专家，并按每个时间截面将残差预测中心化，避免重新引入已失败的市场公共项。
6. 同时训练不分簇的全局残差模型作为对照。只有簇专家在 OOF、最新折、聚类稳定性和冻结 holdout 上同时超过 C4 与全局对照，才会部署；否则自动回退 C4。

## 一行训练与推理

已有 C4 OOF 和缓存时：

```bash
cd ~/xhth && source .venv/bin/activate && python3 examples/lgb_asset_cluster_residual_strategy/train.py --c4-strategy-dir examples/responder_assisted_lgb_catboost_strategy --c4-model-dir examples/responder_assisted_lgb_catboost_strategy/model --c4-cache-dir examples/responder_assisted_lgb_catboost_strategy/work/cache --work-dir examples/lgb_asset_cluster_residual_strategy/work --model-dir examples/lgb_asset_cluster_residual_strategy/model --feature-count 24 --profile-feature-count 12 --clusters 4 --min-assets-per-cluster 3 --walk-forward-blocks 5 --purge-steps 1 --rounds 200 --early-stopping 40 --threads 8 --residual-scales 0,0.25,0.50,0.75,1.0,1.25 --required-positive-fold-rate 0.75 --min-cluster-stability 0.65 --skip-existing-models && python3 timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/lgb_asset_cluster_residual_strategy --output examples/lgb_asset_cluster_residual_strategy/submission.csv
```

如 C4 的 `development_oof_predictions.npz` 尚未生成，需要先按 C4 README 的当前命令运行一次源训练器；`--skip-existing-models` 会复用已有 C4 模型。

## 结果文件

- `model/metadata.json`：部署配置、簇映射、OOF/holdout 分数和晋级门。
- `model/feature_importance.csv`：通过晋级门时各簇专家的重要性。
- `work/cluster_experiment_report.json`：完整实验报告副本。
- `work/feature_stage.json` 与两个 `.npy`：可复用的对齐特征缓存。
- `submission.csv`：通过时为簇专家预测，失败时为未修改 C4。

本机回传使用默认远程地址即可，避免 PowerShell 展开 `~/xhth`：

```powershell
examples\responder_assisted_lgb_catboost_strategy\pull_remote_results.bat
```
