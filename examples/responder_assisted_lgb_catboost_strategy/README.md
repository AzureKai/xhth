# Responder 辅助 LightGBM 时序策略

该策略使用严格时间切分的 stacking：先用输入特征预测 `responder_03` 和 `responder_02`，再将无泄漏的 OOF `responder_hat` 加入 LightGBM target 模型。推理只依赖输入特征和 responder 模型预测，不会读取真实 responder。

## 当前模型

正式模型从 `LGB468_C4` 及其跨折稳定删列版本 `LGB468_C4_STABLE` 中选择。名称中的 `C4` 为兼容现有产物而保留，当前实际只使用两个 responder。

完整候选输入共 470 列：

- 323 个原始 feature。
- 144 个严格历史时序特征。
- 1 个 categorical `asset_id`。
- 2 个 OOF responder 预测：`responder_03_hat`、`responder_02_hat`。

144 个时序特征由 `long_horizon_468_feature_plan.json` 固定：

- 48 个 `rmean5`。
- 8 个重要 `lag1`。
- 7 个重要 `diff1`。
- 27 个源 feature 分别生成 `historical_zscore20`、`minus_ema20` 和 `rolling_std20`，共 81 列。

`asset_id` 作为 LightGBM categorical feature，时序特征只能使用当前 `time_id` 之前的数据。

训练会读取 `LGB468_C4` 的每个 development CV 折模型，统计特征在各折是否被使用以及归一化 gain 的稳定性。默认保留至少 75% 折出现的特征，最多保留 420 列；不足 320 列时按稳定分数回填。`asset_id` 和两个 responder_hat 始终保留。删列候选必须在 walk-forward 和未参与筛选的 terminal holdout 上通过相对完整模型的晋级门槛，才允许部署。

## 主要文件

- `train.py`：预处理、OOF responder、target 选型和最终模型训练。
- `main.py`：时序推理入口。
- `temporal_features.py`：严格历史时序特征状态机。
- `long_horizon_468_feature_plan.json`：当前固定的 468 列基础输入配方。
- `audit_responders.py`：检查 responder 模型和缓存兼容性。
- `requirement.txt`：Python 依赖。
- `work/`：缓存、OOF 模型和中间文件，不提交 Git。
- `model/`：最终模型和训练元数据，不提交 Git。
- `audit/`：模型审计结果。

## 环境

推荐 Python 3.12，并使用独立虚拟环境：

```powershell
python3 -m pip install -r examples/responder_assisted_lgb_catboost_strategy/requirement.txt
```

当前训练参数按 LightGBM 4.6.x 验证。

## 训练

内存充足时推荐一次性读取数据。首次从空目录训练时运行：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode in-memory --temporal-plan examples/responder_assisted_lgb_catboost_strategy/long_horizon_468_feature_plan.json --experiment-suite next-step --target-param-candidates smoothed --rebuild-cache --threads 8
```

后续恢复训练，或在已有 468 列、`responder_03/02` 产物上增加稳定筛选时运行：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode in-memory --temporal-plan examples/responder_assisted_lgb_catboost_strategy/long_horizon_468_feature_plan.json --experiment-suite next-step --target-param-candidates smoothed --skip-existing-models --threads 8
```

内存不足时，将 `--training-data-mode in-memory` 改为 `--training-data-mode out-of-core`。两种模式使用相同的数据切分、特征和标签。

稳定筛选只新增 target 候选；使用 `--skip-existing-models` 时会复用兼容的缓存、responder OOF、部署 responder 和四个基础 target 折模型。不要为这一实验单独传入 `--rebuild-cache`。

### 固定训练配置

- responder：`responder_03`、`responder_02`。
- target 参数组：`smoothed`。
- `num_leaves=47`，`max_depth=10`。
- `min_data_in_leaf=5000`，`min_gain_to_split=0.01`。
- `feature_fraction=0.8`，`feature_fraction_bynode=0.8`，`bagging_fraction=0.8`。
- `lambda_l1=2`，`lambda_l2=30`，`path_smooth=150`。
- 最后 15% 时间作为 terminal holdout。
- development 区间使用 5 折 expanding-window OOF，切分边界 purge 30 个时间步。
- target 使用种子 2026、2027、2028 重训，推理取三个模型均值。
- 稳定筛选默认 `min_fold_rate=0.75`、`min_count=320`、`max_count=420`。

缓存会绑定数据文件指纹、特征配方、responder 列表和 LightGBM 配置版本。配置不兼容时会自动重建，`--skip-existing-models` 不会复用旧模型。

### 当前对照模型

默认 `next-step` 套件训练五个模型：

- `A`：323 个原始 feature 和 categorical `asset_id`。
- `C4`：A 加两个 OOF responder 预测。
- `LGB468`：468 列基础输入，不使用 responder 预测。
- `LGB468_C4`：468 列基础输入加两个 OOF responder 预测，共 470 列。
- `LGB468_C4_STABLE`：从 `LGB468_C4` 各折重要性中筛出的 320～420 列稳定子集。

前三个模型仅作对照；正式部署候选为 `LGB468_C4` 和 `LGB468_C4_STABLE`。选型先比较 development walk-forward 分数，再检查 terminal holdout、预测尺度、裁剪和晋级门槛。稳定删列版本若未通过相对完整模型的全部门槛，不会作为保守回退模型。holdout 不直接用于挑选最高分模型。

### 训练产物

- `model/metadata.json`：特征顺序、responder 列表、验证协议、分数和模型清单。
- `model/target_final_seed*.txt`：三个种子的最终 target 模型。
- `model/target_lightgbm.txt`：首个种子模型的兼容别名。
- `model/responder_*.txt`：最终 responder 模型。
- `model/target_feature_importance.csv`：target 特征重要性。
- `model/stable_feature_report.csv`：逐特征跨折出现率、归一化 gain、稳定分数和保留原因。
- `model/stable_feature_selection.json`：稳定筛选参数及最终特征清单。
- `model/ablation_report.json`：对照模型和选型结果。
- `work/cache/`：预处理后的缓存分片。
- `work/selection_responder_models/`：仅用于验证的 responder 模型。

训练会显示缓存处理、OOF responder、LightGBM boosting、target 实验、验证预测和最终重训进度。

## 审计

审计必须使用训练时对应的 `work/cache`：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/audit_responders.py --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --output-dir examples/responder_assisted_lgb_catboost_strategy/audit
```

报告中的 `model_cache_compatible` 应为 `true`，cache schema、时序引擎版本、配方哈希和特征列必须与模型一致。

## 推理

```powershell
python3 timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/responder_assisted_lgb_catboost_strategy --output examples/responder_assisted_lgb_catboost_strategy/submission.csv
```

推理要求 `time_id` 递增。每个时间步先更新历史状态，再预测两个 responder_hat，最后由三个 target 模型的均值生成结果。推理过程显示完成行数、当前 `time_id`、耗时和预计剩余时间。

## 约束

- 真实 responder 只能用于训练 responder 模型，不能直接进入 target 训练样本或推理输入。
- target 必须使用 expanding-window 产生的 OOF responder 预测，不能使用 responder 模型在自身训练样本上的拟合值。
- 每个验证折和 terminal holdout 都从空时序状态启动，避免历史泄漏。
- `metadata.json` 中的验证分数来自最终重训前的选型模型；最终模型已吸收该验证区间标签。
- 不要通过 `predict_disable_shape_check=true` 绕过特征维度错误，应检查模型与 `metadata.json` 是否匹配。
- 不要提交 `work/`、`model/`、Parquet、NumPy 缓存、虚拟环境或本地回传脚本。
