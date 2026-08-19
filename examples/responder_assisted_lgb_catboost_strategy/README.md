# Responder 辅助 LightGBM 时序策略

该目录实现严格按时间切分的 responder stacking 策略。训练阶段先使用特征预测 responder，再把无泄漏的 OOF `responder_hat` 与原始及时序特征一起用于 target 模型；推理阶段只使用特征和 responder 模型预测值，不会访问真实 responder。

## 目录与产物

- `train.py`：缓存预处理、OOF responder、target 消融和最终模型训练。
- `main.py`：时序 API 推理入口。
- `temporal_features.py`：严格历史时序特征状态机。
- `baseline_468_feature_plan.json`：上一轮强 baseline 的原始 48×3 时序配方，保留作对照。
- `long_horizon_468_feature_plan.json`：当前默认的紧凑 468 列长期时序配方。
- `all_feature_long_horizon_plan.json`：保留作专项消融的全 feature 长期时序配方。
- `build_long_horizon_plan.py`：根据上一轮特征重要性重建长期配方。
- `build_all_feature_long_horizon_plan.py`：把三种 20 步长变换扩展到全部原始 feature。
- `screen_all_responders.py`：对原始数据中的全部 responder 做潜力筛选。
- `analyze_responders.py`：分析 responder 与 target 的相关性和时间稳定性。
- `analyze_feature_temporal_types.py`：判断 feature 适合的时序变换类型。
- `audit_responders.py`：审计 responder 可预测性、均值基线和模型缓存一致性。
- `work/`：缓存、OOF 模型和中间文件，不应提交 Git。
- `model/`：最终模型、metadata、消融报告和特征重要性。
- `analysis/`：responder 与时序特征分析结果。
- `audit/`：responder 审计报告。

## 环境

推荐 Windows Python 3.12，并在独立虚拟环境中安装依赖：

```powershell
python3 -m pip install -r examples/responder_assisted_lgb_catboost_strategy/requirement.txt
```

## 推荐工作流

### 1. 多折稳健筛选全部 responder

推荐先运行4折 expanding-window 多输出 Ridge 筛选、OOF 相关性聚类和轻量 LightGBM 复核：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/screen_responders_multifold.py --data-root data --output-dir examples/responder_assisted_lgb_catboost_strategy/analysis/multifold_responders --max-rows 300000 --folds 4 --refine-count 8 --threads 8
```

筛选器使用 LightGBM 建立 target 基线，Ridge 只负责一次性预测全部 responder。残差融合会按校准段标准化 `responder_hat`、裁剪极端值并加入正则项，同时检查校准段与评估段的预测尺度漂移。严格增量条件没有候选时，只会回退选择尺度稳定且可预测的代表进入 LightGBM 复核，不会直接把它们判为最终候选。安全参数可通过 `--residual-ridge`、`--hat-clip`、`--min-scale-ratio` 和 `--max-scale-ratio` 调整。

每个时间折都按“历史训练→当前块前半段校准残差系数→当前块后半段评估”执行。默认要求 centered R²不低于0.01、至少3/4折 target 增量为正、最后一折为正且稳健分数为正。通过者按 OOF `responder_hat` 绝对相关0.95聚类，每簇只选择一个代表进入轻量 LightGBM 复核。

主要输出：

- `multifold_ridge_ranking.csv`：全部 responder 的 Ridge 多折稳健排名。
- `multifold_ridge_folds.csv`：每个 responder、每个时间折的详细指标。
- `multifold_hat_correlations.csv`：OOF responder_hat 相关矩阵。
- `multifold_lightgbm_ranking.csv`：各簇代表的轻量 LightGBM 多折排名。
- `multifold_candidates.json`：冗余簇和最终建议候选。

### 2. 单区间快速筛选全部 responder

数据 manifest 当前记录47个 responder。脚本自动从 Parquet schema 发现全部 `responder_*`，不会写死编号。数据按连续时间拆成训练60%、校准20%和最终评估20%，输出 responder 的 centered R²、预测相关性，以及加入 baseline target 残差后的样本外增量。

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/screen_all_responders.py --data-root data --output-dir examples/responder_assisted_lgb_catboost_strategy/analysis/all_responders --max-rows 500000 --candidate-count 12 --threads 8
```

主要输出：

- `analysis/all_responders/all_responder_potential.csv`：全部 responder 排名。
- `analysis/all_responders/responder_candidates.json`：同时满足 centered R² 和 target 增量为正的候选。

该步骤是低成本初筛，候选 responder 仍需经过完整 expanding-window OOF 和 target 消融确认。

### 3. 分析 responder 与 target 的直接关系

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/analyze_responders.py --data-root data --output-dir examples/responder_assisted_lgb_catboost_strategy/analysis --max-rows 2000000 --time-bins 10
```

输出全局加权相关、去时间均值相关、去资产均值相关、逐时点截面 IC、分段稳定性和综合筛选分数。

### 4. 生成时序特征路由

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/analyze_feature_temporal_types.py --data-root data --output-dir examples/responder_assisted_lgb_catboost_strategy/analysis
```

`analyze_feature_temporal_types.py` 仍可生成探索性路由。`long_horizon_468_feature_plan.json` 是嵌套在当前大模型中的紧凑对照子集：

- 323 个原始 feature。
- 48 个 `rmean5`，每个历史源特征均保留。
- 在旧 `LGB468` 与 `LGB468_C4` 中分别取 gain 最高的 5 个 `lag1/diff1`，合并去重后保留 8 个 `lag1` 和 7 个 `diff1`。
- 按两个模型内归一化后的时序总 gain 选择 27 个源特征，每个生成 `historical_zscore20`、`minus_ema20`、`rolling_std20`，共 81 列。
- 历史衍生列仍为 48 + 8 + 7 + 81 = 144，因此实验维度保持不变。
- 1 个 categorical `asset_id`。
- 合计 468 个 LightGBM 基础输入；再加入 `responder_03/02` 的两个 OOF `responder_hat` 后，target 输入为 470 列。

正式训练默认读取紧凑的 `long_horizon_468_feature_plan.json`。全量 `all_feature_long_horizon_plan.json` 仅用于专项消融；它为全部 323 个原始 feature 生成 `historical_zscore20`、`minus_ema20`、`rolling_std20`，同时保留紧凑计划中的 48 个 `rmean5`、8 个重要 `lag1` 和 7 个重要 `diff1`：

- 全 feature 长期列：323 × 3 = 969。
- 紧凑计划额外短期列：48 + 8 + 7 = 63。
- 时序衍生列：969 + 63 = 1032。
- 完整基础矩阵：323 + 1032 + categorical `asset_id` = 1356。
- 加入两个 `responder_hat` 后为 1358 列。

两个配方都设置 `exact_recipes=true`，加载器不会自动追加 60 步长变换。需要重新冻结两个配方时依次运行：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/build_long_horizon_plan.py --importance examples/responder_assisted_lgb_catboost_strategy/model/target_feature_importance.csv --base-plan examples/responder_assisted_lgb_catboost_strategy/baseline_468_feature_plan.json --output examples/responder_assisted_lgb_catboost_strategy/long_horizon_468_feature_plan.json
```

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/build_all_feature_long_horizon_plan.py --compact-plan examples/responder_assisted_lgb_catboost_strategy/long_horizon_468_feature_plan.json --output examples/responder_assisted_lgb_catboost_strategy/all_feature_long_horizon_plan.json --raw-feature-count 323
```

如需继续试验分析器生成的路由，可显式传入 `--temporal-plan examples/responder_assisted_lgb_catboost_strategy/analysis/temporal_feature_plan.json`。

### 5. 训练模型

默认 468 列计划在内存充足时推荐一次性装载，以减少 `Sequence` 和磁盘分片开销：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode in-memory --threads 8
```

内存不足或显式试验 1356 列全 feature 计划时使用磁盘分片外存训练：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode out-of-core --threads 8
```

两种模式使用完全相同的时间切分、特征和标签。`in-memory` 使用连续 `float32` 矩阵；`out-of-core` 使用缓存分片和 `lightgbm.Sequence`。

当前默认启用强化版 LightGBM 训练协议：

- 将 `asset_id` 作为 categorical feature，而不是普通连续数值。
- 保留最后 15% 时间作为 terminal holdout；前 85% 内部使用 expanding-window OOF，并在每个边界加入 30 个观测时点的 purge。
- 每个 OOF 验证折和 terminal holdout 都从空时序状态启动，严格模拟测试 API 冷启动。`historical_zscore20` 和 `rolling_std20` 只依赖前 20 期，但 `minus_ema20` 是递归状态；为保证完全无泄漏，当前长期计划会重建整个验证会话的时序列，而不是只修补前 20 个 `time_id`。
- target 固定使用强化正则后的 `smoothed`：47 个叶子、深度 10、每叶至少 5000 行、0.8 行列采样、L1=2、L2=30、`path_smooth=150`。不再在每次训练中重复比较 `reference/guarded`。
- 正式 responder 固定为 `responder_03/02`；`responder_28/29` 已从默认 OOF、训练套件和机制消融中移除，`responder_22/23` 也不进入候选梯队。
- target 晋级同时要求 OOF、holdout 和预测尺度合理；复杂变体还必须在平均折、至少 80% 可评估折、最新折和 holdout 上击败更简单的父模型。默认套件在 CV 冻结后只让 CV 冠军及其必要父模型进入 terminal holdout；专项诊断套件才会评估全部变体。
- 预测裁剪边界只用 OOF 预测拟合，并且只有 OOF 与 holdout 都不变差时才在部署中启用。报告同时保存原始、裁剪和最终部署分数。
- 选型结束后把 OOF 区间与 terminal holdout 标签合并，以 2026/2027/2028 三个种子重训 target；推理取三模型均值。
- 用于验证的 responder 模型保存在 `work/selection_responder_models/`，与最终使用全部数据重训的部署 responder 模型严格分离。
- 缓存绑定输入 Parquet 的路径、大小和修改时间；输入变化会自动重建。缓存同时对最早 500,000 行生成 `feature_health.json`，审计低有限值比例和常数特征，但默认不自动删列。

可通过 `--purge-steps`、`--target-seeds`、`--valid-time-fraction`、`--oof-folds` 和 `--feature-health-rows` 覆盖这些默认值。`--target-param-candidates` 当前只接受 `smoothed`。至少需要 3 个 responder OOF 折，默认 5 折会提供 4 个可用于 target 选型的时间折。当前参数依赖 LightGBM 4.6.x。

断点恢复并跳过兼容的已有模型：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode out-of-core --skip-existing-models --threads 8
```

强制重建缓存：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode out-of-core --rebuild-cache --threads 8
```

### 6. Target 消融套件

默认 `--experiment-suite next-step` 训练：

- `A`：原始特征。
- `C4`：原始特征加 `responder_03/02` 的两个 responder_hat；名称为兼容既有产物而保留。
- `LGB468`：紧凑长期计划的 468 列特征，不加入 responder_hat。
- `LGB468_C4`：紧凑 468 列加两个 OOF responder_hat，共 470 列；名称为兼容既有选型链路而保留。

`A/C4/LGB468` 只作为父级对照，不具有正式部署资格，默认最终模型只能选择 `LGB468_C4`。变体首先按 development walk-forward 平均分排序，然后经过 terminal holdout 晋级门槛；holdout 不直接用于挑选最高分。仅包含对照模型的专项实验默认拒绝覆盖最终模型，必须在独立目录中显式传入 `--allow-control-deployment`。

从上一版 1356 列缓存切回紧凑计划时执行一次：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode in-memory --temporal-plan examples/responder_assisted_lgb_catboost_strategy/long_horizon_468_feature_plan.json --experiment-suite next-step --target-param-candidates smoothed --rebuild-cache --threads 8
```

第一次完成后，后续相同配置使用 `--skip-existing-models`，不要再传 `--rebuild-cache`。

原始 A/B/C/D 套件：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model_legacy --experiment-suite legacy --allow-control-deployment --threads 8
```

当前两个 responder 的专项消融：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model_responder_diagnostic --experiment-suite responder --allow-control-deployment --threads 8
```

解释 C4 收益来源的配对机制消融：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model_c4_mechanism --training-data-mode in-memory --experiment-suite c4-mechanism --allow-control-deployment --skip-existing-models --threads 8
```

该套件统一运行 A、C4、两个单 responder、两个 leave-one-out 和 `C4_SHUFFLED`。打乱对照在每个 `time_id` 内分别重排 `responder_hat`，保留当期分布但破坏样本对应关系；它只用于诊断，永远不会被选为部署模型。结果写入 `ablation_report.json`，并额外生成 `c4_mechanism_report.json`，其中正的 leave-one-out 数值表示删除该 responder 后 C4 变差。建议复用已有 `work/`，避免重新生成相同的 C4 OOF responder。

对筛选梯队中移除 `responder_22/23` 后剩余的10个 responder 运行完整 OOF 单 responder 实验：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work_single_responder --model-dir examples/responder_assisted_lgb_catboost_strategy/model_single_responder --training-data-mode in-memory --experiment-suite single-responder --allow-control-deployment --threads 8
```

该套件默认使用 `responder_14,09,08,10,21,42,07,15,41,24`，训练 A 基线以及每个候选单独加入一个 OOF `responder_hat` 的10个 target 模型。`responder_22/23` 不再进入后续训练。建议使用独立的 `work_single_responder/` 和 `model_single_responder/`，避免覆盖当前正式 C4。

也可以显式指定 responder 列表：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work_custom_responders --model-dir examples/responder_assisted_lgb_catboost_strategy/model_custom_responders --training-data-mode in-memory --experiment-suite single-responder --responders responder_14,responder_09,responder_22 --allow-control-deployment --threads 8
```

每个实验都会输出多折分数、OOF 汇总分、terminal holdout 原始/裁剪分数、预测缩放诊断、晋级门槛、最佳迭代轮数和特征重要性。通过门槛且 walk-forward 平均分最高的变体决定最终特征集合和轮数；若所有变体都未通过，报告会明确警告并回退到 CV 冠军。三种子最终模型保存为 `model/target_final_seed*.txt`，`model/target_lightgbm.txt` 保留为首个种子的兼容别名。精确特征列顺序、responder 子集和模型列表写入 `model/metadata.json`。

### 7. 审计 responder

必须使用与模型相同的 `work/cache`：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/audit_responders.py --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --output-dir examples/responder_assisted_lgb_catboost_strategy/audit
```

报告中的 `model_cache_compatible` 应为 `true`，cache schema、temporal engine version、plan hash 和特征列必须与模型一致。

### 8. 本地时序推理

```powershell
python3 timeseries_api/run_timeseries_api.py --data-root data --strategy-dir examples/responder_assisted_lgb_catboost_strategy --output examples/responder_assisted_lgb_catboost_strategy/submission.csv
```

推理严格要求 `time_id` 递增。模型先更新历史时序状态，再预测需要的 responder_hat，最后预测 target。

## 进度显示

训练过程显示：

- 缓存预处理行数和分片数。
- 每个 LightGBM 的 boosting 轮次。
- OOF responder 模型总体完成比例。
- 最终 responder 模型完成比例。
- target 实验完成比例。
- 验证预测批次完成比例。

推理过程根据 manifest 中的总行数显示百分比、已完成行数、当前 `time_id`、时间步数量、累计耗时和预计剩余时间。

## 重要约束

- 真实 responder 只存在于训练数据，推理时只能使用 responder 模型预测值。
- OOF responder 必须严格按时间 expanding-window 生成，不能用同一训练样本的拟合值训练 target。
- `metadata.json` 中的 `valid_score` 来自重训前的选型模型；最终三种子模型已经吸收末端验证标签，因此不能再把该区间当作最终模型的独立验证集。
- 时序特征只能使用当前 `time_id` 之前的历史状态。
- 不要设置 `predict_disable_shape_check=true` 绕过维度错误；应检查 metadata 和模型列顺序。
- 不要把 `work/`、`model/`、Parquet、NumPy 缓存或虚拟环境提交到 Git。
