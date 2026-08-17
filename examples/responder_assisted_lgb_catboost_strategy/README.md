# Responder 辅助 LightGBM 时序策略

该目录实现严格按时间切分的 responder stacking 策略。训练阶段先使用特征预测 responder，再把无泄漏的 OOF `responder_hat` 与原始及时序特征一起用于 target 模型；推理阶段只使用特征和 responder 模型预测值，不会访问真实 responder。

## 目录与产物

- `train.py`：缓存预处理、OOF responder、target 消融和最终模型训练。
- `main.py`：时序 API 推理入口。
- `temporal_features.py`：严格历史时序特征状态机。
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

训练脚本默认读取 `analysis/temporal_feature_plan.json`。若文件不存在，则对选中的原始特征使用默认时序变换；默认已排除低价值的 `delta1` 和 `xs_rank_delta1`。

### 5. 训练模型

内存充足时推荐一次性装载当前训练和验证区间：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode in-memory --threads 8
```

内存较小时使用磁盘分片外存训练：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode out-of-core --threads 8
```

两种模式使用完全相同的时间切分、特征和标签。`in-memory` 使用连续 `float32` 矩阵；`out-of-core` 使用缓存分片和 `lightgbm.Sequence`。

当前默认启用强化版 LightGBM 训练协议：

- 将 `asset_id` 作为 categorical feature，而不是普通连续数值。
- 使用 15% 末端验证集、5 折 expanding-window OOF，并在每个训练/预测边界加入 30 个观测时点的 purge。
- 使用 63 叶、深度 12、大叶节点、L1/L2、`path_smooth` 和行列采样组成的强正则参数组。
- 先在无泄漏验证集选定 target 变体和轮数，再把 OOF 区间与末端验证标签合并，以 2026/2027/2028 三个种子重训 target；推理取三模型均值。
- 用于验证的 responder 模型保存在 `work/selection_responder_models/`，与最终使用全部数据重训的部署 responder 模型严格分离。

可通过 `--purge-steps`、`--target-seeds`、`--valid-time-fraction` 和 `--oof-folds` 覆盖这些默认值。新参数组依赖 LightGBM 4.6.x。

断点恢复并跳过兼容的已有模型：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode in-memory --skip-existing-models --threads 8
```

强制重建缓存：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode in-memory --rebuild-cache --threads 8
```

### 6. Target 消融套件

默认 `--experiment-suite next-step` 训练：

- `A`：原始特征。
- `C4`：原始特征加四个 responder_hat。
- `C2`：原始特征加 responder_02/03。
- `T60`：C2 加60期波动率和 EMA 偏离。
- `T20_60`：C2 加20/60期波动率和 EMA 偏离。
- `TZ`：T20_60 加20/60期历史 z-score。

只训练指定实验：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --training-data-mode in-memory --target-experiments C2,T60,T20_60 --threads 8
```

原始 A/B/C/D 套件：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --experiment-suite legacy --threads 8
```

当前四个 responder 的专项消融：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model --experiment-suite responder --threads 8
```

解释 C4 收益来源的配对机制消融：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work --model-dir examples/responder_assisted_lgb_catboost_strategy/model_c4_mechanism --training-data-mode in-memory --experiment-suite c4-mechanism --skip-existing-models --threads 8
```

该套件统一运行 A、C4、四个单 responder、四个 leave-one-out 和 `C4_SHUFFLED`。打乱对照在每个 `time_id` 内分别重排 `responder_hat`，保留当期分布但破坏样本对应关系；它只用于诊断，永远不会被选为部署模型。结果写入 `ablation_report.json`，并额外生成 `c4_mechanism_report.json`，其中正的 leave-one-out 数值表示删除该 responder 后 C4 变差。建议复用已有 `work/`，避免重新生成相同的 C4 OOF responder。

对全量筛选结果中的一二梯队12个 responder 运行完整 OOF 单 responder 实验：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work_single_responder --model-dir examples/responder_assisted_lgb_catboost_strategy/model_single_responder --training-data-mode in-memory --experiment-suite single-responder --threads 8
```

该套件默认使用 `responder_14,09,08,10,22,23,21,42,07,15,41,24`，训练 A 基线以及每个候选单独加入一个 OOF `responder_hat` 的12个 target 模型。建议使用独立的 `work_single_responder/` 和 `model_single_responder/`，避免覆盖当前正式 C4。

也可以显式指定 responder 列表：

```powershell
python3 examples/responder_assisted_lgb_catboost_strategy/train.py --data-root data --work-dir examples/responder_assisted_lgb_catboost_strategy/work_custom_responders --model-dir examples/responder_assisted_lgb_catboost_strategy/model_custom_responders --training-data-mode in-memory --experiment-suite single-responder --responders responder_14,responder_09,responder_22 --threads 8
```

每个实验都会输出整体验证 R²、四个连续验证时间段的 R²、分段标准差、最低分、最佳迭代轮数和特征重要性。验证分数最高的可部署变体决定最终特征集合和轮数；三种子最终模型保存为 `model/target_final_seed*.txt`，`model/target_lightgbm.txt` 保留为首个种子的兼容别名。精确特征列顺序、responder 子集和模型列表写入 `model/metadata.json`。

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
