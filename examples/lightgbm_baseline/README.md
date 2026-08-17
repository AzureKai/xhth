# LightGBM low-risk v3

This directory contains the reproducible v3 training and inference package used by team `noob` in the 2026 quantitative trading research competition.

The public-leaderboard score of the included three-seed ensemble is:

```text
0.00293294060904458
```

The package deliberately excludes competition data, generated submissions, caches, logs, and all post-v3 experiments.

## What is included

| Path | Purpose |
| --- | --- |
| `train.py` | In-memory strict walk-forward training entry point |
| `train_low_memory.py` | Disk-backed/memmap full training entry point used for v3 |
| `resume_low_memory.py` | Resume an interrupted final-seed fit |
| `main.py` | Sequential Time-Series API inference implementation |
| `validation.py` | Purged walk-forward, terminal holdout, weighted R2, and gates |
| `data_utils.py`, `features.py`, `preprocess.py` | Causal data and feature pipeline |
| `model_forward_lowrisk_v3/` | Three trained boosters and the complete training report |
| `tests/test_forward_validation.py` | Leakage and validation regression tests |

## Environment

Python 3.11 is recommended. Install the minimal dependencies from the repository root:

```bash
python -m pip install -r examples/lightgbm_baseline/requirements.txt
```

The published competition data is intentionally not committed. Put the official `data/` directory at the repository root. It must contain `manifest.json`, `train/`, and `test/`.

## Reproduce v3 training

The memory-bounded entry point is the recommended path for a machine with about 32 GB RAM:

```bash
python examples/lightgbm_baseline/train_low_memory.py \
  --release-root data \
  --model-dir examples/lightgbm_baseline/model_forward_lowrisk_v3_retrained \
  --cache-dir examples/lightgbm_baseline/.low_memory_cache_forward_v2 \
  --num-threads 24
```

Important protocol details:

- Validation is strictly forward in time with a one-sided 30-step purge.
- The final 15% of time is opened once as a terminal holdout.
- Feature schema and history-feature selection are frozen using the earliest training prefix only.
- The selected model uses seeds `2026`, `2027`, and `2028`.
- `asset_id` is trained as a categorical feature.
- `fitted_oof_scale` is diagnostic only; submitted predictions use scale `1.0`.

If the process is interrupted during final seed fitting, use the paths printed in the original log:

```bash
python examples/lightgbm_baseline/resume_low_memory.py \
  --model-dir examples/lightgbm_baseline/model_forward_lowrisk_v3_retrained \
  --cache-dir examples/lightgbm_baseline/.low_memory_cache_forward_v2 \
  --original-log path/to/training_stdout.log \
  --num-threads 24
```

## Generate a public submission

The repository already contains the accepted v3 weights. Run the official sequential inference runner from the repository root:

### PowerShell

```powershell
$env:LIGHTGBM_BASELINE_MODEL_DIR = "$PWD\examples\lightgbm_baseline\model_forward_lowrisk_v3"
$env:LIGHTGBM_BASELINE_PREDICT_THREADS = "4"
python timeseries_api\run_timeseries_api.py `
  --data-root data `
  --strategy-dir examples\lightgbm_baseline `
  --output submissions\v3_lowrisk.csv
```

### Bash

```bash
LIGHTGBM_BASELINE_MODEL_DIR="$PWD/examples/lightgbm_baseline/model_forward_lowrisk_v3" \
LIGHTGBM_BASELINE_PREDICT_THREADS=4 \
python timeseries_api/run_timeseries_api.py \
  --data-root data \
  --strategy-dir examples/lightgbm_baseline \
  --output submissions/v3_lowrisk.csv
```

The inference process must retain increasing `time_id` order because `main.py` maintains per-asset causal history.

## Verification

Run the published regression tests:

```bash
python -m unittest examples.lightgbm_baseline.tests.test_forward_validation -v
```

Before sharing a generated CSV, verify that it has exactly the official `row_id,target` columns, preserves the sample-submission row order, and contains only finite targets.
