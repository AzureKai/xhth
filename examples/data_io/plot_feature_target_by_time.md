# Plot Feature and Target by Time

This script reads parquet partitions, aggregates selected columns by `time_id`,
and writes one plot per column.

Script path:

```powershell
examples\data_io\plot_feature_target_by_time.py
```

## Requirements

Install the required packages in the Python environment you use to run the
script:

```powershell
pip install pyarrow matplotlib pandas
```

## Basic Usage

Plot all `feature_*` columns and `target` from the training split:

```powershell
python examples\data_io\plot_feature_target_by_time.py --split train --output-dir plots\train_time_series
```

Plot only several columns:

```powershell
python examples\data_io\plot_feature_target_by_time.py --columns feature_000 feature_001 target
```

Plot columns matching a pattern:

```powershell
python examples\data_io\plot_feature_target_by_time.py --columns feature_00* target
```

Plot only one asset:

```powershell
python examples\data_io\plot_feature_target_by_time.py --asset-id 0 --columns feature_000 target
```

## Useful Options

- `--data-root data`: data directory containing `manifest.json`.
- `--split train`: choose `train` or `test`.
- `--output-dir plots/time_series`: directory for output images.
- `--columns feature_000 target`: choose columns or glob patterns.
- `--asset-id 0`: filter to one `asset_id`.
- `--batch-size 200000`: rows read per parquet batch.
- `--column-group-size 24`: value columns read per streaming pass.
- `--max-points 5000`: downsample plotted points per chart; use `0` to disable.
- `--format png`: choose `png`, `pdf`, or `svg`.

## Output

The script writes one image per plotted column, for example:

```text
plots/train_time_series/feature_000.png
plots/train_time_series/feature_001.png
plots/train_time_series/target.png
```

It also writes a summary file:

```text
plots/train_time_series/plot_summary.json
```
