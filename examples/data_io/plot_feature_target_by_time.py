from __future__ import annotations

import argparse
import fnmatch
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot each selected feature and target against time_id. Values are "
            "streamed from parquet partitions and aggregated by time_id."
        )
    )
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2] / "data"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--output-dir", default="plots/time_series")
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help=(
            "Columns or glob patterns to plot, for example feature_000 target "
            "or feature_00*. Defaults to all feature_* columns plus target when present."
        ),
    )
    parser.add_argument("--asset-id", type=int, default=None, help="Optional asset_id filter.")
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument(
        "--column-group-size",
        type=int,
        default=24,
        help="Number of value columns to read per streaming pass.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Downsample plotted points per chart after aggregation. Use 0 to disable.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output image format.",
    )
    return parser.parse_args()


def split_files(data_root: str | Path, split: str) -> list[Path]:
    root = Path(data_root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {}).get(split, [])
        if files:
            return [root / str(file) for file in files]
    return sorted((root / split).glob("*.parquet"))


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def import_plot_deps():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Missing dependency: install pyarrow to read parquet files.") from exc

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Missing dependency: install matplotlib to create plots.") from exc

    return pq, plt


def parquet_columns(pq, path: Path) -> list[str]:
    return pq.ParquetFile(path).schema.names


def selected_value_columns(all_columns: list[str], patterns: list[str] | None, split: str) -> list[str]:
    if patterns:
        selected: list[str] = []
        for pattern in patterns:
            matches = [col for col in all_columns if fnmatch.fnmatch(col, pattern)]
            selected.extend(matches if matches else [pattern])
        missing = [col for col in selected if col not in all_columns]
        if missing:
            raise ValueError(f"columns not found in {split!r} split: {missing}")
        return list(dict.fromkeys(selected))

    value_columns = [col for col in all_columns if col.startswith("feature_")]
    if "target" in all_columns:
        value_columns.append("target")
    return value_columns


def aggregate_by_time(
    pq,
    files: list[Path],
    value_columns: list[str],
    *,
    asset_id: int | None,
    batch_size: int,
) -> dict[str, pd.Series]:
    sum_parts: list[pd.DataFrame] = []
    count_parts: list[pd.DataFrame] = []
    read_columns = ["time_id", *value_columns]
    if asset_id is not None:
        read_columns.insert(1, "asset_id")

    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
            frame = batch.to_pandas()
            if asset_id is not None:
                frame = frame.loc[frame["asset_id"] == asset_id, ["time_id", *value_columns]]
            if frame.empty:
                continue

            grouped = frame.groupby("time_id", sort=True)[value_columns]
            sum_parts.append(grouped.sum(numeric_only=True))
            count_parts.append(grouped.count())

    if not sum_parts:
        return {col: pd.Series(dtype="float64", name=col) for col in value_columns}

    sums = pd.concat(sum_parts).groupby(level=0).sum(numeric_only=True)
    counts = pd.concat(count_parts).groupby(level=0).sum(numeric_only=True)
    means = sums.divide(counts).sort_index()
    return {col: means[col].dropna() for col in value_columns}


def downsample(series: pd.Series, max_points: int) -> pd.Series:
    if max_points <= 0 or len(series) <= max_points:
        return series
    step = max(1, len(series) // max_points)
    return series.iloc[::step]


def plot_series(plt, series: pd.Series, *, column: str, output_dir: Path, fmt: str, max_points: int) -> Path:
    output_path = output_dir / f"{column}.{fmt}"
    plotted = downsample(series, max_points)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plotted.index.to_numpy(), plotted.to_numpy(), linewidth=0.9)
    ax.set_title(f"{column} by time_id")
    ax.set_xlabel("time_id")
    ax.set_ylabel(column)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.column_group_size <= 0:
        raise ValueError("--column-group-size must be positive")

    pq, plt = import_plot_deps()
    files = split_files(args.data_root, args.split)
    if not files:
        raise FileNotFoundError(f"no parquet files found for split={args.split!r}")

    all_columns = parquet_columns(pq, files[0])
    value_columns = selected_value_columns(all_columns, args.columns, args.split)
    if not value_columns:
        raise ValueError("no feature_* or target columns found to plot")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for group in batched(value_columns, args.column_group_size):
        series_by_column = aggregate_by_time(
            pq,
            files,
            group,
            asset_id=args.asset_id,
            batch_size=args.batch_size,
        )
        for column, series in series_by_column.items():
            if series.empty:
                print(f"skip {column}: no rows after filtering")
                continue
            output_path = plot_series(
                plt,
                series,
                column=column,
                output_dir=output_dir,
                fmt=args.format,
                max_points=args.max_points,
            )
            written.append(str(output_path))
            print(f"wrote {output_path}")

    summary_path = output_dir / "plot_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "data_root": str(Path(args.data_root)),
                "split": args.split,
                "asset_id": args.asset_id,
                "columns": value_columns,
                "plot_count": len(written),
                "plots": written,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
