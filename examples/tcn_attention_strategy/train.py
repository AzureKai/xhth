from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def require_package(package: str, install_name: str | None = None) -> None:
    if importlib.util.find_spec(package) is None:
        name = install_name or package
        raise ImportError(f"missing dependency '{name}'. Install requirements from requirement.txt.")


require_package("torch")
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TCN + cross-asset attention strategy.")
    parser.add_argument("--data-root", required=True, help="Release data root containing manifest.json.")
    parser.add_argument("--model-dir", required=True, help="Directory where model artifacts are written.")
    parser.add_argument("--valid-time-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-times", type=int, default=0, help="Use the most recent N train time_id values before validation; 0 means all.")
    parser.add_argument("--max-valid-times", type=int, default=0, help="Use the first N validation time_id values; 0 means all.")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--tcn-layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dilations", default="1,2,4")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--domain-count", type=int, default=5, help="Number of contiguous train time domains for REx loss.")
    parser.add_argument("--rex-lambda", type=float, default=1.0, help="Penalty weight for variance of per-domain training losses; 0 disables REx.")
    parser.add_argument("--batch-size-parquet", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--alpha-grid", default="0.05,0.1,0.2,0.3,0.5,0.8,1.0,1.2,1.5")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def manifest_files(data_root: Path, split: str) -> list[Path]:
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {}).get(split, [])
        if files:
            return [data_root / str(file) for file in files]
    return sorted((data_root / split).glob("*.parquet"))


def parquet_columns(path: Path) -> list[str]:
    require_package("pyarrow")
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


def read_unique_times(files: list[Path], batch_size: int) -> np.ndarray:
    require_package("pyarrow")
    import pyarrow.parquet as pq

    chunks: list[np.ndarray] = []
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["time_id"]):
            chunks.append(np.unique(batch.column(0).to_numpy(zero_copy_only=False)))
    if not chunks:
        raise ValueError("no time_id values found")
    return np.unique(np.concatenate(chunks)).astype(np.int64)


def select_time_ids(
    unique_times: np.ndarray,
    valid_time_fraction: float,
    max_train_times: int,
    max_valid_times: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not 0.0 < valid_time_fraction < 1.0:
        raise ValueError("valid-time-fraction must be in (0, 1)")
    valid_count = max(1, int(round(len(unique_times) * valid_time_fraction)))
    valid_count = min(valid_count, len(unique_times) - 1)
    cutoff_time = int(unique_times[-valid_count])
    train_times = unique_times[unique_times < cutoff_time]
    valid_times = unique_times[unique_times >= cutoff_time]
    if max_train_times > 0:
        train_times = train_times[-max_train_times:]
    if max_valid_times > 0:
        valid_times = valid_times[:max_valid_times]
    if len(train_times) == 0 or len(valid_times) == 0:
        raise ValueError("selected train or validation time_id set is empty")
    return train_times.astype(np.int64), valid_times.astype(np.int64), cutoff_time


def assign_time_domains(time_ids: np.ndarray, domain_count: int) -> tuple[dict[int, int], list[int]]:
    if domain_count <= 0:
        raise ValueError("domain-count must be positive")
    domain_count = min(int(domain_count), len(time_ids))
    chunks = np.array_split(np.asarray(time_ids, dtype=np.int64), domain_count)
    mapping: dict[int, int] = {}
    sizes: list[int] = []
    for domain_id, chunk in enumerate(chunks):
        sizes.append(int(len(chunk)))
        for time_id in chunk:
            mapping[int(time_id)] = int(domain_id)
    return mapping, sizes


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one float")
    return values


def iter_time_frames(files: list[Path], columns: list[str], batch_size: int) -> Iterator[tuple[int, pd.DataFrame]]:
    require_package("pyarrow")
    import pyarrow.parquet as pq

    carry = pd.DataFrame()
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pandas()
            if not carry.empty:
                frame = pd.concat([carry, frame], ignore_index=True)
                carry = pd.DataFrame()
            if frame.empty:
                continue

            previous_group: tuple[int, pd.DataFrame] | None = None
            for time_id, current in frame.groupby("time_id", sort=False):
                if previous_group is not None:
                    yield previous_group[0], previous_group[1].reset_index(drop=True)
                previous_group = (int(time_id), current.copy())
            if previous_group is not None:
                carry = previous_group[1]
    if not carry.empty:
        yield int(carry["time_id"].iloc[0]), carry.reset_index(drop=True)


def compute_feature_stats(
    files: list[Path],
    feature_columns: list[str],
    train_times: set[int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    columns = ["time_id", *feature_columns]
    sums = np.zeros(len(feature_columns), dtype=np.float64)
    sums_sq = np.zeros(len(feature_columns), dtype=np.float64)
    counts = np.zeros(len(feature_columns), dtype=np.float64)

    for time_id, frame in iter_time_frames(files, columns, batch_size):
        if time_id not in train_times:
            continue
        values = frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
        finite = np.isfinite(values)
        safe_values = np.where(finite, values, 0.0).astype(np.float64)
        sums += safe_values.sum(axis=0)
        sums_sq += (safe_values * safe_values).sum(axis=0)
        counts += finite.sum(axis=0)

    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variance = np.divide(sums_sq, counts, out=np.ones_like(sums_sq), where=counts > 0) - mean * mean
    std = np.sqrt(np.maximum(variance, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


def rank_matrix(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    asset_count, feature_dim = values.shape
    rank = np.zeros((asset_count, feature_dim), dtype=np.float32)
    valid_idx = np.flatnonzero(present)
    if len(valid_idx) == 0:
        return rank
    if len(valid_idx) == 1:
        rank[valid_idx[0], :] = 1.0
        return rank
    valid_values = values[valid_idx]
    for col_idx in range(feature_dim):
        column = valid_values[:, col_idx]
        order = np.argsort(column, kind="mergesort")
        sorted_values = column[order]
        sorted_ranks = np.empty(len(valid_idx), dtype=np.float32)
        start = 0
        while start < len(valid_idx):
            end = start + 1
            while end < len(valid_idx) and sorted_values[end] == sorted_values[start]:
                end += 1
            sorted_ranks[start:end] = ((start + 1.0) + end) * 0.5 / len(valid_idx)
            start = end
        ranks = np.empty(len(valid_idx), dtype=np.float32)
        ranks[order] = sorted_ranks
        rank[valid_idx, col_idx] = ranks
    return rank


def cross_section_features(current: np.ndarray, present: np.ndarray) -> np.ndarray:
    asset_count, feature_dim = current.shape
    out = np.zeros((asset_count, feature_dim * 3), dtype=np.float32)
    valid_idx = np.flatnonzero(present)
    if len(valid_idx) == 0:
        return out
    values = current[valid_idx]
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    demean = current - mean
    zscore = demean / std
    rank = rank_matrix(current, present)
    out[:, 0:feature_dim] = np.where(present[:, None], demean, 0.0)
    out[:, feature_dim : feature_dim * 2] = np.where(present[:, None], rank, 0.0)
    out[:, feature_dim * 2 :] = np.where(present[:, None], zscore, 0.0)
    return out


class SampleStore:
    def __init__(
        self,
        root: Path,
        *,
        name: str,
        length: int,
        asset_count: int,
        window_size: int,
        feature_dim: int,
    ):
        root.mkdir(parents=True, exist_ok=True)
        self.length = length
        self.asset_count = asset_count
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.history = np.memmap(root / f"{name}_history.dat", dtype="float32", mode="w+", shape=(length, asset_count, window_size, feature_dim))
        self.cross_section = np.memmap(root / f"{name}_cross_section.dat", dtype="float32", mode="w+", shape=(length, asset_count, feature_dim * 3))
        self.y = np.memmap(root / f"{name}_y.dat", dtype="float32", mode="w+", shape=(length, asset_count))
        self.sample_weight = np.memmap(root / f"{name}_sample_weight.dat", dtype="float32", mode="w+", shape=(length, asset_count))
        self.present = np.memmap(root / f"{name}_present.dat", dtype="bool", mode="w+", shape=(length, asset_count))
        self.domain = np.memmap(root / f"{name}_domain.dat", dtype="int64", mode="w+", shape=(length,))

    def flush(self) -> None:
        self.history.flush()
        self.cross_section.flush()
        self.y.flush()
        self.sample_weight.flush()
        self.present.flush()
        self.domain.flush()


def discover_asset_ids(
    files: list[Path],
    selected_times: set[int],
    batch_size: int,
) -> list[int]:
    seen: set[int] = set()
    for time_id, frame in iter_time_frames(files, ["time_id", "asset_id"], batch_size):
        if time_id in selected_times:
            seen.update(int(value) for value in frame["asset_id"].to_numpy(dtype=np.int64, copy=False))
    if not seen:
        raise ValueError("no asset_id values found in selected times")
    return sorted(seen)


def build_store(
    files: list[Path],
    feature_columns: list[str],
    selected_times: np.ndarray,
    asset_ids: list[int],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    window_size: int,
    batch_size: int,
    store: SampleStore,
    time_to_domain: dict[int, int],
) -> int:
    selected = {int(value) for value in selected_times}
    asset_to_slot = {asset_id: idx for idx, asset_id in enumerate(asset_ids)}
    history = np.zeros((len(asset_ids), window_size, len(feature_columns)), dtype=np.float32)
    columns = ["time_id", "asset_id", "weight", "target", *feature_columns]
    out_idx = 0

    for time_id, frame in iter_time_frames(files, columns, batch_size):
        if time_id not in selected:
            continue
        current = np.zeros((len(asset_ids), len(feature_columns)), dtype=np.float32)
        y = np.zeros(len(asset_ids), dtype=np.float32)
        sample_weight = np.zeros(len(asset_ids), dtype=np.float32)
        present = np.zeros(len(asset_ids), dtype=bool)

        raw = frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = (raw - feature_mean) / feature_std
        frame_asset_ids = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
        targets = pd.to_numeric(frame["target"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        weights = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        weights = np.maximum(weights, 0.0)

        for row_idx, asset_id_raw in enumerate(frame_asset_ids):
            slot = asset_to_slot.get(int(asset_id_raw))
            if slot is None:
                continue
            current[slot] = normalized[row_idx]
            y[slot] = targets[row_idx]
            sample_weight[slot] = weights[row_idx]
            present[slot] = True

        history[:, :-1, :] = history[:, 1:, :]
        history[:, -1, :] = np.where(present[:, None], current, 0.0)
        store.history[out_idx] = history
        store.cross_section[out_idx] = cross_section_features(current, present)
        store.y[out_idx] = y
        store.sample_weight[out_idx] = sample_weight
        store.present[out_idx] = present
        store.domain[out_idx] = int(time_to_domain.get(int(time_id), 0))
        out_idx += 1

    store.flush()
    if out_idx != len(selected_times):
        raise ValueError(f"built {out_idx} samples, expected {len(selected_times)}")
    return out_idx


class MemmapDataset(Dataset):
    def __init__(self, store: SampleStore):
        self.store = store

    def __len__(self) -> int:
        return self.store.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.asarray(self.store.history[idx])),
            torch.from_numpy(np.asarray(self.store.cross_section[idx])),
            torch.from_numpy(np.asarray(self.store.present[idx])),
            torch.from_numpy(np.asarray(self.store.y[idx])),
            torch.from_numpy(np.asarray(self.store.sample_weight[idx])),
            torch.as_tensor(int(self.store.domain[idx]), dtype=torch.long),
        )


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.proj is None else self.proj(x)
        y = F.pad(x, (self.left_padding, 0))
        y = self.conv(y)
        y = torch.relu(y)
        y = y.transpose(1, 2)
        y = self.norm(y)
        y = y.transpose(1, 2)
        y = self.dropout(y)
        return y + residual


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(x, x, x, key_padding_mask=~present.bool(), need_weights=False)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TcnAttentionNet(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        asset_count: int,
        hidden_dim: int,
        tcn_layers: int,
        kernel_size: int,
        dilations: list[int],
        attention_heads: int,
        attention_layers: int,
        dropout: float,
    ):
        super().__init__()
        blocks = []
        in_channels = feature_dim
        for idx in range(tcn_layers):
            dilation = dilations[idx % len(dilations)]
            blocks.append(CausalConvBlock(in_channels, hidden_dim, kernel_size, dilation, dropout))
            in_channels = hidden_dim
        self.temporal = nn.Sequential(*blocks)
        self.cross_attention = nn.ModuleList(
            [AttentionBlock(hidden_dim, attention_heads, dropout) for _ in range(attention_layers)]
        )
        self.cross_section = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.asset_embedding = nn.Embedding(asset_count, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, history: torch.Tensor, cross_section: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        batch_size, asset_count, window_size, feature_dim = history.shape
        temporal_input = history.reshape(batch_size * asset_count, window_size, feature_dim).transpose(1, 2)
        temporal = self.temporal(temporal_input)[:, :, -1].reshape(batch_size, asset_count, -1)
        mixed = temporal
        for block in self.cross_attention:
            mixed = block(mixed, present)
        xs = self.cross_section(cross_section)
        slot_ids = torch.arange(asset_count, device=history.device).unsqueeze(0).expand(batch_size, asset_count)
        asset = self.asset_embedding(slot_ids)
        return self.head(torch.cat([temporal, mixed, xs, asset], dim=-1)).squeeze(-1)


def weighted_loss(pred: torch.Tensor, y: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    denominator = sample_weight.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        return torch.mean((pred - y) ** 2)
    return torch.sum(sample_weight * (pred - y) ** 2) / denominator


def rex_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    sample_weight: torch.Tensor,
    domain: torch.Tensor,
    penalty_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if penalty_weight <= 0.0:
        erm = weighted_loss(pred, y, sample_weight)
        return erm, erm.detach(), torch.zeros((), device=pred.device)

    domain_losses = []
    for domain_id in torch.unique(domain):
        mask = domain == domain_id
        if bool(mask.any()):
            domain_losses.append(weighted_loss(pred[mask], y[mask], sample_weight[mask]))
    if not domain_losses:
        erm = weighted_loss(pred, y, sample_weight)
        return erm, erm.detach(), torch.zeros((), device=pred.device)

    stacked = torch.stack(domain_losses)
    erm = stacked.mean()
    penalty = stacked.var(unbiased=False) if len(domain_losses) > 1 else torch.zeros((), device=pred.device)
    return erm + float(penalty_weight) * penalty, erm.detach(), penalty.detach()


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray) -> float:
    weight = np.maximum(np.asarray(sample_weight, dtype=np.float64), 0.0)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denominator = np.sum(weight * y_true * y_true)
    if denominator <= 0:
        return 0.0
    numerator = np.sum(weight * (y_true - y_pred) ** 2)
    return float(1.0 - numerator / denominator)


def choose_device(raw: str) -> torch.device:
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        return torch.device("cuda")
    if raw == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    preds = []
    ys = []
    weights = []
    with torch.no_grad():
        for history, cross_section, present, y, sample_weight, _domain in loader:
            history = history.to(device=device, dtype=torch.float32)
            cross_section = cross_section.to(device=device, dtype=torch.float32)
            present = present.to(device=device)
            y = y.to(device=device, dtype=torch.float32)
            sample_weight = sample_weight.to(device=device, dtype=torch.float32)
            pred = model(history, cross_section, present)
            loss = weighted_loss(pred, y, sample_weight)
            losses.append(float(loss.detach().cpu()))
            preds.append(pred.detach().cpu().numpy().reshape(-1))
            ys.append(y.detach().cpu().numpy().reshape(-1))
            weights.append(sample_weight.detach().cpu().numpy().reshape(-1))
    pred_arr = np.concatenate(preds) if preds else np.empty(0, dtype=np.float32)
    y_arr = np.concatenate(ys) if ys else np.empty(0, dtype=np.float32)
    weight_arr = np.concatenate(weights) if weights else np.empty(0, dtype=np.float32)
    return float(np.mean(losses) if losses else math.inf), pred_arr, y_arr, weight_arr


def optimize_prediction_scale(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray, alpha_grid: list[float]) -> tuple[float, float]:
    best_alpha = float(alpha_grid[0])
    best_score = weighted_zero_mean_r2(y_true, y_pred * best_alpha, sample_weight)
    for alpha in alpha_grid[1:]:
        score = weighted_zero_mean_r2(y_true, y_pred * float(alpha), sample_weight)
        if score > best_score:
            best_alpha = float(alpha)
            best_score = score
    return best_alpha, best_score


def prediction_clip_bounds(pred: np.ndarray, sample_weight: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(pred) & (sample_weight > 0)
    if not valid.any():
        return -1.0, 1.0
    lower, upper = np.quantile(pred[valid], [0.001, 0.999])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return -1.0, 1.0
    return float(lower), float(upper)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = model_dir / "_sample_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_files = manifest_files(data_root, "train")
    if not train_files:
        raise ValueError(f"no train parquet files found under {data_root}")
    columns = parquet_columns(train_files[0])
    feature_columns = [col for col in columns if str(col).startswith("feature_")]
    if not feature_columns:
        raise ValueError("no feature_* columns found")

    unique_times = read_unique_times(train_files, args.batch_size_parquet)
    train_times, valid_times, cutoff_time = select_time_ids(
        unique_times,
        args.valid_time_fraction,
        args.max_train_times,
        args.max_valid_times,
    )
    train_time_to_domain, train_domain_sizes = assign_time_domains(train_times, args.domain_count)
    valid_time_to_domain, valid_domain_sizes = assign_time_domains(valid_times, args.domain_count)
    selected_times = set(int(value) for value in np.concatenate([train_times, valid_times]))
    asset_ids = discover_asset_ids(train_files, selected_times, args.batch_size_parquet)
    feature_mean, feature_std = compute_feature_stats(
        train_files,
        feature_columns,
        {int(value) for value in train_times},
        args.batch_size_parquet,
    )
    feature_std = np.where(feature_std > 1e-6, feature_std, 1.0).astype(np.float32)

    train_store = SampleStore(
        cache_dir,
        name="train",
        length=len(train_times),
        asset_count=len(asset_ids),
        window_size=args.window_size,
        feature_dim=len(feature_columns),
    )
    valid_store = SampleStore(
        cache_dir,
        name="valid",
        length=len(valid_times),
        asset_count=len(asset_ids),
        window_size=args.window_size,
        feature_dim=len(feature_columns),
    )
    build_store(
        train_files,
        feature_columns,
        train_times,
        asset_ids,
        feature_mean,
        feature_std,
        args.window_size,
        args.batch_size_parquet,
        train_store,
        train_time_to_domain,
    )
    build_store(
        train_files,
        feature_columns,
        valid_times,
        asset_ids,
        feature_mean,
        feature_std,
        args.window_size,
        args.batch_size_parquet,
        valid_store,
        valid_time_to_domain,
    )

    train_loader = DataLoader(
        MemmapDataset(train_store),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        MemmapDataset(valid_store),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    dilations = parse_int_list(args.dilations)
    device = choose_device(args.device)
    model = TcnAttentionNet(
        feature_dim=len(feature_columns),
        asset_count=len(asset_ids),
        hidden_dim=args.hidden_dim,
        tcn_layers=args.tcn_layers,
        kernel_size=args.kernel_size,
        dilations=dilations,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history_report: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_erm_losses = []
        train_rex_penalties = []
        for history, cross_section, present, y, sample_weight, domain in train_loader:
            history = history.to(device=device, dtype=torch.float32)
            cross_section = cross_section.to(device=device, dtype=torch.float32)
            present = present.to(device=device)
            y = y.to(device=device, dtype=torch.float32)
            sample_weight = sample_weight.to(device=device, dtype=torch.float32)
            domain = domain.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(history, cross_section, present)
            loss, erm_loss, rex_penalty = rex_loss(pred, y, sample_weight, domain, args.rex_lambda)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_erm_losses.append(float(erm_loss.detach().cpu()))
            train_rex_penalties.append(float(rex_penalty.detach().cpu()))

        valid_loss, valid_pred, valid_y, valid_weight = evaluate(model, valid_loader, device)
        valid_score = weighted_zero_mean_r2(valid_y, valid_pred, valid_weight)
        report = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses) if train_losses else math.inf),
            "train_erm_loss": float(np.mean(train_erm_losses) if train_erm_losses else math.inf),
            "train_rex_penalty": float(np.mean(train_rex_penalties) if train_rex_penalties else 0.0),
            "valid_loss": valid_loss,
            "valid_score": valid_score,
        }
        history_report.append(report)
        print(json.dumps(report))

        if valid_score > best_score:
            best_score = valid_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping:
                break

    if best_state is None:
        raise ValueError("training did not produce a valid model state")

    model.load_state_dict(best_state)
    _, valid_pred, valid_y, valid_weight = evaluate(model, valid_loader, device)
    prediction_scale, scaled_score = optimize_prediction_scale(valid_y, valid_pred, valid_weight, parse_float_list(args.alpha_grid))
    scaled_pred = valid_pred * prediction_scale
    clip_min, clip_max = prediction_clip_bounds(scaled_pred, valid_weight)

    torch.save({"state_dict": best_state}, model_dir / "model.pt")
    metadata = {
        "strategy": "tcn_attention_strategy",
        "feature_columns": feature_columns,
        "asset_ids": asset_ids,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "window_size": int(args.window_size),
        "hidden_dim": int(args.hidden_dim),
        "tcn_layers": int(args.tcn_layers),
        "kernel_size": int(args.kernel_size),
        "dilations": dilations,
        "attention_heads": int(args.attention_heads),
        "attention_layers": int(args.attention_layers),
        "dropout": float(args.dropout),
        "valid_time_fraction": float(args.valid_time_fraction),
        "valid_cutoff_time_id": int(cutoff_time),
        "train_time_count": int(len(train_times)),
        "valid_time_count": int(len(valid_times)),
        "domain_count": int(min(args.domain_count, len(train_times))),
        "rex_lambda": float(args.rex_lambda),
        "train_domain_sizes": train_domain_sizes,
        "valid_domain_sizes": valid_domain_sizes,
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_score),
        "prediction_scale": float(prediction_scale),
        "scaled_valid_score": float(scaled_score),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "training_history": history_report,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    final_report = {
        "strategy": metadata["strategy"],
        "device": str(device),
        "feature_count": len(feature_columns),
        "asset_ids": asset_ids,
        "train_time_count": metadata["train_time_count"],
        "valid_time_count": metadata["valid_time_count"],
        "domain_count": metadata["domain_count"],
        "rex_lambda": metadata["rex_lambda"],
        "best_epoch": best_epoch,
        "best_valid_score": best_score,
        "prediction_scale": prediction_scale,
        "scaled_valid_score": scaled_score,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "artifacts": ["metadata.json", "model.pt"],
    }
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
