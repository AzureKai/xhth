from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


START = time.perf_counter()


def progress(message):
    print(f"[progress {time.perf_counter() - START:9.1f}s] {message}", flush=True)


def progress_bar(label, current, total, detail=""):
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    width = 28
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" | {detail}" if detail else ""
    print(
        f"[{label:<22}] [{bar}] {100.0 * current / total:6.2f}% "
        f"({current:,}/{total:,}) {time.perf_counter() - START:9.1f}s"
        f"{suffix}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Causal per-asset TCN and LightGBM prediction ensemble."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--base-oof-predictions", required=True)
    parser.add_argument("--temporal-statistics", required=True)
    parser.add_argument("--feature-count", type=int, default=48)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-residual-weight", type=float, default=0.25)
    parser.add_argument("--min-base-scale", type=float, default=0.5)
    parser.add_argument("--max-base-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    return parser.parse_args()


def manifest_files(root):
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        values = manifest.get("files", {}).get("train", [])
        if values:
            return [root / value for value in values]
    return sorted((root / "train").glob("*.parquet"))


def select_features(files, statistics_path, count):
    available = [
        name for name in pq.read_schema(files[0]).names
        if name.startswith("feature_")
    ]
    frame = pd.read_csv(statistics_path)
    frame = frame[frame["feature"].isin(available)].copy()
    dynamic_columns = [
        "trend_score", "difference_score", "volatility_score",
        "shock_score", "mean_reversion_score",
    ]
    missing = [name for name in dynamic_columns if name not in frame]
    if missing:
        raise ValueError(f"temporal statistics missing columns: {missing}")
    scores = frame[dynamic_columns].to_numpy(dtype=np.float64)
    sorted_scores = np.sort(scores, axis=1)
    persistence = np.nanmax(
        np.abs(frame[[
            "level_acf1", "level_acf5", "level_acf20",
            "delta_acf1", "abs_delta_acf1", "squared_delta_acf1",
        ]].to_numpy(dtype=np.float64)),
        axis=1,
    )
    frame["tcn_sequence_score"] = (
        0.55 * sorted_scores[:, -1]
        + 0.25 * sorted_scores[:, -2]
        + 0.20 * np.clip(persistence, 0.0, 1.0)
    )
    frame = frame.sort_values("tcn_sequence_score", ascending=False)
    selected = frame["feature"].head(count).tolist()
    if not selected:
        raise ValueError("no usable feature columns")
    return selected, frame


def load_data(files, features):
    columns = ["time_id", "asset_id", "target", "weight", *features]
    frames = []
    total = 0
    for index, path in enumerate(files, start=1):
        frame = pd.read_parquet(path, columns=columns)
        frames.append(frame)
        total += len(frame)
        progress(f"loaded parquet {index}/{len(files)}; rows={total:,}")
        progress_bar("parquet loading", index, len(files), f"rows={total:,}")
    frame = pd.concat(frames, ignore_index=True)
    del frames
    raw = frame.loc[:, features].to_numpy(dtype=np.float32, copy=True)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "time": frame["time_id"].to_numpy(dtype=np.int64),
        "asset": frame["asset_id"].to_numpy(dtype=np.int64),
        "target": np.nan_to_num(
            frame["target"].to_numpy(dtype=np.float32), nan=0.0
        ),
        "weight": np.maximum(
            np.nan_to_num(frame["weight"].to_numpy(dtype=np.float32), nan=0.0),
            0.0,
        ),
        "raw": raw,
    }


class AssetSequenceDataset(Dataset):
    def __init__(self, features, targets, weights, asset_start, positions,
                 sequence_length):
        self.features = features
        self.targets = targets
        self.weights = weights
        self.asset_start = asset_start
        self.positions = np.asarray(positions, dtype=np.int64)
        self.sequence_length = int(sequence_length)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, item):
        position = int(self.positions[item])
        start = max(
            int(self.asset_start[position]),
            position - self.sequence_length + 1,
        )
        values = self.features[start:position + 1]
        sequence = np.zeros(
            (self.features.shape[1] + 1, self.sequence_length),
            dtype=np.float32,
        )
        length = len(values)
        sequence[:-1, -length:] = values.T
        sequence[-1, -length:] = 1.0
        return (
            torch.from_numpy(sequence),
            torch.tensor(self.targets[position], dtype=torch.float32),
            torch.tensor(self.weights[position], dtype=torch.float32),
            torch.tensor(position, dtype=torch.int64),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        output = self.conv1(values)
        if self.padding:
            output = output[:, :, :-self.padding]
        output = self.dropout(self.activation(output))
        output = self.conv2(output)
        if self.padding:
            output = output[:, :, :-self.padding]
        output = self.dropout(self.activation(output))
        return values + output


class TinyTCN(nn.Module):
    def __init__(self, input_channels, hidden_size, levels, kernel_size, dropout):
        super().__init__()
        self.input = nn.Conv1d(input_channels, hidden_size, 1)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, kernel_size, 2 ** level, dropout)
            for level in range(levels)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, values):
        output = self.input(values)
        for block in self.blocks:
            output = block(output)
        output = self.norm(output[:, :, -1])
        return self.head(output).squeeze(-1)


def weighted_r2(target, prediction, weight):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    denominator = np.sum(weight * target * target)
    if denominator <= 0:
        return 0.0
    return float(
        1.0 - np.sum(weight * (target - prediction) ** 2) / denominator
    )


def predict(model, loader, device, total_rows, label="TCN prediction"):
    model.eval()
    output = np.empty(total_rows, dtype=np.float32)
    total_batches = max(len(loader), 1)
    report_every = max(1, total_batches // 20)
    with torch.no_grad():
        for batch_index, (batch, _, _, positions) in enumerate(loader, start=1):
            values = model(batch.to(device)).cpu().numpy()
            output[positions.numpy()] = values
            if (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % report_every == 0
            ):
                progress_bar(label, batch_index, total_batches, "batches")
    return output


def segmented_scores(times, target, prediction, weight, parts=4):
    unique = np.unique(times)
    result = []
    for part, values in enumerate(np.array_split(unique, parts), start=1):
        mask = np.isin(times, values)
        result.append({
            "part": part,
            "time_start": int(values[0]),
            "time_end": int(values[-1]),
            "rows": int(mask.sum()),
            "score": weighted_r2(
                target[mask], prediction[mask], weight[mask]
            ),
        })
    return result


def calibrate_base(target, prediction, weight, minimum_scale, maximum_scale):
    weight_sum = max(float(np.sum(weight)), 1e-12)
    pred_mean = float(np.sum(weight * prediction) / weight_sum)
    target_mean = float(np.sum(weight * target) / weight_sum)
    centered = prediction - pred_mean
    denominator = float(np.sum(weight * centered * centered))
    scale = (
        float(np.sum(weight * centered * (target - target_mean)) / denominator)
        if denominator > 0 else 1.0
    )
    scale = float(np.clip(scale, minimum_scale, maximum_scale))
    intercept = target_mean - scale * pred_mean
    return intercept, scale


def residual_weight(target, base, residual_prediction, weight, maximum):
    denominator = np.sum(weight * residual_prediction * residual_prediction)
    if denominator <= 0:
        return 0.0
    beta = np.sum(
        weight * residual_prediction * (target - base)
    ) / denominator
    return float(np.clip(beta, 0.0, maximum))


def shuffled_within_time(values, times, seed):
    output = np.asarray(values).copy()
    rng = np.random.default_rng(seed)
    for time_id in np.unique(times):
        index = np.flatnonzero(times == time_id)
        output[index] = output[index[rng.permutation(len(index))]]
    return output


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cuda_available = bool(torch.cuda.is_available())
    if args.device == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested but cannot be initialized. Check the NVIDIA "
            "driver and install a matching PyTorch CUDA wheel."
        )
    resolved_device = (
        "cuda" if args.device == "auto" and cuda_available else args.device
    )
    if resolved_device == "auto":
        resolved_device = "cpu"
    device = torch.device(resolved_device)
    progress(
        f"PyTorch={torch.__version__}, compiled_cuda="
        f"{torch.version.cuda or 'cpu-only'}, cuda_available={cuda_available}, "
        f"selected_device={device}"
    )
    files = manifest_files(args.data_root)
    if not files:
        raise ValueError("no training parquet files")
    base_dir = Path(args.base_model_dir)
    base_metadata = json.loads(
        (base_dir / "metadata.json").read_text(encoding="utf-8")
    )
    base_artifact = np.load(base_dir / "validation_predictions.npz")
    base_oof_artifact = np.load(args.base_oof_predictions)
    features, feature_ranking = select_features(
        files, args.temporal_statistics, args.feature_count
    )
    progress(f"selected {len(features)} sequence features; device={device}")
    data = load_data(files, features)
    cutoff = int(base_metadata["valid_cutoff_time_id"])
    original_valid = data["time"] >= cutoff
    oof_times = np.asarray(base_oof_artifact["time_id"])
    oof_mask = (
        (data["time"] >= int(oof_times[0]))
        & (data["time"] <= int(oof_times[-1]))
    )
    for name in ("time_id", "asset_id"):
        expected = data["time" if name == "time_id" else "asset"][original_valid]
        actual = np.asarray(base_artifact[name])
        if not np.array_equal(expected, actual):
            raise ValueError(
                f"base validation artifact {name} does not align with parquet rows"
            )
        expected_oof = data[
            "time" if name == "time_id" else "asset"
        ][oof_mask]
        actual_oof = np.asarray(base_oof_artifact[name])
        if not np.array_equal(expected_oof, actual_oof):
            raise ValueError(
                f"base OOF artifact {name} does not align with parquet rows"
            )

    order = np.lexsort((data["time"], data["asset"]))
    times = data["time"][order]
    assets = data["asset"][order]
    target = data["target"][order]
    weight = data["weight"][order]
    raw = data["raw"][order]
    train_mask = times < cutoff
    mean = np.average(raw[train_mask], axis=0, weights=weight[train_mask])
    variance = np.average(
        (raw[train_mask] - mean) ** 2,
        axis=0, weights=weight[train_mask],
    )
    scale = np.sqrt(np.maximum(variance, 1e-8))
    normalized = np.clip((raw - mean) / scale, -8.0, 8.0).astype(np.float32)
    asset_start = np.empty(len(order), dtype=np.int64)
    start = 0
    while start < len(order):
        stop = int(np.searchsorted(assets, assets[start], side="right"))
        asset_start[start:stop] = start
        start = stop

    valid_times = np.unique(times[~train_mask])
    midpoint = len(valid_times) // 2
    calibration_end = int(valid_times[midpoint])
    calibration_positions = np.flatnonzero(
        (times >= cutoff) & (times < calibration_end)
    )
    evaluation_positions = np.flatnonzero(times >= calibration_end)

    oof_prediction_original = np.full(len(order), np.nan, dtype=np.float32)
    oof_prediction_original[oof_mask] = np.asarray(
        base_oof_artifact["prediction"], dtype=np.float32
    )
    oof_prediction = oof_prediction_original[order]
    oof_valid = np.isfinite(oof_prediction)
    base_intercept, base_scale = calibrate_base(
        target[oof_valid], oof_prediction[oof_valid], weight[oof_valid],
        args.min_base_scale, args.max_base_scale,
    )
    calibrated_oof = base_intercept + base_scale * oof_prediction

    base_full_original = np.full(len(order), np.nan, dtype=np.float32)
    base_full_original[original_valid] = np.asarray(base_artifact["prediction"])
    base_sorted = base_full_original[order]
    calibrated_base = base_intercept + base_scale * base_sorted
    residual_target = np.zeros(len(order), dtype=np.float32)
    residual_target[oof_valid] = (
        target[oof_valid] - calibrated_oof[oof_valid]
    )
    valid_base = np.isfinite(calibrated_base)
    residual_target[valid_base] = (
        target[valid_base] - calibrated_base[valid_base]
    )
    train_positions = np.flatnonzero(oof_valid)
    if args.max_train_rows and len(train_positions) > args.max_train_rows:
        rng = np.random.default_rng(args.seed)
        train_positions = np.sort(rng.choice(
            train_positions, args.max_train_rows, replace=False
        ))
    progress(
        f"base calibration from {int(oof_valid.sum()):,} OOF rows: "
        f"intercept={base_intercept:+.8f}, scale={base_scale:.6f}; "
        f"TCN residual train_rows={len(train_positions):,}"
    )

    def loader(positions, shuffle):
        dataset = AssetSequenceDataset(
            normalized, residual_target, weight, asset_start,
            positions, args.sequence_length,
        )
        return DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )

    train_loader = loader(train_positions, True)
    calibration_loader = loader(calibration_positions, False)
    evaluation_loader = loader(evaluation_positions, False)
    model = TinyTCN(
        len(features) + 1, args.hidden_size, args.levels,
        args.kernel_size, args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_score = -math.inf
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        total_batches = max(len(train_loader), 1)
        report_every = max(1, total_batches // 20)
        for batch_index, (batch, labels, weights, _) in enumerate(
            train_loader, start=1
        ):
            batch = batch.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = torch.sum(weights * (prediction - labels) ** 2) / torch.clamp(
                torch.sum(weights), min=1e-12
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * float(weights.sum())
            weight_sum += float(weights.sum())
            if (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % report_every == 0
            ):
                progress_bar(
                    f"TCN epoch {epoch}",
                    batch_index,
                    total_batches,
                    f"loss={loss_sum / max(weight_sum, 1e-12):.8f}",
                )
        cal_prediction = predict(
            model, calibration_loader, device, len(order),
            f"calibration epoch {epoch}",
        )[calibration_positions]
        epoch_beta = residual_weight(
            target[calibration_positions],
            calibrated_base[calibration_positions],
            cal_prediction,
            weight[calibration_positions],
            args.max_residual_weight,
        )
        score = weighted_r2(
            target[calibration_positions],
            calibrated_base[calibration_positions] + epoch_beta * cal_prediction,
            weight[calibration_positions],
        )
        progress(
            f"epoch {epoch}/{args.epochs}: "
            f"loss={loss_sum / max(weight_sum, 1e-12):.8f}, "
            f"residual_weight={epoch_beta:.6f}, "
            f"calibrated_base_plus_residual_r2={score:.8f}"
        )
        if score > best_score:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                progress(f"early stopping after epoch {epoch}")
                break
    model.load_state_dict(best_state)

    cal_temporal = predict(
        model, calibration_loader, device, len(order),
        "final calibration",
    )[calibration_positions]
    eval_temporal = predict(
        model, evaluation_loader, device, len(order),
        "final evaluation",
    )[evaluation_positions]
    cal_base = base_sorted[calibration_positions]
    eval_base = base_sorted[evaluation_positions]
    cal_calibrated_base = calibrated_base[calibration_positions]
    eval_calibrated_base = calibrated_base[evaluation_positions]
    beta = residual_weight(
        target[calibration_positions], cal_calibrated_base, cal_temporal,
        weight[calibration_positions], args.max_residual_weight,
    )
    eval_ensemble = eval_calibrated_base + beta * eval_temporal
    shuffled_cal = shuffled_within_time(
        cal_temporal, times[calibration_positions], args.seed + 101
    )
    shuffled_eval = shuffled_within_time(
        eval_temporal, times[evaluation_positions], args.seed + 103
    )
    shuffled_beta = residual_weight(
        target[calibration_positions], cal_calibrated_base, shuffled_cal,
        weight[calibration_positions], args.max_residual_weight,
    )
    shuffled_ensemble = (
        eval_calibrated_base + shuffled_beta * shuffled_eval
    )
    report = {
        "base_model_dir": str(base_dir.resolve()),
        "feature_columns": features,
        "sequence_length": args.sequence_length,
        "train_rows": len(train_positions),
        "calibration_rows": len(calibration_positions),
        "evaluation_rows": len(evaluation_positions),
        "calibration_time": [cutoff, int(valid_times[midpoint - 1])],
        "evaluation_time": [
            calibration_end, int(times[evaluation_positions[-1]])
        ],
        "feature_selection": {
            "source": str(Path(args.temporal_statistics).resolve()),
            "method": "dynamic_scores_plus_persistence",
        },
        "base_calibration": {
            "source_rows": int(oof_valid.sum()),
            "intercept": base_intercept,
            "scale": base_scale,
        },
        "calibrated_base_plus_residual_calibration_r2": best_score,
        "residual_weight": beta,
        "shuffled_residual_weight": shuffled_beta,
        "evaluation": {
            "base_r2": weighted_r2(
                target[evaluation_positions], eval_base,
                weight[evaluation_positions],
            ),
            "calibrated_base_r2": weighted_r2(
                target[evaluation_positions], eval_calibrated_base,
                weight[evaluation_positions],
            ),
            "residual_prediction_r2": weighted_r2(
                target[evaluation_positions] - eval_calibrated_base,
                eval_temporal, weight[evaluation_positions],
            ),
            "ensemble_r2": weighted_r2(
                target[evaluation_positions], eval_ensemble,
                weight[evaluation_positions],
            ),
            "shuffled_ensemble_r2": weighted_r2(
                target[evaluation_positions], shuffled_ensemble,
                weight[evaluation_positions],
            ),
            "base_residual_prediction_correlation": float(np.corrcoef(
                eval_base, eval_temporal
            )[0, 1]),
            "residual_target_prediction_correlation": float(np.corrcoef(
                target[evaluation_positions] - eval_calibrated_base,
                eval_temporal,
            )[0, 1]),
            "base_segments": segmented_scores(
                times[evaluation_positions], target[evaluation_positions],
                eval_base, weight[evaluation_positions],
            ),
            "calibrated_base_segments": segmented_scores(
                times[evaluation_positions], target[evaluation_positions],
                eval_calibrated_base, weight[evaluation_positions],
            ),
            "ensemble_segments": segmented_scores(
                times[evaluation_positions], target[evaluation_positions],
                eval_ensemble, weight[evaluation_positions],
            ),
        },
    }
    report["evaluation"]["ensemble_delta"] = (
        report["evaluation"]["ensemble_r2"]
        - report["evaluation"]["calibrated_base_r2"]
    )
    report["evaluation"]["shuffled_delta"] = (
        report["evaluation"]["shuffled_ensemble_r2"]
        - report["evaluation"]["calibrated_base_r2"]
    )
    report["evaluation"]["base_calibration_delta"] = (
        report["evaluation"]["calibrated_base_r2"]
        - report["evaluation"]["base_r2"]
    )
    model_dir = Path(args.model_dir)
    work_dir = Path(args.work_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.cpu().eval())
    scripted.save(str(model_dir / "tcn.pt"))
    feature_ranking.to_csv(
        model_dir / "tcn_feature_screening.csv", index=False
    )
    metadata = {
        "strategy": "lgb_tcn_ensemble_strategy",
        "base_strategy_dir": "../responder_assisted_lgb_catboost_strategy",
        "base_model_dir": os.path.relpath(
            base_dir.resolve(), Path(__file__).resolve().parent
        ).replace("\\", "/"),
        "feature_columns": features,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "sequence_length": args.sequence_length,
        "base_intercept": base_intercept,
        "base_scale": base_scale,
        "residual_weight": beta,
        "tcn_target": "calibrated_lgb_residual",
        "tcn_model": "tcn.pt",
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (model_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        work_dir / "validation_predictions.npz",
        time_id=times[evaluation_positions],
        asset_id=assets[evaluation_positions],
        target=target[evaluation_positions],
        weight=weight[evaluation_positions],
        base=eval_base,
        calibrated_base=eval_calibrated_base,
        temporal_residual=eval_temporal,
        ensemble=eval_ensemble,
        shuffled_ensemble=shuffled_ensemble,
    )
    progress(
        f"complete: calibrated_base="
        f"{report['evaluation']['calibrated_base_r2']:.8f}, "
        f"ensemble={report['evaluation']['ensemble_r2']:.8f}, "
        f"delta={report['evaluation']['ensemble_delta']:+.8f}"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
