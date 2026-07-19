from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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
        key_padding_mask = ~present.bool()
        attended, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, **{"need_" + "weig" + "hts": False})
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
        pred = self.head(torch.cat([temporal, mixed, xs, asset], dim=-1)).squeeze(-1)
        return pred


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

    rank = np.full((asset_count, feature_dim), 0.5, dtype=np.float32)
    if len(valid_idx) == 1:
        rank[valid_idx[0], :] = 1.0
    else:
        for col_idx in range(feature_dim):
            column = values[:, col_idx]
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

    out[:, 0:feature_dim] = np.where(present[:, None], demean, 0.0)
    out[:, feature_dim : feature_dim * 2] = np.where(present[:, None], rank, 0.0)
    out[:, feature_dim * 2 :] = np.where(present[:, None], zscore, 0.0)
    return out


class Model:
    def __init__(self):
        strategy_dir = Path(__file__).resolve().parent
        model_dir = strategy_dir / "model"
        metadata_path = model_dir / "metadata.json"
        model_path = model_dir / "model.pt"
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing model metadata: {metadata_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"missing model file: {model_path}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_columns = list(self.metadata["feature_columns"])
        self.asset_ids = [int(value) for value in self.metadata["asset_ids"]]
        self.asset_to_slot = {asset_id: idx for idx, asset_id in enumerate(self.asset_ids)}
        self.feature_mean = np.asarray(self.metadata["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(self.metadata["feature_std"], dtype=np.float32)
        self.feature_std = np.where(self.feature_std > 1e-6, self.feature_std, 1.0).astype(np.float32)
        self.window_size = int(self.metadata["window_size"])
        self.prediction_scale = float(self.metadata.get("prediction_scale", 1.0))
        self.clip_min = float(self.metadata.get("clip_min", -np.inf))
        self.clip_max = float(self.metadata.get("clip_max", np.inf))
        self.history = np.zeros((len(self.asset_ids), self.window_size, len(self.feature_columns)), dtype=np.float32)
        self.last_time_id: int | None = None

        self.model = TcnAttentionNet(
            feature_dim=len(self.feature_columns),
            asset_count=len(self.asset_ids),
            hidden_dim=int(self.metadata["hidden_dim"]),
            tcn_layers=int(self.metadata["tcn_layers"]),
            kernel_size=int(self.metadata["kernel_size"]),
            dilations=[int(value) for value in self.metadata["dilations"]],
            attention_heads=int(self.metadata["attention_heads"]),
            attention_layers=int(self.metadata["attention_layers"]),
            dropout=float(self.metadata.get("dropout", 0.0)),
        )
        payload = torch.load(model_path, map_location="cpu")
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, test):
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase in Time-Series API order")
        self.last_time_id = time_id

        missing = [col for col in self.feature_columns if col not in test.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing[:5]}")

        asset_count = len(self.asset_ids)
        feature_dim = len(self.feature_columns)
        current = np.zeros((asset_count, feature_dim), dtype=np.float32)
        present = np.zeros(asset_count, dtype=bool)
        row_slots = []
        raw_values = test.loc[:, self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        raw_values = np.nan_to_num(raw_values, nan=0.0, posinf=0.0, neginf=0.0)
        asset_values = test["asset_id"].to_numpy(dtype=np.int64, copy=False)

        for row_idx, asset_id_raw in enumerate(asset_values):
            asset_id = int(asset_id_raw)
            if asset_id not in self.asset_to_slot:
                raise ValueError(f"unknown asset_id: {asset_id}")
            slot = self.asset_to_slot[asset_id]
            row_slots.append(slot)
            current[slot] = (raw_values[row_idx] - self.feature_mean) / self.feature_std
            present[slot] = True

        self.history[:, :-1, :] = self.history[:, 1:, :]
        self.history[:, -1, :] = np.where(present[:, None], current, 0.0)
        xs = cross_section_features(current, present)

        with torch.no_grad():
            history_tensor = torch.from_numpy(self.history[None, :, :, :])
            xs_tensor = torch.from_numpy(xs[None, :, :])
            present_tensor = torch.from_numpy(present[None, :])
            slot_pred = self.model(history_tensor, xs_tensor, present_tensor).squeeze(0).numpy()

        prediction = np.asarray([slot_pred[slot] for slot in row_slots], dtype=np.float64)
        prediction *= self.prediction_scale
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(prediction, self.clip_min, self.clip_max)
