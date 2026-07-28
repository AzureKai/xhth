from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export compatible LightGBM validation predictions, then train "
            "and evaluate the causal TCN ensemble."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--base-work-dir",
        default="examples/responder_assisted_lgb_catboost_strategy/work",
    )
    parser.add_argument(
        "--base-model-dir",
        default="examples/responder_assisted_lgb_catboost_strategy/model",
    )
    parser.add_argument(
        "--work-dir",
        default="examples/lgb_tcn_ensemble_strategy/work",
    )
    parser.add_argument(
        "--model-dir",
        default="examples/lgb_tcn_ensemble_strategy/model",
    )
    parser.add_argument("--base-experiment-suite", default="next-step")
    parser.add_argument("--training-data-mode", default="in-memory")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--feature-count", type=int, default=48)
    parser.add_argument("--temporal-screen-rows", type=int, default=500_000)
    parser.add_argument("--target-oof-folds", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def run_stage(label, command, stage, total):
    started = time.perf_counter()
    print(
        f"[pipeline] stage {stage}/{total}: {label}\n"
        f"[pipeline] command: {' '.join(command)}",
        flush=True,
    )
    subprocess.run(command, check=True)
    print(
        f"[pipeline] stage {stage}/{total} complete: {label}; "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )


def check_torch(device):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed for this Python interpreter.\n"
            "CUDA 12.4 install command:\n"
            "python3 -m pip install torch==2.5.1 "
            "--index-url https://download.pytorch.org/whl/cu124"
        ) from exc
    available = bool(torch.cuda.is_available())
    compiled_cuda = str(torch.version.cuda or "cpu-only")
    print(
        f"[pipeline] PyTorch={torch.__version__}, "
        f"compiled_cuda={compiled_cuda}, cuda_available={available}",
        flush=True,
    )
    if device == "cuda" and not available:
        raise SystemExit(
            "CUDA was explicitly requested but PyTorch cannot initialize it.\n"
            "This machine reports a CUDA 12.4-capable driver. Reinstall the "
            "matching official wheel:\n"
            "python3 -m pip install --force-reinstall torch==2.5.1 "
            "--index-url https://download.pytorch.org/whl/cu124\n"
            "Then verify with:\n"
            "python3 -c \"import torch; print(torch.__version__, "
            "torch.version.cuda, torch.cuda.is_available())\""
        )
    if device == "auto" and not available:
        print(
            "[pipeline] warning: CUDA is unavailable; TCN will run on CPU. "
            "For this CUDA 12.4 driver, install torch==2.5.1 from the cu124 "
            "wheel index, or pass --device cpu intentionally.",
            flush=True,
        )


def main():
    args = parse_args()
    check_torch(args.device)
    root = Path(__file__).resolve().parents[2]
    base_train = (
        root / "examples" / "responder_assisted_lgb_catboost_strategy"
        / "train.py"
    )
    temporal_analyzer = (
        root / "examples" / "responder_assisted_lgb_catboost_strategy"
        / "analyze_feature_temporal_types.py"
    )
    tcn_train = Path(__file__).resolve().parent / "train.py"
    oof_exporter = Path(__file__).resolve().parent / "export_base_oof.py"
    temporal_output = Path(args.work_dir) / "temporal_analysis"
    base_oof_output = Path(args.work_dir) / "base_target_oof_predictions.npz"
    temporal_command = [
        sys.executable, str(temporal_analyzer),
        "--data-root", args.data_root,
        "--output-dir", str(temporal_output),
        "--max-rows", str(args.temporal_screen_rows),
    ]
    base_command = [
        sys.executable, str(base_train),
        "--data-root", args.data_root,
        "--work-dir", args.base_work_dir,
        "--model-dir", args.base_model_dir,
        "--training-data-mode", args.training_data_mode,
        "--experiment-suite", args.base_experiment_suite,
        "--skip-existing-models",
        "--threads", str(args.threads),
    ]
    tcn_command = [
        sys.executable, str(tcn_train),
        "--data-root", args.data_root,
        "--base-model-dir", args.base_model_dir,
        "--base-oof-predictions", str(base_oof_output),
        "--temporal-statistics",
        str(temporal_output / "feature_temporal_statistics.csv"),
        "--work-dir", args.work_dir,
        "--model-dir", args.model_dir,
        "--feature-count", str(args.feature_count),
        "--sequence-length", str(args.sequence_length),
        "--hidden-size", str(args.hidden_size),
        "--levels", str(args.levels),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
    ]
    oof_command = [
        sys.executable, str(oof_exporter),
        "--base-work-dir", args.base_work_dir,
        "--base-model-dir", args.base_model_dir,
        "--output", str(base_oof_output),
        "--folds", str(args.target_oof_folds),
        "--threads", str(args.threads),
    ]
    if args.max_train_rows:
        tcn_command.extend(["--max-train-rows", str(args.max_train_rows)])
    run_stage("temporal feature screening", temporal_command, 1, 4)
    run_stage("LightGBM validation export", base_command, 2, 4)
    run_stage("strict target OOF export", oof_command, 3, 4)
    run_stage("TCN residual training and evaluation", tcn_command, 4, 4)
    print("[pipeline] all training stages complete", flush=True)


if __name__ == "__main__":
    main()
