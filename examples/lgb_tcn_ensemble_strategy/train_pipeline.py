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


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    base_train = (
        root / "examples" / "responder_assisted_lgb_catboost_strategy"
        / "train.py"
    )
    tcn_train = Path(__file__).resolve().parent / "train.py"
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
        "--work-dir", args.work_dir,
        "--model-dir", args.model_dir,
        "--feature-count", str(args.feature_count),
        "--sequence-length", str(args.sequence_length),
        "--hidden-size", str(args.hidden_size),
        "--levels", str(args.levels),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
    ]
    if args.max_train_rows:
        tcn_command.extend(["--max-train-rows", str(args.max_train_rows)])
    run_stage("LightGBM validation export", base_command, 1, 2)
    run_stage("TCN training and ensemble evaluation", tcn_command, 2, 2)
    print("[pipeline] all training stages complete", flush=True)


if __name__ == "__main__":
    main()
