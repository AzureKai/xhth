#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash pull_remote_results.sh [remote_host] [remote_repo]
#
# Defaults:
#   remote_host = ustc-lab
#   remote_repo = ~/xhth

REMOTE_HOST="${1:-ustc-lab}"
REMOTE_REPO="${2:-~/xhth}"
STRATEGY_REL="examples/responder_assisted_lgb_catboost_strategy"
REMOTE_STRATEGY="${REMOTE_REPO%/}/${STRATEGY_REL}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_STRATEGY="${SCRIPT_DIR}"

copy_required_dir() {
    local name="$1"
    echo "[pull] ${REMOTE_HOST}:${REMOTE_STRATEGY}/${name}/ -> ${LOCAL_STRATEGY}/${name}/"
    mkdir -p "${LOCAL_STRATEGY}/${name}"
    scp -r \
        "${REMOTE_HOST}:${REMOTE_STRATEGY}/${name}/." \
        "${LOCAL_STRATEGY}/${name}/"
}

copy_optional_dir() {
    local name="$1"
    echo "[pull] trying optional directory: ${name}/"
    mkdir -p "${LOCAL_STRATEGY}/${name}"
    if ! scp -r \
        "${REMOTE_HOST}:${REMOTE_STRATEGY}/${name}/." \
        "${LOCAL_STRATEGY}/${name}/"; then
        echo "[warn] remote ${name}/ is unavailable; skipped" >&2
    fi
}

copy_optional_file() {
    local name="$1"
    echo "[pull] trying optional file: ${name}"
    if ! scp \
        "${REMOTE_HOST}:${REMOTE_STRATEGY}/${name}" \
        "${LOCAL_STRATEGY}/${name}"; then
        echo "[warn] remote ${name} is unavailable; skipped" >&2
    fi
}

echo "[pull] remote repository: ${REMOTE_HOST}:${REMOTE_REPO}"
echo "[pull] local strategy:    ${LOCAL_STRATEGY}"

# Required for result analysis and local inference/submission.
copy_required_dir "model"
copy_required_dir "audit"

# Useful reports and an already generated submission are optional.
copy_optional_dir "analysis"
copy_optional_file "submission.csv"

required_files=(
    "${LOCAL_STRATEGY}/model/metadata.json"
    "${LOCAL_STRATEGY}/model/ablation_report.json"
    "${LOCAL_STRATEGY}/model/target_feature_importance.csv"
    "${LOCAL_STRATEGY}/model/target_lightgbm.txt"
    "${LOCAL_STRATEGY}/audit/responder_audit_report.json"
)

for path in "${required_files[@]}"; do
    if [[ ! -s "${path}" ]]; then
        echo "[error] required result is missing or empty: ${path}" >&2
        exit 1
    fi
done

echo "[done] remote model and audit results were copied successfully"
echo "[note] work/cache and OOF artifacts were intentionally not copied"
