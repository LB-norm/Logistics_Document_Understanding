#!/usr/bin/env bash
# Run the complete hybrid campaign and attempt every experiment.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

if [[ -n "${QWEN_PYTHON:-}" ]]; then
    python_command="$QWEN_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    python_command=python
elif [[ -x .venv/bin/python ]]; then
    python_command=.venv/bin/python
else
    python_command=python3
fi

runs_dir="${QWEN_HYBRID_RUNS_DIR:-runs/qwen-hybrid}"
mkdir -p "$runs_dir"

# Extra arguments are shared training overrides, e.g. --cache-dir or --dry-run.
# pipefail preserves the final failure summary's exit status through tee.
"$python_command" -u src/Qwen/run_qwen_experiment_queue.py \
    experiments/qwen/hybrid/queue.json \
    --continue-on-error \
    -- --runs-dir "$runs_dir" "$@" \
    2>&1 | tee -a "$runs_dir/queue.log"
