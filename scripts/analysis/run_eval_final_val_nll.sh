#!/usr/bin/env bash
set -euo pipefail

# Fixed-noise final validation (eval_final_val_nll.py) for every finished run
# under the given output roots, using all 4 GPUs of one node: each GPU takes
# every 4th run.  Finished evaluations are skipped, so rerun it as runs finish.
#
# On an interactive GPU node, inside the container, from the repo root:
#   scripts/analysis/run_eval_final_val_nll.sh <outputs root> [<root> ...]
# As a Slurm job instead (NODES=k spreads the runs over 4k GPUs; the numbers
# do not depend on how many GPUs share the work):
#   SBATCH=1 [NODES=4] scripts/analysis/run_eval_final_val_nll.sh <outputs root> [<root> ...]
# Never run two evaluators on the same root at once: both would split the same
# pending list and evaluate every run twice.
# Extra eval_final_val_nll.py arguments go in EVAL_ARGS (e.g. "--force").

(( $# >= 1 )) || { echo "usage: $0 <outputs root> [...]" >&2; exit 1; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH_DIR="${SCRATCH:-/iopsstor/scratch/cscs/${USER}}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH_DIR}/SwissAI-DLM-data}"
PYTHON="${PYTHON:-${REPO_ROOT}/venv-docker/bin/python}"
NUM_GPUS="${NUM_GPUS:-4}"
EVAL_ARGS="${EVAL_ARGS:-}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
NODES="${NODES:-1}"

ENV_SETUP="cd '${REPO_ROOT}'
export PYTHONPATH='${REPO_ROOT}:${REPO_ROOT}/src' PYTHONNOUSERSITE=1 HYDRA_FULL_ERROR=1
export HF_HOME='${DATA_ROOT}/cache/hf'
export HF_DATASETS_CACHE=\"\${HF_HOME}/datasets\" HUGGINGFACE_HUB_CACHE=\"\${HF_HOME}/hub\" TRANSFORMERS_CACHE=\"\${HF_HOME}/transformers\"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled"
ROOTS="$(printf "'%s' " "$@")"

if [[ "${SBATCH:-0}" == "1" ]]; then
  mkdir -p "${REPO_ROOT}/logs/eval_final_val_nll"
  sbatch --job-name=eval_final_val_nll --time="${TIME_LIMIT}" --nodes="${NODES}" \
    --ntasks-per-node="${NUM_GPUS}" --cpus-per-task=72 --gpus-per-node=4 \
    --account="${ACCOUNT:-ab035}" \
    --output="${REPO_ROOT}/logs/eval_final_val_nll/%x_%j.out" <<EOF
#!/bin/bash
set -euo pipefail
${ENV_SETUP}
srun --environment=uni-d2 --ntasks=$((NUM_GPUS * NODES)) --ntasks-per-node=${NUM_GPUS} --cpus-per-task=72 \\
  "${PYTHON}" -u scripts/analysis/eval_final_val_nll.py ${ROOTS} ${EVAL_ARGS}
EOF
  exit 0
fi

eval "${ENV_SETUP}"
pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
  SLURM_PROCID=$i SLURM_LOCALID=$i SLURM_NTASKS=${NUM_GPUS} \
    "${PYTHON}" -u scripts/analysis/eval_final_val_nll.py "$@" ${EVAL_ARGS} \
    > "${DATA_ROOT}/eval_final_val_nll_gpu$i.log" 2>&1 &
  pids+=($!)
done
echo "Running ${NUM_GPUS} evaluators; logs: ${DATA_ROOT}/eval_final_val_nll_gpu*.log"
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
