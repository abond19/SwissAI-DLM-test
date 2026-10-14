#!/usr/bin/env bash
set -euo pipefail

# Joint (B, eta, c) hyperparameter grid for the base GIDD model (L12/H768) with the bst-scaling
# recipe: muP + CompleteP (model=gidd_hf_ref: Peri-LN, dense-LM architecture, no EMA) trained
# with AdamW (optim=adamw_mup: decoupled weight decay, betas 0.9/0.95, eps 1e-14) and a WSD
# schedule (10% warmup, 20% linear decay to 0), at 20 tokens per parameter.
#
#   B   in {32, 64, 128, 256}                       global batch in sequences
#   eta in {0.125, 0.25, 0.5, 1, 2, 4} x ETA_BASE   peak LR (width-256 convention, see
#                                                   model.base_width)
#   c = eta * lambda in {1e-7, ..., 1e-3}           weight_decay = c: with the fully decoupled
#                                                   decay the per-step shrink is sched(t) * c,
#                                                   independent of eta
#
# 4 x 6 x 5 = 120 cells.  Every cell trains for TOKENS_PER_PARAM * N tokens, N = total
# parameters (untied embeddings included), so the number of steps is TOKENS / (B * SEQUENCE_LENGTH).
#
# Dry run by default (prints the grid, token budgets and topology).  Submit with DRY_RUN=0.
# Restrict with e.g. BATCH_SIZES="128" ETA_MULTS="1 2" CS="1e-4" for a sub-grid.  Cells are
# resumable: AUTO_RESUME=1 + SBATCH_ARGS=--dependency=singleton chains time-limited jobs.

DRY_RUN="${DRY_RUN:-1}"
TOKENS_PER_PARAM="${TOKENS_PER_PARAM:-20}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
N_BLOCKS="${N_BLOCKS:-12}"
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
N_HEADS="${N_HEADS:-$((HIDDEN_SIZE / 64))}"
VOCAB_SIZE="${VOCAB_SIZE:-131072}"
PARAMETRIZATION="${PARAMETRIZATION:-completep}"   # standard | mup | completep

ETA_BASE="${ETA_BASE:-2.19116e-2}"                # bst-scaling peri-LN anchor (w768 L12)
BATCH_SIZES="${BATCH_SIZES:-32 64 128 256}"
ETA_MULTS="${ETA_MULTS:-0.125 0.25 0.5 1 2 4}"
CS="${CS:-1e-7 1e-6 1e-5 1e-4 1e-3}"

# fp32 master weights + bf16 compute (the decoupled decay is rounded away in bf16 weights).
LOCAL_BATCH="${LOCAL_BATCH:-4}"
MAX_RANKS="${MAX_RANKS:-16}"
NTASKS_PER_NODE=4
DEVICES_PER_NODE=4
GPUS_PER_NODE=4
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
ACCOUNT="${ACCOUNT:-ab035}"
PARTITION="${PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
SEED="${SEED:-4}"
# Short test runs: MAX_STEPS_OVERRIDE=<steps> replaces the 20-tokens-per-parameter step count.
MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
# Throughput-probe knobs (defaults = the production setting).
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-false}"
TORCH_COMPILE="${TORCH_COMPILE:-auto}"   # auto (compile GIDD) | true | false
VALIDATE_AFTER_TRAINING="${VALIDATE_AFTER_TRAINING:-true}"
PROFILER="${PROFILER:-false}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WARMUP_FRAC="${WARMUP_FRAC:-0.1}"
DECAY_FRAC="${DECAY_FRAC:-0.2}"
NUM_WORKERS="${NUM_WORKERS:-1}"
# Validation interval in optimizer steps (passed to Lightning as micro-batches).
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.5}"
TRAIN_LOG_EVERY_N_STEPS="${TRAIN_LOG_EVERY_N_STEPS:-25}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
AUTO_RESUME="${AUTO_RESUME:-1}"
SBATCH_ARGS="${SBATCH_ARGS:-}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-adamw_mup_${PARAMETRIZATION}_hp_grid_L${N_BLOCKS}H${HIDDEN_SIZE}_${TOKENS_PER_PARAM}tpp}"
WANDB_ENTITY="${WANDB_ENTITY:-SwissAI_DLM}"
WANDB_PROJECT="${WANDB_PROJECT:-SwissAI_AdamW_muP_HP_Tuning}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
# Read-only data (pretokenized parquet + packed Arrow cache) lives in zdelbari's storage.
# Everything the job writes (outputs, logs, W&B, HF cache) goes to the submitting user's scratch.
DATA_SRC_ROOT="${DATA_SRC_ROOT:-/iopsstor/scratch/cscs/zdelbari/SwissAI-DLM-data}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH:-/iopsstor/scratch/cscs/${USER}}/SwissAI-DLM-data}"
PRETOK_LOCAL_DIR="${PRETOK_LOCAL_DIR:-${DATA_SRC_ROOT}/training-data/gidd-nemotron-cc-pretok}"
# Own cache dir holding symlinks to the shared packed train/valid datasets (bs = sequence length),
# so anything the loader writes next to the cache lands in the user's scratch.
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs/hparam_tuning/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/${EXPERIMENT_NAME}}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb_logs}"

# Total parameters of the gidd_hf_ref architecture (no biases, Peri-LN, untied embeddings):
#   2*V*n  +  L*(4n^2 attn + 8n^2 mlp + 4n norm gains)  +  n final norm.
N_PARAMS=$((2 * VOCAB_SIZE * HIDDEN_SIZE + N_BLOCKS * (12 * HIDDEN_SIZE * HIDDEN_SIZE + 4 * HIDDEN_SIZE) + HIDDEN_SIZE))
TOKEN_BUDGET=$((TOKENS_PER_PARAM * N_PARAMS))

if (( LOCAL_BATCH <= 0 || MAX_RANKS <= 0 )); then
  echo "LOCAL_BATCH and MAX_RANKS must be positive." >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "1" ]]; then
  for p in "${PRETOK_LOCAL_DIR}"; do
    [[ -d "${p}" ]] || { echo "Missing data directory: ${p}" >&2; exit 1; }
  done
fi

# Ranks and gradient accumulation for a global batch: as many ranks as fit in MAX_RANKS.
topology() {  # $1 = B  ->  sets RANKS, ACC, NODES
  local gbs="$1" ranks
  ranks=$((gbs / LOCAL_BATCH))
  (( ranks > MAX_RANKS )) && ranks=${MAX_RANKS}
  if (( ranks % NTASKS_PER_NODE != 0 || gbs % (LOCAL_BATCH * ranks) != 0 )); then
    echo "B=${gbs} is not compatible with LOCAL_BATCH=${LOCAL_BATCH}, MAX_RANKS=${MAX_RANKS}." >&2
    exit 1
  fi
  RANKS=${ranks}
  ACC=$((gbs / (LOCAL_BATCH * ranks)))
  NODES=$((ranks / NTASKS_PER_NODE))
  VAL_CHECK_BATCHES=$((VAL_CHECK_INTERVAL * ACC))
}

fmt() { awk -v a="$1" -v m="$2" 'BEGIN { printf "%.6e", a * m }'; }

echo "AdamW + ${PARAMETRIZATION} HP grid: L${N_BLOCKS}/H${HIDDEN_SIZE}, N=${N_PARAMS} params, ${TOKENS_PER_PARAM} tokens/param"
echo "Tokens per cell: ${TOKEN_BUDGET} | sequence length ${SEQUENCE_LENGTH} | eta_base=${ETA_BASE}"
echo "Schedule: WSD warmup=${WARMUP_FRAC} decay=${DECAY_FRAC} | AdamW betas 0.9/0.95 eps 1e-14 | grad clip ${GRAD_CLIP}"
NCELLS=0
for B in ${BATCH_SIZES}; do
  topology "${B}"
  STEPS=$((TOKEN_BUDGET / (SEQUENCE_LENGTH * B)))
  printf 'B=%-4s steps=%-7s tokens/step=%-8s nodes=%s ranks=%s acc=%s\n' \
    "${B}" "${STEPS}" "$((SEQUENCE_LENGTH * B))" "${NODES}" "${RANKS}" "${ACC}"
done
for B in ${BATCH_SIZES}; do for M in ${ETA_MULTS}; do for C in ${CS}; do NCELLS=$((NCELLS + 1)); done; done; done
echo "Cells: ${NCELLS}"
if [[ "${DRY_RUN}" == "1" ]]; then
  for B in ${BATCH_SIZES}; do for M in ${ETA_MULTS}; do for C in ${CS}; do
    printf '  B=%-4s eta=%s (x%s) c=%s\n' "${B}" "$(fmt "${ETA_BASE}" "${M}")" "${M}" "${C}"
  done; done; done
  echo "Dry run only. Use DRY_RUN=0 to submit."
  exit 0
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${WANDB_DIR}" "${DATA_CACHE_DIR}"
for split in train valid; do
  name="nemotron-cc-pretok-${split}_$([[ ${split} == train ]] && echo train || echo validation)_bs${SEQUENCE_LENGTH}_unwrapped_eosFalse_specialFalse.dat"
  src="${DATA_SRC_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok/${name}"
  [[ -d "${src}" ]] || { echo "Missing shared packed dataset: ${src}" >&2; exit 1; }
  [[ -e "${DATA_CACHE_DIR}/${name}" ]] || ln -s "${src}" "${DATA_CACHE_DIR}/${name}"
done
PARTITION_DIRECTIVE=""
[[ -n "${PARTITION}" ]] && PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"

SUBMITTED=0
for B in ${BATCH_SIZES}; do
  topology "${B}"
  NTASKS=${RANKS}
  MAX_STEPS=$((TOKEN_BUDGET / (SEQUENCE_LENGTH * B)))
  [[ -n "${MAX_STEPS_OVERRIDE}" ]] && MAX_STEPS="${MAX_STEPS_OVERRIDE}"
  for M in ${ETA_MULTS}; do
    ETA="$(fmt "${ETA_BASE}" "${M}")"
    for C in ${CS}; do
      RUN_NAME="${PARAMETRIZATION}_L${N_BLOCKS}H${HIDDEN_SIZE}_S${SEQUENCE_LENGTH}_${TOKENS_PER_PARAM}tpp_B${B}_lrx${M//./p}_c${C//-/m}_seed${SEED}${RUN_NAME_SUFFIX}"
      RUN_STORAGE_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
      JOB_RESUME=false
      JOB_RESUME_PATH=""
      if [[ "${AUTO_RESUME}" == "1" ]]; then
        JOB_RESUME=true
        JOB_RESUME_PATH="${RUN_STORAGE_DIR}/checkpoints/last.ckpt"
      fi
      if [[ "${ALLOW_EXISTING_OUTPUT}" != "1" && "${AUTO_RESUME}" != "1" && -d "${RUN_STORAGE_DIR}" && -n "$(ls -A "${RUN_STORAGE_DIR}")" ]]; then
        echo "Refusing to reuse non-empty output: ${RUN_STORAGE_DIR}" >&2
        exit 1
      fi

      "${SBATCH_BIN}" ${SBATCH_ARGS} <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --account=${ACCOUNT}
${PARTITION_DIRECTIVE}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err

set -euo pipefail
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export WANDB_MODE=${WANDB_MODE}
export WANDB_DIR="${WANDB_DIR}"
export WANDB_SAVE_CODE=true
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export JOB_TMPDIR="\${SLURM_TMPDIR:-/tmp/SwissAI-DLM-\${USER:-unknown}-\${SLURM_JOB_ID:-manual}}"
export TMPDIR="\${JOB_TMPDIR}"
export TMP="\${JOB_TMPDIR}"
export TEMP="\${JOB_TMPDIR}"
export XDG_CACHE_HOME="\${JOB_TMPDIR}/xdg"
export HF_HOME="${DATA_ROOT}/cache/hf"
export HF_DATASETS_CACHE="\${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="\${HF_HOME}/hub"
export TRANSFORMERS_CACHE="\${HF_HOME}/transformers"
mkdir -p "\${TMPDIR}" "\${XDG_CACHE_HOME}" "\${WANDB_DIR}" "${RUN_STORAGE_DIR}"

srun --environment=uni-d2 \\
  --ntasks=${NTASKS} \\
  --ntasks-per-node=${NTASKS_PER_NODE} \\
  --cpus-per-task=${CPUS_PER_TASK} \\
  "${VENV_PYTHON}" -u -m discrete_diffusion \\
  seed=${SEED} \\
  data=nemotron-cc-pretok \\
  data.pretok_local_dir="${PRETOK_LOCAL_DIR}" \\
  data.cache_dir="${DATA_CACHE_DIR}" \\
  model=gidd_hf_ref \\
  model.parametrization=${PARAMETRIZATION} \\
  model.length=${SEQUENCE_LENGTH} \\
  model.max_tokens=1024 \\
  model.n_blocks=${N_BLOCKS} \\
  model.hidden_size=${HIDDEN_SIZE} \\
  model.n_heads=${N_HEADS} \\
  model.dropout=0.0 \\
  model.activation_checkpointing=${ACTIVATION_CHECKPOINTING} \\
  ++training.forward_autocast_dtype=bf16 \\
  algo=gidd \\
  algo.loss_type=gidd_easydel_lowmem \\
  algo.hybrid_mixing_shift=-1000 \\
  algo.loss_weighting=dynamic \\
  algo.low_discrepancy_sampling=true \\
  algo.time_sampling_scope=global_batch \\
  lr_scheduler=wsd \\
  lr_scheduler.warmup_frac=${WARMUP_FRAC} \\
  lr_scheduler.decay_frac=${DECAY_FRAC} \\
  strategy=ddp \\
  trainer.deterministic=false \\
  trainer.num_nodes=${NODES} \\
  trainer.devices=${DEVICES_PER_NODE} \\
  trainer.accumulate_grad_batches=${ACC} \\
  trainer.max_steps=${MAX_STEPS} \\
  trainer.log_every_n_steps=${TRAIN_LOG_EVERY_N_STEPS} \\
  trainer.gradient_clip_val=${GRAD_CLIP} \\
  trainer.num_sanity_val_steps=0 \\
  trainer.val_check_interval=${VAL_CHECK_BATCHES} \\
  trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \\
  trainer.precision=bf16-mixed \\
  loader.global_batch_size=${B} \\
  loader.eval_global_batch_size=${B} \\
  loader.batch_size=${LOCAL_BATCH} \\
  loader.eval_batch_size=${LOCAL_BATCH} \\
  loader.multiprocessing_context=spawn \\
  +loader.lazy_spawn_dataset=true \\
  loader.num_workers=${NUM_WORKERS} \\
  loader.pin_memory=true \\
  training.antithetic_sampling=true \\
  training.loss_precision=bf16 \\
  training.fault_tolerant=true \\
  training.torch_compile=${TORCH_COMPILE} \\
  training.log_train_aux_metrics=true \\
  training.sync_train_loss=true \\
  training.validate_after_training=${VALIDATE_AFTER_TRAINING} \\
  eval.generate_samples=false \\
  callbacks.pytorch_profiler.enabled=${PROFILER} \\
  perf.enabled=true \\
  perf.theoretical_peak_tflops_per_gpu=989 \\
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY} \\
  callbacks.checkpoint_every_n_steps.save_top_k=1 \\
  callbacks.checkpoint_every_n_steps.save_last=true \\
  +callbacks.checkpoint_every_n_steps.save_on_train_epoch_end=true \\
  callbacks.checkpoint_monitor.save_top_k=0 \\
  checkpointing.save_dir="${RUN_STORAGE_DIR}" \\
  checkpointing.resume_from_ckpt=${JOB_RESUME} \\
  checkpointing.resume_ckpt_path="${JOB_RESUME_PATH}" \\
  optim=adamw_mup \\
  optim.lr=${ETA} \\
  optim.weight_decay=${C} \\
  optim.beta1=0.9 \\
  optim.beta2=0.95 \\
  optim.eps=1e-14 \\
  hydra.run.dir="${RUN_STORAGE_DIR}/hydra/\${SLURM_JOB_ID}" \\
  wandb.save_dir="${WANDB_DIR}" \\
  +wandb.entity="${WANDB_ENTITY}" \\
  wandb.project="${WANDB_PROJECT}" \\
  wandb.group="${WANDB_GROUP}" \\
  wandb.name="${RUN_NAME}" \\
  +wandb.save_code=true \\
  wandb.job_type=train \\
  +wandb.tags="[adamw_mup,${PARAMETRIZATION},peri_ln,hp_grid,L${N_BLOCKS},H${HIDDEN_SIZE},batch_size_${B},lr_${ETA},c_${C},${TOKENS_PER_PARAM}tpp,seed_${SEED}]"
SBATCH_EOF
      SUBMITTED=$((SUBMITTED + 1))
    done
  done
done

echo "Submitted ${SUBMITTED} jobs."
