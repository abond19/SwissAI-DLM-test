#!/usr/bin/env bash
set -euo pipefail

# Three-pair SCION follow-up for the L12/H768 model at GBS=64.
# The pairs are intentionally explicit (not a Cartesian product):
#   (beta=1e-5, alpha=0.015), (beta=1e-4, alpha=0.2),
#   (beta=1e-3, alpha=0.015).
#
# This variant keeps the Set-C training radii, but initializes every B.3
# group at radius 1.0 so initialization is not multiplied by its rho/radius.
# Input embeddings and the LM head remain untied, and the historical
# head_scale/hidden_size forward-logit multiplier remains disabled.

TOKEN_BUDGET="${TOKEN_BUDGET:-2000000000}"
BUDGET_LABEL="${BUDGET_LABEL:-2B}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
# Model shape; defaults are the L12/H768 base model.
N_BLOCKS="${N_BLOCKS:-12}"
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-$((4 * HIDDEN_SIZE))}"
N_HEADS="${N_HEADS:-$((HIDDEN_SIZE / 64))}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-false}"
# torch.compile: auto compiles each GIDD layer and the loss; set false to disable.
TORCH_COMPILE="${TORCH_COMPILE:-auto}"
# Parameter storage dtype (configs/model/gidd_hf.yaml default bf16).  fp32 keeps
# fp32 master weights while trainer.precision=bf16-mixed still computes in bf16.
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
# Backbone forward autocast dtype: fp32 (historical default: compute runs in the
# parameter dtype) or bf16 (bf16 compute with fp32 master weights).
FORWARD_AUTOCAST="${FORWARD_AUTOCAST:-fp32}"
# Optional fixed step count for diagnostic traces at different global batches.
MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-}"
# Override this when reusing the launcher for another global batch size.
GBS="${GBS:-64}"
LOCAL_BATCH="${LOCAL_BATCH:-8}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
DEVICES_PER_NODE=4
NTASKS_PER_NODE=4
GPUS_PER_NODE=4
NTASKS=$((GBS / (LOCAL_BATCH * ACCUMULATION_STEPS)))
NODES=$((NTASKS / NTASKS_PER_NODE))
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
ACCOUNT="${ACCOUNT:-ab035}"
PARTITION="${PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SEED="${SEED:-4}"
# Use the verified deterministic 2B-token subset.  Its validation entry is a
# read-only link to the standard validation cache, so validation is unchanged.
NUM_WORKERS="${NUM_WORKERS:-1}"
LAZY_SPAWN_DATASET="${LAZY_SPAWN_DATASET:-true}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
TRAIN_LOG_EVERY_N_STEPS="${TRAIN_LOG_EVERY_N_STEPS:-25}"
# Validation interval in optimizer steps (passed to Lightning as micro-batches).
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-250}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.5}"
VALIDATE_AFTER_TRAINING="${VALIDATE_AFTER_TRAINING:-true}"
LATE_VALIDATION_ENABLED="${LATE_VALIDATION_ENABLED:-false}"
LATE_VALIDATION_START_FRACTION="${LATE_VALIDATION_START_FRACTION:-0.72}"
LATE_VALIDATION_EVERY_N_STEPS="${LATE_VALIDATION_EVERY_N_STEPS:-100}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
CHECKPOINT_SAVE_TOP_K="${CHECKPOINT_SAVE_TOP_K:-1}"
CHECKPOINT_SAVE_LAST="${CHECKPOINT_SAVE_LAST:-true}"
CHECKPOINT_MONITOR_SAVE_TOP_K="${CHECKPOINT_MONITOR_SAVE_TOP_K:-0}"
# Legacy trace supplies ordinary-step L and dual-gradient diagnostics; the periodic collector supplies rho and sigma.
TRACE_ENABLED="${TRACE_ENABLED:-false}"
TRACE_COLLECT_NOISE_STATS="${TRACE_COLLECT_NOISE_STATS:-false}"
TRACE_NOISE_STATS_EVERY="${TRACE_NOISE_STATS_EVERY:-200}"
TRACE_NOISE_MIN_SAMPLES="${TRACE_NOISE_MIN_SAMPLES:-3}"
TRACE_M="${TRACE_M:-3}"
TRACE_PROGRESS_EVERY="${TRACE_PROGRESS_EVERY:-0}"
# Legacy-trace local-smoothness (L) cadence: 1 = every step (historical),
# N = every N-th step, 0 = off (gradient norms for mu only, near free).
TRACE_EVERY="${TRACE_EVERY:-1}"
# Empty by default: ordinary training starts from fresh initialization.  Replay
# launchers can set this to a full Lightning checkpoint without sharing their
# output directory with the source run.
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-false}"
# AUTO_RESUME=1: resume from the run's own checkpoints/last.ckpt when it
# exists (fresh start otherwise), so the same submission can be queued
# repeatedly as a chain of time-limited jobs (e.g. SBATCH_ARGS=--dependency=singleton).
AUTO_RESUME="${AUTO_RESUME:-0}"
# false: on resume, do not skip the batches the checkpoint consumed (needed
# when resuming with a different accumulation, which read_resume_position
# cannot align, or onto a different data cache where skipping is meaningless).
RESUME_SKIP_SEEN_BATCHES="${RESUME_SKIP_SEEN_BATCHES:-true}"
SBATCH_ARGS="${SBATCH_ARGS:-}"
# Continue an existing W&B run instead of creating a new one (resume jobs).
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-must}"
# Resumed lr sweeps: make each step's noise depend only on (seed, step, rank)
# so runs resumed at different steps keep sharing noise.
RNG_RESEED_EVERY_STEP="${RNG_RESEED_EVERY_STEP:-false}"
RNG_RESEED_FROM_STEP="${RNG_RESEED_FROM_STEP:-0}"
LATE_TRAIN_LOGGING_ENABLED="${LATE_TRAIN_LOGGING_ENABLED:-false}"
LATE_TRAIN_LOGGING_LAST_N_STEPS="${LATE_TRAIN_LOGGING_LAST_N_STEPS:-100}"
DRY_RUN="${DRY_RUN:-1}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
# Set GRID_MODE=full for the Cartesian beta x momentum grid requested after
# the explicit three-pair pilot.  The default keeps the original pilot pairs.
GRID_MODE="${GRID_MODE:-explicit}"
# Optional comma-separated explicit pairs, for example
# "8e-5:0.1,8e-5:0.05,1e-4:0.06". This takes precedence over GRID_MODE.
PAIR_SPECS_OVERRIDE="${PAIR_SPECS_OVERRIDE:-}"

# Fixed Set-C training-time SCION radii.
SCALE_EMBED="${SCALE_EMBED:-3500}"
SCALE_BIAS="${SCALE_BIAS:-90}"
SCALE_LN="${SCALE_LN:-6.8}"
SCALE_MATRIX="${SCALE_MATRIX:-336}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-scion_b3_equalLR_fixedSetC_initScale1_untied_noHeadScale_subset2Bseed4_gbs${GBS}}"
WANDB_ENTITY="${WANDB_ENTITY:-SwissAI_DLM}"
WANDB_PROJECT="${WANDB_PROJECT:-SwissAI_Scion_B3_HP_Tuning}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
DATA_ROOT="${DATA_ROOT:-/iopsstor/scratch/cscs/zdelbari/SwissAI-DLM-data}"
PRETOK_LOCAL_DIR="${PRETOK_LOCAL_DIR:-${DATA_ROOT}/training-data/gidd-nemotron-cc-pretok}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok-2b-seed4}"
# Set to 0 only for a verified full packed cache.  The default protects the
# short HP runs from silently falling back to a much larger dataset.
REQUIRE_SUBSET_MANIFEST="${REQUIRE_SUBSET_MANIFEST:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs/hparam_tuning/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/${EXPERIMENT_NAME}}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb_logs}"

if (( TOKEN_BUDGET <= 0 || SEQUENCE_LENGTH <= 0 || LOCAL_BATCH <= 0 || ACCUMULATION_STEPS <= 0 )); then
  echo "TOKEN_BUDGET, SEQUENCE_LENGTH, LOCAL_BATCH, and ACCUMULATION_STEPS must be positive." >&2
  exit 1
fi
VAL_CHECK_BATCHES=$((VAL_CHECK_INTERVAL * ACCUMULATION_STEPS))
if (( TRAIN_LOG_EVERY_N_STEPS <= 0 || VAL_CHECK_INTERVAL <= 0 || LATE_VALIDATION_EVERY_N_STEPS <= 0 )); then
  echo "Training and validation logging intervals must be positive." >&2
  exit 1
fi
if (( GBS % (LOCAL_BATCH * ACCUMULATION_STEPS * NTASKS_PER_NODE) != 0 )); then
  echo "GBS=${GBS} must be divisible by LOCAL_BATCH * ACCUMULATION_STEPS * NTASKS_PER_NODE." >&2
  exit 1
fi
if [[ ! -d "${PRETOK_LOCAL_DIR}" ]]; then
  echo "Missing pretokenized dataset: ${PRETOK_LOCAL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_CACHE_DIR}" ]]; then
  echo "Missing dataset cache: ${DATA_CACHE_DIR}" >&2
  exit 1
fi
if [[ "${RESUME_FROM_CKPT}" == "true" && ! -f "${RESUME_CKPT_PATH}" ]]; then
  echo "Missing resume checkpoint: ${RESUME_CKPT_PATH}" >&2
  exit 1
fi
if [[ "${REQUIRE_SUBSET_MANIFEST}" == "1" && ! -f "${DATA_CACHE_DIR}/subset_manifest.json" ]]; then
  echo "Refusing to run without the verified subset manifest: ${DATA_CACHE_DIR}/subset_manifest.json" >&2
  exit 1
fi

if [[ -n "${MAX_STEPS_OVERRIDE}" ]]; then
  MAX_STEPS="${MAX_STEPS_OVERRIDE}"
else
  MAX_STEPS=$((TOKEN_BUDGET / (SEQUENCE_LENGTH * GBS)))
fi
ACTUAL_TOKENS=$((MAX_STEPS * SEQUENCE_LENGTH * GBS))
WARMDOWN_ITERS="${WARMDOWN_ITERS:-$((MAX_STEPS * 28 / 100))}"
PARTITION_DIRECTIVE=""
if [[ -n "${PARTITION}" ]]; then
  PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
fi

declare -a PAIR_SPECS=()
if [[ -n "${PAIR_SPECS_OVERRIDE}" ]]; then
  IFS=, read -r -a PAIR_SPECS <<< "${PAIR_SPECS_OVERRIDE}"
elif [[ "${GRID_MODE}" == "full" ]]; then
  BETAS=("1e-4" "3e-4" "5e-4" "7e-4" "9e-4")
  MOMENTA=("0.2" "0.1" "0.05" "0.02")
  for BETA in "${BETAS[@]}"; do
    for MOMENTUM in "${MOMENTA[@]}"; do
      PAIR_SPECS+=("${BETA}:${MOMENTUM}")
    done
  done
else
  PAIR_SPECS=("1e-5:0.015" "1e-4:0.2" "1e-3:0.015")
fi

# A resume checkpoint and a W&B run id each belong to exactly one run.
if [[ ( "${RESUME_FROM_CKPT}" == "true" || -n "${WANDB_RUN_ID}" ) && ${#PAIR_SPECS[@]} -ne 1 ]]; then
  echo "RESUME_CKPT_PATH / WANDB_RUN_ID continue one run; got ${#PAIR_SPECS[@]} beta:momentum pairs." >&2
  exit 1
fi
WANDB_RESUME_ARGS=""
if [[ -n "${WANDB_RUN_ID}" ]]; then
  WANDB_RESUME_ARGS="wandb.id=${WANDB_RUN_ID} +wandb.resume=${WANDB_RESUME}"
fi

echo "SCION B.3 explicit-pair follow-up (GBS=${GBS})"
echo "Budget: requested=${TOKEN_BUDGET}, steps=${MAX_STEPS}, actual=${ACTUAL_TOKENS}"
echo "Model: n_blocks=${N_BLOCKS}, hidden=${HIDDEN_SIZE}, intermediate=${INTERMEDIATE_SIZE}, heads=${N_HEADS}, activation_checkpointing=${ACTIVATION_CHECKPOINTING}, weights=${MODEL_DTYPE}, forward autocast=${FORWARD_AUTOCAST}"
echo "Topology: nodes=${NODES}, ranks=${NTASKS}, micro-batch=${LOCAL_BATCH}, accumulation=${ACCUMULATION_STEPS}, effective local batch=$((LOCAL_BATCH * ACCUMULATION_STEPS))"
echo "Grid mode: ${GRID_MODE} | pairs: ${#PAIR_SPECS[@]}"
printf '  %s\n' "${PAIR_SPECS[@]}"
echo "Training radii: embed=${SCALE_EMBED}, bias=${SCALE_BIAS}, norm=${SCALE_LN}, matrix=${SCALE_MATRIX}"
if [[ "${REQUIRE_SUBSET_MANIFEST}" == "1" ]]; then
  echo "Dataset cache: ${DATA_CACHE_DIR} (verified 2B-token subset; validation link retained)"
else
  echo "Dataset cache: ${DATA_CACHE_DIR} (full packed cache; no subset repetition)"
fi
echo "B.3 boundary init: enabled, init radius override=1.0 (no rho/radius scaling)"
echo "Tied embeddings: false | head output scaling: false | gradient clipping: off"
echo "Logging: train every ${TRAIN_LOG_EVERY_N_STEPS} steps | validation every ${VAL_CHECK_INTERVAL} steps"
echo "Trace: legacy=${TRACE_ENABLED} (L every ${TRACE_EVERY}), rho/sigma=${TRACE_COLLECT_NOISE_STATS}, every=${TRACE_NOISE_STATS_EVERY}, m=${TRACE_M}"
echo "Late validation: enabled=${LATE_VALIDATION_ENABLED}, start=${LATE_VALIDATION_START_FRACTION}, every=${LATE_VALIDATION_EVERY_N_STEPS}"
echo "Final-window train logging: enabled=${LATE_TRAIN_LOGGING_ENABLED}, last ${LATE_TRAIN_LOGGING_LAST_N_STEPS} steps"
if [[ "${RESUME_FROM_CKPT}" == "true" ]]; then
  echo "Resume checkpoint: ${RESUME_CKPT_PATH}"
fi
if [[ -n "${WANDB_RUN_ID}" ]]; then
  echo "W&B: continuing run id ${WANDB_RUN_ID} (resume=${WANDB_RESUME})"
fi
if [[ "${RNG_RESEED_EVERY_STEP}" == "true" ]]; then
  echo "RNG: reseed every optimizer step from (seed, step, rank), from step ${RNG_RESEED_FROM_STEP}"
fi
echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT}, group=${WANDB_GROUP}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%-12s %-12s\n' beta momentum
  for PAIR in "${PAIR_SPECS[@]}"; do
    IFS=: read -r BETA MOMENTUM <<< "${PAIR}"
    printf '%-12s %-12s\n' "${BETA}" "${MOMENTUM}"
  done
  echo "Dry run only. Use DRY_RUN=0 to submit the grid."
  exit 0
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${WANDB_DIR}"
SUBMITTED=0
for PAIR in "${PAIR_SPECS[@]}"; do
  IFS=: read -r BETA MOMENTUM <<< "${PAIR}"
  BETA_ID="${BETA//./p}"
  BETA_ID="${BETA_ID//-/m}"
  MOM_ID="${MOMENTUM//./p}"
  RUN_NAME="scion_b3_equalLR_fixedSetC_initScale1_untied_noHeadScale_L${N_BLOCKS}H${HIDDEN_SIZE}_S2048_${BUDGET_LABEL}_gbs${GBS}_beta${BETA_ID}_alpha${MOM_ID}_seed${SEED}${RUN_NAME_SUFFIX}"
  RUN_STORAGE_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

  JOB_RESUME_FROM_CKPT="${RESUME_FROM_CKPT}"
  JOB_RESUME_CKPT_PATH="${RESUME_CKPT_PATH}"
  if [[ "${AUTO_RESUME}" == "1" ]]; then
    JOB_RESUME_FROM_CKPT=true
    JOB_RESUME_CKPT_PATH="${RUN_STORAGE_DIR}/checkpoints/last.ckpt"
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
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR}"
export WANDB_SAVE_CODE=true
export HYDRA_FULL_ERROR=1
# Avoid a large contiguous allocation failing solely because PyTorch's cached
# blocks are fragmented across CUDA segments.
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
  model=gidd_hf \\
  model.length=${SEQUENCE_LENGTH} \\
  model.max_tokens=1024 \\
  model.n_blocks=${N_BLOCKS} \\
  model.hidden_size=${HIDDEN_SIZE} \\
  model.intermediate_size=${INTERMEDIATE_SIZE} \\
  model.n_heads=${N_HEADS} \\
  model.activation_checkpointing=${ACTIVATION_CHECKPOINTING} \\
  model.torch_dtype=${MODEL_DTYPE} \\
  ++training.forward_autocast_dtype=${FORWARD_AUTOCAST} \\
  model.dropout=0.0 \\
  model.activation_scale=2.0 \\
  model.attn_soft_cap=30.0 \\
  model.resid_scale=4.0 \\
  model.head_output_scaling_enabled=false \\
  model.tie_word_embeddings=false \\
  algo=gidd \\
  algo.loss_type=gidd_easydel_lowmem \\
  algo.hybrid_mixing_shift=-1000 \\
  algo.loss_weighting=dynamic \\
  algo.low_discrepancy_sampling=true \\
  algo.time_sampling_scope=global_batch \\
  lr_scheduler=constant_warmup \\
  strategy=ddp \\
  trainer.deterministic=false \\
  trainer.num_nodes=${NODES} \\
  trainer.devices=${DEVICES_PER_NODE} \\
  trainer.accumulate_grad_batches=${ACCUMULATION_STEPS} \\
  trainer.max_steps=${MAX_STEPS} \\
  trainer.log_every_n_steps=${TRAIN_LOG_EVERY_N_STEPS} \\
  trainer.gradient_clip_val=0.0 \\
  trainer.num_sanity_val_steps=${NUM_SANITY_VAL_STEPS} \\
  trainer.val_check_interval=${VAL_CHECK_BATCHES} \\
  trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \\
  trainer.precision=bf16-mixed \\
  loader.global_batch_size=${GBS} \\
  loader.eval_global_batch_size=${GBS} \\
  loader.batch_size=${LOCAL_BATCH} \\
  loader.eval_batch_size=${LOCAL_BATCH} \\
  loader.multiprocessing_context=spawn \\
  +loader.lazy_spawn_dataset=${LAZY_SPAWN_DATASET} \\
  loader.num_workers=${NUM_WORKERS} \\
  loader.pin_memory=true \\
  training.ema=0.0 \\
  training.antithetic_sampling=true \\
  training.loss_precision=bf16 \\
  training.fault_tolerant=true \\
  training.torch_compile=${TORCH_COMPILE} \\
  training.log_train_aux_metrics=true \\
  training.sync_train_loss=true \\
  training.validate_after_training=${VALIDATE_AFTER_TRAINING} \\
  eval.generate_samples=false \\
  callbacks.pytorch_profiler.enabled=false \\
  callbacks.late_validation_frequency.enabled=${LATE_VALIDATION_ENABLED} \\
  callbacks.late_validation_frequency.start_fraction=${LATE_VALIDATION_START_FRACTION} \\
  callbacks.late_validation_frequency.every_n_steps=${LATE_VALIDATION_EVERY_N_STEPS} \\
  callbacks.late_train_logging.enabled=${LATE_TRAIN_LOGGING_ENABLED} \\
  callbacks.late_train_logging.last_n_steps=${LATE_TRAIN_LOGGING_LAST_N_STEPS} \\
  perf.enabled=true \\
  perf.theoretical_peak_tflops_per_gpu=989 \\
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY} \\
  callbacks.checkpoint_every_n_steps.save_top_k=${CHECKPOINT_SAVE_TOP_K} \\
  callbacks.checkpoint_every_n_steps.save_last=${CHECKPOINT_SAVE_LAST} \\
  +callbacks.checkpoint_every_n_steps.save_on_train_epoch_end=true \\
  callbacks.checkpoint_monitor.save_top_k=${CHECKPOINT_MONITOR_SAVE_TOP_K} \\
  checkpointing.save_dir="${RUN_STORAGE_DIR}" \\
  checkpointing.resume_from_ckpt=${JOB_RESUME_FROM_CKPT} \\
  checkpointing.resume_ckpt_path="${JOB_RESUME_CKPT_PATH}" \\
  optim=scion \\
  optim.lr=${BETA} \\
  optim.equal_group_lr=true \\
  optim.boundary_init=true \\
  +optim.boundary_init_scale=1.0 \\
  optim.weight_decay=0.0 \\
  optim.momentum=${MOMENTUM} \\
  optim.scale_embed=${SCALE_EMBED} \\
  optim.scale_bias=${SCALE_BIAS} \\
  optim.scale_layer_norm=${SCALE_LN} \\
  optim.scale_matrix=${SCALE_MATRIX} \\
  optim.bias_norm=RowNorm \\
  optim.norm_layer_norm=BiasRMS \\
  optim.spectral_norm_steps=5 \\
  optim.unconstrained=false \\
  optim.warmup_iters=0 \\
  optim.warmdown_iters=${WARMDOWN_ITERS} \\
  optim.min_lr=1e-8 \\
  optim.trace_enabled=${TRACE_ENABLED} \\
  optim.trace_collect_noise_stats=${TRACE_COLLECT_NOISE_STATS} \\
  optim.trace_noise_stats_every=${TRACE_NOISE_STATS_EVERY} \\
  optim.trace_noise_min_samples=${TRACE_NOISE_MIN_SAMPLES} \\
  optim.trace_m=${TRACE_M} \\
  +optim.trace_progress_every=${TRACE_PROGRESS_EVERY} \\
  ++optim.trace_every=${TRACE_EVERY} \\
  hydra.run.dir="${RUN_STORAGE_DIR}/hydra/\${SLURM_JOB_ID}" \\
  wandb.save_dir="${WANDB_DIR}" \\
  +wandb.entity="${WANDB_ENTITY}" \\
  wandb.project="${WANDB_PROJECT}" \\
  wandb.group="${WANDB_GROUP}" \\
  wandb.name="${RUN_NAME}" \\
  ${WANDB_RESUME_ARGS} \\
  ++training.rng_reseed_every_step=${RNG_RESEED_EVERY_STEP} \\
  ++training.rng_reseed_from_step=${RNG_RESEED_FROM_STEP} \\
  ++training.resume_skip_seen_batches=${RESUME_SKIP_SEEN_BATCHES} \\
  +wandb.save_code=true \\
  wandb.job_type=train \\
  +wandb.tags="[scion,equal_group_lr,explicit_pair_grid,fixedSetC,init_radius_1_no_rho,untied_embeddings,no_head_output_scaling,L${N_BLOCKS},H${HIDDEN_SIZE},batch_size_${GBS},fw_stepsize_${BETA},momentum_${MOMENTUM},token_budget_${BUDGET_LABEL},accumulation_${ACCUMULATION_STEPS},seed_${SEED},no_gradient_clipping,global_time_sampling,residual_4_over_D]"
SBATCH_EOF
  SUBMITTED=$((SUBMITTED + 1))
done

echo "Submitted ${SUBMITTED} jobs."
