#!/usr/bin/env bash
set -euo pipefail

# Speed benchmark on random data (no dataset on disk) for two production setups:
#
#   BENCH_SETUP=scion (default)  submit_scion_b3_fixed_init_no_rho_gbs64.sh
#                                gidd_hf, SCION, bf16 weights, micro-batch 8, GBS 64
#   BENCH_SETUP=mup              submit_adamw_mup_hp_grid.sh (one grid cell)
#                                gidd_hf_ref + completep, AdamW-muP, WSD, fp32 weights,
#                                micro-batch 4, GBS 64, grad clip 1.0
#
# Each setup passes the same training overrides as its production launcher, except:
#   - data=synthetic-random-gidd: fresh random tokens for every sample, with the GIDD
#     tokenizer (same vocab size / special ids as nemotron-cc-pretok).  Batches are
#     reproducible across runs (seeded per sample index).
#   - loader.lazy_spawn_dataset=false: the lazy wrapper only applies to an on-disk
#     Arrow cache; the random dataset is already tiny to pickle.
#   - a short run (BENCH_STEPS), one hyperparameter point, always a fresh start.
#   - BenchmarkTimerCallback measures step time independently of perf.*, so changing
#     the perf settings does not change the yardstick.
#   - W&B defaults to offline mode in a separate project.
#
# Because the data never touches disk, data-pipeline changes (workers, prefetch,
# on-disk format) are NOT measured by this benchmark.
#
# Each run writes <run dir>/benchmark_timing.json and appends one JSON line to
# RESULTS_LOG.  Compare runs with, e.g.:
#   jq -r '[.setup,.run_label,.git_commit,.step_time_median_s,.step_time_mean_train_only_s,.peak_memory_allocated_gib]|@tsv' \
#     logs/speed_benchmark/results.jsonl
#
# LAUNCH=sbatch (default) submits a job with the production topology (2 nodes for
# scion, 4 for mup).  LAUNCH=local runs the same job script right here with torchrun
# on this node's GPUs (run it inside the uni-d2 environment).  Local mode keeps the
# per-GPU micro-batch and one micro-batch per optimizer step, so GBS defaults to
# LOCAL_BATCH x GPUs (scion 8 x 4 = 32, mup 4 x 4 = 16); per-GPU work matches
# production, but there is no inter-node gradient all-reduce.
#
# Usage:
#   LAUNCH=local BENCH_LABEL=baseline DRY_RUN=0 scripts/training/submit_speed_benchmark_random_data.sh
#   BENCH_LABEL=baseline DRY_RUN=0 scripts/training/submit_speed_benchmark_random_data.sh
#   BENCH_SETUP=mup BENCH_LABEL=baseline DRY_RUN=0 scripts/training/submit_speed_benchmark_random_data.sh
#   BENCH_LABEL=fp32_master MODEL_DTYPE=fp32 FORWARD_AUTOCAST=bf16 DRY_RUN=0 scripts/training/submit_speed_benchmark_random_data.sh

BENCH_SETUP="${BENCH_SETUP:-scion}"
LAUNCH="${LAUNCH:-sbatch}"   # sbatch | local
# Free-form tag for this run in the results log (letters, digits, . _ - only).
BENCH_LABEL="${BENCH_LABEL:-baseline}"
# Optimizer steps to run, and steps excluded from timing at the start.
BENCH_STEPS="${BENCH_STEPS:-500}"
BENCH_WARMUP_STEPS="${BENCH_WARMUP_STEPS:-50}"
# Synthetic dataset sizes (rows are generated on demand; no memory cost).
# limit_val_batches is a fraction, so SYNTH_VALID_ROWS sets the validation
# workload; set it to the real validation row count to match production.
SYNTH_TRAIN_ROWS="${SYNTH_TRAIN_ROWS:-1000000}"
SYNTH_VALID_ROWS="${SYNTH_VALID_ROWS:-4096}"
SYNTH_SEED="${SYNTH_SEED:-1234}"

# ---------------------------------------------------------------------------
# Settings shared by both production launchers (same defaults in each).
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
N_BLOCKS="${N_BLOCKS:-12}"
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
N_HEADS="${N_HEADS:-$((HIDDEN_SIZE / 64))}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-false}"
case "${LAUNCH}" in
  sbatch)
    DEVICES_PER_NODE=4
    ;;
  local)
    # Must equal torch.cuda.device_count(): the loader asserts
    # GBS == micro-batch x nodes x visible GPUs x accumulation.
    if [[ -z "${NUM_GPUS:-}" ]]; then
      if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        NUM_GPUS=$(awk -F, '{ print NF }' <<< "${CUDA_VISIBLE_DEVICES}")
      else
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
      fi
    fi
    if (( ${NUM_GPUS:-0} <= 0 )); then
      echo "LAUNCH=local found no GPUs; set NUM_GPUS." >&2
      exit 1
    fi
    DEVICES_PER_NODE=${NUM_GPUS}
    ;;
  *)
    echo "LAUNCH must be 'sbatch' or 'local', got '${LAUNCH}'." >&2
    exit 1
    ;;
esac
NTASKS_PER_NODE=${DEVICES_PER_NODE}
GPUS_PER_NODE=${DEVICES_PER_NODE}
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
ACCOUNT="${ACCOUNT:-ab035}"
PARTITION="${PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
SEED="${SEED:-4}"
NUM_WORKERS="${NUM_WORKERS:-1}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
TRAIN_LOG_EVERY_N_STEPS="${TRAIN_LOG_EVERY_N_STEPS:-25}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.5}"
VALIDATE_AFTER_TRAINING="${VALIDATE_AFTER_TRAINING:-true}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
CHECKPOINT_SAVE_TOP_K="${CHECKPOINT_SAVE_TOP_K:-1}"
CHECKPOINT_SAVE_LAST="${CHECKPOINT_SAVE_LAST:-true}"
CHECKPOINT_MONITOR_SAVE_TOP_K="${CHECKPOINT_MONITOR_SAVE_TOP_K:-0}"
TORCH_COMPILE="${TORCH_COMPILE:-false}"
PROFILER="${PROFILER:-false}"
SBATCH_ARGS="${SBATCH_ARGS:-}"
DRY_RUN="${DRY_RUN:-1}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

# ---------------------------------------------------------------------------
# Setup-specific defaults and overrides, copied from each production launcher.
declare -a SETUP_ARGS=()
case "${BENCH_SETUP}" in
  scion)
    LOCAL_BATCH="${LOCAL_BATCH:-8}"
    ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
    if [[ "${LAUNCH}" == local ]]; then
      GBS="${GBS:-$((LOCAL_BATCH * DEVICES_PER_NODE * ACCUMULATION_STEPS))}"
    else
      GBS="${GBS:-64}"
    fi
    if (( LOCAL_BATCH <= 0 || ACCUMULATION_STEPS <= 0 )) \
        || (( GBS % (LOCAL_BATCH * ACCUMULATION_STEPS * NTASKS_PER_NODE) != 0 )); then
      echo "GBS=${GBS} must be divisible by LOCAL_BATCH * ACCUMULATION_STEPS * NTASKS_PER_NODE." >&2
      exit 1
    fi
    NTASKS=$((GBS / (LOCAL_BATCH * ACCUMULATION_STEPS)))
    NODES=$((NTASKS / NTASKS_PER_NODE))
    ACC="${ACCUMULATION_STEPS}"
    MODEL_CONFIG=gidd_hf
    OPTIM_CONFIG=scion
    LR_SCHEDULER=constant_warmup
    INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-$((4 * HIDDEN_SIZE))}"
    # Parameter storage dtype (configs/model/gidd_hf.yaml default bf16).
    MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
    FORWARD_AUTOCAST="${FORWARD_AUTOCAST:-fp32}"
    GRAD_CLIP="${GRAD_CLIP:-0.0}"
    VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-250}"
    # One pair from the production pilot grid.  Step time does not depend on it.
    BETA="${BETA:-1e-4}"
    MOMENTUM="${MOMENTUM:-0.2}"
    LR="${BETA}"
    WEIGHT_DECAY=0.0
    # Fixed Set-C training-time SCION radii.
    SCALE_EMBED="${SCALE_EMBED:-3500}"
    SCALE_BIAS="${SCALE_BIAS:-90}"
    SCALE_LN="${SCALE_LN:-6.8}"
    SCALE_MATRIX="${SCALE_MATRIX:-336}"
    LATE_VALIDATION_ENABLED="${LATE_VALIDATION_ENABLED:-false}"
    LATE_VALIDATION_START_FRACTION="${LATE_VALIDATION_START_FRACTION:-0.72}"
    LATE_VALIDATION_EVERY_N_STEPS="${LATE_VALIDATION_EVERY_N_STEPS:-100}"
    LATE_TRAIN_LOGGING_ENABLED="${LATE_TRAIN_LOGGING_ENABLED:-false}"
    LATE_TRAIN_LOGGING_LAST_N_STEPS="${LATE_TRAIN_LOGGING_LAST_N_STEPS:-100}"
    TRACE_ENABLED="${TRACE_ENABLED:-false}"
    TRACE_COLLECT_NOISE_STATS="${TRACE_COLLECT_NOISE_STATS:-false}"
    TRACE_NOISE_STATS_EVERY="${TRACE_NOISE_STATS_EVERY:-200}"
    TRACE_NOISE_MIN_SAMPLES="${TRACE_NOISE_MIN_SAMPLES:-3}"
    TRACE_M="${TRACE_M:-3}"
    TRACE_PROGRESS_EVERY="${TRACE_PROGRESS_EVERY:-0}"
    TRACE_EVERY="${TRACE_EVERY:-1}"
    RNG_RESEED_EVERY_STEP="${RNG_RESEED_EVERY_STEP:-false}"
    RNG_RESEED_FROM_STEP="${RNG_RESEED_FROM_STEP:-0}"
    RESUME_SKIP_SEEN_BATCHES="${RESUME_SKIP_SEEN_BATCHES:-true}"
    WARMDOWN_ITERS="${WARMDOWN_ITERS:-$((BENCH_STEPS * 28 / 100))}"
    SETUP_ARGS=(
      "model.intermediate_size=${INTERMEDIATE_SIZE}"
      "model.activation_scale=2.0"
      "model.attn_soft_cap=30.0"
      "model.resid_scale=4.0"
      "model.head_output_scaling_enabled=false"
      "model.tie_word_embeddings=false"
      "training.ema=0.0"
      "callbacks.late_validation_frequency.enabled=${LATE_VALIDATION_ENABLED}"
      "callbacks.late_validation_frequency.start_fraction=${LATE_VALIDATION_START_FRACTION}"
      "callbacks.late_validation_frequency.every_n_steps=${LATE_VALIDATION_EVERY_N_STEPS}"
      "callbacks.late_train_logging.enabled=${LATE_TRAIN_LOGGING_ENABLED}"
      "callbacks.late_train_logging.last_n_steps=${LATE_TRAIN_LOGGING_LAST_N_STEPS}"
      "optim.equal_group_lr=true"
      "optim.boundary_init=true"
      "+optim.boundary_init_scale=1.0"
      "optim.momentum=${MOMENTUM}"
      "optim.scale_embed=${SCALE_EMBED}"
      "optim.scale_bias=${SCALE_BIAS}"
      "optim.scale_layer_norm=${SCALE_LN}"
      "optim.scale_matrix=${SCALE_MATRIX}"
      "optim.bias_norm=RowNorm"
      "optim.norm_layer_norm=BiasRMS"
      "optim.spectral_norm_steps=5"
      "optim.unconstrained=false"
      "optim.warmup_iters=0"
      "optim.warmdown_iters=${WARMDOWN_ITERS}"
      "optim.min_lr=1e-8"
      "optim.trace_enabled=${TRACE_ENABLED}"
      "optim.trace_collect_noise_stats=${TRACE_COLLECT_NOISE_STATS}"
      "optim.trace_noise_stats_every=${TRACE_NOISE_STATS_EVERY}"
      "optim.trace_noise_min_samples=${TRACE_NOISE_MIN_SAMPLES}"
      "optim.trace_m=${TRACE_M}"
      "+optim.trace_progress_every=${TRACE_PROGRESS_EVERY}"
      "++optim.trace_every=${TRACE_EVERY}"
      "++training.rng_reseed_every_step=${RNG_RESEED_EVERY_STEP}"
      "++training.rng_reseed_from_step=${RNG_RESEED_FROM_STEP}"
      "++training.resume_skip_seen_batches=${RESUME_SKIP_SEEN_BATCHES}"
    )
    SETUP_SUMMARY="SCION beta=${BETA}, momentum=${MOMENTUM}, radii embed=${SCALE_EMBED} bias=${SCALE_BIAS} norm=${SCALE_LN} matrix=${SCALE_MATRIX}"
    BETA_ID="${BETA//./p}"
    HP_TAG="beta${BETA_ID//-/m}_alpha${MOMENTUM//./p}"
    ;;
  mup)
    LOCAL_BATCH="${LOCAL_BATCH:-4}"
    if [[ "${LAUNCH}" == local ]]; then
      MAX_RANKS="${MAX_RANKS:-${DEVICES_PER_NODE}}"
      GBS="${GBS:-$((LOCAL_BATCH * MAX_RANKS))}"
    else
      MAX_RANKS="${MAX_RANKS:-16}"
      GBS="${GBS:-64}"
    fi
    # Same rank / accumulation rule as submit_adamw_mup_hp_grid.sh: as many ranks
    # as fit in MAX_RANKS, the rest as gradient accumulation.
    if (( LOCAL_BATCH <= 0 || MAX_RANKS <= 0 )); then
      echo "LOCAL_BATCH and MAX_RANKS must be positive." >&2
      exit 1
    fi
    NTASKS=$((GBS / LOCAL_BATCH))
    if (( NTASKS > MAX_RANKS )); then
      NTASKS=${MAX_RANKS}
    fi
    if (( NTASKS <= 0 || NTASKS % NTASKS_PER_NODE != 0 || GBS % (LOCAL_BATCH * NTASKS) != 0 )); then
      echo "GBS=${GBS} is not compatible with LOCAL_BATCH=${LOCAL_BATCH}, MAX_RANKS=${MAX_RANKS}." >&2
      exit 1
    fi
    ACC=$((GBS / (LOCAL_BATCH * NTASKS)))
    NODES=$((NTASKS / NTASKS_PER_NODE))
    MODEL_CONFIG=gidd_hf_ref
    OPTIM_CONFIG=adamw_mup
    LR_SCHEDULER=wsd
    # gidd_hf_std / gidd_hf_ref store fp32 weights; passing it explicitly is a no-op
    # by default and lets MODEL_DTYPE=bf16 be compared.
    MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
    FORWARD_AUTOCAST="${FORWARD_AUTOCAST:-bf16}"
    GRAD_CLIP="${GRAD_CLIP:-1.0}"
    VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
    PARAMETRIZATION="${PARAMETRIZATION:-completep}"   # standard | mup | completep
    # One grid cell: the bst-scaling anchor LR (ETA_MULT=1) and c=1e-4.  Step time
    # does not depend on them.
    ETA_BASE="${ETA_BASE:-2.19116e-2}"
    ETA_MULT="${ETA_MULT:-1}"
    LR="$(awk -v a="${ETA_BASE}" -v m="${ETA_MULT}" 'BEGIN { printf "%.6e", a * m }')"
    WEIGHT_DECAY="${C:-1e-4}"
    WARMUP_FRAC="${WARMUP_FRAC:-0.1}"
    DECAY_FRAC="${DECAY_FRAC:-0.2}"
    SETUP_ARGS=(
      "model.parametrization=${PARAMETRIZATION}"
      "lr_scheduler.warmup_frac=${WARMUP_FRAC}"
      "lr_scheduler.decay_frac=${DECAY_FRAC}"
      "optim.beta1=0.9"
      "optim.beta2=0.95"
      "optim.eps=1e-14"
    )
    SETUP_SUMMARY="AdamW-muP (${PARAMETRIZATION}) lr=${LR}, c=${WEIGHT_DECAY}, WSD warmup=${WARMUP_FRAC} decay=${DECAY_FRAC}"
    LR_ID="${LR//./p}"
    HP_TAG="${PARAMETRIZATION}_lr${LR_ID//-/m}_c${WEIGHT_DECAY//-/m}"
    ;;
  *)
    echo "BENCH_SETUP must be 'scion' or 'mup', got '${BENCH_SETUP}'." >&2
    exit 1
    ;;
esac
if [[ "${LAUNCH}" == local ]] && (( NTASKS != DEVICES_PER_NODE )); then
  echo "LAUNCH=local runs one rank per GPU on this node (${DEVICES_PER_NODE}); GBS=${GBS}," \
       "LOCAL_BATCH=${LOCAL_BATCH} and accumulation=${ACC} need ${NTASKS} ranks." >&2
  exit 1
fi
# One "  override \" line per setup-specific override, spliced into the job script.
SETUP_ARGS_TEXT="$(printf '  %s \\\n' "${SETUP_ARGS[@]}")"

WANDB_MODE_SETTING="${WANDB_MODE_SETTING:-offline}"
WANDB_ENTITY="${WANDB_ENTITY:-SwissAI_DLM}"
WANDB_PROJECT="${WANDB_PROJECT:-SwissAI_DLM_Speed_Benchmark}"
WANDB_GROUP="${WANDB_GROUP:-speed_benchmark_random_data_${BENCH_SETUP}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Python with the project's dependencies: VENV_PYTHON if set, else the uni-d2
# container's venv when present (CSCS), else the active conda environment.
if [[ -z "${VENV_PYTHON:-}" ]]; then
  if [[ -x "${REPO_ROOT}/venv-docker/bin/python" ]]; then
    VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    VENV_PYTHON="${CONDA_PREFIX}/bin/python"
  else
    # The sbatch path runs inside the uni-d2 container, where the venv exists
    # even if it is not visible from this shell.
    VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
  fi
fi
# Only used for the Hugging Face cache (GIDD tokenizer) and run outputs.
DATA_ROOT="${DATA_ROOT:-/iopsstor/scratch/cscs/zdelbari/SwissAI-DLM-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs/speed_benchmark}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/speed_benchmark}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb_logs}"
RESULTS_LOG="${RESULTS_LOG:-${LOG_DIR}/results.jsonl}"

if [[ ! "${BENCH_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "BENCH_LABEL may contain only letters, digits, '.', '_' and '-'." >&2
  exit 1
fi
if (( BENCH_STEPS <= BENCH_WARMUP_STEPS )); then
  echo "BENCH_STEPS (${BENCH_STEPS}) must exceed BENCH_WARMUP_STEPS (${BENCH_WARMUP_STEPS})." >&2
  exit 1
fi
if (( SEQUENCE_LENGTH <= 0 || TRAIN_LOG_EVERY_N_STEPS <= 0 || VAL_CHECK_INTERVAL <= 0 )); then
  echo "SEQUENCE_LENGTH and the logging / validation intervals must be positive." >&2
  exit 1
fi

MAX_STEPS="${BENCH_STEPS}"

PARTITION_DIRECTIVE=""
if [[ -n "${PARTITION}" ]]; then
  PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="speedbench_${BENCH_SETUP}_${BENCH_LABEL}_L${N_BLOCKS}H${HIDDEN_SIZE}_S${SEQUENCE_LENGTH}_gbs${GBS}_lb${LOCAL_BATCH}_acc${ACC}_${MODEL_DTYPE}_${HP_TAG}_${TIMESTAMP}"
RUN_STORAGE_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

echo "Speed benchmark on random data: setup=${BENCH_SETUP}, label=${BENCH_LABEL}"
echo "Steps: ${BENCH_STEPS} total, ${BENCH_WARMUP_STEPS} warmup excluded from timing"
echo "Model: ${MODEL_CONFIG}, n_blocks=${N_BLOCKS}, hidden=${HIDDEN_SIZE}, heads=${N_HEADS}, activation_checkpointing=${ACTIVATION_CHECKPOINTING}, weights=${MODEL_DTYPE}, forward autocast=${FORWARD_AUTOCAST}"
echo "Optimizer: ${SETUP_SUMMARY}, grad clip ${GRAD_CLIP}"
echo "Launch: ${LAUNCH} | python: ${VENV_PYTHON}"
echo "Topology: nodes=${NODES}, ranks=${NTASKS}, micro-batch=${LOCAL_BATCH}, accumulation=${ACC}, GBS=${GBS}"
echo "Data: synthetic-random-gidd, train rows=${SYNTH_TRAIN_ROWS}, valid rows=${SYNTH_VALID_ROWS}, seed=${SYNTH_SEED}"
echo "Validation every ${VAL_CHECK_INTERVAL} micro-batches (limit_val_batches=${LIMIT_VAL_BATCHES}); checkpoint every ${CHECKPOINT_EVERY} steps"
echo "Output: ${RUN_STORAGE_DIR}"
echo "Results log: ${RESULTS_LOG}"
echo "W&B: mode=${WANDB_MODE_SETTING}, ${WANDB_ENTITY}/${WANDB_PROJECT}, group=${WANDB_GROUP}"

if [[ "${LAUNCH}" == local && ! -x "${VENV_PYTHON}" ]]; then
  echo "Python not found: ${VENV_PYTHON}. Activate the conda environment or set VENV_PYTHON." >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run only. Use DRY_RUN=0 to submit."
  exit 0
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${WANDB_DIR}"

# The job script is written to LOG_DIR (kept for reference), then either submitted
# with sbatch or run here.  Both modes run the same body; only the header and the
# launcher line (srun vs torchrun) differ.
if [[ "${LAUNCH}" == sbatch ]]; then
  LOCAL_ENV=""
  LAUNCH_LINES="$(printf '%s \\\n' \
    "srun --environment=uni-d2" \
    "  --ntasks=${NTASKS}" \
    "  --ntasks-per-node=${NTASKS_PER_NODE}" \
    "  --cpus-per-task=${CPUS_PER_TASK}" \
    "  \"${VENV_PYTHON}\" -u -m discrete_diffusion")"
else
  LOCAL_ENV="$(printf '%s\n' \
    "# Lightning prefers SLURMEnvironment over the torchrun variables whenever" \
    "# SLURM_NTASKS is set (e.g. inside a batch allocation); hide it so torchrun" \
    "# defines the ranks." \
    "unset SLURM_NTASKS SLURM_NTASKS_PER_NODE" \
    "export PYTHONUNBUFFERED=1" \
    "# torchrun defaults to 1 OpenMP thread per rank; srun --cpus-per-task gives" \
    "# each rank its share of the node cores instead." \
    "export OMP_NUM_THREADS=\"\${OMP_NUM_THREADS:-\$(( \$(nproc) / ${NTASKS} ))}\"")"
  LAUNCH_LINES="$(printf '%s \\\n' \
    "\"${VENV_PYTHON}\" -u -m torch.distributed.run --standalone --nproc_per_node=${NTASKS}" \
    "  -m discrete_diffusion")"
fi

JOB_FILE="${LOG_DIR}/${RUN_NAME}.sh"
{
  echo '#!/bin/bash'
  if [[ "${LAUNCH}" == sbatch ]]; then
    echo "#SBATCH --job-name=${RUN_NAME}"
    echo "#SBATCH --time=${TIME_LIMIT}"
    echo "#SBATCH --nodes=${NODES}"
    echo "#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}"
    echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
    echo "#SBATCH --gpus-per-node=${GPUS_PER_NODE}"
    echo "#SBATCH --account=${ACCOUNT}"
    if [[ -n "${PARTITION_DIRECTIVE}" ]]; then
      echo "${PARTITION_DIRECTIVE}"
    fi
    echo "#SBATCH --output=${LOG_DIR}/%x_%j.out"
    echo "#SBATCH --error=${LOG_DIR}/%x_%j.err"
  else
    echo "# Local run: ${NTASKS} ranks via torchrun on one node."
  fi
  echo
} > "${JOB_FILE}"

cat >> "${JOB_FILE}" <<SBATCH_EOF
set -euo pipefail
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export WANDB_MODE=${WANDB_MODE_SETTING}
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
${LOCAL_ENV}

# Record the code actually being run (the job may start after later edits).
GIT_COMMIT="\$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ "\${GIT_COMMIT}" != unknown ]] && ! git -C "${REPO_ROOT}" diff --quiet HEAD 2>/dev/null; then
  GIT_COMMIT="\${GIT_COMMIT}-dirty"
  git -C "${REPO_ROOT}" diff HEAD > "${RUN_STORAGE_DIR}/uncommitted.diff" || true
fi
echo "Benchmarking code at \${GIT_COMMIT}"

${LAUNCH_LINES}
  seed=${SEED} \\
  data=synthetic-random-gidd \\
  data.cache_dir="\${JOB_TMPDIR}/synthetic-random-gidd" \\
  data.synthetic_train_size=${SYNTH_TRAIN_ROWS} \\
  data.synthetic_validation_size=${SYNTH_VALID_ROWS} \\
  data.synthetic_seed=${SYNTH_SEED} \\
  model=${MODEL_CONFIG} \\
  model.length=${SEQUENCE_LENGTH} \\
  model.max_tokens=1024 \\
  model.n_blocks=${N_BLOCKS} \\
  model.hidden_size=${HIDDEN_SIZE} \\
  model.n_heads=${N_HEADS} \\
  model.activation_checkpointing=${ACTIVATION_CHECKPOINTING} \\
  model.torch_dtype=${MODEL_DTYPE} \\
  ++training.forward_autocast_dtype=${FORWARD_AUTOCAST} \\
  model.dropout=0.0 \\
  algo=gidd \\
  algo.loss_type=gidd_easydel_lowmem \\
  algo.hybrid_mixing_shift=-1000 \\
  algo.loss_weighting=dynamic \\
  algo.low_discrepancy_sampling=true \\
  algo.time_sampling_scope=global_batch \\
  lr_scheduler=${LR_SCHEDULER} \\
  strategy=ddp \\
  trainer.deterministic=false \\
  trainer.num_nodes=${NODES} \\
  trainer.devices=${DEVICES_PER_NODE} \\
  trainer.accumulate_grad_batches=${ACC} \\
  trainer.max_steps=${MAX_STEPS} \\
  trainer.log_every_n_steps=${TRAIN_LOG_EVERY_N_STEPS} \\
  trainer.gradient_clip_val=${GRAD_CLIP} \\
  trainer.num_sanity_val_steps=${NUM_SANITY_VAL_STEPS} \\
  trainer.val_check_interval=${VAL_CHECK_INTERVAL} \\
  trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \\
  trainer.precision=bf16-mixed \\
  loader.global_batch_size=${GBS} \\
  loader.eval_global_batch_size=${GBS} \\
  loader.batch_size=${LOCAL_BATCH} \\
  loader.eval_batch_size=${LOCAL_BATCH} \\
  loader.multiprocessing_context=spawn \\
  +loader.lazy_spawn_dataset=false \\
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
  +callbacks.benchmark_timer._target_=discrete_diffusion.callbacks.benchmark_timer.BenchmarkTimerCallback \\
  +callbacks.benchmark_timer.warmup_steps=${BENCH_WARMUP_STEPS} \\
  +callbacks.benchmark_timer.output_dir="${RUN_STORAGE_DIR}" \\
  +callbacks.benchmark_timer.results_log="${RESULTS_LOG}" \\
  +callbacks.benchmark_timer.run_label="'${BENCH_LABEL}'" \\
  +callbacks.benchmark_timer.setup="'${BENCH_SETUP}'" \\
  +callbacks.benchmark_timer.git_commit="'\${GIT_COMMIT}'" \\
  perf.enabled=true \\
  perf.theoretical_peak_tflops_per_gpu=989 \\
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY} \\
  callbacks.checkpoint_every_n_steps.save_top_k=${CHECKPOINT_SAVE_TOP_K} \\
  callbacks.checkpoint_every_n_steps.save_last=${CHECKPOINT_SAVE_LAST} \\
  callbacks.checkpoint_monitor.save_top_k=${CHECKPOINT_MONITOR_SAVE_TOP_K} \\
  checkpointing.save_dir="${RUN_STORAGE_DIR}" \\
  checkpointing.resume_from_ckpt=false \\
  checkpointing.resume_ckpt_path="${RUN_STORAGE_DIR}/checkpoints/last.ckpt" \\
  optim=${OPTIM_CONFIG} \\
  optim.lr=${LR} \\
  optim.weight_decay=${WEIGHT_DECAY} \\
${SETUP_ARGS_TEXT}
  hydra.run.dir="${RUN_STORAGE_DIR}/hydra/\${SLURM_JOB_ID:-local}" \\
  wandb.save_dir="${WANDB_DIR}" \\
  +wandb.entity="${WANDB_ENTITY}" \\
  wandb.project="${WANDB_PROJECT}" \\
  wandb.group="${WANDB_GROUP}" \\
  wandb.name="${RUN_NAME}" \\
  +wandb.save_code=true \\
  wandb.job_type=speed_benchmark \\
  +wandb.tags="[speed_benchmark,random_data,${BENCH_SETUP},${BENCH_LABEL},L${N_BLOCKS},H${HIDDEN_SIZE},batch_size_${GBS},local_batch_${LOCAL_BATCH},accumulation_${ACC},${MODEL_DTYPE}]"
SBATCH_EOF

if [[ "${LAUNCH}" == sbatch ]]; then
  "${SBATCH_BIN}" ${SBATCH_ARGS} < "${JOB_FILE}"
  echo "Submitted ${RUN_NAME} (job script: ${JOB_FILE})."
else
  JOB_LOG="${LOG_DIR}/${RUN_NAME}.log"
  echo "Running ${JOB_FILE} (log: ${JOB_LOG})"
  set +e
  bash "${JOB_FILE}" 2>&1 | tee "${JOB_LOG}"
  status=${PIPESTATUS[0]}
  set -e
  exit "${status}"
fi
