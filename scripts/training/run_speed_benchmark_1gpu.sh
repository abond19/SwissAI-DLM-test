#!/usr/bin/env bash
set -euo pipefail

# Single-GPU speed benchmark on random data, run directly on a GPU node with the
# active conda environment (no Slurm, no container).
#
#   BENCH_SETUP=scion (default)  as submit_scion_b3_fixed_init_no_rho_gbs64.sh:
#                                gidd_hf, SCION, bf16 weights, micro-batch 8
#   BENCH_SETUP=mup              as submit_adamw_mup_hp_grid.sh (one grid cell):
#                                gidd_hf_ref + completep, AdamW-muP, WSD, fp32
#                                weights, micro-batch 4, grad clip 1.0
#
# Training overrides match the production launchers except: random data
# (data=synthetic-random-gidd), one GPU (global batch = one micro-batch, so the
# per-GPU work is the production one, with no gradient all-reduce between GPUs),
# a short run, a fresh start, W&B offline, and BenchmarkTimerCallback for timing.
# Data-pipeline changes are not measured (nothing is read from disk).
#
# One-time setup: download the GIDD tokenizer into the cache this script uses
# (only the tokenizer files, not the 10B model):
#   huggingface-cli download dvruette/gidd-unif-10b \
#     tokenizer.json tokenizer_config.json special_tokens_map.json config.json \
#     --cache-dir "$DATA_ROOT/cache/hf/transformers"
# (`hf download` with the same arguments if your huggingface_hub is >= 0.34.)
#
# Usage (on a GPU node):
#   conda activate <env>
#   export DATA_ROOT=/path/to/SwissAI-DLM-data
#   BENCH_LABEL=baseline scripts/training/run_speed_benchmark_1gpu.sh
#   BENCH_SETUP=mup BENCH_LABEL=baseline scripts/training/run_speed_benchmark_1gpu.sh
#
# Results: printed as a [BENCHMARK] line, written to <run dir>/benchmark_timing.json
# and appended to RESULTS_LOG.  Compare runs with, e.g.:
#   jq -r '[.setup,.run_label,.git_commit,.step_time_median_s,.peak_memory_allocated_gib]|@tsv' \
#     logs/speed_benchmark_1gpu/results.jsonl

BENCH_SETUP="${BENCH_SETUP:-scion}"
# Free-form tag for this run in the results log (letters, digits, . _ - only).
BENCH_LABEL="${BENCH_LABEL:-baseline}"
# Optimizer steps to run, and steps excluded from timing at the start.
BENCH_STEPS="${BENCH_STEPS:-500}"
BENCH_WARMUP_STEPS="${BENCH_WARMUP_STEPS:-50}"
DRY_RUN="${DRY_RUN:-0}"   # 1: print the command and exit
# Extra Hydra overrides appended to the command, space-separated, for keys the
# script does not already set, e.g. EXTRA_ARGS="+trainer.inference_mode=false".
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ---------------------------------------------------------------------------
# Paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Holds the Hugging Face cache (tokenizer) and, by default, the run outputs.
DATA_ROOT="${DATA_ROOT:-${SCRATCH:-${HOME}}/SwissAI-DLM-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs/speed_benchmark_1gpu}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/speed_benchmark_1gpu}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb_logs}"
RESULTS_LOG="${RESULTS_LOG:-${LOG_DIR}/results.jsonl}"
HF_HOME="${DATA_ROOT}/cache/hf"
# Python of the active conda environment unless PYTHON is set.
PYTHON="${PYTHON:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}"

# ---------------------------------------------------------------------------
# Shared settings (same values in both production launchers).
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
N_BLOCKS="${N_BLOCKS:-12}"
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
N_HEADS="${N_HEADS:-$((HIDDEN_SIZE / 64))}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-false}"
SEED="${SEED:-4}"
NUM_WORKERS="${NUM_WORKERS:-1}"
TRAIN_LOG_EVERY_N_STEPS="${TRAIN_LOG_EVERY_N_STEPS:-25}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.5}"
VALIDATE_AFTER_TRAINING="${VALIDATE_AFTER_TRAINING:-true}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
TORCH_COMPILE="${TORCH_COMPILE:-false}"
PROFILER="${PROFILER:-false}"
SYNTH_TRAIN_ROWS="${SYNTH_TRAIN_ROWS:-1000000}"
SYNTH_SEED="${SYNTH_SEED:-1234}"

# ---------------------------------------------------------------------------
# Setup-specific settings, copied from each production launcher.
declare -a SETUP_ARGS=()
case "${BENCH_SETUP}" in
  scion)
    LOCAL_BATCH="${LOCAL_BATCH:-8}"
    # 4096 rows over the production 8 ranks: same validation work per GPU.
    SYNTH_VALID_ROWS="${SYNTH_VALID_ROWS:-512}"
    MODEL_CONFIG=gidd_hf
    OPTIM_CONFIG=scion
    LR_SCHEDULER=constant_warmup
    MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
    FORWARD_AUTOCAST="${FORWARD_AUTOCAST:-fp32}"
    GRAD_CLIP="${GRAD_CLIP:-0.0}"
    VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-250}"
    BETA="${BETA:-1e-4}"
    MOMENTUM="${MOMENTUM:-0.2}"
    LR="${BETA}"
    WEIGHT_DECAY=0.0
    WARMDOWN_ITERS="${WARMDOWN_ITERS:-$((BENCH_STEPS * 28 / 100))}"
    SETUP_ARGS=(
      "model.intermediate_size=${INTERMEDIATE_SIZE:-$((4 * HIDDEN_SIZE))}"
      model.activation_scale=2.0
      model.attn_soft_cap=30.0
      model.resid_scale=4.0
      model.head_output_scaling_enabled=false
      model.tie_word_embeddings=false
      training.ema=0.0
      callbacks.late_validation_frequency.enabled=false
      callbacks.late_train_logging.enabled=false
      optim.equal_group_lr=true
      optim.boundary_init=true
      +optim.boundary_init_scale=1.0
      "optim.momentum=${MOMENTUM}"
      "optim.scale_embed=${SCALE_EMBED:-3500}"
      "optim.scale_bias=${SCALE_BIAS:-90}"
      "optim.scale_layer_norm=${SCALE_LN:-6.8}"
      "optim.scale_matrix=${SCALE_MATRIX:-336}"
      optim.bias_norm=RowNorm
      optim.norm_layer_norm=BiasRMS
      optim.spectral_norm_steps=5
      optim.unconstrained=false
      optim.warmup_iters=0
      "optim.warmdown_iters=${WARMDOWN_ITERS}"
      optim.min_lr=1e-8
      optim.trace_enabled=false
      optim.trace_collect_noise_stats=false
    )
    BETA_ID="${BETA//./p}"
    HP_TAG="beta${BETA_ID//-/m}_alpha${MOMENTUM//./p}"
    ;;
  mup)
    LOCAL_BATCH="${LOCAL_BATCH:-4}"
    # 4096 rows over the production 16 ranks: same validation work per GPU.
    SYNTH_VALID_ROWS="${SYNTH_VALID_ROWS:-256}"
    MODEL_CONFIG=gidd_hf_ref
    OPTIM_CONFIG=adamw_mup
    LR_SCHEDULER=wsd
    MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
    FORWARD_AUTOCAST="${FORWARD_AUTOCAST:-bf16}"
    GRAD_CLIP="${GRAD_CLIP:-1.0}"
    VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
    PARAMETRIZATION="${PARAMETRIZATION:-completep}"
    LR="$(awk -v a="${ETA_BASE:-2.19116e-2}" -v m="${ETA_MULT:-1}" 'BEGIN { printf "%.6e", a * m }')"
    WEIGHT_DECAY="${C:-1e-4}"
    SETUP_ARGS=(
      "model.parametrization=${PARAMETRIZATION}"
      "lr_scheduler.warmup_frac=${WARMUP_FRAC:-0.1}"
      "lr_scheduler.decay_frac=${DECAY_FRAC:-0.2}"
      optim.beta1=0.9
      optim.beta2=0.95
      optim.eps=1e-14
    )
    LR_ID="${LR//./p}"
    HP_TAG="${PARAMETRIZATION}_lr${LR_ID//-/m}_c${WEIGHT_DECAY//-/m}"
    ;;
  *)
    echo "BENCH_SETUP must be 'scion' or 'mup', got '${BENCH_SETUP}'." >&2
    exit 1
    ;;
esac
GBS="${LOCAL_BATCH}"

# ---------------------------------------------------------------------------
# Checks.
if [[ ! "${BENCH_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "BENCH_LABEL may contain only letters, digits, '.', '_' and '-'." >&2
  exit 1
fi
if (( BENCH_STEPS <= BENCH_WARMUP_STEPS )); then
  echo "BENCH_STEPS (${BENCH_STEPS}) must exceed BENCH_WARMUP_STEPS (${BENCH_WARMUP_STEPS})." >&2
  exit 1
fi
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "No Python found: activate the conda environment or set PYTHON." >&2
  exit 1
fi
if [[ ! -d "${HF_HOME}/transformers/models--dvruette--gidd-unif-10b" ]]; then
  echo "WARNING: GIDD tokenizer not found under ${HF_HOME}/transformers." >&2
  echo "  Download it first (see the header of this script) or the run will try the Hub." >&2
  if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Environment.
# The loader asserts global batch == micro-batch x visible GPUs, so expose exactly
# one GPU (the first of any already-visible set).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
# Inside a Slurm allocation Lightning would otherwise take the rank layout from
# SLURM_NTASKS instead of running the single local process.
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Disabled: on this cluster some dependencies (e.g. lightning_utilities) live in
# the user site-packages (~/.local), which PYTHONNOUSERSITE=1 would hide.
# export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
# transformers 4.45 reads the tokenizer from here when it is set.
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ "${GIT_COMMIT}" != unknown ]] && ! git -C "${REPO_ROOT}" diff --quiet HEAD 2>/dev/null; then
  GIT_COMMIT="${GIT_COMMIT}-dirty"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="speedbench1gpu_${BENCH_SETUP}_${BENCH_LABEL}_L${N_BLOCKS}H${HIDDEN_SIZE}_S${SEQUENCE_LENGTH}_lb${LOCAL_BATCH}_${MODEL_DTYPE}_${HP_TAG}_${TIMESTAMP}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

# ---------------------------------------------------------------------------
# Command.
CMD=(
  "${PYTHON}" -u -m discrete_diffusion
  "seed=${SEED}"
  data=synthetic-random-gidd
  "data.cache_dir=${RUN_DIR}/data_cache"
  "data.synthetic_train_size=${SYNTH_TRAIN_ROWS}"
  "data.synthetic_validation_size=${SYNTH_VALID_ROWS}"
  "data.synthetic_seed=${SYNTH_SEED}"
  "model=${MODEL_CONFIG}"
  "model.length=${SEQUENCE_LENGTH}"
  model.max_tokens=1024
  "model.n_blocks=${N_BLOCKS}"
  "model.hidden_size=${HIDDEN_SIZE}"
  "model.n_heads=${N_HEADS}"
  "model.activation_checkpointing=${ACTIVATION_CHECKPOINTING}"
  "model.torch_dtype=${MODEL_DTYPE}"
  "++training.forward_autocast_dtype=${FORWARD_AUTOCAST}"
  model.dropout=0.0
  algo=gidd
  algo.loss_type=gidd_easydel_lowmem
  algo.hybrid_mixing_shift=-1000
  algo.loss_weighting=dynamic
  algo.low_discrepancy_sampling=true
  algo.time_sampling_scope=global_batch
  "lr_scheduler=${LR_SCHEDULER}"
  strategy=ddp
  trainer.deterministic=false
  trainer.num_nodes=1
  trainer.devices=1
  trainer.accumulate_grad_batches=1
  "trainer.max_steps=${BENCH_STEPS}"
  "trainer.log_every_n_steps=${TRAIN_LOG_EVERY_N_STEPS}"
  "trainer.gradient_clip_val=${GRAD_CLIP}"
  trainer.num_sanity_val_steps=0
  "trainer.val_check_interval=${VAL_CHECK_INTERVAL}"
  "trainer.limit_val_batches=${LIMIT_VAL_BATCHES}"
  trainer.precision=bf16-mixed
  "loader.global_batch_size=${GBS}"
  "loader.eval_global_batch_size=${GBS}"
  "loader.batch_size=${LOCAL_BATCH}"
  "loader.eval_batch_size=${LOCAL_BATCH}"
  loader.multiprocessing_context=spawn
  +loader.lazy_spawn_dataset=false
  "loader.num_workers=${NUM_WORKERS}"
  loader.pin_memory=true
  training.antithetic_sampling=true
  training.loss_precision=bf16
  training.fault_tolerant=true
  "training.torch_compile=${TORCH_COMPILE}"
  training.log_train_aux_metrics=true
  training.sync_train_loss=true
  "training.validate_after_training=${VALIDATE_AFTER_TRAINING}"
  eval.generate_samples=false
  "callbacks.pytorch_profiler.enabled=${PROFILER}"
  +callbacks.benchmark_timer._target_=discrete_diffusion.callbacks.benchmark_timer.BenchmarkTimerCallback
  "+callbacks.benchmark_timer.warmup_steps=${BENCH_WARMUP_STEPS}"
  "+callbacks.benchmark_timer.output_dir=${RUN_DIR}"
  "+callbacks.benchmark_timer.results_log=${RESULTS_LOG}"
  "+callbacks.benchmark_timer.run_label='${BENCH_LABEL}'"
  "+callbacks.benchmark_timer.setup='${BENCH_SETUP}'"
  "+callbacks.benchmark_timer.git_commit='${GIT_COMMIT}'"
  perf.enabled=true
  perf.theoretical_peak_tflops_per_gpu=989
  "callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY}"
  callbacks.checkpoint_every_n_steps.save_top_k=1
  callbacks.checkpoint_every_n_steps.save_last=true
  callbacks.checkpoint_monitor.save_top_k=0
  "checkpointing.save_dir=${RUN_DIR}"
  checkpointing.resume_from_ckpt=false
  "optim=${OPTIM_CONFIG}"
  "optim.lr=${LR}"
  "optim.weight_decay=${WEIGHT_DECAY}"
  "${SETUP_ARGS[@]}"
  "hydra.run.dir=${RUN_DIR}/hydra"
  "wandb.save_dir=${WANDB_DIR}"
  wandb.project=SwissAI_DLM_Speed_Benchmark
  "wandb.group=speed_benchmark_1gpu_${BENCH_SETUP}"
  "wandb.name=${RUN_NAME}"
  wandb.job_type=speed_benchmark
  "+wandb.tags=[speed_benchmark,random_data,1gpu,${BENCH_SETUP},${BENCH_LABEL},${MODEL_DTYPE}]"
)
if [[ -n "${EXTRA_ARGS}" ]]; then
  read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_ARGS}"
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "Single-GPU speed benchmark: setup=${BENCH_SETUP}, label=${BENCH_LABEL}, commit=${GIT_COMMIT}"
echo "Model ${MODEL_CONFIG} L${N_BLOCKS}/H${HIDDEN_SIZE}, weights ${MODEL_DTYPE}, optimizer ${OPTIM_CONFIG}, micro-batch ${LOCAL_BATCH}"
echo "Steps ${BENCH_STEPS} (${BENCH_WARMUP_STEPS} warmup), GPU ${CUDA_VISIBLE_DEVICES}, python ${PYTHON}"
echo "Run dir ${RUN_DIR}"
if [[ -n "${EXTRA_ARGS}" ]]; then
  echo "Extra overrides: ${EXTRA_ARGS}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${WANDB_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
echo "Log ${LOG_FILE}"
cd "${REPO_ROOT}"
set +e
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
