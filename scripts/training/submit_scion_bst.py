#!/usr/bin/env python3
"""Launcher for the SCION BST experiment (L12/H768 GIDD, B.3 parameterization).

Step 1 tunes the base model at T0; step 2 multiplies the token budget by m and
sets the global batch by the BST rule B = B_base * m^(2/3), sweeping the same
Frank-Wolfe step-size (beta) grid at every budget.  One Slurm job (or a chain
of auto-resuming jobs) is submitted per point of

    multipliers x batch sizes x radii x momenta x betas x seeds.

Dry run by default: prints every run with its batch size, topology, steps and
estimated GPU-hours, plus totals.  Add --submit to submit.

Training setup (fixed): B.3 SCION with equal group step sizes, boundary init at
radius 1, no gradient clipping, constant step size with a 28% linear warmdown,
fp32 master weights with bf16 compute, regional torch.compile, full Nemotron-CC
packed cache.  Each run directory gets run_meta.json (what was asked for),
train_curve.csv (per-step train ELBO and every validation), and a DONE marker
when training finishes.  Final ranking numbers come from
scripts/analysis/eval_final_val_nll.py, not from the in-run validation.

Examples (from the repo root, on a login node):
  # Step 1, beta x batch-size grid at T0
  scripts/training/submit_scion_bst.py --multipliers 1 --batch-sizes 256 \\
      --betas 2e-5 3e-5 4e-5 5e-5 6e-5 7.5e-5 9e-5 1.2e-4
  # Step 2, the 2x and 4x budgets on the full grid
  scripts/training/submit_scion_bst.py --multipliers 2 4 --betas 2e-5 3e-5 4e-5 5e-5 6e-5 7.5e-5 9e-5 1.2e-4
See also submit_scion_bst_step1.sh / submit_scion_bst_step2.sh.
"""

import argparse
import getpass
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RADII = '3500:336:90:6.8'  # embed:matrix:bias:norm (Set-C)
GPUS_PER_NODE = 4


def fmt_num(x):
  """6e-05 -> 6e-5, 1.2e-4 -> 1.2e-4, 0.06 -> 0.06 (for Hydra values and names)."""
  x = float(x)
  if x != 0 and abs(x) < 1e-3:
    mantissa, exponent = f'{x:.6e}'.split('e')
    return f'{mantissa.rstrip("0").rstrip(".")}e{int(exponent)}'
  return f'{x:g}'


def tag(x):
  return fmt_num(x).replace('.', 'p').replace('-', 'm').replace('+', '')


def parse_hours(limit):
  parts = [int(p) for p in limit.replace('-', ':').split(':')]
  if len(parts) == 4:  # D-HH:MM:SS
    d, h, m, s = parts
  else:
    d, (h, m, s) = 0, ([0] * (3 - len(parts)) + parts)
  return d * 24 + h + m / 60 + s / 3600


def ranks_for(gbs, local_batch, max_ranks):
  micro = gbs // local_batch
  r = min(max_ranks, micro) // GPUS_PER_NODE * GPUS_PER_NODE
  while r >= GPUS_PER_NODE:
    if micro % r == 0:
      return r
    r -= GPUS_PER_NODE
  return None


def choose_gbs(target, local_batch, max_ranks, tol):
  """Multiple of one node's micro-batch near target; within the tolerance,
  prefer the size that spreads over the most GPUs (shorter wall-clock)."""
  unit = local_batch * GPUS_PER_NODE
  lo, hi = target * (1 - tol), target * (1 + tol)
  cands = [k * unit for k in range(max(1, math.floor(lo / unit)), math.ceil(hi / unit) + 1)
           if lo <= k * unit <= hi]
  if not cands:
    cands = [max(unit, round(target / unit) * unit)]
  return max(cands, key=lambda g: (ranks_for(g, local_batch, max_ranks) or 0, -abs(g - target)))


def build_runs(args):
  runs = []
  max_ranks = args.max_nodes * GPUS_PER_NODE
  local_batches = [args.local_batch] + (
    [args.fallback_local_batch] if args.fallback_local_batch != args.local_batch else [])
  for m in args.multipliers:
    tokens = int(round(args.t0 * m))
    plan = []  # (gbs, local batch)
    if args.batch_sizes:
      for b in map(int, args.batch_sizes):
        lb = next((lb for lb in local_batches if b % (lb * GPUS_PER_NODE) == 0), None)
        if lb is None:
          sys.exit(f'GBS {b} is not a multiple of {local_batches} x {GPUS_PER_NODE}.')
        plan.append((b, lb))
    else:
      # BST batch; fall back to the smaller micro-batch when the larger one
      # cannot represent it within the tolerance.
      target = args.b_base * m ** (2.0 / 3.0)
      choices = [(choose_gbs(target, lb, max_ranks, args.gbs_tol), lb) for lb in local_batches]
      plan.append(next((c for c in choices if abs(c[0] - target) <= args.gbs_tol * target),
                       min(choices, key=lambda c: abs(c[0] - target))))
    for gbs, lb in plan:
      ranks = ranks_for(gbs, lb, max_ranks)
      acc = gbs // (lb * ranks)
      steps = tokens // (gbs * args.seq_len)
      for radii in args.radii:
        embed, matrix, bias, norm = radii.split(':')
        for alpha in args.momenta:
          for beta in args.betas:
            for seed in args.seeds:
              name = (f'scion_bst_m{tag(m)}_gbs{gbs}_beta{tag(beta)}_a{tag(alpha)}'
                      + ('' if radii == DEFAULT_RADII else
                         f'_r{tag(embed)}-{tag(matrix)}-{tag(bias)}-{tag(norm)}')
                      + f'_s{seed}{args.name_suffix}')
              gpu_h = steps * gbs * args.seq_len / args.throughput / 3600 * 1.03
              wall_h = gpu_h / ranks + 0.1
              runs.append(dict(
                name=name, multiplier=m, t0=args.t0, tokens_requested=tokens,
                tokens=steps * gbs * args.seq_len, gbs=gbs,
                gbs_bst_target=args.b_base * m ** (2.0 / 3.0), b_base=args.b_base,
                beta=fmt_num(beta), momentum=fmt_num(alpha),
                scale_embed=embed, scale_matrix=matrix, scale_bias=bias, scale_norm=norm,
                seed=int(seed), local_batch=lb, ranks=ranks,
                nodes=ranks // GPUS_PER_NODE, accumulation=acc, max_steps=steps,
                warmdown_steps=steps * 28 // 100, est_gpu_hours=gpu_h, est_wall_hours=wall_h))
  return runs


def job_script(run, args, run_dir, cache_dir):
  steps = run['max_steps']
  acc = run['accumulation']
  val_every = max(1, steps // args.val_count)
  if args.checkpoint_every:
    ckpt_every = args.checkpoint_every
  elif run['est_wall_hours'] < 0.85 * parse_hours(args.time_limit):
    ckpt_every = steps  # one job is enough: only the final checkpoint
  else:
    ckpt_every = max(100, int(steps / run['est_wall_hours']))  # about hourly
  data_root = args.data_root
  overrides = [
    f'seed={run["seed"]}',
    'data=nemotron-cc-pretok',
    f'data.pretok_local_dir={args.data_src_root}/training-data/gidd-nemotron-cc-pretok',
    f'data.cache_dir={cache_dir}',
    'model=gidd_hf',
    f'model.length={args.seq_len}', 'model.max_tokens=1024',
    f'model.n_blocks={args.n_blocks}', f'model.hidden_size={args.hidden_size}',
    f'model.intermediate_size={4 * args.hidden_size}', f'model.n_heads={args.hidden_size // 64}',
    'model.activation_checkpointing=false',
    f'model.torch_dtype={args.model_dtype}',
    f'++training.forward_autocast_dtype={args.forward_autocast}',
    'model.dropout=0.0', 'model.activation_scale=2.0', 'model.attn_soft_cap=30.0',
    'model.resid_scale=4.0', 'model.head_output_scaling_enabled=false',
    'model.tie_word_embeddings=false',
    'algo=gidd', 'algo.loss_type=gidd_easydel_lowmem', 'algo.hybrid_mixing_shift=-1000',
    'algo.loss_weighting=dynamic', 'algo.low_discrepancy_sampling=true',
    'algo.time_sampling_scope=global_batch',
    'lr_scheduler=constant_warmup',
    'strategy=ddp', 'trainer.deterministic=false',
    f'trainer.num_nodes={run["nodes"]}', f'trainer.devices={GPUS_PER_NODE}',
    f'trainer.accumulate_grad_batches={acc}',
    f'trainer.max_steps={steps}',
    f'trainer.log_every_n_steps={args.train_log_every}',
    'trainer.gradient_clip_val=0.0', 'trainer.num_sanity_val_steps=0',
    f'trainer.val_check_interval={val_every * acc}',
    f'trainer.limit_val_batches={args.limit_val_batches}',
    'trainer.precision=bf16-mixed',
    f'loader.global_batch_size={run["gbs"]}',
    f'loader.eval_global_batch_size={run["local_batch"] * run["ranks"]}',
    f'loader.batch_size={run["local_batch"]}', f'loader.eval_batch_size={run["local_batch"]}',
    'loader.multiprocessing_context=spawn', '+loader.lazy_spawn_dataset=true',
    f'loader.num_workers={args.num_workers}', 'loader.pin_memory=true',
    'training.ema=0.0', 'training.antithetic_sampling=true', 'training.loss_precision=bf16',
    'training.fault_tolerant=true', f'training.torch_compile={args.torch_compile}',
    'training.log_train_aux_metrics=true', 'training.sync_train_loss=true',
    'training.validate_after_training=false',
    'eval.generate_samples=false', 'callbacks.pytorch_profiler.enabled=false',
    'callbacks.late_validation_frequency.enabled=false',
    'callbacks.late_train_logging.enabled=false',
    'perf.enabled=true', 'perf.theoretical_peak_tflops_per_gpu=989',
    f'callbacks.checkpoint_every_n_steps.every_n_train_steps={ckpt_every}',
    'callbacks.checkpoint_every_n_steps.save_top_k=1',
    'callbacks.checkpoint_every_n_steps.save_last=true',
    '+callbacks.checkpoint_every_n_steps.save_on_train_epoch_end=true',
    'callbacks.checkpoint_monitor.save_top_k=0',
    f'checkpointing.save_dir={run_dir}',
    'checkpointing.resume_from_ckpt=true',
    f'checkpointing.resume_ckpt_path={run_dir}/checkpoints/last.ckpt',
    '+callbacks.fixed_noise_validation._target_='
    'discrete_diffusion.callbacks.fixed_noise_validation.FixedNoiseValidationCallback',
    f'+callbacks.fixed_noise_validation.seed={args.val_seed}',
    '+callbacks.train_curve_csv._target_='
    'discrete_diffusion.callbacks.train_curve_csv.TrainCurveCSVCallback',
    f'+callbacks.train_curve_csv.path={run_dir}/train_curve.csv',
    f'+callbacks.train_curve_csv.tokens_per_step={run["gbs"] * args.seq_len}',
    'optim=scion', f'optim.lr={run["beta"]}', 'optim.equal_group_lr=true',
    'optim.boundary_init=true', '+optim.boundary_init_scale=1.0',
    'optim.weight_decay=0.0', f'optim.momentum={run["momentum"]}',
    f'optim.scale_embed={run["scale_embed"]}', f'optim.scale_bias={run["scale_bias"]}',
    f'optim.scale_layer_norm={run["scale_norm"]}', f'optim.scale_matrix={run["scale_matrix"]}',
    'optim.bias_norm=RowNorm', 'optim.norm_layer_norm=BiasRMS', 'optim.spectral_norm_steps=5',
    'optim.unconstrained=false', 'optim.warmup_iters=0',
    f'optim.warmdown_iters={run["warmdown_steps"]}', 'optim.min_lr=1e-8',
    'optim.trace_enabled=false', 'optim.trace_collect_noise_stats=false',
    f'hydra.run.dir={run_dir}/hydra/${{SLURM_JOB_ID}}',
    f'wandb.save_dir={args.wandb_dir}',
    f'+wandb.entity={args.wandb_entity}', f'wandb.project={args.wandb_project}',
    f'wandb.group={args.experiment}', f'wandb.name={run["name"]}',
    f'wandb.id={run["name"][-64:]}', '+wandb.resume=allow',
    'wandb.job_type=train',
    f'+wandb.tags=[bst,scion,b3,m{tag(run["multiplier"])},gbs{run["gbs"]},beta{run["beta"]}]',
  ] + list(args.extra)
  partition = f'#SBATCH --partition={args.partition}' if args.partition else ''
  cmd = ' \\\n  '.join(f'"{o}"' for o in overrides)
  return f'''#!/bin/bash
#SBATCH --job-name={run["name"]}
#SBATCH --time={args.time_limit}
#SBATCH --nodes={run["nodes"]}
#SBATCH --ntasks-per-node={GPUS_PER_NODE}
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node={GPUS_PER_NODE}
#SBATCH --account={args.account}
{partition}
#SBATCH --output={args.log_dir}/%x_%j.out
#SBATCH --error={args.log_dir}/%x_%j.err

set -euo pipefail
if [[ -f "{run_dir}/DONE" ]]; then
  echo "Already finished: {run_dir}"; exit 0
fi
cd "{REPO_ROOT}"
export PYTHONPATH="{REPO_ROOT}:{REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export WANDB_MODE={args.wandb_mode}
export WANDB_DIR="{args.wandb_dir}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export JOB_TMPDIR="${{SLURM_TMPDIR:-/tmp/SwissAI-DLM-${{USER:-unknown}}-${{SLURM_JOB_ID:-manual}}}}"
export TMPDIR="${{JOB_TMPDIR}}" TMP="${{JOB_TMPDIR}}" TEMP="${{JOB_TMPDIR}}"
export XDG_CACHE_HOME="${{JOB_TMPDIR}}/xdg"
# Persistent compile cache: later jobs and resumes reuse compiled kernels.
export TORCHINDUCTOR_CACHE_DIR="{data_root}/cache/torchinductor"
export HF_HOME="{data_root}/cache/hf"
export HF_DATASETS_CACHE="${{HF_HOME}}/datasets" HUGGINGFACE_HUB_CACHE="${{HF_HOME}}/hub"
export TRANSFORMERS_CACHE="${{HF_HOME}}/transformers"
mkdir -p "${{TMPDIR}}" "${{XDG_CACHE_HOME}}" "{args.wandb_dir}" "{run_dir}" "${{TORCHINDUCTOR_CACHE_DIR}}"

srun --environment=uni-d2 --ntasks={run["ranks"]} --ntasks-per-node={GPUS_PER_NODE} --cpus-per-task=72 \\
  "{REPO_ROOT}/venv-docker/bin/python" -u -m discrete_diffusion \\
  {cmd}
touch "{run_dir}/DONE"
'''


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  g = p.add_argument_group('grid')
  g.add_argument('--multipliers', type=float, nargs='+', default=[1.0])
  g.add_argument('--t0', type=float, default=2e9, help='base token budget T0')
  g.add_argument('--b-base', type=float, default=256, help='base global batch (sequences)')
  g.add_argument('--batch-sizes', nargs='+', default=None,
                 help='explicit global batch sizes instead of the BST rule')
  g.add_argument('--betas', type=float, nargs='+', required=True)
  g.add_argument('--momenta', type=float, nargs='+', default=[0.06])
  g.add_argument('--radii', nargs='+', default=[DEFAULT_RADII],
                 help='embed:matrix:bias:norm, one or more')
  g.add_argument('--seeds', type=int, nargs='+', default=[4])
  t = p.add_argument_group('topology and estimates')
  t.add_argument('--local-batch', type=int, default=8, help='micro-batch per GPU')
  t.add_argument('--fallback-local-batch', type=int, default=8,
                 help='used when --local-batch cannot represent a BST batch size')
  t.add_argument('--max-nodes', type=int, default=4,
                 help='cap per run; the rest is gradient accumulation (amortizes the SCION update)')
  t.add_argument('--gbs-tol', type=float, default=0.02,
                 help='relative tolerance when rounding the BST batch size')
  t.add_argument('--throughput', type=float, default=140000,
                 help='tokens/s per GPU used for the estimates (update after benchmarking)')
  t.add_argument('--time-limit', default='12:00:00')
  t.add_argument('--chain', default='auto',
                 help='jobs per run chained with --dependency=singleton (auto: from the estimate)')
  t.add_argument('--checkpoint-every', type=int, default=0,
                 help='0: final checkpoint only if one job suffices, else about hourly')
  m = p.add_argument_group('model and training')
  m.add_argument('--seq-len', type=int, default=2048)
  m.add_argument('--n-blocks', type=int, default=12)
  m.add_argument('--hidden-size', type=int, default=768)
  m.add_argument('--model-dtype', default='fp32')
  m.add_argument('--forward-autocast', default='bf16')
  m.add_argument('--torch-compile', default='auto')
  m.add_argument('--num-workers', type=int, default=1)
  m.add_argument('--val-count', type=int, default=16, help='in-run validations per run')
  m.add_argument('--limit-val-batches', type=float, default=0.1,
                 help='fraction of the validation set for in-run curves')
  m.add_argument('--val-seed', type=int, default=1234)
  m.add_argument('--train-log-every', type=int, default=10)
  m.add_argument('--extra', nargs='*', default=[], help='extra Hydra overrides')
  o = p.add_argument_group('paths and submission')
  scratch = os.environ.get('SCRATCH', f'/iopsstor/scratch/cscs/{getpass.getuser()}')
  o.add_argument('--experiment', default='scion_bst_L12H768_T0_2B')
  o.add_argument('--name-suffix', default='')
  o.add_argument('--data-src-root', default='/iopsstor/scratch/cscs/zdelbari/SwissAI-DLM-data')
  o.add_argument('--data-root', default=f'{scratch}/SwissAI-DLM-data')
  o.add_argument('--output-root', default=None)
  o.add_argument('--account', default='ab035')
  o.add_argument('--partition', default='')
  o.add_argument('--wandb-mode', default='offline')
  o.add_argument('--wandb-entity', default='SwissAI_DLM')
  o.add_argument('--wandb-project', default='SwissAI_Scion_BST')
  o.add_argument('--submit', action='store_true')
  args = p.parse_args()

  args.output_root = Path(args.output_root or f'{args.data_root}/outputs/bst_scion/{args.experiment}')
  args.log_dir = REPO_ROOT / 'logs' / args.experiment
  args.wandb_dir = REPO_ROOT / 'wandb_logs'

  runs = build_runs(args)
  limit_h = parse_hours(args.time_limit)
  print(f'{"run":<62} {"T":>7} {"GBS":>5} {"(BST)":>6} {"nodes":>5} {"mb":>3} {"acc":>4} '
        f'{"steps":>6} {"GPU-h":>6} {"wall-h":>6} {"jobs":>4}')
  total_gpu_h = total_tokens = 0
  for r in runs:
    r['chain'] = (max(1, math.ceil(r['est_wall_hours'] / (0.9 * limit_h)))
                  if args.chain == 'auto' else int(args.chain))
    total_gpu_h += r['est_gpu_hours']
    total_tokens += r['tokens']
    print(f'{r["name"]:<62} {r["tokens"] / 1e9:6.1f}B {r["gbs"]:>5} {r["gbs_bst_target"]:6.0f} '
          f'{r["nodes"]:>5} {r["local_batch"]:>3} {r["accumulation"]:>4} {r["max_steps"]:>6} '
          f'{r["est_gpu_hours"]:6.1f} {r["est_wall_hours"]:6.1f} {r["chain"]:>4}')
  print(f'\n{len(runs)} runs, {total_tokens / 1e9:.0f}B tokens, ~{total_gpu_h:.0f} GPU-h '
        f'(~{total_gpu_h / GPUS_PER_NODE:.0f} node-h) at {args.throughput:.0f} tok/s/GPU')
  if not args.submit:
    print('Dry run. Add --submit to submit.')
    return

  # Own cache dir of links to the shared packed dataset: anything the loader
  # writes next to the cache lands in this user's scratch.
  cache_dir = Path(args.data_root) / 'cache/discrete_diffusion/nemotron-cc-pretok'
  cache_dir.mkdir(parents=True, exist_ok=True)
  src_cache = Path(args.data_src_root) / 'cache/discrete_diffusion/nemotron-cc-pretok'
  for split, mode in (('train', 'train'), ('valid', 'validation')):
    name = f'nemotron-cc-pretok-{split}_{mode}_bs{args.seq_len}_unwrapped_eosFalse_specialFalse.dat'
    if not (src_cache / name).is_dir():
      sys.exit(f'Missing shared packed dataset: {src_cache / name}')
    if not (cache_dir / name).exists():
      (cache_dir / name).symlink_to(src_cache / name)
  args.log_dir.mkdir(parents=True, exist_ok=True)

  submitted = 0
  for r in runs:
    run_dir = args.output_root / r['name']
    if (run_dir / 'DONE').exists():
      print(f'skip (finished): {r["name"]}')
      continue
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in r.items()}
    meta.update(experiment=args.experiment, model_dtype=args.model_dtype,
                forward_autocast=args.forward_autocast, seq_len=args.seq_len)
    (run_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2))
    script = job_script(r, args, run_dir, cache_dir)
    (run_dir / 'job.sbatch').write_text(script)
    for _ in range(r['chain']):
      extra = ['--dependency=singleton'] if r['chain'] > 1 else []
      out = subprocess.run(['sbatch', *extra, str(run_dir / 'job.sbatch')],
                           capture_output=True, text=True)
      print(out.stdout.strip() or out.stderr.strip())
      submitted += out.returncode == 0
  print(f'Submitted {submitted} jobs.')


if __name__ == '__main__':
  main()
