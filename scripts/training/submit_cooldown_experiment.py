#!/usr/bin/env python3
"""Cooldown-shape experiment for the AdamW-muP GIDD model (L12/H768, completep).

Compares learning-rate tails of a warmup-stable-decay schedule over the final
fraction rho of training (see the "two terminal cooldowns" note):
  linear      eta0 (1 - j/M)
  cosine      eta0 (1 + cos(pi j/M)) / 2
  sqrt        eta0 (1 - sqrt(j/M))
  reciprocal  eta0 k0/(k0 + j) (1 - j/M), k0 = rec_k0_frac * M  (the proposed tail)
plus the classic cosine schedule (warmup, then cosine to 0: the cosine tail at
rho = 0.9 with 10% warmup).

All tails for one (peak LR, seed) share the warmup and constant phase, so that
prefix is trained once: a trunk run saves full checkpoints at every switch step
and each tail resumes from its switch checkpoint (Slurm afterok dependency).
Every run reseeds its RNG from (seed, step, rank) at each step, so all arms see
the same data and the same diffusion noise: they differ only in the schedule,
and small differences between tails are measurable from one seed.

Dry run by default (prints every arm with its steps and GPU-hours); --submit
submits.  Run on a login node, outside the container.  Then rank with
scripts/analysis/run_eval_final_val_nll.sh and cooldown_analysis.py.
"""

import argparse
import getpass
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit_scion_bst import GPUS_PER_NODE, REPO_ROOT, fmt_num, parse_hours, ranks_for, tag  # noqa: E402

# gidd_hf_ref total parameters (untied embeddings), as in submit_adamw_mup_hp_grid.sh.
VOCAB, HIDDEN, BLOCKS = 131072, 768, 12
N_PARAMS = 2 * VOCAB * HIDDEN + BLOCKS * (12 * HIDDEN * HIDDEN + 4 * HIDDEN) + HIDDEN


def switch_step(total_steps, rho):
  """First tail step, with the scheduler's rounding (WSDLRScheduler)."""
  return total_steps - int(round(rho * total_steps))


def build(args):
  steps = int(args.tokens // (args.gbs * args.seq_len))
  ranks = ranks_for(args.gbs, args.local_batch, args.max_nodes * GPUS_PER_NODE)
  if ranks is None:
    sys.exit(f'GBS {args.gbs} is not a multiple of {args.local_batch} x {GPUS_PER_NODE}.')
  acc = args.gbs // (args.local_batch * ranks)
  tails = []
  for rho in args.rhos:
    for shape in args.shapes:
      for k0 in (args.rec_k0_fracs if shape == 'reciprocal' else [None]):
        tails.append((rho, shape, k0))
  if args.classic_cosine:
    tails.append((1.0 - args.warmup_frac, 'cosine', None))
  rho_min = min(r for r, _, _ in tails)
  gpu_h_per_step = args.gbs * args.seq_len / args.throughput / 3600
  runs = []
  for mult in args.eta_mults:
    for seed in args.seeds:
      base = f'{args.prefix}_eta{tag(mult)}_s{seed}'
      trunk_end = switch_step(steps, rho_min)
      trunk = dict(kind='trunk', name=f'{base}_trunk', eta_mult=mult, seed=seed,
                   rho=rho_min, shape='linear', rec_k0_frac=None, start_step=0,
                   max_steps=trunk_end,
                   branch_steps=sorted({switch_step(steps, r) for r, _, _ in tails}))
      runs.append(trunk)
      for rho, shape, k0 in tails:
        name = f'{base}_rho{tag(rho)}_{shape}' + (f'_k{tag(k0)}' if k0 is not None else '')
        runs.append(dict(kind='tail', name=name, eta_mult=mult, seed=seed, rho=rho,
                         shape=shape, rec_k0_frac=k0, start_step=switch_step(steps, rho),
                         max_steps=steps, trunk=trunk['name']))
  for r in runs:
    r.update(total_steps=steps, gbs=args.gbs, local_batch=args.local_batch, ranks=ranks,
             nodes=ranks // GPUS_PER_NODE, accumulation=acc, tokens=steps * args.gbs * args.seq_len,
             eta=args.eta_base * r['eta_mult'], weight_decay=args.weight_decay,
             warmup_frac=args.warmup_frac, seq_len=args.seq_len)
    r['est_gpu_hours'] = (r['max_steps'] - r['start_step']) * gpu_h_per_step * 1.03 + 0.05 * ranks
    r['est_wall_hours'] = r['est_gpu_hours'] / ranks
  return runs


def overrides(r, args, run_dir, cache_dir):
  steps = r['total_steps']
  acc = r['accumulation']
  val_every = max(1, steps // args.val_count)
  o = [
    f'seed={r["seed"]}', 'data=nemotron-cc-pretok',
    f'data.pretok_local_dir={args.data_src_root}/training-data/gidd-nemotron-cc-pretok',
    f'data.cache_dir={cache_dir}',
    'model=gidd_hf_ref', 'model.parametrization=completep',
    f'model.length={args.seq_len}', 'model.max_tokens=1024',
    f'model.n_blocks={r.get("n_blocks", BLOCKS)}', f'model.hidden_size={r.get("hidden_size", HIDDEN)}',
    f'model.n_heads={r.get("hidden_size", HIDDEN) // 64}',
    'model.dropout=0.0', 'model.activation_checkpointing=false',
    '++training.forward_autocast_dtype=bf16',
    'algo=gidd', 'algo.loss_type=gidd_easydel_lowmem', 'algo.hybrid_mixing_shift=-1000',
    'algo.loss_weighting=dynamic', 'algo.low_discrepancy_sampling=true',
    'algo.time_sampling_scope=global_batch',
    'lr_scheduler=wsd', f'lr_scheduler.total_steps={steps}',
    f'lr_scheduler.warmup_frac={r["warmup_frac"]}', f'lr_scheduler.decay_frac={r["rho"]}',
    f'lr_scheduler.decay_shape={r["shape"]}',
    f'lr_scheduler.rec_k0_frac={r["rec_k0_frac"] if r["rec_k0_frac"] is not None else 0.1}',
    'strategy=ddp', 'trainer.deterministic=false',
    f'trainer.num_nodes={r["nodes"]}', f'trainer.devices={GPUS_PER_NODE}',
    f'trainer.accumulate_grad_batches={acc}', f'trainer.max_steps={r["max_steps"]}',
    f'trainer.log_every_n_steps={args.train_log_every}',
    f'trainer.gradient_clip_val={args.grad_clip}', 'trainer.num_sanity_val_steps=0',
    f'trainer.val_check_interval={val_every * acc}',
    f'trainer.limit_val_batches={args.limit_val_batches}', 'trainer.precision=bf16-mixed',
    f'loader.global_batch_size={r["gbs"]}',
    f'loader.eval_global_batch_size={r["local_batch"] * r["ranks"]}',
    f'loader.batch_size={r["local_batch"]}', f'loader.eval_batch_size={r["local_batch"]}',
    'loader.multiprocessing_context=spawn', '+loader.lazy_spawn_dataset=true',
    f'loader.num_workers={args.num_workers}', 'loader.pin_memory=true',
    'training.antithetic_sampling=true', 'training.loss_precision=bf16',
    'training.fault_tolerant=true', f'training.torch_compile={args.torch_compile}',
    'training.log_train_aux_metrics=true', 'training.sync_train_loss=true',
    'training.validate_after_training=false',
    # Noise depends only on (seed, step, rank): every arm sees identical noise.
    '++training.rng_reseed_every_step=true',
    'eval.generate_samples=false', 'callbacks.pytorch_profiler.enabled=false',
    'perf.enabled=true', 'perf.theoretical_peak_tflops_per_gpu=989',
    f'callbacks.checkpoint_every_n_steps.every_n_train_steps={r["max_steps"]}',
    'callbacks.checkpoint_every_n_steps.save_top_k=1',
    'callbacks.checkpoint_every_n_steps.save_last=true',
    '+callbacks.checkpoint_every_n_steps.save_on_train_epoch_end=true',
    'callbacks.checkpoint_monitor.save_top_k=0',
    f'checkpointing.save_dir={run_dir}', 'checkpointing.resume_from_ckpt=true',
    'checkpointing.resume_ckpt_path=${RESUME_CKPT}',
    '+callbacks.fixed_noise_validation._target_='
    'discrete_diffusion.callbacks.fixed_noise_validation.FixedNoiseValidationCallback',
    f'+callbacks.fixed_noise_validation.seed={args.val_seed}',
    '+callbacks.train_curve_csv._target_='
    'discrete_diffusion.callbacks.train_curve_csv.TrainCurveCSVCallback',
    f'+callbacks.train_curve_csv.path={run_dir}/train_curve.csv',
    f'+callbacks.train_curve_csv.tokens_per_step={r["gbs"] * args.seq_len}',
    'optim=adamw_mup', f'optim.lr={fmt_num(r["eta"])}',
    f'optim.weight_decay={fmt_num(r["weight_decay"])}',
    'optim.beta1=0.9', 'optim.beta2=0.95', 'optim.eps=1e-14',
    f'hydra.run.dir={run_dir}/hydra/${{SLURM_JOB_ID}}',
    f'wandb.save_dir={args.wandb_dir}', f'+wandb.entity={args.wandb_entity}',
    f'wandb.project={args.wandb_project}', f'wandb.group={args.experiment}',
    f'wandb.name={r["name"]}', f'wandb.id={r["name"][-64:]}', '+wandb.resume=allow',
    'wandb.job_type=train',
  ]
  if r['kind'] == 'trunk' and r.get('branch_steps'):
    o += ['+callbacks.checkpoint_at_steps._target_='
          'discrete_diffusion.callbacks.checkpoint_at_steps.CheckpointAtStepsCallback',
          f'+callbacks.checkpoint_at_steps.dirpath={run_dir}/branch_ckpts',
          '+callbacks.checkpoint_at_steps.steps=[' + ','.join(map(str, r['branch_steps'])) + ']']
  return o + list(args.extra)


def job_script(r, args, run_dir, cache_dir, trunk_dir):
  own = f'{run_dir}/checkpoints/last.ckpt'
  if r['kind'] == 'tail':
    resume = (f'RESUME_CKPT="{own}"\n'
              f'[[ -f "${{RESUME_CKPT}}" ]] || RESUME_CKPT="{trunk_dir}/branch_ckpts/step{r["start_step"]}.ckpt"\n'
              '[[ -f "${RESUME_CKPT}" ]] || { echo "Missing trunk checkpoint ${RESUME_CKPT}"; exit 1; }')
  else:
    resume = f'RESUME_CKPT="{own}"'
  partition = f'#SBATCH --partition={args.partition}' if args.partition else ''
  cmd = ' \\\n  '.join(f'"{o}"' for o in overrides(r, args, run_dir, cache_dir))
  data_root = args.data_root
  return f'''#!/bin/bash
#SBATCH --job-name={r["name"]}
#SBATCH --time={args.time_limit}
#SBATCH --nodes={r["nodes"]}
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
{resume}
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
export TORCHINDUCTOR_CACHE_DIR="{data_root}/cache/torchinductor"
export HF_HOME="{data_root}/cache/hf"
export HF_DATASETS_CACHE="${{HF_HOME}}/datasets" HUGGINGFACE_HUB_CACHE="${{HF_HOME}}/hub"
export TRANSFORMERS_CACHE="${{HF_HOME}}/transformers"
mkdir -p "${{TMPDIR}}" "${{XDG_CACHE_HOME}}" "{args.wandb_dir}" "{run_dir}" "${{TORCHINDUCTOR_CACHE_DIR}}"

srun --environment=uni-d2 --ntasks={r["ranks"]} --ntasks-per-node={GPUS_PER_NODE} --cpus-per-task=72 \\
  "{REPO_ROOT}/venv-docker/bin/python" -u -m discrete_diffusion \\
  {cmd}
touch "{run_dir}/DONE"
'''


def run_cmd(cmd):
  out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
  return out.returncode, out.stdout.strip(), out.stderr.strip()


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  g = p.add_argument_group('experiment')
  g.add_argument('--tokens', type=float, default=20 * N_PARAMS,
                 help='token budget (default 20 per parameter, 5.73B)')
  g.add_argument('--gbs', type=int, default=256)
  g.add_argument('--eta-base', type=float, default=2.19116e-2)
  g.add_argument('--eta-mults', type=float, nargs='+', default=[0.5, 1.0, 2.0])
  g.add_argument('--weight-decay', type=float, default=1e-4, help='decoupled c = eta * lambda')
  g.add_argument('--warmup-frac', type=float, default=0.1)
  g.add_argument('--rhos', type=float, nargs='+', default=[0.2, 0.5],
                 help='cooldown fractions for the tail shapes')
  g.add_argument('--shapes', nargs='+', default=['linear', 'cosine', 'reciprocal'],
                 choices=['linear', 'cosine', 'sqrt', 'reciprocal'])
  g.add_argument('--rec-k0-fracs', type=float, nargs='+', default=[0.05, 0.2],
                 help='reciprocal offset k0 as a fraction of the tail length M')
  g.add_argument('--no-classic-cosine', dest='classic_cosine', action='store_false',
                 help='omit the warmup + cosine-to-zero baseline')
  g.add_argument('--seeds', type=int, nargs='+', default=[4])
  g.add_argument('--grad-clip', type=float, default=1.0)
  t = p.add_argument_group('topology')
  t.add_argument('--local-batch', type=int, default=16)
  t.add_argument('--max-nodes', type=int, default=2)
  t.add_argument('--throughput', type=float, default=200000,
                 help='tokens/s per GPU for the estimates (muP, micro-batch 16)')
  t.add_argument('--time-limit', default='06:00:00')
  o = p.add_argument_group('run settings and paths')
  scratch = os.environ.get('SCRATCH', f'/iopsstor/scratch/cscs/{getpass.getuser()}')
  o.add_argument('--seq-len', type=int, default=2048)
  o.add_argument('--torch-compile', default='auto')
  o.add_argument('--num-workers', type=int, default=1)
  o.add_argument('--val-count', type=int, default=20)
  o.add_argument('--limit-val-batches', type=float, default=0.1)
  o.add_argument('--val-seed', type=int, default=1234)
  o.add_argument('--train-log-every', type=int, default=10)
  o.add_argument('--extra', nargs='*', default=[])
  o.add_argument('--experiment', default='cooldown_mup_L12H768_20tpp')
  o.add_argument('--prefix', default='cd')
  o.add_argument('--data-src-root', default='/iopsstor/scratch/cscs/zdelbari/SwissAI-DLM-data')
  o.add_argument('--data-root', default=f'{scratch}/SwissAI-DLM-data')
  o.add_argument('--output-root', default=None)
  o.add_argument('--account', default='ab035')
  o.add_argument('--partition', default='')
  o.add_argument('--wandb-mode', default='offline')
  o.add_argument('--wandb-entity', default='SwissAI_DLM')
  o.add_argument('--wandb-project', default='SwissAI_Cooldown')
  o.add_argument('--submit', action='store_true')
  args = p.parse_args()
  args.output_root = Path(args.output_root or f'{args.data_root}/outputs/cooldown/{args.experiment}')
  args.log_dir = REPO_ROOT / 'logs' / args.experiment
  args.wandb_dir = REPO_ROOT / 'wandb_logs'

  runs = build(args)
  r0 = runs[0]
  print(f'T={r0["tokens"] / 1e9:.2f}B, GBS={args.gbs}, {r0["total_steps"]} steps; '
        f'{r0["nodes"]} nodes x {GPUS_PER_NODE}, micro-batch {args.local_batch}, '
        f'accumulation {r0["accumulation"]}')
  print(f'{"run":<44} {"steps":>13} {"GPU-h":>6} {"wall-h":>6}')
  for r in runs:
    print(f'{r["name"]:<44} {r["start_step"]:>6}-{r["max_steps"]:<6} '
          f'{r["est_gpu_hours"]:6.1f} {r["est_wall_hours"]:6.1f}')
  total = sum(r['est_gpu_hours'] for r in runs)
  full = sum(1 for r in runs if r['kind'] == 'tail') * runs[1]['total_steps'] * (
    args.gbs * args.seq_len / args.throughput / 3600) * 1.03
  print(f'\n{len(runs)} jobs, ~{total:.0f} GPU-h (~{total / GPUS_PER_NODE:.0f} node-h); '
        f'without branching ~{full:.0f} GPU-h')
  if max(r['est_wall_hours'] for r in runs) > 0.9 * parse_hours(args.time_limit):
    print('WARNING: some runs may exceed --time-limit; resubmitting resumes them.')
  if not args.submit:
    print('Dry run. Add --submit to submit (from a login node, outside the container).')
    return

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

  code, out, err = run_cmd(['squeue', '-u', getpass.getuser(), '--noheader', '--format=%i %j'])
  if code != 0:
    sys.exit('squeue failed: run on a login node, outside the container.\n' + err)
  queued = {}
  for line in out.splitlines():
    job_id, _, job_name = line.partition(' ')
    queued[job_name] = job_id
  job_ids = {}
  submitted = 0
  for r in runs:  # trunks come before their tails
    run_dir = args.output_root / r['name']
    trunk_dir = args.output_root / r.get('trunk', r['name'])
    if (run_dir / 'DONE').exists():
      print(f'skip (finished): {r["name"]}')
      continue
    if r['name'] in queued:
      job_ids[r['name']] = queued[r['name']]
      print(f'skip (queued as {queued[r["name"]]}): {r["name"]}')
      continue
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(r, experiment=args.experiment)
    (run_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2))
    (run_dir / 'job.sbatch').write_text(job_script(r, args, run_dir, cache_dir, trunk_dir))
    cmd = ['sbatch']
    if r['kind'] == 'tail' and not (trunk_dir / 'DONE').exists():
      if r['trunk'] not in job_ids:
        sys.exit(f'Trunk {r["trunk"]} is neither finished nor queued; not submitting its tails.')
      cmd.append(f'--dependency=afterok:{job_ids[r["trunk"]]}')
    code, out, err = run_cmd(cmd + [str(run_dir / 'job.sbatch')])
    print(out or err)
    if code != 0:
      sys.exit(f'sbatch failed after {submitted} jobs (submit from a login node, outside the '
               'container).  Rerunning skips finished and queued runs.')
    match = re.search(r'(\d+)\s*$', out)
    job_ids[r['name']] = match.group(1) if match else None
    submitted += 1
  print(f'Submitted {submitted} jobs.')


if __name__ == '__main__':
  main()
