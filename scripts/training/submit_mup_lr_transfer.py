#!/usr/bin/env python3
"""Learning-rate transfer check for the AdamW-muP GIDD model across widths.

Trains the same model (gidd_hf_ref, CompleteP, depth --n-blocks) at several widths
over the same grid of learning-rate multipliers, on the same token budget and
global batch.  muP transfers the learning rate if the best multiplier is the same
at every width (in base-width units, the width-256 convention of optim=adamw_mup).

Defaults apply the two confirmed fixes from the coordinate checks
(model.rms_norm_eps=1e-6, model.final_norm_affine=false) and keep Peri-LN on,
which is the configuration under test; --no-peri-ln runs the control.

One Slurm job per (width, multiplier, seed).  Dry run by default; --submit to
submit (login node, outside the container).  Rank with
scripts/analysis/run_eval_final_val_nll.sh, then lr_transfer_analysis.py.
"""

import argparse
import getpass
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit_scion_bst import GPUS_PER_NODE, ranks_for, tag  # noqa: E402
from submit_cooldown_experiment import VOCAB, job_script  # noqa: E402

FIXES = ['model.rms_norm_eps=1e-6', '++model.final_norm_affine=false']


def flops_per_token(width, blocks, seq_len):
  """Training FLOPs per token: 6 x (matrices + head) + attention scores."""
  return 6 * (12 * blocks * width * width + width * VOCAB) + 12 * blocks * seq_len * width


def parse_map(text):
  return {int(k): int(v) for k, v in (item.split(':') for item in text.split(','))}


def build(args):
  steps = int(args.tokens // (args.gbs * args.seq_len))
  local_batches = parse_map(args.local_batch_by_width)
  ref = flops_per_token(768, 12, args.seq_len)
  runs = []
  for width in args.widths:
    if width % 64:
      sys.exit(f'width {width} must be a multiple of the head dim 64')
    lb = local_batches.get(width, min(local_batches.values()))
    ranks = ranks_for(args.gbs, lb, args.max_nodes * GPUS_PER_NODE)
    if ranks is None:
      sys.exit(f'GBS {args.gbs} is not a multiple of {lb} x {GPUS_PER_NODE} (width {width}).')
    acc = args.gbs // (lb * ranks)
    # Throughput scaled from the measured width-768 number by FLOPs per token;
    # small models run at lower utilisation, so the gain is capped at 3x.
    tput = args.throughput_768 * min(3.0, ref / flops_per_token(width, args.n_blocks, args.seq_len))
    gpu_h = steps * args.gbs * args.seq_len / tput / 3600 * 1.05 + 0.05 * ranks
    for mult in args.eta_mults:
      for seed in args.seeds:
        runs.append(dict(
          kind='single', name=f'{args.prefix}_w{width}_eta{tag(mult)}_s{seed}',
          hidden_size=width, n_blocks=args.n_blocks, eta_mult=mult, seed=seed,
          eta=args.eta_base * mult, weight_decay=args.weight_decay,
          warmup_frac=args.warmup_frac, rho=args.decay_frac, shape='linear',
          rec_k0_frac=None, start_step=0, max_steps=steps, total_steps=steps,
          gbs=args.gbs, local_batch=lb, ranks=ranks, nodes=ranks // GPUS_PER_NODE,
          accumulation=acc, tokens=steps * args.gbs * args.seq_len, seq_len=args.seq_len,
          peri_ln=args.peri_ln, est_gpu_hours=gpu_h, est_wall_hours=gpu_h / ranks))
  return runs


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  g = p.add_argument_group('experiment')
  g.add_argument('--widths', type=int, nargs='+', default=[256, 512, 1024])
  g.add_argument('--n-blocks', type=int, default=12)
  g.add_argument('--eta-base', type=float, default=2.19116e-2)
  g.add_argument('--eta-mults', type=float, nargs='+', default=[0.25, 0.5, 1.0, 2.0, 4.0])
  g.add_argument('--seeds', type=int, nargs='+', default=[4])
  g.add_argument('--tokens', type=float, default=1e9, help='same budget at every width')
  g.add_argument('--gbs', type=int, default=128)
  g.add_argument('--weight-decay', type=float, default=1e-4)
  g.add_argument('--warmup-frac', type=float, default=0.1)
  g.add_argument('--decay-frac', type=float, default=0.2, help='linear cooldown fraction')
  g.add_argument('--no-peri-ln', dest='peri_ln', action='store_false',
                 help='control: Pre-LN only (model.peri_ln=false)')
  g.add_argument('--no-fixes', dest='fixes', action='store_false',
                 help='omit the epsilon and final-norm fixes')
  g.add_argument('--grad-clip', type=float, default=1.0)
  t = p.add_argument_group('topology')
  t.add_argument('--local-batch-by-width', default='256:16,512:16,768:16,1024:8,2048:4')
  t.add_argument('--max-nodes', type=int, default=1)
  t.add_argument('--throughput-768', type=float, default=200000,
                 help='tokens/s/GPU at width 768 (measured, micro-batch 16)')
  t.add_argument('--time-limit', default='04:00:00')
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
  o.add_argument('--experiment', default=None)
  o.add_argument('--prefix', default='lrt')
  o.add_argument('--data-src-root', default='/iopsstor/scratch/cscs/zdelbari/SwissAI-DLM-data')
  o.add_argument('--data-root', default=f'{scratch}/SwissAI-DLM-data')
  o.add_argument('--output-root', default=None)
  o.add_argument('--account', default='ab035')
  o.add_argument('--partition', default='')
  o.add_argument('--wandb-mode', default='offline')
  o.add_argument('--wandb-entity', default='SwissAI_DLM')
  o.add_argument('--wandb-project', default='SwissAI_muP_LR_Transfer')
  o.add_argument('--submit', action='store_true')
  args = p.parse_args()
  args.experiment = args.experiment or (
    'mup_lr_transfer_' + ('periln' if args.peri_ln else 'noperiln')
    + ('' if args.fixes else '_nofixes'))
  args.extra = (FIXES if args.fixes else []) + ([] if args.peri_ln else ['model.peri_ln=false']) \
    + list(args.extra)
  args.output_root = Path(args.output_root or
                          f'{args.data_root}/outputs/lr_transfer/{args.experiment}')
  args.log_dir = Path(__file__).resolve().parents[2] / 'logs' / args.experiment
  args.wandb_dir = Path(__file__).resolve().parents[2] / 'wandb_logs'

  runs = build(args)
  print(f'{args.experiment}: {runs[0]["tokens"] / 1e9:.2f}B tokens, GBS {args.gbs}, '
        f'{runs[0]["max_steps"]} steps, depth {args.n_blocks}; extra overrides: {args.extra}')
  print(f'{"run":<28} {"nodes":>5} {"mb":>3} {"acc":>4} {"GPU-h":>6} {"wall-h":>6}')
  for r in runs:
    print(f'{r["name"]:<28} {r["nodes"]:>5} {r["local_batch"]:>3} {r["accumulation"]:>4} '
          f'{r["est_gpu_hours"]:6.1f} {r["est_wall_hours"]:6.1f}')
  total = sum(r['est_gpu_hours'] for r in runs)
  print(f'\n{len(runs)} jobs, ~{total:.0f} GPU-h (~{total / GPUS_PER_NODE:.0f} node-h)')
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
  queue = subprocess.run(['squeue', '-u', getpass.getuser(), '--noheader', '--format=%j'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
  if queue.returncode != 0:
    sys.exit('squeue failed: run on a login node, outside the container.\n' + queue.stderr)
  queued = set(queue.stdout.split())
  submitted = 0
  for r in runs:
    run_dir = args.output_root / r['name']
    if (run_dir / 'DONE').exists() or r['name'] in queued:
      print(f'skip (finished or queued): {r["name"]}')
      continue
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'run_meta.json').write_text(json.dumps(dict(r, experiment=args.experiment,
                                                            extra=args.extra), indent=2))
    (run_dir / 'job.sbatch').write_text(job_script(r, args, run_dir, cache_dir, run_dir))
    out = subprocess.run(['sbatch', str(run_dir / 'job.sbatch')], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, universal_newlines=True)
    print(out.stdout.strip() or out.stderr.strip())
    if out.returncode != 0:
      sys.exit(f'sbatch failed after {submitted} jobs (submit from a login node, outside the '
               'container).  Rerunning skips finished and queued runs.')
    submitted += 1
  print(f'Submitted {submitted} jobs.')


if __name__ == '__main__':
  main()
