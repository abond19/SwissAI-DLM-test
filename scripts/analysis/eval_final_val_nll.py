#!/usr/bin/env python3
"""Final validation NLL on the full validation set with identical noise.

The validation loss logged during training depends on the run's topology
(ranks and eval batch size decide which examples meet which diffusion noise),
so runs with different global batch sizes are not directly comparable.  This
script re-evaluates each run's final checkpoint on ONE GPU with a fixed eval
batch size and a fixed seed: every run sees the same sequences in the same
order with the same diffusion times and masks, and differences between runs
come only from the weights.

For every run directory under the given roots (a directory containing
``run_meta.json`` from submit_scion_bst.py, or any run with
``hydra/*/.hydra/config.yaml`` and ``checkpoints/last.ckpt``) it writes
``<run>/final_val_nll.json`` and appends one line to ``--results``.  Finished
evaluations are skipped unless --force.  Runs that have not reached their
``trainer.max_steps`` are skipped.

Parallel use: under ``srun --ntasks=N`` each task takes every N-th run still to
evaluate (finished, no result yet), so the shards stay balanced, and
binds to GPU ``SLURM_LOCALID``; see run_eval_final_val_nll.sh.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# One GPU per task, chosen before torch initializes CUDA.  The loaders assert
# global batch == micro-batch x visible GPUs, so exactly one must be visible.
_LOCAL_ID = int(os.environ.get('SLURM_LOCALID', 0))
_SHARD = int(os.environ.get('SLURM_PROCID', 0))
_NUM_SHARDS = int(os.environ.get('SLURM_NTASKS', 1))
_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
if not _visible:
  os.environ['CUDA_VISIBLE_DEVICES'] = str(_LOCAL_ID)
elif ',' in _visible:
  _devices = _visible.split(',')
  os.environ['CUDA_VISIBLE_DEVICES'] = _devices[_LOCAL_ID % len(_devices)]
# Otherwise Lightning would read the srun task layout as a distributed job.
for _var in ('SLURM_NTASKS', 'SLURM_NTASKS_PER_NODE'):
  os.environ.pop(_var, None)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))

import hydra  # noqa: E402
import lightning as L  # noqa: E402
import omegaconf  # noqa: E402
import torch  # noqa: E402

import discrete_diffusion.__main__  # noqa: E402,F401  (registers OmegaConf resolvers)
from discrete_diffusion.data import get_dataloaders, get_tokenizer  # noqa: E402


def find_runs(roots):
  runs = []
  for root in roots:
    root = Path(root)
    candidates = [root] if (root / 'checkpoints').is_dir() else sorted(root.iterdir())
    for run in candidates:
      if run.is_dir() and (run / 'checkpoints' / 'last.ckpt').exists():
        runs.append(run)
  return runs


def load_config(run):
  configs = sorted((run / 'hydra').glob('*/.hydra/config.yaml'),
                   key=lambda p: p.stat().st_mtime)
  if not configs:
    raise FileNotFoundError(f'no hydra config under {run}/hydra')
  return omegaconf.OmegaConf.load(configs[-1])


def checkpoint_step(run):
  # mmap: read only the counters, not the weights.
  state = torch.load(run / 'checkpoints' / 'last.ckpt', map_location='cpu',
                     mmap=True, weights_only=False)
  return int(state.get('global_step', -1))


def pending_runs(roots, args):
  """Runs still to evaluate: finished training, no result yet (or --force).

  Every shard computes this same list and then takes every N-th entry, so the
  work is split evenly however many runs are already evaluated or unfinished.
  """
  pending = []
  for run in find_runs(roots):
    if (run / 'final_val_nll.json').exists() and not args.force:
      continue
    if not args.allow_unfinished:
      try:
        step, max_steps = checkpoint_step(run), int(load_config(run).trainer.max_steps)
      except Exception as e:  # noqa: BLE001 - leave it for evaluate() to report
        step, max_steps = None, f'{type(e).__name__}: {e}'
      if step is not None and step < max_steps:
        if _SHARD == 0:
          print(f'  skip (unfinished, step {step}/{max_steps}): {run.name}', flush=True)
        continue
    pending.append(run)
  return pending


class Progress(L.Callback):
  """Prints batches done, s/batch and an ETA (prints only; no RNG use)."""

  def __init__(self, total, every):
    self.total, self.every, self.start = total, max(1, every), None

  def on_validation_start(self, trainer, pl_module):
    self.start = time.time()

  def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                              dataloader_idx=0):
    done = batch_idx + 1
    if done % self.every and done != self.total:
      return
    elapsed = time.time() - self.start
    left = (self.total - done) * elapsed / done
    print(f'    batch {done}/{self.total}: {elapsed / done:.3f} s/batch, '
          f'{elapsed / 60:.1f} min elapsed, ~{left / 60:.1f} min left', flush=True)


def evaluate(run, args):
  cfg = load_config(run)
  ckpt = run / 'checkpoints' / 'last.ckpt'
  global_step = checkpoint_step(run)
  start = time.time()

  # Fixed evaluation topology: one device, fixed eval batch, whole split.
  cfg.trainer.devices = 1
  cfg.trainer.num_nodes = 1
  cfg.trainer.accumulate_grad_batches = 1
  cfg.trainer.limit_val_batches = args.limit_val_batches
  cfg.trainer.num_sanity_val_steps = 0
  cfg.loader.batch_size = args.eval_batch_size
  cfg.loader.global_batch_size = args.eval_batch_size
  cfg.loader.eval_batch_size = args.eval_batch_size
  cfg.loader.eval_global_batch_size = args.eval_batch_size
  cfg.loader.num_workers = args.num_workers
  cfg.eval.generate_samples = False
  cfg.training.torch_compile = False
  cfg.strategy = omegaconf.OmegaConf.create({
    '_target_': 'lightning.pytorch.strategies.SingleDeviceStrategy',
    'device': 'cuda:0'})

  torch.set_float32_matmul_precision('high')
  L.seed_everything(args.seed, verbose=False)
  tokenizer = get_tokenizer(cfg)
  _, valid_loader = get_dataloaders(cfg, tokenizer, skip_train=True)
  model = hydra.utils.get_class(cfg.algo._target_)(cfg, tokenizer=tokenizer)
  num_batches = len(valid_loader)
  if args.limit_val_batches <= 1:
    num_batches = max(1, int(num_batches * args.limit_val_batches))
  print(f'  {len(valid_loader.dataset)} sequences, {num_batches} batches of '
        f'{args.eval_batch_size}', flush=True)
  trainer = L.Trainer(
    **omegaconf.OmegaConf.to_container(cfg.trainer, resolve=True),
    default_root_dir=str(run / 'final_eval_scratch'),
    callbacks=[Progress(num_batches, args.progress_every)], logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False,
    strategy=hydra.utils.instantiate(cfg.strategy))
  # The seed above fixes the noise for the whole pass (validation reads the
  # global RNG only).
  L.seed_everything(args.seed, verbose=False)
  setup_seconds = time.time() - start
  metrics = trainer.validate(model, dataloaders=valid_loader,
                             ckpt_path=str(ckpt), verbose=False)
  result = {k: float(v) for k, v in (metrics[0] if metrics else {}).items()}
  result.update(status='ok', global_step=global_step, eval_seed=args.seed,
                eval_batch_size=args.eval_batch_size,
                limit_val_batches=args.limit_val_batches,
                setup_seconds=round(setup_seconds, 1),
                validate_seconds=round(time.time() - start - setup_seconds, 1))
  del model, trainer, valid_loader
  torch.cuda.empty_cache()
  return result


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('roots', nargs='+', help='run directories or directories of runs')
  p.add_argument('--results', type=Path, default=None,
                 help='JSONL collecting all results (default <first root>/final_val_nll.jsonl)')
  p.add_argument('--seed', type=int, default=1234)
  p.add_argument('--eval_batch_size', type=int, default=8)
  p.add_argument('--limit_val_batches', type=float, default=1.0)
  p.add_argument('--num_workers', type=int, default=2)
  p.add_argument('--progress_every', type=int, default=250,
                 help='print progress every this many batches')
  p.add_argument('--force', action='store_true')
  p.add_argument('--allow_unfinished', action='store_true')
  args = p.parse_args()

  results_path = args.results or Path(args.roots[0]) / 'final_val_nll.jsonl'
  pending = pending_runs(args.roots, args)
  runs = pending[_SHARD::_NUM_SHARDS]
  print(f'[shard {_SHARD}/{_NUM_SHARDS}] {len(runs)} of {len(pending)} runs to evaluate',
        flush=True)
  for i, run in enumerate(runs):
    out = run / 'final_val_nll.json'
    print(f'[{i + 1}/{len(runs)}] {run.name}', flush=True)
    try:
      result = evaluate(run, args)
    except Exception as e:  # noqa: BLE001 - keep going through the other runs
      result = {'status': f'error: {type(e).__name__}: {e}'}
      torch.cuda.empty_cache()
    result['run_dir'] = str(run)
    meta = run / 'run_meta.json'
    if meta.exists():
      result['meta'] = json.loads(meta.read_text())
    print(f'  {result.get("status")} val/nll={result.get("val/nll")} '
          f'(setup {result.get("setup_seconds")} s, validate {result.get("validate_seconds")} s)',
          flush=True)
    if result['status'] == 'ok':
      out.write_text(json.dumps(result, indent=2))
    with open(results_path, 'a') as f:
      f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
  main()
