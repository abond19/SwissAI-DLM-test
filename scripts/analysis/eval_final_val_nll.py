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

Parallel use: under ``srun --ntasks=N`` each task takes every N-th run and
binds to GPU ``SLURM_LOCALID``; see submit_eval_final_val_nll.sh.
"""

import argparse
import json
import os
import sys
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


def evaluate(run, args):
  cfg = load_config(run)
  ckpt = run / 'checkpoints' / 'last.ckpt'
  # mmap: read only the counters, not the weights.
  state = torch.load(ckpt, map_location='cpu', mmap=True, weights_only=False)
  global_step = int(state.get('global_step', -1))
  del state
  max_steps = int(cfg.trainer.max_steps)
  if global_step < max_steps and not args.allow_unfinished:
    return {'status': f'unfinished: step {global_step} < {max_steps}'}

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
  trainer = L.Trainer(
    **omegaconf.OmegaConf.to_container(cfg.trainer, resolve=True),
    default_root_dir=str(run / 'final_eval_scratch'),
    callbacks=[], logger=False, enable_checkpointing=False,
    enable_progress_bar=False,
    strategy=hydra.utils.instantiate(cfg.strategy))
  # The seed above fixes the noise for the whole pass (validation reads the
  # global RNG only).
  L.seed_everything(args.seed, verbose=False)
  metrics = trainer.validate(model, dataloaders=valid_loader,
                             ckpt_path=str(ckpt), verbose=False)
  result = {k: float(v) for k, v in (metrics[0] if metrics else {}).items()}
  result.update(status='ok', global_step=global_step, eval_seed=args.seed,
                eval_batch_size=args.eval_batch_size,
                limit_val_batches=args.limit_val_batches)
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
  p.add_argument('--force', action='store_true')
  p.add_argument('--allow_unfinished', action='store_true')
  args = p.parse_args()

  results_path = args.results or Path(args.roots[0]) / 'final_val_nll.jsonl'
  runs = find_runs(args.roots)[_SHARD::_NUM_SHARDS]
  print(f'[shard {_SHARD}/{_NUM_SHARDS}] {len(runs)} runs', flush=True)
  for i, run in enumerate(runs):
    out = run / 'final_val_nll.json'
    if out.exists() and not args.force:
      print(f'  skip (done): {run.name}', flush=True)
      continue
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
    print(f'  {result.get("status")} val/nll={result.get("val/nll")}', flush=True)
    if result['status'] == 'ok':
      out.write_text(json.dumps(result, indent=2))
    with open(results_path, 'a') as f:
      f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
  main()
