"""Training-speed benchmark timer that does not rely on the perf.* metrics."""

from __future__ import annotations

import datetime
import gc
import json
import logging
import os
import socket
import statistics
import time

import lightning as L
import omegaconf
import torch


LOGGER = logging.getLogger(__name__)


def _cuda_sync():
  if torch.cuda.is_available():
    torch.cuda.synchronize()


def _percentile(values, q):
  if not values:
    return None
  ordered = sorted(values)
  k = (len(ordered) - 1) * q
  lo = int(k)
  hi = min(lo + 1, len(ordered) - 1)
  return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class BenchmarkTimerCallback(L.Callback):
  """Measure optimizer-step time over a steady-state window.

  The window starts once ``warmup_steps`` optimizer steps have completed
  (excluding compilation, allocator warmup and first-batch worker startup)
  and ends in ``on_train_end``.  The callback adds only a handful of CUDA
  synchronizations (window start/end and around each validation), so it does
  not perturb the per-step behaviour it is measuring.

  Reported step times:

  - ``step_time_median_s``: median of per-step wall-clock deltas, skipping
    steps that contained a validation or checkpoint.  Most robust number for
    comparing code changes to the training step itself.
  - ``step_time_mean_train_only_s``: window wall time minus validation time,
    divided by window steps.  Includes checkpoint writes.
  - ``step_time_mean_end_to_end_s``: window wall time divided by window
    steps.  Includes validation and checkpoints.

  Validation after ``trainer.fit`` returns (``validate_after_training``) and
  the final ``last.ckpt`` write in ``train.py`` happen after the window and
  are not counted.

  Opt-in memory diagnostics for validation (both off by default):

  - ``gc_each_val_step``: run ``gc.collect()`` before every validation batch
    and print GPU memory before/after it.  If memory that grows across
    validation batches is freed by this, it is held by Python reference cycles.
  - ``memory_snapshot``: ``'validate'`` records CUDA allocation history from
    the start of the post-training validation (``trainer.validate``); ``'all'``
    records from the start of fit (slows training; timings are then not
    comparable).  Snapshots are written to ``output_dir`` at the start and end
    of the post-training validation and on any exception (e.g. CUDA OOM).
    Open them at https://pytorch.org/memory_viz.

  With either enabled, GPU memory is printed at each validation batch.
  """

  def __init__(
      self,
      enabled: bool = True,
      warmup_steps: int = 50,
      output_dir: str | None = None,
      results_log: str | None = None,
      run_label: str = "",
      git_commit: str = "",
      setup: str = "",
      gc_each_val_step: bool = False,
      memory_snapshot: str = "",
  ) -> None:
    super().__init__()
    self.enabled = bool(enabled)
    self.warmup_steps = max(1, int(warmup_steps))
    self.output_dir = output_dir
    self.results_log = results_log
    self.run_label = str(run_label)
    self.git_commit = str(git_commit)
    self.setup = str(setup)
    self.gc_each_val_step = bool(gc_each_val_step)
    snapshot = str(memory_snapshot or '').strip().lower()
    if snapshot in ('1', 'true', 'validate'):
      snapshot = 'validate'
    elif snapshot in ('', '0', 'false', 'none'):
      snapshot = ''
    elif snapshot != 'all':
      raise ValueError(
        "memory_snapshot must be '', 'validate' (or 1) or 'all', got "
        f"{memory_snapshot!r}")
    self.memory_snapshot = snapshot
    self._diagnostics = self.gc_each_val_step or bool(self.memory_snapshot)
    self._recording = False

    self._last_step = -1
    self._window_start = None
    self._window_start_step = None
    self._prev_step_time = None
    self._step_dirty = False
    self._deltas = []
    self._val_start = None
    self._val_total_s = 0.0
    self._val_count = 0
    self._ckpt_count = 0
    self._finished = False

  @staticmethod
  def _validation_stage(trainer) -> str:
    fn = getattr(trainer.state.fn, 'value', trainer.state.fn)
    return 'post-training' if fn == 'validate' else 'in-training'

  def _print_memory(self, trainer, what: str) -> None:
    if not trainer.is_global_zero or not torch.cuda.is_available():
      return
    print(f'[BENCHMARK-MEM] {what}: allocated '
          f'{torch.cuda.memory_allocated() / 2**30:.2f} GiB, reserved '
          f'{torch.cuda.memory_reserved() / 2**30:.2f} GiB', flush=True)

  def _start_recording(self, trainer, when: str) -> None:
    if self._recording or not torch.cuda.is_available():
      return
    torch.cuda.memory._record_memory_history(max_entries=200_000)
    self._recording = True
    if trainer.is_global_zero:
      print(f'[BENCHMARK-MEM] recording CUDA allocation history from {when}',
            flush=True)

  def _dump_snapshot(self, trainer, name: str) -> None:
    if not self._recording:
      return
    output_dir = self.output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(
      output_dir, f'memory_snapshot_{name}_rank{trainer.global_rank}.pickle')
    torch.cuda.memory._dump_snapshot(path)
    if trainer.is_global_zero:
      print(f'[BENCHMARK-MEM] wrote {path}', flush=True)

  def on_fit_start(self, trainer, pl_module):
    del pl_module
    if self.memory_snapshot == 'all':
      self._start_recording(trainer, 'the start of fit')

  def on_validation_batch_start(
      self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
    del pl_module, batch, dataloader_idx
    if not self._diagnostics:
      return
    stage = self._validation_stage(trainer)
    if not self.gc_each_val_step:
      self._print_memory(trainer, f'{stage} validation batch {batch_idx}')
      return
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    collected = gc.collect()
    if trainer.is_global_zero and torch.cuda.is_available():
      after = torch.cuda.memory_allocated()
      print(f'[BENCHMARK-MEM] {stage} validation batch {batch_idx}: allocated '
            f'{before / 2**30:.2f} GiB before gc, {after / 2**30:.2f} GiB after '
            f'({collected} objects collected)', flush=True)

  def on_exception(self, trainer, pl_module, exception):
    del pl_module
    if not self._diagnostics:
      return
    if trainer.is_global_zero:
      print(f'[BENCHMARK-MEM] exception: {type(exception).__name__}', flush=True)
    self._print_memory(trainer, 'at exception')
    try:
      self._dump_snapshot(trainer, 'exception')
    except Exception as dump_error:  # never mask the original exception
      print(f'[BENCHMARK-MEM] snapshot dump failed: {dump_error}', flush=True)

  def _in_window(self, trainer) -> bool:
    return (self.enabled
            and self._window_start is not None
            and not self._finished
            and not trainer.sanity_checking)

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    del pl_module, outputs, batch, batch_idx
    if not self.enabled or self._finished:
      return
    step = int(trainer.global_step)
    # Called once per microbatch; only optimizer-step boundaries count.
    if step == self._last_step:
      return
    self._last_step = step

    if self._window_start is None:
      if step >= self.warmup_steps:
        _cuda_sync()
        now = time.perf_counter()
        self._window_start = now
        self._window_start_step = step
        self._prev_step_time = now
        self._step_dirty = False
      return

    now = time.perf_counter()
    if not self._step_dirty:
      self._deltas.append(now - self._prev_step_time)
    self._prev_step_time = now
    self._step_dirty = False

  def on_validation_start(self, trainer, pl_module):
    del pl_module
    if self._diagnostics:
      stage = self._validation_stage(trainer)
      if stage == 'post-training' and self.memory_snapshot:
        self._start_recording(trainer, 'the post-training validation')
        self._dump_snapshot(trainer, 'postval_start')
      self._print_memory(trainer, f'{stage} validation start')
    if not self._in_window(trainer):
      return
    _cuda_sync()
    self._val_start = time.perf_counter()

  def on_validation_end(self, trainer, pl_module):
    del pl_module
    if self._diagnostics:
      stage = self._validation_stage(trainer)
      self._print_memory(trainer, f'{stage} validation end')
      if stage == 'post-training':
        self._dump_snapshot(trainer, 'postval_end')
    if not self._in_window(trainer) or self._val_start is None:
      return
    _cuda_sync()
    now = time.perf_counter()
    self._val_total_s += now - self._val_start
    self._val_count += 1
    self._val_start = None
    # Validation runs between two optimizer steps; restart the step clock so
    # the next delta measures only training work.
    self._prev_step_time = now

  def on_save_checkpoint(self, trainer, pl_module, checkpoint):
    del pl_module, checkpoint
    if not self._in_window(trainer):
      return
    self._ckpt_count += 1
    self._step_dirty = True

  def on_train_end(self, trainer, pl_module):
    del pl_module
    if not self.enabled or self._finished:
      return
    self._finished = True
    if self._window_start is None:
      if trainer.is_global_zero:
        LOGGER.warning(
          '[BENCHMARK] training ended at step %d before the %d warmup steps '
          'completed; no timing recorded.', trainer.global_step,
          self.warmup_steps)
      return
    _cuda_sync()
    window_s = time.perf_counter() - self._window_start
    window_steps = int(trainer.global_step) - int(self._window_start_step)

    peak_alloc = peak_reserved = 0.0
    if torch.cuda.is_available():
      peak = torch.tensor(
        [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
        dtype=torch.float64, device=torch.cuda.current_device())
      if (torch.distributed.is_available()
          and torch.distributed.is_initialized()):
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
      peak_alloc, peak_reserved = (float(v) / 2**30 for v in peak.tolist())

    if not trainer.is_global_zero:
      return

    config = getattr(trainer.lightning_module, 'config', None)
    def _sel(key, default=None):
      if config is None:
        return default
      return omegaconf.OmegaConf.select(config, key, default=default)

    world_size = int(trainer.world_size)
    global_batch = _sel('loader.global_batch_size')
    seq_len = _sel('model.length')
    tokens_per_step = (int(global_batch) * int(seq_len)
                       if global_batch and seq_len else None)

    median = statistics.median(self._deltas) if self._deltas else None
    train_only = ((window_s - self._val_total_s) / window_steps
                  if window_steps > 0 else None)
    end_to_end = window_s / window_steps if window_steps > 0 else None

    def _tok_per_s(step_s):
      if not step_s or not tokens_per_step:
        return None
      return tokens_per_step / step_s

    tps_median = _tok_per_s(median)
    tps_train_only = _tok_per_s(train_only)
    result = {
      'run_label': self.run_label,
      'setup': self.setup,
      'git_commit': self.git_commit,
      'model_name': _sel('model.name'),
      'model_torch_dtype': _sel('model.torch_dtype'),
      'optimizer': _sel('optim.name'),
      'timestamp': datetime.datetime.now().astimezone().isoformat(
        timespec='seconds'),
      'hostname': socket.gethostname(),
      'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
      'torch_version': torch.__version__,
      'gpu_name': (torch.cuda.get_device_name()
                   if torch.cuda.is_available() else None),
      'world_size': world_size,
      'num_nodes': int(trainer.num_nodes),
      'local_batch': _sel('loader.batch_size'),
      'accumulate_grad_batches': int(trainer.accumulate_grad_batches),
      'global_batch': global_batch,
      'seq_len': seq_len,
      'tokens_per_step': tokens_per_step,
      'max_steps': int(trainer.max_steps),
      'warmup_steps': self.warmup_steps,
      'window_steps': window_steps,
      'window_wall_s': window_s,
      'clean_steps_measured': len(self._deltas),
      'step_time_median_s': median,
      'step_time_mean_clean_s': (statistics.fmean(self._deltas)
                                 if self._deltas else None),
      'step_time_p10_s': _percentile(self._deltas, 0.10),
      'step_time_p90_s': _percentile(self._deltas, 0.90),
      'step_time_mean_train_only_s': train_only,
      'step_time_mean_end_to_end_s': end_to_end,
      'tokens_per_s_median': tps_median,
      'tokens_per_s_per_gpu_median': (tps_median / world_size
                                      if tps_median else None),
      'tokens_per_s_train_only': tps_train_only,
      'validation_count': self._val_count,
      'validation_total_s': self._val_total_s,
      'checkpoint_count': self._ckpt_count,
      'peak_memory_allocated_gib': peak_alloc,
      'peak_memory_reserved_gib': peak_reserved,
    }

    def _fmt(v, unit='', scale=1.0, digits=1):
      return 'n/a' if v is None else f'{v * scale:.{digits}f}{unit}'

    summary = (
      f"[BENCHMARK] setup={self.setup or '-'} "
      f"label={self.run_label or '-'} "
      f"commit={self.git_commit or '-'} "
      f"window={window_steps} steps after {self.warmup_steps} warmup | "
      f"median step {_fmt(median, ' ms', 1e3)} "
      f"(p10 {_fmt(result['step_time_p10_s'], '', 1e3)}, "
      f"p90 {_fmt(result['step_time_p90_s'], '', 1e3)}) | "
      f"train-only mean {_fmt(train_only, ' ms', 1e3)} | "
      f"end-to-end mean {_fmt(end_to_end, ' ms', 1e3)} | "
      f"{_fmt(result['tokens_per_s_per_gpu_median'], ' tok/s/GPU', 1.0, 0)} | "
      f"validation {self._val_count}x {_fmt(self._val_total_s, ' s')} | "
      f"checkpoints {self._ckpt_count} | "
      f"peak mem {peak_alloc:.1f} GiB alloc / {peak_reserved:.1f} GiB reserved")
    LOGGER.info(summary)
    print(summary, flush=True)

    output_dir = self.output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'benchmark_timing.json')
    with open(output_path, 'w') as f:
      json.dump(result, f, indent=2)
    LOGGER.info('[BENCHMARK] wrote %s', output_path)
    if self.results_log:
      os.makedirs(os.path.dirname(os.path.abspath(self.results_log)),
                  exist_ok=True)
      with open(self.results_log, 'a') as f:
        f.write(json.dumps(result) + '\n')
      LOGGER.info('[BENCHMARK] appended to %s', self.results_log)
