"""Per-optimizer-step training curve written to a CSV file."""

import csv
import os

import lightning as L
import torch


class TrainCurveCSVCallback(L.Callback):
  """Write the per-step mean of a training metric and each validation result.

  Rows go to ``path`` from global rank 0, one per optimizer step:
  ``step, tokens, lr, <metric>`` where ``<metric>`` is the mean over the step's
  micro-batches of the (already rank-synced) logged value.  Validation rows
  carry ``val/*`` metrics instead.  Values stay on the GPU and are converted in
  blocks of ``flush_every`` steps, so logging adds no per-step host sync.  On
  resume, rows past the restored step are dropped before appending.
  """

  def __init__(self, path: str, metric: str = 'train/elbo',
               tokens_per_step: int = 0, flush_every: int = 50,
               val_prefix: str = 'val/') -> None:
    super().__init__()
    self.path = path
    self.metric = metric
    self.tokens_per_step = int(tokens_per_step)
    self.flush_every = max(1, int(flush_every))
    self.val_prefix = val_prefix
    self._sum = None
    self._count = 0
    self._pending = []  # (step, lr, tensor)
    self._initialized = False
    self._last_step = None

  def _prepare_file(self, trainer):
    self._initialized = True
    if not trainer.is_global_zero:
      return
    os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
    start_step = int(trainer.global_step)
    if os.path.exists(self.path):
      with open(self.path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if int(r['step']) <= start_step]
    else:
      rows = []
    with open(self.path, 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['kind', 'step', 'tokens', 'lr', 'name', 'value'])
      for r in rows:
        writer.writerow([r['kind'], r['step'], r['tokens'], r['lr'],
                         r['name'], r['value']])

  def on_train_start(self, trainer, pl_module):
    if not self._initialized:
      self._prepare_file(trainer)
    self._last_step = int(trainer.global_step)

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if not trainer.is_global_zero:
      return
    value = trainer.callback_metrics.get(self.metric)
    if value is None:
      return
    value = value.detach().float()
    self._sum = value if self._sum is None else self._sum + value
    self._count += 1
    # global_step advances after the last micro-batch of an accumulation group.
    step = int(trainer.global_step)
    if self._last_step is not None and step == self._last_step:
      return
    self._last_step = step
    lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else float('nan')
    self._pending.append((step, float(lr), self._sum / self._count))
    self._sum = None
    self._count = 0
    if len(self._pending) >= self.flush_every:
      self._flush()

  def _flush(self):
    if not self._pending:
      return
    values = torch.stack([v for _, _, v in self._pending]).cpu().tolist()
    with open(self.path, 'a', newline='') as f:
      writer = csv.writer(f)
      for (step, lr, _), v in zip(self._pending, values):
        writer.writerow(['train', step, step * self.tokens_per_step, lr,
                         self.metric, v])
    self._pending = []

  def on_validation_end(self, trainer, pl_module):
    if not trainer.is_global_zero or trainer.sanity_checking:
      return
    self._flush()
    step = int(trainer.global_step)
    rows = [(k, float(v)) for k, v in trainer.callback_metrics.items()
            if k.startswith(self.val_prefix)]
    if not rows:
      return
    with open(self.path, 'a', newline='') as f:
      writer = csv.writer(f)
      for k, v in rows:
        writer.writerow(['val', step, step * self.tokens_per_step, '', k, v])

  def on_train_end(self, trainer, pl_module):
    if trainer.is_global_zero:
      self._flush()

  def on_save_checkpoint(self, trainer, pl_module, checkpoint):
    # Rows up to a checkpoint must be on disk before a later resume truncates.
    if trainer.is_global_zero:
      self._flush()
