"""Opt-in validation-frequency increase for the final training phase."""

from __future__ import annotations

import logging

import lightning as L


LOGGER = logging.getLogger(__name__)


class LateValidationFrequencyCallback(L.Callback):
  """Increase periodic validation frequency after a chosen training fraction.

  Lightning converts an integer ``trainer.val_check_interval`` into
  ``trainer.val_check_batch``.  Our pretraining runs remain in their first
  very long epoch, so changing the latter at the chosen global optimizer step
  takes effect for the following batches without restarting the dataloader.
  """

  def __init__(
      self,
      enabled: bool = False,
      start_fraction: float = 0.72,
      every_n_steps: int = 100,
  ) -> None:
    super().__init__()
    if not 0.0 <= float(start_fraction) <= 1.0:
      raise ValueError("start_fraction must be in [0, 1]")
    if int(every_n_steps) <= 0:
      raise ValueError("every_n_steps must be positive")
    self.enabled = bool(enabled)
    self.start_fraction = float(start_fraction)
    self.every_n_steps = int(every_n_steps)
    self._switched = False

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if not self.enabled or self._switched:
      return
    max_steps = int(getattr(trainer, "max_steps", 0) or 0)
    if max_steps <= 0:
      raise RuntimeError(
          "LateValidationFrequencyCallback requires a finite trainer.max_steps")
    start_step = int(max_steps * self.start_fraction)
    if int(trainer.global_step) < start_step:
      return

    # Integer validation intervals are measured in training batches
    # (micro-batches); convert from optimizer steps.
    accumulation = max(1, int(getattr(trainer, "accumulate_grad_batches", 1) or 1))
    trainer.val_check_interval = self.every_n_steps * accumulation
    trainer.val_check_batch = self.every_n_steps * accumulation
    self._switched = True

    if getattr(trainer, "is_global_zero", False):
      LOGGER.info(
          "[LATE_VALIDATION] switched at global_step=%d: validating every %d "
          "optimizer steps for the remaining %.1f%% of training",
          int(trainer.global_step), self.every_n_steps,
          100.0 * (1.0 - self.start_fraction))
      pl_module.log(
          "stats/late_validation_interval_steps", float(self.every_n_steps),
          on_step=True, on_epoch=False, sync_dist=False, prog_bar=False)

