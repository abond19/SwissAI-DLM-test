"""Validation with the same diffusion noise at every evaluation."""

import lightning as L
import torch


class FixedNoiseValidationCallback(L.Callback):
  """Reseed the RNG for each validation pass and restore it afterwards.

  Validation draws diffusion times and masks from the global RNG, so by
  default every evaluation uses different noise and the logged ``val/*``
  values jump between evaluations by more than the differences being compared.
  With this callback each pass starts from ``seed + global_rank``: runs with the
  same validation topology (devices, eval batch size) see identical noise, and
  successive evaluations of one run differ only through the model.  The
  training RNG state is restored afterwards, so training is unaffected by when
  (or whether) validation runs.
  """

  def __init__(self, seed: int = 1234, enabled: bool = True) -> None:
    super().__init__()
    self.seed = int(seed)
    self.enabled = bool(enabled)
    self._cpu_state = None
    self._cuda_states = None

  def on_validation_epoch_start(self, trainer, pl_module):
    if not self.enabled:
      return
    self._cpu_state = torch.get_rng_state()
    self._cuda_states = (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None)
    torch.manual_seed(self.seed + int(trainer.global_rank))

  def on_validation_epoch_end(self, trainer, pl_module):
    if not self.enabled or self._cpu_state is None:
      return
    torch.set_rng_state(self._cpu_state)
    if self._cuda_states is not None:
      torch.cuda.set_rng_state_all(self._cuda_states)
    self._cpu_state = None
    self._cuda_states = None
