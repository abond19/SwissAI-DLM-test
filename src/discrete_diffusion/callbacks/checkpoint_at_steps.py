"""Full checkpoints at a fixed list of optimizer steps."""

import os

import lightning as L


class CheckpointAtStepsCallback(L.Callback):
  """Save ``<dirpath>/step<N>.ckpt`` right after optimizer step N, for each N in ``steps``.

  Used to branch several runs (e.g. different cooldowns) from one shared prefix:
  each branch resumes from the checkpoint at its switch step.  All ranks call
  ``save_checkpoint``; Lightning writes from rank 0.
  """

  def __init__(self, dirpath: str, steps) -> None:
    super().__init__()
    self.dirpath = str(dirpath)
    self.steps = sorted({int(s) for s in steps})
    self._done = set()

  def on_train_start(self, trainer, pl_module):
    # After a resume, steps already passed were saved by the earlier job.  Decided
    # from the restored step so every rank agrees (save_checkpoint has a barrier).
    start = int(trainer.global_step)
    self._done = {s for s in self.steps if s <= start}

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    step = int(trainer.global_step)
    if step not in self.steps or step in self._done:
      return
    self._done.add(step)
    trainer.save_checkpoint(os.path.join(self.dirpath, f'step{step}.ckpt'))
