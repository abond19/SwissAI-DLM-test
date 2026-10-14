"""AdamW with fully decoupled weight decay and muP / CompleteP hyperparameter scaling.

Port of the bst-scaling AdamW setup (``model.configure_optimizers`` + the ``decoupled_wd``
patch in ``train.py``).  Two things differ from ``torch.optim.AdamW`` as used by
``optim=adamw``:

1. **Decoupled weight decay.**  Loshchilov & Hutter (Alg. 2) shrink the weights by
   ``eta_t * lambda`` where ``eta_t`` is the schedule multiplier in [0, 1] and the peak LR
   does not appear.  ``torch.optim.AdamW`` shrinks by ``lr_t * wd`` with the full learning
   rate, so ``wd`` is entangled with the LR and, under muP, with each group's LR scale.
   Here every group shrinks by ``sched(t) * weight_decay`` -- independent of the peak LR,
   of the width and of the group's muP scale -- so ``weight_decay`` can be held fixed while
   the LR is swept or the model is scaled.  ``weight_decay`` is therefore a per-step
   fraction (the reference's ``lambda_decoupled == peak_lr * conventional_wd``), e.g. 1e-4.
2. **Per-role LR and epsilon scaling** from :mod:`discrete_diffusion.models.scaling`.

The Adam update itself is delegated to ``torch.optim.AdamW`` (fused / foreach kernels,
checkpoint format, DDP / FSDP behaviour all unchanged); the decay is applied just before it,
which is exactly the order ``torch.optim.AdamW`` uses.

The schedule multiplier is ``group['lr'] / group['peak_lr']``: ``peak_lr`` is recorded when the
optimizer is built (global LR times the group's muP scale) and any LR scheduler
(Lambda / timm cosine / ...) rescales ``group['lr']`` from there.

Note: with bf16 master weights the shrink ``1 - sched * wd`` is rounded away for small
``wd``; use fp32 master weights (``model.torch_dtype=fp32``).
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch

from ..models.scaling import (
    ROLES, Parametrization, classify_parameter, get_parametrization,
    validate_model_config)


class DecoupledAdamW(torch.optim.AdamW):
  """``torch.optim.AdamW`` whose weight decay is ``sched(t) * group['lambda_decay']``.

  Param groups take the usual AdamW keys (``lr``, ``eps``, ``betas``) plus
  ``lambda_decay`` (default 0).  ``weight_decay`` is forced to 0 so the stock
  coupled decay never runs.
  """

  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
               lambda_decay=0.0, fused=None, foreach=None):
    super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0.0,
                     fused=fused, foreach=foreach)
    # fused=True sets this, which makes Lightning refuse gradient clipping.
    self._step_supports_amp_scaling = False
    for group in self.param_groups:
      group.setdefault('lambda_decay', float(lambda_decay))
      group.setdefault('peak_lr', float(group['lr']))
      wd = group['lambda_decay']
      if not 0.0 <= wd < 1.0:
        raise ValueError(
            f"lambda_decay is a per-step shrink fraction in [0, 1), got {wd}")

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    for group in self.param_groups:
      wd = group['lambda_decay']
      if wd == 0.0:
        continue
      peak = group['peak_lr']
      sched = group['lr'] / peak if peak > 0 else 0.0
      params = [p for p in group['params'] if p.grad is not None]
      if params and sched > 0.0:
        torch._foreach_mul_(params, 1.0 - sched * wd)
    super().step()
    return loss


def build_adamw_param_groups(
    named_params: Iterable[Tuple[str, torch.nn.Parameter]],
    parametrization: Parametrization,
    *, lr: float, eps: float, weight_decay: float):
  """One param group per scaling role, with muP-scaled ``lr`` / ``eps``.

  Every group gets the same ``lambda_decay`` (all tensors are decayed, as in the
  reference).  Shared tensors are grouped once.
  """
  by_role = {role: [] for role in ROLES}
  seen = set()
  for name, param in named_params:
    if not param.requires_grad or id(param) in seen:
      continue
    seen.add(id(param))
    by_role[classify_parameter(name)].append(param)

  groups = []
  for role in ROLES:
    if not by_role[role]:
      continue
    lr_scale = parametrization.lr_scale(role)
    groups.append({
        'params': by_role[role],
        'name': role,
        'lr': lr * lr_scale,
        'lr_scale': lr_scale,
        'eps': eps * parametrization.eps_scale(role),
        'lambda_decay': weight_decay,
    })
  if not groups:
    raise ValueError('No trainable parameters found for AdamW parameter groups.')
  return groups


def build_adamw_mup(config, named_params) -> DecoupledAdamW:
  """Build the optimizer from a Hydra config (``optim=adamw_mup``).

  The parametrization (``standard | mup | completep``) and the reference ``base_width`` /
  ``base_depth`` the LR was tuned on are read from ``config.model``, next to the init that
  uses them; sizes come from ``model.hidden_size`` / ``model.n_blocks``.
  """
  optim_cfg = config.optim
  model_cfg = config.model
  name = getattr(model_cfg, 'parametrization', None)
  if name is None:
    raise ValueError(
        'optim=adamw_mup needs model.parametrization (standard | mup | completep), e.g. '
        'model=gidd_hf_std; the model init and the optimizer scales must come from the '
        'same parametrization.')
  parametrization = get_parametrization(
      name, hidden_size=model_cfg.hidden_size, n_blocks=model_cfg.n_blocks,
      base_width=getattr(model_cfg, 'base_width', 256),
      base_depth=getattr(model_cfg, 'base_depth', 12))
  validate_model_config(parametrization, model_cfg)

  groups = build_adamw_param_groups(
      named_params, parametrization, lr=float(optim_cfg.lr),
      eps=float(optim_cfg.eps), weight_decay=float(optim_cfg.weight_decay))
  fused = getattr(optim_cfg, 'fused', None)
  optimizer = DecoupledAdamW(
      groups, lr=float(optim_cfg.lr),
      betas=(float(optim_cfg.beta1), float(optim_cfg.beta2)),
      eps=float(optim_cfg.eps),
      fused=None if fused is None else bool(fused))
  optimizer.parametrization = parametrization
  return optimizer
