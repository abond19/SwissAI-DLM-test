"""EasyDel-style GIDD mixing schedule in PyTorch.

This mirrors the JAX EasyDel parameterization where:
  alpha = sigmoid(log_snr)
  q_t(z|x) = alpha * delta_x(z) + (1 - alpha) * pi_lambda(z; log_snr)
and
  pi_lambda(log_snr) = sigmoid(scale * log_snr + shift) * Uniform_non_mask
                      + (1 - sigmoid(scale * log_snr + shift)) * Mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


def _select(config, key: str, default=None):
  """Optional config lookup that never raises.

  ``getattr(dictconfig, missing_key, default)`` (and ``.get`` / ``in``) make
  OmegaConf raise and catch a ConfigAttributeError internally.  The exception
  sits in a reference cycle with its traceback, whose frames keep the caller's
  locals (e.g. a training/validation step's logits) alive until Python's cyclic
  GC runs.  ``OmegaConf.select`` returns the default without raising.
  """
  if isinstance(config, DictConfig):
    return OmegaConf.select(config, key, default=default)
  node = config
  for part in key.split('.'):
    node = getattr(node, part, None)
    if node is None:
      return default
  return node


#for now it is tha same as gidd. It does not support per token sampling.
def sample_t(config, batch_size: int, eps: float | None = None, device=None):
  if eps is None:
    eps = _select(config, 'algo.t_eps', _select(config, 'model.t_eps', 1e-4))

  low_disc = bool(_select(config, 'training.low_discrepancy_sampling',
                          _select(config, 'algo.low_discrepancy_sampling', False)))
  if low_disc:
    t = torch.arange(batch_size, device=device, dtype=torch.float32) / max(batch_size, 1)
    t = (t + torch.rand(1, device=device, dtype=torch.float32)).fmod(1.0)
  else:
    t = torch.rand(batch_size, device=device, dtype=torch.float32)

  t = (1 - 2 * eps) * t + eps
  return t

  
class EasyDelHybridDiffusion(nn.Module):
  """Hybrid mixing schedule using EasyDel's log-SNR parameterization."""

  def __init__(
    self,
    tokenizer,
    min_log_snr: float = -10.0,
    max_log_snr: float = 10.0,
    hybrid_scale: float = 1.0,
    hybrid_shift: float = 0.0,
    prior_distribution: str = "masked",
  ):
    super().__init__()
    self.tokenizer = tokenizer
    self.mask_id = int(tokenizer.mask_token_id)
    self.vocab_size = int(len(tokenizer))

    self.min_log_snr = float(min_log_snr)
    self.max_log_snr = float(max_log_snr)
    self.hybrid_scale = float(hybrid_scale)
    self.hybrid_shift = float(hybrid_shift)
    self.prior_distribution = str(prior_distribution)

    mask = torch.zeros(self.vocab_size)
    mask[self.mask_id] = 1.0
    self.register_buffer('mask', mask, persistent=False)

    unif = (1.0 - mask) / max(self.vocab_size - 1, 1)
    self.register_buffer('unif', unif, persistent=False)

    # Keep compatibility with existing code paths that inspect these fields.
    pr = torch.full((self.vocab_size,), -1e3)
    pr[self.mask_id] = 0.0
    self.register_buffer('log_prior', pr - pr.logsumexp(-1, keepdim=True))

  @staticmethod
  def _safe_log(x: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(x.dtype).tiny
    return torch.log(x.clamp_min(tiny))

  @staticmethod
  def _safe_sigmoid(x: torch.Tensor, precision: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.sigmoid(x.to(dtype=precision)).to(dtype=x.dtype)

  def alpha_from_log_snr(self, log_snr: torch.Tensor) -> torch.Tensor:
    log_snr = log_snr.clamp(self.min_log_snr, self.max_log_snr)
    return self._safe_sigmoid(log_snr, precision=torch.float32)

  def log_snr_from_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
    log_snr = self._safe_log(alpha) - self._safe_log(1 - alpha)
    return log_snr.clamp(self.min_log_snr, self.max_log_snr)

  def log_snr_from_time(self, t: torch.Tensor) -> torch.Tensor:
    alpha = 1 - t
    return self.log_snr_from_alpha(alpha)

  def pi_lambda(self, log_snr: torch.Tensor) -> torch.Tensor:
    alpha = self._safe_sigmoid(self.hybrid_scale * log_snr + self.hybrid_shift)
    alpha_dtype = alpha.to(dtype=self.unif.dtype)
    pi_at_logsnr = alpha_dtype[..., None] * self.unif
    pi_at_logsnr[..., self.mask_id].add_((1 - alpha).to(dtype=self.unif.dtype))
    return pi_at_logsnr

  def pi_lambda_at_ids(self, log_snr: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    alpha = self._safe_sigmoid(
      self.hybrid_scale * log_snr.to(dtype=torch.float32) + self.hybrid_shift,
      precision=torch.float32,
    )
    is_mask = (input_ids == self.mask_id).to(alpha.dtype)
    pi_vals = alpha * (1 - is_mask) / max(self.vocab_size - 1, 1) + (1 - alpha) * is_mask
    return pi_vals.to(dtype=log_snr.dtype)

  def pi_lambda_prime_at_ids(self, log_snr: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    alpha = self._safe_sigmoid(
      self.hybrid_scale * log_snr.to(dtype=torch.float32) + self.hybrid_shift,
      precision=torch.float32,
    )
    alpha_prime = self.hybrid_scale * alpha * (1 - alpha)
    diff = (self.unif - self.mask).to(dtype=alpha.dtype)
    pi_prime_at_z = alpha_prime * diff[input_ids]
    return pi_prime_at_z.to(dtype=log_snr.dtype)

  def pi_lambda_prime(self, log_snr: torch.Tensor) -> torch.Tensor:
    alpha = self._safe_sigmoid(
      self.hybrid_scale * log_snr.to(dtype=torch.float32) + self.hybrid_shift,
      precision=torch.float32,
    )[..., None]
    alpha_prime = self.hybrid_scale * alpha * (1 - alpha)
    pi_prime = alpha_prime.to(dtype=log_snr.dtype) * (self.unif - self.mask).to(dtype=log_snr.dtype)
    return pi_prime

  def p_log_snr(self, log_snr: torch.Tensor) -> torch.Tensor:
    sigm = self._safe_sigmoid(log_snr, precision=torch.float32)
    return sigm * (1 - sigm)

  @staticmethod
  def _expand_to_token_shape(x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    if x.shape == token_ids.shape:
      return x
    if x.ndim == 1 and token_ids.ndim == 2 and x.shape[0] == token_ids.shape[0]:
      return x[:, None].expand_as(token_ids)
    if x.ndim == 2 and token_ids.ndim == 2 and x.shape[0] == token_ids.shape[0] and x.shape[1] == 1:
      return x.expand_as(token_ids)
    raise ValueError(
      f"Time/schedule shape {tuple(x.shape)} is incompatible with token shape {tuple(token_ids.shape)}")

  def _sample_uniform_non_mask_ids(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    if self.vocab_size <= 1:
      return torch.full(shape, self.mask_id, device=device, dtype=torch.long)
    ids = torch.randint(0, self.vocab_size - 1, shape, device=device)
    return ids + (ids >= self.mask_id).to(ids.dtype)

  def get_alpha_betapi(self, t: torch.Tensor):
    log_snr = self.log_snr_from_time(t)
    alpha = self.alpha_from_log_snr(log_snr)
    pi = self.pi_lambda(log_snr).to(dtype=alpha.dtype)
    beta_pi = (1 - alpha)[..., None] * pi
    if alpha.ndim == 1:
      alpha = alpha[:, None]
    return alpha, beta_pi

  def probs_at_t(self, prs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    orig_dtype = prs.dtype
    alpha_t, beta_pi = self.get_alpha_betapi(t)
    probs = prs.mul(alpha_t.unsqueeze(-1))
    if beta_pi.ndim == 2:
      probs[..., :beta_pi.shape[-1]].add_(beta_pi.unsqueeze(1))
    else:
      probs[..., :beta_pi.shape[-1]].add_(beta_pi)
    return probs.to(orig_dtype)

  def sample_zt(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    log_snr = self.log_snr_from_time(t)
    log_snr = self._expand_to_token_shape(log_snr, input_ids)

    alpha = self.alpha_from_log_snr(log_snr)
    pi_u_at_logsnr = self._safe_sigmoid(
      self.hybrid_scale * log_snr + self.hybrid_shift, precision=torch.float32)

    is_noise_free = torch.bernoulli(alpha).to(torch.bool)
    is_uniform = torch.bernoulli(pi_u_at_logsnr).to(torch.bool)

    uniform_ids = self._sample_uniform_non_mask_ids(tuple(input_ids.shape), input_ids.device)
    mask_ids = torch.full_like(input_ids, self.mask_id)
    noise_ids = torch.where(is_uniform, uniform_ids, mask_ids)
    return torch.where(is_noise_free, input_ids, noise_ids)

  @torch.no_grad()
  def sample_prior(self, shape, *, device: torch.device | None = None) -> torch.Tensor:
    if device is None:
      device = self.log_prior.device
    shape = tuple(shape)
    if self.prior_distribution == "masked":
      return torch.full(shape, self.mask_id, device=device, dtype=torch.long)
    if self.prior_distribution == "uniform":
      return torch.randint(0, self.vocab_size, shape, device=device)
    raise ValueError(
      f"Unknown prior_distribution={self.prior_distribution!r}. Expected 'masked' or 'uniform'.")


__all__ = ['EasyDelHybridDiffusion', 'sample_t']
