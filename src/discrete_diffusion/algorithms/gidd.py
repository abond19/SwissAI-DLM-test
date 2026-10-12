"""GIDD algorithm implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

from . import base as trainer_base
from ..noise_schedules import LogLinear
from ..noise_schedules import HybridDiffusion, sample_t as sample_t_hybrid
from ..noise_schedules.gidd_constant_pi import GiddLinearNoise, sample_t as sample_t_constant_pi
from ..noise_schedules.gidd_easydel import (
  EasyDelHybridDiffusion,
  sample_t as sample_t_easydel,
)


class GiddLoss(nn.Module):
  def __init__(self, config, tokenizer, noise_schedule):
    super().__init__()
    self.config = config
    self.tokenizer = tokenizer
    self.noise_schedule = noise_schedule
    self.vocab_size = len(tokenizer)

    try:
      self.loss_weighting = config.loss.loss_weighting
      self.min_loss_weight = config.loss.min_loss_weight
      self.max_loss_weight = config.loss.max_loss_weight
    except Exception:
      self.loss_weighting = getattr(config.algo, 'loss_weighting', 'dynamic')
      self.min_loss_weight = float(getattr(config.algo, 'min_loss_weight', 0.0))
      self.max_loss_weight = float(getattr(config.algo, 'max_loss_weight', 2.0))
    assert self.max_loss_weight > 0, "max_loss_weight must be positive"

    self.mask_id = tokenizer.mask_token_id

  def get_weights(self, t: torch.Tensor, z_t: torch.Tensor, input_ids: torch.Tensor):
    orig_dtype = t.dtype
    t = t.unsqueeze(-1).to(torch.float64)
    t1m = (1 - t)

    gamma = self.noise_schedule.log_gamma.exp()
    t_gamma = t.pow(gamma)
    t1m_gamma = t1m.pow(gamma)
    B = self.noise_schedule.log_B.exp()

    c_t = t_gamma.sqrt() * t1m_gamma.sqrt() * B
    c_t_prime = (gamma / 2) * (1 - 2 * t) / (t * t1m) * c_t

    is_mask = (z_t == self.mask_id).to(t.dtype)
    is_x = (z_t == input_ids).to(t.dtype)

    alpha_ratio = -1 / (1 - t) - c_t_prime / (1 + c_t)
    N = self.vocab_size - 1
    weight_on_x = (c_t + (1 - t) * c_t_prime) / N / ((1 - t) * (1 - t + c_t / N))
    weight_on_u = (c_t + (1 - t) * c_t_prime) / ((1 - t) * c_t)
    weight_on_m = 1 / ((1 - t) * t)

    elbo_weights = is_x * weight_on_x + is_mask * weight_on_m + (1 - is_x - is_mask) * weight_on_u

    loss_weights = elbo_weights.clone()
    if self.loss_weighting == "clip":
      loss_weights = loss_weights.clip(self.min_loss_weight, self.max_loss_weight)
    elif self.loss_weighting == "dynamic":
      log_snr_like = torch.sigmoid(-t).clip(-20, 20)
      x_scale = B / self.vocab_size * torch.exp(gamma / 2 * log_snr_like)
      loss_weights = (1 - is_x) * ((1 - is_mask) + 2 * is_mask) + is_x * x_scale
      loss_weights = loss_weights.clip(self.min_loss_weight, self.max_loss_weight)
    elif self.loss_weighting == "raw":
      loss_weights = elbo_weights

    return (alpha_ratio.to(orig_dtype),
            elbo_weights.to(orig_dtype),
            loss_weights.to(orig_dtype))

  def forward(
    self,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    reduction: str = "tokenmean",
  ):
    dtype = logits.dtype
    _, elbo_weights, ws = self.get_weights(t, z_t, input_ids)

    logits[..., self.mask_id] = torch.finfo(dtype).min

    x = F.one_hot(input_ids, logits.shape[-1]).to(dtype)
    x_hat = logits.softmax(-1).to(dtype)
    log_q_t = self.noise_schedule.probs_at_t(x, t).log_().clip_(min=-1e6)
    log_p_t = self.noise_schedule.probs_at_t(x_hat, t).log_().clip_(min=-1e6)

    kl_loss = F.kl_div(log_p_t, log_q_t, reduction="none", log_target=True).sum(-1)

    log_q_zt = log_q_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
    log_p_zt = log_p_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
    log_ratio = log_q_zt - log_p_zt

    is_loss = log_ratio.exp() - log_ratio - 1
    elbo = elbo_weights * (kl_loss + is_loss)
    loss = ws * (kl_loss + is_loss)

    eps = torch.finfo(loss.dtype).eps
    denom_ws = (ws * attention_mask).sum().clamp_min(eps)
    metrics = {
      "kl_loss": (ws * kl_loss.detach() * attention_mask).sum() / denom_ws,
      "is_loss": (ws * is_loss.detach() * attention_mask).sum() / denom_ws,
      "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum().clamp_min(eps),
    }

    if reduction == "tokenmean":
      num_tokens = attention_mask.numel()
      loss = loss.sum() / num_tokens

    return loss, elbo, metrics

class GiddLossConstantPi(nn.Module):
  def __init__(self, config, tokenizer, noise_schedule):
    super().__init__()
    self.config = config
    self.tokenizer = tokenizer
    self.noise_schedule = noise_schedule
    self.vocab_size = len(tokenizer)

    try:
      self.loss_weighting = config.loss.loss_weighting
      self.min_loss_weight = config.loss.min_loss_weight
      self.max_loss_weight = config.loss.max_loss_weight
    except Exception:
      self.loss_weighting = getattr(config.algo, 'loss_weighting', 'dynamic')
      self.min_loss_weight = float(getattr(config.algo, 'min_loss_weight', 0.0))
      self.max_loss_weight = float(getattr(config.algo, 'max_loss_weight', 2.0))
    assert self.max_loss_weight > 0, "max_loss_weight must be positive"

    self.mask_id = tokenizer.mask_token_id
    self.supports_mask = bool((self.noise_schedule.pi[self.mask_id] > 0).item())

  def _drop_mask_dim(self, x: torch.Tensor) -> torch.Tensor:
    mid = self.mask_id
    return torch.cat([x[..., :mid], x[..., mid + 1:]], dim=-1)

  def _remap_no_mask(self, idx: torch.Tensor) -> torch.Tensor:
    mid = self.mask_id
    return idx - (idx > mid).to(idx.dtype)

  def get_weights(self, t: torch.Tensor, z_t: torch.Tensor, input_ids: torch.Tensor):
    orig_dtype = t.dtype
    t = t.unsqueeze(-1).to(torch.float64)

    alpha_ratio = 1 / (1-t)

    is_x = (z_t == input_ids).to(t.dtype)

    pi_values = self.noise_schedule.gather_pi(input_ids)

    weight_on_not_x = 1 / t
    weight_on_x = pi_values / ( 1 - t + t * pi_values)

    elbo_weights = (is_x * weight_on_x + (1 - is_x) * weight_on_not_x) * alpha_ratio

    loss_weights = elbo_weights.clone()
    if self.loss_weighting == "clip":
      loss_weights = loss_weights.clip(self.min_loss_weight, self.max_loss_weight)
    elif self.loss_weighting == "dynamic":
      log_snr_like = torch.sigmoid(-t).clip(-20, 20)
      x_scale = pi_values * torch.exp(log_snr_like)
      loss_weights = (1 - is_x) * ((1 - is_x) + 2 * is_x) + is_x * x_scale
      loss_weights = loss_weights.clip(self.min_loss_weight, self.max_loss_weight)
    elif self.loss_weighting == "raw":
      loss_weights = elbo_weights

    return (alpha_ratio.to(orig_dtype),
            elbo_weights.to(orig_dtype),
            loss_weights.to(orig_dtype))

  def forward(
    self,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    reduction: str = "tokenmean",
  ):
    dtype = logits.dtype
    _, elbo_weights, ws = self.get_weights(t, z_t, input_ids)

    logits[..., self.mask_id] = torch.finfo(dtype).min

    if self.supports_mask:
      x = F.one_hot(input_ids, logits.shape[-1]).to(dtype)
      x_hat = logits.softmax(-1).to(dtype)
      log_q_t = self.noise_schedule.probs_at_t(x, t).log_().clip_(min=-1e6)
      log_p_t = self.noise_schedule.probs_at_t(x_hat, t).log_().clip_(min=-1e6)

      kl_loss = F.kl_div(log_p_t, log_q_t, reduction="none", log_target=True).sum(-1)

      log_q_zt = log_q_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
      log_p_zt = log_p_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
    else:
      logits_nm = self._drop_mask_dim(logits)
      x_hat = logits_nm.softmax(-1).to(dtype)
      x = F.one_hot(input_ids, logits.shape[-1]).to(dtype)
      x_nm = self._drop_mask_dim(x)

      alpha_t, beta_pi = self.noise_schedule.get_alpha_betapi(t)
      beta_pi_nm = self._drop_mask_dim(beta_pi)

      probs_q = x_nm.mul(alpha_t.unsqueeze(-1))
      probs_q[..., :beta_pi_nm.shape[-1]].add_(beta_pi_nm.unsqueeze(1))
      probs_p = x_hat.mul(alpha_t.unsqueeze(-1))
      probs_p[..., :beta_pi_nm.shape[-1]].add_(beta_pi_nm.unsqueeze(1))
      log_q_t = probs_q.log_().clip_(min=-1e6)
      log_p_t = probs_p.log_().clip_(min=-1e6)

      kl_loss = F.kl_div(log_p_t, log_q_t, reduction="none", log_target=True).sum(-1)

      z_t_nm = self._remap_no_mask(z_t)
      log_q_zt = log_q_t.gather(-1, z_t_nm.unsqueeze(-1)).squeeze(-1)
      log_p_zt = log_p_t.gather(-1, z_t_nm.unsqueeze(-1)).squeeze(-1)
    log_ratio = log_q_zt - log_p_zt

    is_loss = torch.expm1(log_ratio) - log_ratio
    elbo = elbo_weights * (kl_loss + is_loss)
    loss = ws * (kl_loss + is_loss)

    eps = torch.finfo(loss.dtype).eps
    denom_ws = (ws * attention_mask).sum().clamp_min(eps)
    metrics = {
      "kl_loss": (ws * kl_loss.detach() * attention_mask).sum() / denom_ws,
      "is_loss": (ws * is_loss.detach() * attention_mask).sum() / denom_ws,
      "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum().clamp_min(eps),
    }

    if reduction == "tokenmean":
      num_tokens = attention_mask.numel()
      loss = loss.sum() / num_tokens

    return loss, elbo, metrics


class GiddLossEasyDel(nn.Module):
  """PyTorch port of the EasyDel GIDD loss parameterization."""

  def __init__(self, config, tokenizer, noise_schedule: EasyDelHybridDiffusion):
    super().__init__()
    self.config = config
    self.tokenizer = tokenizer
    self.noise_schedule = noise_schedule
    self.vocab_size = len(tokenizer)
    self.mask_id = tokenizer.mask_token_id
    self.beta_is_div = float(getattr(config.algo, 'beta_is_div', 1.0))

  # @staticmethod
  # def _safe_log(x: torch.Tensor) -> torch.Tensor:
  #   tiny = torch.finfo(x.dtype).tiny
  #   neg_inf = torch.log(torch.tensor(tiny, dtype=x.dtype, device=x.device))
  #   return torch.where(x > 0, torch.log(x.clamp_min(tiny)), neg_inf)

  def _marginal_probs(self, log_snr: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    alpha = self.noise_schedule.alpha_from_log_snr(log_snr)
    pi = self.noise_schedule.pi_lambda(log_snr).to(dtype=probs.dtype)
    return alpha[..., None] * probs + (1 - alpha)[..., None] * pi

  def _marginal_log_probs(self, log_snr: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    return self.noise_schedule._safe_log(self._marginal_probs(log_snr, probs))

  def _get_loss_weights(self, log_snr: torch.Tensor, z_t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    pi_at_z = self.noise_schedule.pi_lambda_at_ids(log_snr, z_t)
    pi_prime_at_z = self.noise_schedule.pi_lambda_prime_at_ids(log_snr, z_t)
    snr = torch.exp(log_snr)
    delta = (z_t == x0).to(dtype=log_snr.dtype)
    return (pi_at_z - pi_prime_at_z) / (pi_at_z + snr * delta).clamp_min(1e-8)

  def _get_elbo_weights(self, log_snr: torch.Tensor, z_t: torch.Tensor, x0: torch.Tensor):
    loss_weights = self._get_loss_weights(log_snr, z_t, x0)
    p_log_snr = self.noise_schedule.p_log_snr(log_snr)
    elbo_weights = loss_weights / p_log_snr.clamp_min(1e-12)
    return elbo_weights, loss_weights

  # @staticmethod
  # def _expand_log_snr_to_tokens(log_snr: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
  #   """Ensure log_snr is token-wise [B, L] for EasyDel-style formulas."""
  #   if log_snr.shape == token_ids.shape:
  #     return log_snr
  #   if log_snr.ndim == 1 and token_ids.ndim == 2 and log_snr.shape[0] == token_ids.shape[0]:
  #     return log_snr[:, None].expand_as(token_ids)
  #   if log_snr.ndim == 2 and token_ids.ndim == 2 and log_snr.shape[0] == token_ids.shape[0] and log_snr.shape[1] == 1:
  #     return log_snr.expand_as(token_ids)
  #   raise ValueError(
  #     f"log_snr shape {tuple(log_snr.shape)} is incompatible with token shape {tuple(token_ids.shape)}")

  def forward(
    self,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    reduction: str = "tokenmean",
  ):
    dtype = logits.dtype
    logits = logits.clone()
    logits[..., self.mask_id] = torch.finfo(dtype).min

    log_snr = self.noise_schedule.log_snr_from_time(t)
    log_snr = self.noise_schedule._expand_to_token_shape(log_snr, input_ids)
    elbo_weights, ws = self._get_elbo_weights(log_snr, z_t, input_ids)
    elbo_weights = elbo_weights.clamp(0, 1e6)
    ws = ws.clamp(0, 1e3)

    x = F.one_hot(input_ids, logits.shape[-1]).to(dtype)
    x_hat = logits.softmax(-1).to(dtype)
    log_q_t = self._marginal_log_probs(log_snr, x).clamp(min=-1e6)
    log_p_t = self._marginal_log_probs(log_snr, x_hat).clamp(min=-1e6)

    kl_loss = (torch.exp(log_q_t) * (log_q_t - log_p_t)).sum(-1)

    log_q_zt = log_q_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
    log_p_zt = log_p_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
    log_ratio = log_q_zt - log_p_zt
    ratio = torch.exp(log_q_zt) / (torch.exp(log_p_zt) + 1e-12)
    is_loss = ratio - log_ratio - 1

    elbo = elbo_weights * (kl_loss + is_loss)
    loss = ws * kl_loss + self.beta_is_div * ws * is_loss

    eps = torch.finfo(loss.dtype).eps
    denom_ws = (ws * attention_mask).sum().clamp_min(eps)
    metrics = {
      "kl_loss": (ws * kl_loss.detach() * attention_mask).sum() / denom_ws,
      "is_loss": (ws * is_loss.detach() * attention_mask).sum() / denom_ws,
      "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum().clamp_min(eps),
    }

    if reduction == "tokenmean":
      loss = loss.sum() / attention_mask.numel()

    return loss, elbo, metrics


class GiddLossEasyDelLowMem(GiddLossEasyDel):
  """EasyDel GIDD loss without materializing full-vocab q/log-q/log-p tensors."""

  def __init__(self, config, tokenizer, noise_schedule: EasyDelHybridDiffusion):
    super().__init__(config, tokenizer, noise_schedule)
    self.chunk_size = int(getattr(config.algo, 'easydel_lowmem_chunk_size', 8192))
    if self.chunk_size <= 0:
      raise ValueError('algo.easydel_lowmem_chunk_size must be positive')
    self.checkpoint_chunks = bool(
      getattr(config.algo, 'easydel_lowmem_checkpoint_chunks', True))

    scale = float(self.noise_schedule.hybrid_scale)
    shift = float(self.noise_schedule.hybrid_shift)
    min_log_snr = float(self.noise_schedule.min_log_snr)
    max_log_snr = float(self.noise_schedule.max_log_snr)
    max_mixing_logit = max(scale * min_log_snr + shift,
                           scale * max_log_snr + shift)
    self.mask_prior_only = max_mixing_logit < -80.0

  @staticmethod
  def _xlogx(x: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(x.dtype).tiny
    return torch.where(x > 0, x * torch.log(x.clamp_min(tiny)), torch.zeros_like(x))

  def _logsumexp_without_mask(self, logits: torch.Tensor) -> torch.Tensor:
    mid = int(self.mask_id)
    if logits.shape[-1] != self.vocab_size:
      raise ValueError(
        f"logits vocab size {logits.shape[-1]} does not match tokenizer size {self.vocab_size}")
    if self.vocab_size <= 1:
      raise ValueError("GIDD requires vocab_size > 1")

    if mid == 0:
      return logits[..., 1:].logsumexp(dim=-1)
    if mid == self.vocab_size - 1:
      return logits[..., :-1].logsumexp(dim=-1)
    left = logits[..., :mid].logsumexp(dim=-1)
    right = logits[..., mid + 1:].logsumexp(dim=-1)
    return torch.logaddexp(left, right)

  def _log_model_probs_at_ids(
    self,
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    logsumexp_without_mask: torch.Tensor,
  ) -> torch.Tensor:
    gathered_logits = logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    log_probs = gathered_logits - logsumexp_without_mask
    neg_inf = torch.full_like(log_probs, float('-inf'))
    return torch.where(token_ids == self.mask_id, neg_inf, log_probs)

  def _sum_nonmask_log_model_mixture(
    self,
    logits: torch.Tensor,
    logsumexp_without_mask: torch.Tensor,
    log_alpha: torch.Tensor,
    log_beta_u: torch.Tensor,
  ) -> torch.Tensor:
    total = torch.zeros_like(logsumexp_without_mask)

    for start in range(0, self.vocab_size, self.chunk_size):
      end = min(start + self.chunk_size, self.vocab_size)
      chunk_logits = logits[..., start:end]
      mask_offset = int(self.mask_id) - start
      has_mask = 0 <= mask_offset < (end - start)

      def chunk_sum(
        chunk: torch.Tensor,
        lse: torch.Tensor,
        log_a: torch.Tensor,
        log_bu: torch.Tensor,
      ) -> torch.Tensor:
        log_model_probs = chunk - lse.unsqueeze(-1)
        log_p = torch.logaddexp(
          log_a.unsqueeze(-1) + log_model_probs,
          log_bu.unsqueeze(-1),
        )
        if has_mask:
          log_p = log_p.clone()
          log_p[..., mask_offset] = 0
        return log_p.sum(dim=-1)

      if self.checkpoint_chunks and torch.is_grad_enabled() and chunk_logits.requires_grad:
        total = total + checkpoint(
          chunk_sum,
          chunk_logits,
          logsumexp_without_mask,
          log_alpha,
          log_beta_u,
          use_reentrant=False,
        )
      else:
        total = total + chunk_sum(
          chunk_logits, logsumexp_without_mask, log_alpha, log_beta_u)

    return total

  def _finish_loss(
    self,
    kl_loss: torch.Tensor,
    is_loss: torch.Tensor,
    elbo_weights: torch.Tensor,
    ws: torch.Tensor,
    attention_mask: torch.Tensor,
    reduction: str,
  ):
    elbo = elbo_weights * (kl_loss + is_loss)
    loss = ws * kl_loss + self.beta_is_div * ws * is_loss

    eps = torch.finfo(loss.dtype).eps
    denom_ws = (ws * attention_mask).sum().clamp_min(eps)
    metrics = {
      "kl_loss": (ws * kl_loss.detach() * attention_mask).sum() / denom_ws,
      "is_loss": (ws * is_loss.detach() * attention_mask).sum() / denom_ws,
      "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum().clamp_min(eps),
    }

    if reduction == "tokenmean":
      loss = loss.sum() / attention_mask.numel()

    return loss, elbo, metrics

  def _forward_mask_prior(
    self,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    z_t: torch.Tensor,
    log_snr: torch.Tensor,
    elbo_weights: torch.Tensor,
    ws: torch.Tensor,
    reduction: str,
  ):
    alpha = self.noise_schedule.alpha_from_log_snr(log_snr)
    beta = 1 - alpha
    log_alpha = self.noise_schedule._safe_log(alpha)
    log_beta = self.noise_schedule._safe_log(beta)

    lse = self._logsumexp_without_mask(logits)
    log_model_x = self._log_model_probs_at_ids(logits, input_ids, lse)
    log_model_z = self._log_model_probs_at_ids(logits, z_t, lse)

    x_is_mask = input_ids == self.mask_id
    z_is_mask = z_t == self.mask_id
    z_is_x = z_t == input_ids

    q_x = torch.where(x_is_mask, torch.ones_like(alpha), alpha)
    log_q_x = torch.where(x_is_mask, torch.zeros_like(log_alpha), log_alpha)
    log_p_x = torch.where(x_is_mask, log_beta, log_alpha + log_model_x)
    kl_loss = q_x * (log_q_x - log_p_x)

    tiny_log = log_alpha.new_full(
      log_alpha.shape, torch.finfo(log_alpha.dtype).tiny).log()
    log_q_z = torch.where(
      z_is_x,
      log_q_x,
      torch.where(z_is_mask, log_beta, tiny_log),
    )
    log_p_z = torch.where(z_is_mask, log_beta, log_alpha + log_model_z)
    log_ratio = log_q_z - log_p_z
    ratio = torch.exp(log_q_z) / (torch.exp(log_p_z) + 1e-12)
    is_loss = ratio - log_ratio - 1

    return self._finish_loss(
      kl_loss, is_loss, elbo_weights, ws, attention_mask, reduction)

  def _forward_chunked(
    self,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    z_t: torch.Tensor,
    log_snr: torch.Tensor,
    elbo_weights: torch.Tensor,
    ws: torch.Tensor,
    reduction: str,
  ):
    alpha = self.noise_schedule.alpha_from_log_snr(log_snr)
    beta = 1 - alpha
    pi_uniform_mass = self.noise_schedule._safe_sigmoid(
      self.noise_schedule.hybrid_scale * log_snr.to(dtype=torch.float32)
      + self.noise_schedule.hybrid_shift,
      precision=torch.float32,
    ).to(dtype=alpha.dtype)
    pi_mask = 1 - pi_uniform_mass
    pi_nonmask = pi_uniform_mass / max(self.vocab_size - 1, 1)

    beta_pi_nonmask = beta * pi_nonmask
    beta_pi_mask = beta * pi_mask
    log_alpha = self.noise_schedule._safe_log(alpha)
    log_beta_pi_nonmask = self.noise_schedule._safe_log(beta_pi_nonmask)
    log_beta_pi_mask = self.noise_schedule._safe_log(beta_pi_mask)

    x_is_mask = input_ids == self.mask_id
    z_is_mask = z_t == self.mask_id
    pi_x = torch.where(x_is_mask, pi_mask, pi_nonmask)
    beta_pi_x = beta * pi_x
    q_x = alpha + beta_pi_x

    all_beta_pi_log_beta_pi = (
      (self.vocab_size - 1) * self._xlogx(beta_pi_nonmask)
      + self._xlogx(beta_pi_mask)
    )
    qlogq = (
      self._xlogx(q_x)
      + all_beta_pi_log_beta_pi
      - self._xlogx(beta_pi_x)
    )

    lse = self._logsumexp_without_mask(logits)
    log_model_x = self._log_model_probs_at_ids(logits, input_ids, lse)
    log_model_z = self._log_model_probs_at_ids(logits, z_t, lse)

    nonmask_logp_sum = self._sum_nonmask_log_model_mixture(
      logits, lse, log_alpha, log_beta_pi_nonmask)
    pi_logp_sum = pi_nonmask * nonmask_logp_sum + pi_mask * log_beta_pi_mask

    log_p_x_nonmask = torch.logaddexp(log_alpha + log_model_x, log_beta_pi_nonmask)
    log_p_x = torch.where(x_is_mask, log_beta_pi_mask, log_p_x_nonmask)
    qlogp = alpha * log_p_x + beta * pi_logp_sum
    kl_loss = qlogq - qlogp

    pi_z = torch.where(z_is_mask, pi_mask, pi_nonmask)
    q_z = torch.where(z_t == input_ids, alpha, torch.zeros_like(alpha)) + beta * pi_z
    log_q_z = self.noise_schedule._safe_log(q_z)
    log_p_z_nonmask = torch.logaddexp(log_alpha + log_model_z, log_beta_pi_nonmask)
    log_p_z = torch.where(z_is_mask, log_beta_pi_mask, log_p_z_nonmask)
    log_ratio = log_q_z - log_p_z
    ratio = torch.exp(log_q_z) / (torch.exp(log_p_z) + 1e-12)
    is_loss = ratio - log_ratio - 1

    return self._finish_loss(
      kl_loss, is_loss, elbo_weights, ws, attention_mask, reduction)

  def forward(
    self,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    reduction: str = "tokenmean",
  ):
    log_snr = self.noise_schedule.log_snr_from_time(t)
    log_snr = self.noise_schedule._expand_to_token_shape(log_snr, input_ids)
    elbo_weights, ws = self._get_elbo_weights(log_snr, z_t, input_ids)
    elbo_weights = elbo_weights.clamp(0, 1e6)
    ws = ws.clamp(0, 1e3)

    if self.mask_prior_only:
      return self._forward_mask_prior(
        logits, input_ids, attention_mask, z_t, log_snr, elbo_weights, ws, reduction)

    return self._forward_chunked(
      logits, input_ids, attention_mask, z_t, log_snr, elbo_weights, ws, reduction)


class _MaskTokenizerAdapter:

  def __init__(self, base_tokenizer, vocab_size: int, mask_token_id: int):
    self._base = base_tokenizer
    self.mask_token_id = int(mask_token_id)
    self._len = int(vocab_size)

  def __len__(self):
    return self._len


class GIDD(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    self.mask_id, vocab_size = trainer_base.ensure_mask_token(tokenizer)

    super().__init__(config, tokenizer, vocab_size=vocab_size)
    self.save_hyperparameters()

    if not getattr(self.config.algo, 'time_conditioning', False):
      raise ValueError('GIDD requires algo.time_conditioning=True')

    self._mask_tok = _MaskTokenizerAdapter(
      base_tokenizer=tokenizer, vocab_size=vocab_size, mask_token_id=self.mask_id)

    p_uniform = float(getattr(self.config.algo, 'p_uniform', None))
    loss_type = str(getattr(self.config.algo, 'loss_type', None))
    if loss_type == 'gidd_constant_pi':
      self.hybrid_noise = GiddLinearNoise(tokenizer=self._mask_tok, p_uniform=p_uniform)
      self.loss_fn = GiddLossConstantPi(self.config, self._mask_tok, self.hybrid_noise)
      self._sample_t_fn = sample_t_constant_pi
    elif loss_type == 'gidd':
      gamma = 1.0
      self.hybrid_noise = HybridDiffusion(
        tokenizer=self._mask_tok,
        p_uniform=p_uniform,
        clip_noise=20,
        gamma=gamma,
      )
      self.loss_fn = GiddLoss(self.config, self._mask_tok, self.hybrid_noise)
      self._sample_t_fn = sample_t_hybrid
    elif loss_type in {'gidd_easydel', 'gidd_easydel_lowmem'}:
      self.hybrid_noise = EasyDelHybridDiffusion(
        tokenizer=self._mask_tok,
        min_log_snr=float(getattr(self.config.algo, 'min_log_snr', -10.0)),
        max_log_snr=float(getattr(self.config.algo, 'max_log_snr', 10.0)),
        hybrid_scale=float(getattr(self.config.algo, 'hybrid_mixing_scale', 1.0)),
        hybrid_shift=float(getattr(self.config.algo, 'hybrid_mixing_shift', 0.0)),
        prior_distribution=str(getattr(self.config.algo, 'prior_distribution', 'masked')),
      )
      if loss_type == 'gidd_easydel_lowmem':
        self.loss_fn = GiddLossEasyDelLowMem(self.config, self._mask_tok, self.hybrid_noise)
      else:
        self.loss_fn = GiddLossEasyDel(self.config, self._mask_tok, self.hybrid_noise)
      self._sample_t_fn = sample_t_easydel
    else:
      raise ValueError(
        f"Unknown GIDD loss_type={loss_type!r}. Expected one of "
        "{'gidd', 'gidd_constant_pi', 'gidd_easydel', 'gidd_easydel_lowmem'}.")

    self._loglinear = LogLinear()
    model_name = str(getattr(self.config.model, 'name', '')).lower()
    model_type = str(getattr(self.config.model, 'type', '')).lower()
    model_target = str(getattr(self.config.model, '_target_', '')).lower()
    self._sigma_free_backbone = (
      'gidd_hf' in model_name
      or 'gidd_hf' in model_type
      or 'gidd_hf_wrapper' in model_target
    )
    # Ephemeral shift shared by all microbatches of one accumulated global
    # batch.  It is created at accumulation step zero and cleared after the
    # final slice; it is intentionally not checkpointed.
    self._global_time_lattice_shift = None

  def compile_for_training(self):
    """Compile the per-step hot paths in place with ``torch.compile``.

    ``nll`` calls ``self.backbone`` directly rather than ``self.forward``, so
    compiling the LightningModule leaves the model eager.  Instead compile each
    transformer layer (regional compilation: short compile times) and the loss.
    ``nn.Module.compile`` works in place, so parameter names and checkpoints
    are unchanged.
    """
    layers = getattr(getattr(getattr(self.backbone, 'model', None), 'model', None),
                     'layers', None)
    if layers is not None:
      for layer in layers:
        layer.compile()
    else:
      self.backbone.compile()
    self.loss_fn.compile()

  def _process_model_input(self, x0, valid_tokens):
    return x0, valid_tokens

  def _process_sigma(self, sigma):
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def _sample_t(self, batch_size: int, current_accumulation_step=None,
                train_mode: bool = False):
    """Sample diffusion times for one local micro-batch.

    ``local`` (the historical/default mode) constructs a low-discrepancy grid
    independently on every rank.  With identical rank RNG streams, that means
    every rank uses the same local grid.  ``global_batch`` instead constructs
    one randomized low-discrepancy grid over the *logical global batch* and
    assigns this rank/micro-batch its contiguous slice.  The random shift is
    broadcast from rank zero, so this remains correct even if ranks use
    different RNG streams for other diffusion noise.

    This mode intentionally changes only time sampling.  Token corruption is
    still sampled locally by ``EasyDelHybridDiffusion.sample_zt``; keeping that
    separate makes it possible to measure the effect of global time coverage
    without changing the corruption distribution at the same time.
    """
    eps = float(getattr(self.config.algo, 't_eps', 1e-4))
    scope = str(getattr(
      self.config.algo, 'time_sampling_scope', 'local')).lower()
    if scope not in {'local', 'global_batch'}:
      raise ValueError(
        "algo.time_sampling_scope must be 'local' or 'global_batch', got "
        f"{scope!r}.")

    # Validation/evaluation batches are not part of an optimizer global batch;
    # retain the usual local sampling for those calls.
    if scope == 'local' or not train_mode or current_accumulation_step is None:
      return self._sample_t_fn(
        self.config, batch_size, eps=eps, device=self.device)

    # OmegaConf.select: a getattr default on a missing DictConfig key raises
    # internally and leaks this step's frames (and logits) into a GC cycle.
    low_disc = bool(OmegaConf.select(
      self.config, 'training.low_discrepancy_sampling',
      default=OmegaConf.select(
        self.config, 'algo.low_discrepancy_sampling', default=False)))
    if not low_disc:
      raise ValueError(
        "algo.time_sampling_scope='global_batch' currently requires "
        "low-discrepancy time sampling. Set "
        "algo.low_discrepancy_sampling=true.")

    accumulation_steps = max(1, int(
      getattr(self.trainer, 'accumulate_grad_batches', 1) or 1))
    if (torch.distributed.is_available()
        and torch.distributed.is_initialized()):
      world_size = torch.distributed.get_world_size()
      global_rank = torch.distributed.get_rank()
    else:
      world_size = 1
      global_rank = 0

    global_batch_size = int(self.config.loader.global_batch_size)
    expected_global_batch_size = batch_size * world_size * accumulation_steps
    if global_batch_size != expected_global_batch_size:
      raise ValueError(
        "Global time sampling requires loader.global_batch_size to equal "
        "local_batch * world_size * accumulate_grad_batches; got "
        f"{global_batch_size} != {batch_size} * {world_size} * "
        f"{accumulation_steps} = {expected_global_batch_size}.")

    accumulation_step = int(current_accumulation_step)
    if not 0 <= accumulation_step < accumulation_steps:
      raise ValueError(
        f"Invalid accumulation step {accumulation_step}; expected [0, "
        f"{accumulation_steps}).")

    # Rank 0 selects one randomized lattice shift for the entire logical
    # global batch.  Every rank and every accumulated microbatch must reuse
    # that same shift; drawing once per microbatch would no longer be one
    # B_global-point low-discrepancy lattice.
    if accumulation_step == 0:
      shift = torch.empty(1, device=self.device, dtype=torch.float32)
      if global_rank == 0:
        shift.uniform_()
      if world_size > 1:
        torch.distributed.broadcast(shift, src=0)
      self._global_time_lattice_shift = shift
    else:
      shift = self._global_time_lattice_shift
      if shift is None:
        raise RuntimeError(
          "Global time sampling reached accumulation step "
          f"{accumulation_step} without the lattice shift from step 0. "
          "Accumulated microbatches must be processed in order.")

    local_offset = (
      (global_rank * accumulation_steps + accumulation_step) * batch_size)
    indices = torch.arange(
      local_offset, local_offset + batch_size,
      device=self.device, dtype=torch.float32)
    t = (indices / float(global_batch_size) + shift).fmod(1.0)
    if accumulation_step == accumulation_steps - 1:
      self._global_time_lattice_shift = None
    return (1 - 2 * eps) * t + eps

  def _sigma_from_alphat(self, alpha_t: torch.Tensor) -> torch.Tensor:
    return -torch.log(alpha_t)

  # def _mask_logits_forbidden_classes(self, logits: torch.Tensor) -> torch.Tensor:
  #   logits = logits.clone()
  #   logits[..., self.mask_id] = self.neg_infinity
  #   return logits

  def nll(self, input_tokens,
          current_accumulation_step=None, train_mode=False):
    t = self._sample_t(
      input_tokens.shape[0],
      current_accumulation_step=current_accumulation_step,
      train_mode=train_mode)
    z_t = self.hybrid_noise.sample_zt(input_tokens, t)

    if self._sigma_free_backbone:
        logits = self.backbone(z_t)
    else:
      alpha_t = self._loglinear.alpha_t(t)
      sigma = self._sigma_from_alphat(alpha_t.unsqueeze(-1))
      sigma = self._process_sigma(sigma)
      logits = self.backbone(z_t, sigma)

    attention_mask = torch.ones_like(input_tokens, dtype=logits.dtype)

    # Use 'none' reduction to return per-token loss [batch, seq_len]
    # TrainerObjectiveMixin._loss() will then do the tokenmean reduction
    loss, elbo, metrics = self.loss_fn(
      logits=logits,
      input_ids=input_tokens,
      attention_mask=attention_mask,
      z_t=z_t,
      t=t,
      reduction='none',
    )

    aux_metrics = None
    if train_mode and self.training:
      aux_metrics = {
        'train/elbo': metrics['elbo'],
        'train/kl_loss': metrics['kl_loss'],
        'train/is_loss': metrics['is_loss'],
      }

    return loss, elbo, aux_metrics


__all__ = ['GIDD']
