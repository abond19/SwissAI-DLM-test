"""SwissAI wrapper for the HF-style GIDD architecture from gidd-easydel."""

from __future__ import annotations

import torch

from .configuration_gidd import GiddConfig
from .modeling_gidd_hf import GiddForDiffusionLM
from .parametrized_init import init_parametrized, set_residual_scale
from .scaling import get_parametrization, validate_model_config


def validate_parametrized_model(config) -> None:
  model_cfg = config.model
  validate_model_config(
    get_parametrization(
      model_cfg.parametrization,
      hidden_size=model_cfg.hidden_size,
      n_blocks=model_cfg.n_blocks,
      base_width=getattr(model_cfg, 'base_width', 256),
      base_depth=getattr(model_cfg, 'base_depth', 12)),
    model_cfg)


def build_gidd_config(config, vocab_size: int) -> GiddConfig:
  """Build the HF-style GIDD config shared by full-model and pipeline paths."""
  model_cfg = config.model
  algo_cfg = config.algo

  hidden_size = model_cfg.hidden_size
  intermediate_size = int(getattr(model_cfg, 'intermediate_size', hidden_size * 4))
  base_init_scale = model_cfg.init_scale
  aux_init_scale = model_cfg.aux_init_scale
  head_scale = model_cfg.head_scale
  # BST/NanoGPT applies the unembedding directly, without an additional
  # width-dependent multiplier on the logits.  Keep the historical EasyDeL
  # behavior available only as an explicit opt-in for reproduction runs.
  head_output_scaling_enabled = bool(
    getattr(model_cfg, 'head_output_scaling_enabled', False))
  head_scaling = (
    head_scale / hidden_size if head_output_scaling_enabled else 1.0)
  trainer_precision = str(
    getattr(getattr(config, 'trainer', object()), 'precision', '')).lower()
  loss_precision = str(
    getattr(getattr(config, 'training', object()), 'loss_precision', '')).lower()
  model_dtype = str(getattr(model_cfg, 'torch_dtype', '')).lower()

  if model_dtype in {'bf16', 'bfloat16'}:
    torch_dtype = torch.bfloat16
  elif model_dtype in {'fp16', 'float16'}:
    torch_dtype = torch.float16
  elif model_dtype in {'fp32', 'float32'}:
    torch_dtype = torch.float32
  elif 'bf16' in trainer_precision or loss_precision == 'bf16':
    torch_dtype = torch.bfloat16
  elif '16' in trainer_precision:
    torch_dtype = torch.float16
  else:
    torch_dtype = torch.float32

  # ``model.parametrization`` set: the legacy init / scale knobs are neutral and the init comes
  # from ``init_parametrized`` (see models/parametrized_init.py).  Unset: historical behaviour.
  parametrized = getattr(model_cfg, 'parametrization', None) is not None
  if parametrized:
    validate_parametrized_model(config)

  return GiddConfig(
    vocab_size=int(vocab_size),
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_hidden_layers=model_cfg.n_blocks,
    num_attention_heads=int(getattr(model_cfg, 'n_heads', 12)),
    is_causal=bool(getattr(algo_cfg, 'causal_attention', False)),
    max_position_embeddings=model_cfg.length,
    attn_performer=model_cfg.attn_attn_backend,
    tie_word_embeddings=bool(getattr(model_cfg, 'tie_word_embeddings', False)),
    rms_norm_eps=(
      None if getattr(model_cfg, 'rms_norm_eps', 1e-6) is None
      else float(getattr(model_cfg, 'rms_norm_eps', 1e-6))),
    attn_soft_cap=float(getattr(model_cfg, 'attn_soft_cap', 30.0)),
    resid_scale=model_cfg.resid_scale,
    init_scale=0.0 if parametrized else base_init_scale / max(1, hidden_size ** 0.5),
    head_init_scale=(
      0.0 if (parametrized or model_cfg.zero_head_init) else aux_init_scale),
    emb_init_scale=aux_init_scale,
    weight_scaling=1.0,
    head_scaling=1.0 if parametrized else head_scaling,
    rope_theta=float(getattr(model_cfg, 'rope_theta', 10000.0)),
    attention_bias=model_cfg.attn_bias,
    mlp_bias=model_cfg.mlp_bias,
    use_qk_norm=bool(getattr(model_cfg, 'use_qk_norm', True)),
    min_log_snr=float(getattr(algo_cfg, 'min_log_snr', -9.0)),
    max_log_snr=float(getattr(algo_cfg, 'max_log_snr', 9.0)),
    noise_type=algo_cfg.hybrid_mixing_shift,
    torch_dtype=torch_dtype,
    mlp_activation=str(getattr(model_cfg, 'mlp_activation', 'relu2')).lower(),
    norm_zero_centered=bool(getattr(model_cfg, 'norm_zero_centered', True)),
    qk_norm_per_head=bool(getattr(model_cfg, 'qk_norm_per_head', False)),
    qk_norm_affine=bool(getattr(model_cfg, 'qk_norm_affine', True)),
    peri_ln=bool(getattr(model_cfg, 'peri_ln', False)),
    peri_ln_affine=bool(getattr(model_cfg, 'peri_ln_affine', True)),
    peri_ln_embed=bool(getattr(model_cfg, 'peri_ln_embed', False)),
    final_norm_affine=bool(getattr(model_cfg, 'final_norm_affine', True)),
    activation_checkpointing=bool(
      getattr(model_cfg, 'activation_checkpointing', False)),
    activation_checkpoint_preserve_rng_state=bool(
      getattr(model_cfg, 'activation_checkpoint_preserve_rng_state', False)),
  )


class GiddHFWrapper(torch.nn.Module):
  """Adapter exposing SwissAI's expected `forward(x, sigma) -> logits` contract."""

  def __init__(self, config, vocab_size: int):
    super().__init__()
    hf_cfg = build_gidd_config(config, vocab_size)
    self.model = GiddForDiffusionLM(hf_cfg)
    self.config = hf_cfg
    if getattr(config.model, 'parametrization', None) is not None:
      init_parametrized(self.model, config.model)
      set_residual_scale(self.model, config.model)

    # PyTorch FSDP flattens the parameters it manages and requires every
    # parameter in that flat handle to have the same dtype.  GIDD intentionally
    # keeps RMSNorm scale parameters in fp32 while the main matrices are bf16,
    # which is valid for DDP but cannot be flattened by a root-only FSDP wrap.
    #
    # This opt-in compatibility mode is for the current root-wrapped FSDP
    # strategy.  It converts only floating-point parameters (not buffers such
    # as RoPE frequencies) to the configured model dtype.  The default keeps
    # the existing mixed-dtype behavior used by DDP training.
    if bool(getattr(config.model, 'fsdp_flatten_compatible', False)):
      target_dtype = hf_cfg.torch_dtype
      for parameter in self.model.parameters():
        if parameter.is_floating_point() and parameter.dtype != target_dtype:
          parameter.data = parameter.data.to(dtype=target_dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # attention_mask = torch.ones_like(x, dtype=torch.bool)
    outputs = self.model(
      input_ids=x,
      attention_mask=None,
      output_attentions=False,
      output_hidden_states=False,
      use_cache=False,
    )
    return outputs.logits
