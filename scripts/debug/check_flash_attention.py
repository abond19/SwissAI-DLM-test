"""Show, on the GPU, which attention path the GIDD model runs before and after the dtype fixes.

For the model code at an old git revision (default ``c9d0284``, the commit the
original baselines ran) and the current working tree, builds three configs
(gidd_hf with bf16 weights, gidd_hf with fp32 weights, gidd_hf_ref/muP with
fp32 weights), runs forward+backward under bf16 autocast (as Lightning's
``bf16-mixed`` does) and reports:

  - calls to flash-attn's ``flash_attn_func`` vs the eager fallback,
  - the q/k/v dtypes that reached the attention module,
  - whether the PyTorch profiler saw flash-attn CUDA kernels,
  - peak memory and time of one forward+backward.

Usage (on a GPU node, conda env active, from the repo root):
  python scripts/debug/check_flash_attention.py [--old-ref c9d0284]
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = 'src/discrete_diffusion/models'

COMMON = dict(vocab_size=8192, hidden_size=768, intermediate_size=3072, num_hidden_layers=2,
              num_attention_heads=12, max_position_embeddings=2048, attn_performer='flash_attn2',
              init_scale=0.02, emb_init_scale=0.02, head_init_scale=0.02)
CONFIGS = {
  'gidd_hf, bf16 weights (SCION default)': dict(
    torch_dtype=torch.bfloat16, attention_bias=True, attn_soft_cap=30.0, activation_scale=2.0, resid_scale=4.0),
  'gidd_hf, fp32 weights (SCION fp32 runs)': dict(
    torch_dtype=torch.float32, attention_bias=True, attn_soft_cap=30.0, activation_scale=2.0, resid_scale=4.0),
  'gidd_hf_ref, fp32 weights (muP runs)': dict(
    torch_dtype=torch.float32, attention_bias=False, mlp_bias=False, attn_soft_cap=0.0, mlp_activation='gelu',
    qk_norm_per_head=True, qk_norm_affine=False, norm_zero_centered=False, rms_norm_eps=None, peri_ln=True),
}


def load_model_package(label: str, ref: str | None):
  """Import configuration_gidd + modeling_gidd_hf from a git ref (or the working tree)."""
  pkg_dir = Path(tempfile.mkdtemp()) / f'gidd_{label}'
  pkg_dir.mkdir()
  (pkg_dir / '__init__.py').write_text('')
  for name in ('configuration_gidd.py', 'modeling_gidd_hf.py'):
    if ref is None:
      text = (REPO / MODEL_DIR / name).read_text()
    else:
      text = subprocess.run(['git', '-C', str(REPO), 'show', f'{ref}:{MODEL_DIR}/{name}'],
                            check=True, capture_output=True, text=True).stdout
    (pkg_dir / name).write_text(text)
  sys.path.insert(0, str(pkg_dir.parent))
  config_mod = importlib.import_module(f'gidd_{label}.configuration_gidd')
  model_mod = importlib.import_module(f'gidd_{label}.modeling_gidd_hf')
  return config_mod, model_mod


def instrument(model_mod, stats):
  """Count flash vs eager calls and record the dtypes reaching attention."""
  flash_fn = model_mod._flash_attn2_func
  if flash_fn is None:
    raise SystemExit('flash-attn is not importable in this environment.')

  def counting_flash(*args, **kwargs):
    stats['flash_attn_func calls'] += 1
    return flash_fn(*args, **kwargs)
  model_mod._flash_attn2_func = counting_flash

  eager_forward = model_mod.VanillaSoftcapAttention.forward

  def counting_eager(self, *args, **kwargs):
    stats['eager fallback calls'] += 1
    return eager_forward(self, *args, **kwargs)
  model_mod.VanillaSoftcapAttention.forward = counting_eager

  flash_module_forward = model_mod.FlashAttn2SoftcapAttention.forward

  def recording_forward(self, q, k, v, attention_mask=None):
    stats['dtypes'].add(f'q={str(q.dtype)[6:]} k={str(k.dtype)[6:]} v={str(v.dtype)[6:]}')
    return flash_module_forward(self, q, k, v, attention_mask=attention_mask)
  model_mod.FlashAttn2SoftcapAttention.forward = recording_forward


def run(config_mod, model_mod, overrides, batch):
  stats = collections.defaultdict(int)
  stats['dtypes'] = set()
  instrument(model_mod, stats)
  torch.manual_seed(0)
  model = model_mod.GiddForDiffusionLM(config_mod.GiddConfig(**COMMON, **overrides)).cuda()
  x = torch.randint(4, COMMON['vocab_size'], (batch, COMMON['max_position_embeddings']), device='cuda')

  def fwd_bwd():
    with torch.autocast('cuda', dtype=torch.bfloat16):
      logits = model(input_ids=x).logits
    logits.float().square().mean().backward()
    model.zero_grad(set_to_none=True)

  fwd_bwd()  # warm-up
  torch.cuda.synchronize()
  stats.clear()
  stats['dtypes'] = set()
  torch.cuda.reset_peak_memory_stats()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    fwd_bwd()
    torch.cuda.synchronize()
  flash_kernels = sorted({e.key for e in prof.key_averages() if 'flash' in e.key.lower()})
  one_pass = {k: (set(v) if isinstance(v, set) else v) for k, v in stats.items()}
  peak = torch.cuda.max_memory_allocated() / 2**30
  start = time.perf_counter()
  for _ in range(3):
    fwd_bwd()
  torch.cuda.synchronize()
  ms = (time.perf_counter() - start) / 3 * 1e3
  del model
  torch.cuda.empty_cache()
  return one_pass, flash_kernels, peak, ms


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--old-ref', default='c9d0284')
  parser.add_argument('--batch', type=int, default=4)
  args = parser.parse_args()
  if not torch.cuda.is_available():
    raise SystemExit('Needs a GPU.')
  versions = {f'old ({args.old_ref})': args.old_ref, 'current': None}
  for vlabel, ref in versions.items():
    for clabel, overrides in CONFIGS.items():
      # Fresh module copy per run so the instrumentation does not stack.
      tag = f"{'old' if ref else 'cur'}_{abs(hash(clabel)) % 10**6}"
      config_mod, model_mod = load_model_package(tag, ref)
      stats, kernels, peak, ms = run(config_mod, model_mod, overrides, args.batch)
      print(f'{vlabel:16s} | {clabel:40s} | per fwd+bwd: flash_attn_func {stats.get("flash_attn_func calls", 0):2d}, '
            f'eager fallback {stats.get("eager fallback calls", 0):2d} | {", ".join(sorted(stats["dtypes"]))} | '
            f'flash kernels on GPU: {"yes" if kernels else "no"} ({len(kernels)}) | '
            f'peak {peak:.1f} GiB | fwd+bwd {ms:.1f} ms')


if __name__ == '__main__':
  main()
