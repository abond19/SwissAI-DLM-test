"""Check the batched ``Scion.step`` against the original per-parameter loop on GPU.

Builds SCION groups shaped like the L12/H768 GIDD model (embedding/head: Sign,
biases: RowNorm, norm scales: BiasRMS, matrices: Spectral), applies identical
gradients to two copies, steps one with the original loop (reproduced verbatim
below) and one with ``Scion.step``, and reports per-group differences and the
GPU time of both.

Usage (on a GPU node, conda env active, from the repo root):
  python scripts/debug/check_scion_batched.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from discrete_diffusion.optimizers import scion as S  # noqa: E402

H, I, L, V, HEADS = 768, 3072, 12, 131072, 12
STEPS, TIMED = 3, 20


@torch.no_grad()
def reference_step(opt):
  """The pre-batching ``Scion.step`` loop, unchanged."""
  for group in opt.param_groups:
    lr = group['lr']
    momentum = group['momentum']
    scale = group['scale']
    unconstrained = group['unconstrained']
    norm_backend = S.norm_dict[group['norm']](**group['norm_kwargs'])
    for p in group['params']:
      g = p.grad
      if g is None:
        continue
      state = opt.state[p]
      if momentum != 1:
        if 'momentum_buffer' not in state.keys():
          state['momentum_buffer'] = torch.zeros_like(g)
        buf = state['momentum_buffer']
        buf.mul_(1 - momentum).add_(g, alpha=momentum)
        g = buf
      update = scale * norm_backend.lmo(g)
      if not unconstrained:
        p.data.mul_(1 - lr)
      p.data.add_(update, alpha=-lr)


def make_groups(dtype, seed=0):
  gen = torch.Generator(device='cuda').manual_seed(seed)

  def param(*shape, dt=None):
    t = torch.randn(*shape, generator=gen, device='cuda').to(dt or dtype) * 0.02
    return torch.nn.Parameter(t)

  emb = [param(V, H), param(V, H)]
  bias, one_d, mat = [], [], []
  for _ in range(L):
    bias += [param(HEADS, H // HEADS), param(HEADS, H // HEADS), param(I), param(H)]
    one_d += [param(H, dt=torch.float32) for _ in range(4)]
    mat += [param(H, H) for _ in range(4)] + [param(I, H), param(H, I)]
  one_d.append(param(H, dt=torch.float32))
  common = {'lr': 1e-4, 'momentum': 0.2}
  return [
    {'params': emb, 'name': 'embedding', 'norm': 'Sign', 'norm_kwargs': {}, 'scale': 3500.0, **common},
    {'params': bias, 'name': 'bias', 'norm': 'RowNorm', 'norm_kwargs': {}, 'scale': 90.0, **common},
    {'params': one_d, 'name': 'one_d', 'norm': 'BiasRMS', 'norm_kwargs': {}, 'scale': 6.8, **common},
    {'params': mat, 'name': 'matrix', 'norm': 'Spectral', 'norm_kwargs': {'steps': 5}, 'scale': 336.0, **common},
  ]


def set_grads(groups_a, groups_b, seed):
  gen = torch.Generator(device='cuda').manual_seed(seed)
  for ga, gb in zip(groups_a, groups_b):
    for pa, pb in zip(ga['params'], gb['params']):
      g = torch.randn(pa.shape, generator=gen, device='cuda').to(pa.dtype)
      pa.grad, pb.grad = g.clone(), g.clone()


def _max_rel_diff(refs, others):
  exact = all(torch.equal(r, o) for r, o in zip(refs, others))
  worst = max(((r.float() - o.float()).abs().max() / r.float().abs().max()).item()
              for r, o in zip(refs, others))
  return 'identical' if exact else f'max relative diff {worst:.1e}'


@torch.no_grad()
def compare_spectral_lmo(dtype):
  """Newton-Schulz on identical inputs: batched vs per-matrix, plus a noise floor.

  The noise floor compares two versions of the *original* per-matrix code
  (compiled vs eager).  Newton-Schulz runs in bf16 and amplifies small
  rounding differences, so batching is fine if it differs by a similar amount.
  """
  gen = torch.Generator(device='cuda').manual_seed(7)
  backend = S.Spectral(steps=5)
  mats = [torch.randn(shape, generator=gen, device='cuda').to(dtype)
          for _ in range(L) for shape in [(H, H)] * 4 + [(I, H), (H, I)]]
  ref = [backend.lmo(m.clone()) for m in mats]
  new = S._lmo_all(backend, [m.clone() for m in mats])
  eager_ns = S.zeropower_via_newtonschulz5._torchdynamo_orig_callable
  eager = []
  for m in mats:
    x = eager_ns(m.clone().reshape(len(m), -1), steps=backend.steps).view(m.shape)
    x *= S._spectral_lmo_scale(backend, m.shape)
    eager.append(x)
  return _max_rel_diff(ref, new), _max_rel_diff(ref, eager)


def gpu_ms(fn, n):
  start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  torch.cuda.synchronize()
  start.record()
  for _ in range(n):
    fn()
  end.record()
  torch.cuda.synchronize()
  return start.elapsed_time(end) / n


def main():
  if not torch.cuda.is_available():
    raise SystemExit('Needs a GPU.')
  for dtype in (torch.bfloat16, torch.float32):
    # Fresh compile caches per dtype, as in a training run (one dtype only).
    torch._dynamo.reset()
    ga, gb = make_groups(dtype), make_groups(dtype)
    init = [[p.detach().clone() for p in g['params']] for g in ga]
    ref = S.Scion(ga, lr=1e-4, momentum=0.2)
    new = S.Scion(gb, lr=1e-4, momentum=0.2)
    for step in range(STEPS):
      set_grads(ga, gb, seed=100 + step)
      reference_step(ref)
      new.step()
    torch.cuda.synchronize()
    lines = []
    for g_a, g_b, p0 in zip(ga, gb, init):
      exact = all(torch.equal(a, b) for a, b in zip(g_a['params'], g_b['params']))
      # Compare the accumulated change, not the weights (lr is small).
      worst = max(
        ((a.float() - b.float()).abs().max()
         / (a.float() - z.float()).abs().max().clamp_min(1e-30)).item()
        for a, b, z in zip(g_a['params'], g_b['params'], p0))
      bufs = all(torch.equal(ref.state[a]['momentum_buffer'], new.state[b]['momentum_buffer'])
                 for a, b in zip(g_a['params'], g_b['params']))
      lines.append(f"  {g_a['name']:9s} weights {'identical' if exact else f'differ (max {worst:.1e} of the update)'}"
                   f", momentum buffers {'identical' if bufs else 'differ'}")
    print(f'{dtype} after {STEPS} steps:')
    print('\n'.join(lines))
    batched, floor = compare_spectral_lmo(dtype)
    print(f'  Newton-Schulz, same input: batched vs per-matrix {batched}; '
          f'noise floor (per-matrix compiled vs eager) {floor}')

    ref_ms = gpu_ms(lambda: reference_step(ref), TIMED)
    new_ms = gpu_ms(new.step, TIMED)
    print(f'  GPU time per step: original loop {ref_ms:.1f} ms, batched {new_ms:.1f} ms '
          f'({ref_ms / new_ms:.2f}x)')


if __name__ == '__main__':
  main()
