#!/usr/bin/env python3
"""Fit a two-term learning-rate law to the cooldown runs and predict other schedules.

Model, with eta_j the LR multiplier at step j (peak 1) and S_s = sum_{j<s} eta_j the
cumulative LR area:

    L(s) = L0 + A * S_s^(-alpha) + Y_s,      Y_{j+1} = (1 - a eta_j) Y_j + b eta_j^2

The power law is the slow progress that needs LR area; Y is the noise excess of
the "two terminal cooldowns" note (it settles at b eta / a at a constant LR and
relaxes on a time scale of 1/(a eta) steps).  All arms of one (peak LR, seed)
share data and noise, so a joint fit separates the two terms.  The note's
reciprocal tail uses k0 = 4/a steps.

Outputs: fitted parameters, measured vs fitted final loss per arm, predictions
for candidate tails (linear, 1-sqrt, reciprocal with k0 = c/a), and
<out>/cooldown_law_fit.png (loss vs LR area, data vs fit).  Train ELBO is used
(every step; smoothed), so predictions are for the final train ELBO; their
ranking is what matters.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cooldown_analysis import read_curves  # noqa: E402
import json  # noqa: E402


def schedule(total, warmup_frac, rho, shape, k0=None):
  """Multiplier per step (numpy), matching WSDLRScheduler; k0 in steps."""
  it = np.arange(total, dtype=float)
  warm = int(round(warmup_frac * total))
  cool = int(round(rho * total))
  start = total - cool
  eta = np.ones(total)
  eta[:warm] = (it[:warm] + 1) / (warm + 1)
  j = it[start:] - start
  frac = j / max(1, cool)
  if shape == 'linear':
    tail = 1 - frac
  elif shape == 'cosine':
    tail = 0.5 * (1 + np.cos(np.pi * frac))
  elif shape == 'sqrt':
    tail = 1 - np.sqrt(frac)
  elif shape == 'reciprocal':
    tail = k0 / (k0 + j) * (1 - frac)
  else:
    raise ValueError(shape)
  eta[start:] = np.clip(tail, 0, None)
  if warm > start:  # cosine tail covering the warmup (classic cosine)
    eta[:warm] = np.minimum(eta[:warm], (it[:warm] + 1) / (warm + 1))
  return eta


def model(eta, params):
  l0, log_A, alpha, log_a, log_b = params
  A, a, b = math.exp(log_A), math.exp(log_a), math.exp(log_b)
  area = np.concatenate([[0.0], np.cumsum(eta)])[:-1] + eta  # area after step s
  y = np.empty_like(eta)
  cur = 0.0
  for j, e in enumerate(eta):  # Y after step j
    cur = (1 - a * e) * cur + b * e * e
    y[j] = cur
  return l0 + A * np.maximum(area, 1e-9) ** (-alpha) + y


def load(root, eta_mult, seed):
  arms = []
  for meta in sorted(Path(root).glob('*/run_meta.json')):
    r = json.loads(meta.read_text())
    if r['seed'] != seed or not math.isclose(r['eta_mult'], eta_mult):
      continue
    train, _ = read_curves(meta.parent / 'train_curve.csv')
    if not train:
      continue
    k0 = (r['rec_k0_frac'] * int(round(r['rho'] * r['total_steps']))
          if r['shape'] == 'reciprocal' else None)
    r['eta_sched'] = schedule(r['total_steps'], r['warmup_frac'], r['rho'], r['shape'], k0)
    steps = np.array([s for s, _ in train])
    vals = np.array([v for _, v in train])
    keep = steps > r['start_step']  # each run contributes only the steps it trained
    r['steps'], r['vals'] = steps[keep], vals[keep]
    arms.append(r)
  return arms


def smoothed(steps, vals, window, every):
  if len(vals) < window:
    return steps, vals
  kernel = np.ones(window) / window
  sm = np.convolve(vals, kernel, mode='valid')
  st = steps[window - 1:]
  return st[::every], sm[::every]


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('root')
  p.add_argument('--eta-mult', type=float, default=1.0)
  p.add_argument('--seed', type=int, default=4)
  p.add_argument('--fit-from', type=float, default=0.15,
                 help='ignore steps before this fraction of the run (early transient)')
  p.add_argument('--smooth', type=int, default=100, help='running-mean window (steps)')
  p.add_argument('--out', type=Path, default=None)
  args = p.parse_args()
  from scipy.optimize import least_squares

  arms = load(args.root, args.eta_mult, args.seed)
  if not arms:
    sys.exit('No runs with train curves for this peak LR and seed.')
  total = arms[0]['total_steps']
  data = []
  for r in arms:
    st, sm = smoothed(r['steps'], r['vals'], args.smooth, max(1, args.smooth // 4))
    keep = st >= args.fit_from * total
    data.append((r, st[keep] - 1, sm[keep]))  # row for step s is the loss after step s-1

  def residuals(params):
    out = []
    for r, idx, vals in data:
      pred = model(r['eta_sched'], params)
      out.append(pred[idx] - vals)
    return np.concatenate(out)

  final_drop = max(r['vals'][-50:].mean() for r in arms) - min(r['vals'][-50:].mean() for r in arms)
  x0 = [min(r['vals'].min() for r in arms) - 0.3, math.log(1.0), 0.3,
        math.log(1.0 / 500), math.log(max(final_drop, 0.05) / 500)]
  fits = []
  for log_a0 in (math.log(1 / 100), math.log(1 / 500), math.log(1 / 2000), math.log(1 / 8000)):
    start = list(x0)
    start[3] = log_a0
    start[4] = log_a0 + math.log(0.15)
    fits.append(least_squares(residuals, start, loss='soft_l1', f_scale=0.01,
                              bounds=([0, -10, 0.01, -14, -20], [10, 10, 3, 0, 0])))
  fit = min(fits, key=lambda f: f.cost)
  l0, log_A, alpha, log_a, log_b = fit.x
  a, b = math.exp(log_a), math.exp(log_b)
  rms = math.sqrt(np.mean(fit.fun ** 2))
  print(f'Fit on {len(arms)} runs, peak x{args.eta_mult:g}, seed {args.seed} '
        f'(rms residual {rms:.4f}):')
  print(f'  slow progress: L0={l0:.4f}  A={math.exp(log_A):.4g}  alpha={alpha:.3f}')
  print(f'  noise term:    a={a:.3g}  b={b:.3g}  -> excess at peak b/a={b / a:.4f}, '
        f'relaxation 1/a={1 / a:.0f} steps at peak')
  print(f'  note\'s reciprocal offset k0 = 4/a = {4 / a:.0f} steps')
  print('\nfinal train ELBO, measured (mean of last 50 steps) vs fitted:')
  for r in arms:
    if r['kind'] != 'tail':
      continue
    pred = model(r['eta_sched'], fit.x)[-1]
    label = f'rho={r["rho"]:g} {r["shape"]}' + (
      f' k0={r["rec_k0_frac"]:g}M (ak0={a * r["rec_k0_frac"] * round(r["rho"] * total):.1f})'
      if r['shape'] == 'reciprocal' else '')
    print(f'  {label:<42} {r["vals"][-50:].mean():.4f}  fit {pred:.4f}')

  warm = arms[0]['warmup_frac']
  cands = []
  for rho in (0.1, 0.2, 0.3, 0.5):
    m = int(round(rho * total))
    cands.append((f'rho={rho:g} linear', schedule(total, warm, rho, 'linear')))
    cands.append((f'rho={rho:g} 1-sqrt', schedule(total, warm, rho, 'sqrt')))
    for c in (1, 4, 16, 64):
      cands.append((f'rho={rho:g} reciprocal ak0={c} (k0={c / a:.0f}={c / a / m:.2f}M)',
                    schedule(total, warm, rho, 'reciprocal', c / a)))
  cands.append(('classic cosine', schedule(total, warm, 1 - warm, 'cosine')))
  preds = sorted((model(e, fit.x)[-1], n) for n, e in cands)
  print('\npredicted final train ELBO for candidate tails (best first):')
  for v, n in preds:
    print(f'  {v:.4f}  {n}')

  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    return
  out = args.out or Path(args.root) / 'analysis'
  out.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(1, 2, figsize=(14, 5))
  for i, (r, idx, vals) in enumerate(data):
    color = plt.get_cmap('tab10')(i % 10)
    area = np.cumsum(r['eta_sched'])
    pred = model(r['eta_sched'], fit.x)
    name = 'trunk' if r['kind'] == 'trunk' else f'rho={r["rho"]:g} {r["shape"]}'
    ax[0].plot(area[idx], vals, '.', ms=2, color=color, label=name)
    ax[0].plot(area[idx], pred[idx], '-', lw=1, color=color)
    ax[1].plot(idx, vals, '.', ms=2, color=color, label=name)
    ax[1].plot(idx, pred[idx], '-', lw=1, color=color)
  ax[0].set(title='Train ELBO vs cumulative LR area (dots: data, lines: fit)',
            xlabel='LR area (steps at peak)')
  ax[1].set(title='Train ELBO vs step', xlabel='step')
  for x in ax:
    x.grid(alpha=0.3)
    x.legend(fontsize=7, markerscale=4)
  fig.tight_layout()
  fig.savefig(out / 'cooldown_law_fit.png', dpi=150)
  print(f'\nWrote {out / "cooldown_law_fit.png"}')


if __name__ == '__main__':
  main()
