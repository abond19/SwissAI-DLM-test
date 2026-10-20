#!/usr/bin/env python3
"""Learning-rate transfer across widths (runs from submit_mup_lr_transfer.py).

For every width: final fixed-noise validation NLL (eval_final_val_nll.py) and the
train ELBO averaged over the last --train-frac of steps, against the learning-rate
multiplier; the interpolated optimum (quadratic in log LR through the best point
and its neighbours); fitted with a parabola in log LR through all points (mean over seeds), with an
uncertainty, so single noisy runs do not decide the optimum.  muP transfers the learning rate
if the optimal multiplier is the same at every width.  The fitted drift
d log2(multiplier*) / d log2(width) is printed: 0 means perfect transfer.

Writes <root>/analysis/lr_transfer.png and lr_transfer.csv.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cooldown_analysis import read_curves  # noqa: E402


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('roots', nargs='+', help='one or more experiment output roots')
  p.add_argument('--metric', choices=['final_val', 'train_final'], default='final_val')
  p.add_argument('--train-frac', type=float, default=0.05)
  p.add_argument('--out', type=Path, default=None)
  args = p.parse_args()
  out = args.out or Path(args.roots[0]) / 'analysis'
  out.mkdir(parents=True, exist_ok=True)

  runs = []
  for root in args.roots:
    for meta in sorted(Path(root).glob('*/run_meta.json')):
      r = json.loads(meta.read_text())
      final = meta.parent / 'final_val_nll.json'
      r['final_val'] = json.loads(final.read_text()).get('val/nll') if final.exists() else None
      train, _ = read_curves(meta.parent / 'train_curve.csv')
      tail = [v for s, v in train if s > (1 - args.train_frac) * r['max_steps']]
      r['train_final'] = sum(tail) / len(tail) if tail else None
      r['curve'] = train
      for k in ('final_val', 'train_final'):  # diverged runs (NaN / inf) are dropped
        if r[k] is not None and not math.isfinite(r[k]):
          r[k] = None
          r['diverged'] = True
      runs.append(r)
  if not runs:
    sys.exit('No runs found.')

  import numpy as np

  # One curve per (experiment, width): mean over seeds at each multiplier.
  series = defaultdict(list)
  for r in runs:
    series[(r['experiment'], r['hidden_size'])].append(r)
  rows, optima = [], {}
  for key in sorted(series):
    exp, width = key
    by_mult = defaultdict(list)
    for r in series[key]:
      if r[args.metric] is not None:
        by_mult[r['eta_mult']].append(r[args.metric])
      rows.append(dict(experiment=exp, width=width, seed=r['seed'], eta_mult=r['eta_mult'],
                       eta=r['eta'], final_val=r['final_val'], train_final=r['train_final']))
    diverged = sorted({r['eta_mult'] for r in series[key] if r.get('diverged')})
    if diverged:
      print(f'  diverged (left out of the fit): ' + ', '.join(f'x{m:g}' for m in diverged))
    mults = sorted(by_mult)
    means = [float(np.mean(by_mult[m])) for m in mults]
    print(f'{exp}  width={width}  ({args.metric}, mean over seeds)')
    for m_, v in zip(mults, means):
      n = len(by_mult[m_])
      spread = f' +- {np.std(by_mult[m_], ddof=1) / np.sqrt(n):.4f}' if n > 1 else ''
      print(f'  x{m_:<6g} {v:.4f}{spread}  (n={n})')
    if len(mults) < 5:
      print('  need >= 5 multipliers for the parabola fit\n')
      continue
    # Parabola in u = log2(multiplier) through ALL points: robust to single noisy runs.
    u = np.log2(mults)
    coef, cov = np.polyfit(u, means, 2, cov=True)
    c, b, a = coef
    resid = np.array(means) - np.polyval(coef, u)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    if c <= 0:
      print(f'  no minimum: the fitted parabola opens downward (residual rms {rms:.4f}); '
            'noise dominates or the grid misses the walls\n')
      continue
    u_star = -b / (2 * c)
    grad = np.array([b / (2 * c * c), -1 / (2 * c)])  # d u* / d(c, b)
    se = float(np.sqrt(grad @ cov[:2, :2] @ grad))
    inside = u.min() <= u_star <= u.max()
    optima[key] = (u_star, se, inside)
    print(f'  parabola optimum x{2 ** u_star:.3g}  (log2 = {u_star:+.2f} +- {se:.2f}; '
          f'residual rms {rms:.4f}; curvature {c:.4f} +- {np.sqrt(cov[0, 0]):.4f})'
          + ('' if inside else '  OUTSIDE the grid: extend it'))
    print()

  by_exp = defaultdict(list)
  for (exp, width), (u_star, se, inside) in optima.items():
    by_exp[exp].append((width, u_star, se, inside))
  for exp, pts in by_exp.items():
    if len(pts) < 2:
      continue
    x = np.log2([w for w, *_ in pts])
    y = np.array([us for _, us, _, _ in pts])
    wts = 1 / np.maximum(np.array([se for _, _, se, _ in pts]), 1e-3) ** 2
    xm = np.sum(wts * x) / np.sum(wts)
    slope = float(np.sum(wts * (x - xm) * y) / np.sum(wts * (x - xm) ** 2))
    slope_se = float(np.sqrt(1 / np.sum(wts * (x - xm) ** 2)))
    span = x.max() - x.min()
    print(f'{exp}: optimum drifts as width^{slope:+.3f} +- {slope_se:.3f}  '
          f'(x{2 ** (slope * span):.2f} across the widths; 0 = perfect transfer)')
    if abs(slope) < 2 * slope_se:
      print('  -> consistent with no drift at this precision')

  with open(out / 'lr_transfer.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    print('matplotlib not available; skipped the figure.')
    return
  fig, ax = plt.subplots(1, 2, figsize=(13, 5))
  cmap = plt.get_cmap('viridis')
  keys = sorted(series)
  widths = sorted({k[1] for k in keys})
  for key in keys:
    exp, width = key
    color = cmap(widths.index(width) / max(1, len(widths) - 1))
    ls = '-' if 'noperiln' not in exp else '--'
    for i, metric in enumerate(('final_val', 'train_final')):
      pts = sorted((r['eta_mult'], r[metric]) for r in series[key] if r[metric] is not None)
      if pts:
        ax[i].plot(*zip(*pts), 'o' + ls, color=color, label=f'{exp} w={width}')
        b = min(pts, key=lambda q: q[1])
        ax[i].plot(*b, 'D', color='gold', mec='k', zorder=5)
  for i, title in enumerate(('Final validation NLL (fixed noise)', 'Train ELBO (end of run)')):
    ax[i].set(xscale='log', title=title, xlabel='LR multiplier (width-256 units)')
    ax[i].grid(alpha=0.3)
  ax[0].legend(fontsize=7)
  fig.tight_layout()
  fig.savefig(out / 'lr_transfer.png', dpi=150)
  print(f'Wrote {out / "lr_transfer.csv"} and {out / "lr_transfer.png"}')


if __name__ == '__main__':
  main()
