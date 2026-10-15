#!/usr/bin/env python3
"""Learning-rate transfer across widths (runs from submit_mup_lr_transfer.py).

For every width: final fixed-noise validation NLL (eval_final_val_nll.py) and the
train ELBO averaged over the last --train-frac of steps, against the learning-rate
multiplier; the interpolated optimum (quadratic in log LR through the best point
and its neighbours); and whether it is bracketed.  muP transfers the learning rate
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
from bst_scion_analysis import optimum  # noqa: E402
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
      runs.append(r)
  if not runs:
    sys.exit('No runs found.')

  series = defaultdict(list)
  for r in runs:
    series[(r['experiment'], r['hidden_size'], r['seed'])].append(r)
  rows, optima = [], {}
  for key in sorted(series):
    exp, width, seed = key
    rs = sorted(series[key], key=lambda r: r['eta_mult'])
    pts = [(r['eta_mult'], r[args.metric]) for r in rs]
    best, star, bracketed = optimum(pts)
    print(f'{exp}  width={width}  seed={seed}  ({args.metric})')
    for r in rs:
      v = r[args.metric]
      mark = ' <- best' if r['eta_mult'] == best else ''
      print(f'  x{r["eta_mult"]:<6g} {"pending" if v is None else f"{v:.4f}"}{mark}')
      rows.append(dict(experiment=exp, width=width, seed=seed, eta_mult=r['eta_mult'],
                       eta=r['eta'], final_val=r['final_val'], train_final=r['train_final']))
    if star is not None:
      optima[key] = (star, bracketed)
      print(f'  optimum x{star:.3g} ({"bracketed" if bracketed else "NOT bracketed: extend the grid"})')
    print()

  by_exp = defaultdict(list)
  for (exp, width, seed), (star, bracketed) in optima.items():
    by_exp[exp].append((width, star, bracketed))
  for exp, pts in by_exp.items():
    if len({w for w, _, _ in pts}) < 2:
      continue
    xs = [math.log2(w) for w, _, _ in pts]
    ys = [math.log2(s) for _, s, _ in pts]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = (sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
             / max(1e-12, sum((x - xm) ** 2 for x in xs)))
    widths = sorted({w for w, _, _ in pts})
    shift = 2 ** (slope * (math.log2(widths[-1]) - math.log2(widths[0])))
    print(f'{exp}: optimum drifts as width^{slope:+.3f}  '
          f'(x{shift:.2f} from width {widths[0]} to {widths[-1]}; 0 = perfect transfer)')
    if not all(b for _, _, b in pts):
      print('  some optima are at the grid edge; the drift is a lower bound')

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
    exp, width, seed = key
    color = cmap(widths.index(width) / max(1, len(widths) - 1))
    ls = '-' if 'noperiln' not in exp else '--'
    for i, metric in enumerate(('final_val', 'train_final')):
      pts = sorted((r['eta_mult'], r[metric]) for r in series[key] if r[metric] is not None)
      if pts:
        ax[i].plot(*zip(*pts), 'o' + ls, color=color, label=f'{exp} w={width} s={seed}')
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
