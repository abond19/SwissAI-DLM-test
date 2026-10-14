#!/usr/bin/env python3
"""Analysis for the SCION BST experiment (submit_scion_bst.py runs).

Reads each run's run_meta.json, final_val_nll.json (eval_final_val_nll.py) and
train_curve.csv, then:
  * writes <out>/summary.csv: one row per run with the final fixed-noise
    validation NLL, the train ELBO averaged over the last --train-window tokens,
    and the train/validation loss just before the warmdown;
  * prints, per token budget, the loss against beta with the best point marked,
    the interpolated optimum beta* (quadratic in log beta through the best point
    and its neighbours), and whether the minimum is bracketed;
  * with --suggest, prints the betas to run next so each minimum is bracketed
    at the target grid spacing;
  * fits beta* ~ T^k across budgets: k = 0 is the constant-step-size
    hypothesis, k = -1/3 the SCG (Islamov et al., 2026) prediction;
  * draws <out>/bst_scion.png in the layout of the BST paper's Figure 2: loss
    against beta per budget (top) and loss against tokens at each budget's best
    beta with a Huber fit a0 + b0 * T^-c through the pre-warmdown points (bottom).

Only the base series is compared across budgets (default radii, momentum and
seed unless --momentum/--radii/--seed say otherwise); step-1 batch-size and
radius sweeps appear in summary.csv and the per-budget tables.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def load_runs(roots):
  runs = []
  for root in roots:
    for meta_path in sorted(Path(root).glob('*/run_meta.json')):
      run_dir = meta_path.parent
      r = json.loads(meta_path.read_text())
      r['run_dir'] = str(run_dir)
      r['beta'] = float(r['beta'])
      r['radii'] = ':'.join(str(r[k]) for k in
                            ('scale_embed', 'scale_matrix', 'scale_bias', 'scale_norm'))
      final = run_dir / 'final_val_nll.json'
      r['final_val'] = (json.loads(final.read_text()).get('val/nll')
                        if final.exists() else None)
      r['done'] = (run_dir / 'DONE').exists()
      r['curve'] = read_curve(run_dir / 'train_curve.csv')
      runs.append(r)
  return runs


def read_curve(path):
  train, val = [], []
  if not path.exists():
    return {'train': train, 'val': val}
  with open(path, newline='') as f:
    for row in csv.DictReader(f):
      point = (int(row['step']), float(row['tokens']), float(row['value']))
      if row['kind'] == 'train':
        train.append(point)
      elif row['name'] == 'val/nll':
        val.append(point)
  return {'train': train, 'val': val}


def mean(xs):
  return sum(xs) / len(xs) if xs else None


def summarize(r, train_window, precooldown_steps):
  train = r['curve']['train']
  if train:
    end = train[-1][1]
    r['train_final'] = mean([v for _, t, v in train if t > end - train_window])
  else:
    r['train_final'] = None
  ws = r['max_steps'] - r['warmdown_steps']
  pre = [v for s, _, v in train if ws - precooldown_steps < s <= ws]
  r['train_precooldown'] = mean(pre)
  r['tokens_precooldown'] = ws * r['gbs'] * r['seq_len']
  vals_before = [p for p in r['curve']['val'] if p[0] <= ws]
  r['val_precooldown'] = vals_before[-1][2] if vals_before else None
  r['val_precooldown_tokens'] = vals_before[-1][1] if vals_before else None


def optimum(points):
  """points: [(beta, loss)].  Returns (best beta, interpolated beta*, bracketed)."""
  pts = sorted(p for p in points if p[1] is not None)
  if not pts:
    return None, None, False
  i = min(range(len(pts)), key=lambda j: pts[j][1])
  best = pts[i][0]
  if i == 0 or i == len(pts) - 1:
    return best, best, False
  (x0, y0), (x1, y1), (x2, y2) = [(math.log(b), l) for b, l in pts[i - 1:i + 2]]
  denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
  a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
  b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
  star = math.exp(-b / (2 * a)) if a > 0 else best
  return best, min(max(star, pts[i - 1][0]), pts[i + 1][0]), True


def suggest(points, ratio):
  pts = sorted(b for b, l in points if l is not None)
  if not pts:
    return []
  best, _, bracketed = optimum(points)
  if not bracketed:
    return [best / ratio] if best == pts[0] else [best * ratio]
  i = pts.index(best)
  out = []
  for nb in (pts[i - 1], pts[i + 1]):
    if max(nb, best) / min(nb, best) > ratio * 1.05:
      out.append(math.sqrt(nb * best))
  return out


def huber_fit(xs, ys, delta=1e-3):
  try:
    import numpy as np
    from scipy.optimize import least_squares
  except ImportError:
    return None
  xs, ys = np.asarray(xs, float), np.asarray(ys, float)
  if len(xs) < 3:
    return None
  res = least_squares(lambda p: p[0] + p[1] * xs ** (-p[2]) - ys,
                      x0=[ys.min() - 0.1, 0.5, 0.33], loss='huber', f_scale=delta,
                      bounds=([0, 0, 0], [np.inf, np.inf, 3]))
  return res.x


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('roots', nargs='+')
  p.add_argument('--out', type=Path, default=None)
  p.add_argument('--metric', choices=['final_val', 'train_final'], default='final_val',
                 help='loss used to pick the best beta')
  p.add_argument('--train-window', type=float, default=250e6,
                 help='tokens averaged for the final train ELBO')
  p.add_argument('--precooldown-steps', type=int, default=100)
  p.add_argument('--momentum', type=float, default=0.06)
  p.add_argument('--radii', default='3500:336:90:6.8')
  p.add_argument('--seed', type=int, default=4)
  p.add_argument('--suggest', action='store_true')
  p.add_argument('--grid-ratio', type=float, default=1.26,
                 help='target spacing between neighbouring betas near the optimum')
  args = p.parse_args()
  out = args.out or Path(args.roots[0]) / 'analysis'
  out.mkdir(parents=True, exist_ok=True)

  runs = load_runs(args.roots)
  for r in runs:
    summarize(r, args.train_window, args.precooldown_steps)
  cols = ['name', 'multiplier', 'tokens', 'gbs', 'beta', 'momentum', 'radii', 'seed',
          'done', 'final_val', 'train_final', 'train_precooldown', 'val_precooldown',
          'tokens_precooldown', 'run_dir']
  with open(out / 'summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    w.writerows(runs)

  # Per (budget, batch, momentum, radii, seed) series: loss against beta.
  series = defaultdict(list)
  for r in runs:
    series[(r['multiplier'], r['gbs'], float(r['momentum']), r['radii'], r['seed'])].append(r)
  base = {}
  print(f'Metric: {args.metric}\n')
  for key in sorted(series):
    m, gbs, mom, radii, seed = key
    rs = sorted(series[key], key=lambda r: r['beta'])
    pts = [(r['beta'], r[args.metric]) for r in rs]
    best, star, bracketed = optimum(pts)
    print(f'm={m:g} T={rs[0]["tokens"] / 1e9:.1f}B GBS={gbs} momentum={mom:g} '
          f'radii={radii} seed={seed}')
    for r in rs:
      v = r[args.metric]
      mark = ' <- best' if r['beta'] == best else ''
      vstr = 'pending' if v is None else f'{v:.4f}'
      print(f'  beta={r["beta"]:<9.3g} {vstr}{mark}')
    if best is not None:
      print(f'  beta* ~ {star:.3g} ({"bracketed" if bracketed else "NOT bracketed"})')
      if args.suggest:
        nxt = suggest(pts, args.grid_ratio)
        print('  next: ' + (' '.join(f'{b:.3g}' for b in nxt) if nxt else 'none needed'))
    print()
    if (math.isclose(mom, args.momentum) and radii == args.radii and seed == args.seed
        and (m not in base or len(rs) > len(base[m]))):
      base[m] = rs

  # Optimal step size against budget.
  trend = []
  for m in sorted(base):
    best, star, bracketed = optimum([(r['beta'], r[args.metric]) for r in base[m]])
    if star is not None:
      trend.append((m, star, bracketed))
  if len(trend) >= 2:
    xs = [math.log(m) for m, _, _ in trend]
    ys = [math.log(s) for _, s, _ in trend]
    xm, ym = mean(xs), mean(ys)
    k = (sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
         / max(1e-12, sum((x - xm) ** 2 for x in xs)))
    print('beta* by budget: ' + ', '.join(
      f'{m:g}x {s:.3g}{"" if b else " (edge)"}' for m, s, b in trend))
    print(f'fit beta* ~ T^{k:.3f}   (constant: 0, SCG theory: -0.333)\n')

  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    print('matplotlib not available; skipped the figure.')
    return
  fig, ax = plt.subplots(2, 2, figsize=(12, 8))
  cmap = plt.get_cmap('viridis')
  ms = sorted(base)
  fits = {}
  for col, metric, curve_key, pre_key in ((0, 'final_val', 'val', 'val_precooldown'),
                                          (1, 'train_final', 'train', 'train_precooldown')):
    pre_x, pre_y = [], []
    for i, m in enumerate(ms):
      color = cmap(i / max(1, len(ms) - 1))
      rs = [r for r in base[m] if r[metric] is not None]
      label = f'T={base[m][0]["tokens"] / 1e9:.1f}B, B={base[m][0]["gbs"]}'
      if rs:
        ax[0, col].plot([r['beta'] for r in rs], [r[metric] for r in rs], 'o-',
                        color=color, label=label)
        b = min(rs, key=lambda r: r[metric])
        ax[0, col].plot(b['beta'], b[metric], 'D', color='gold', mec='k', zorder=5)
      done = [r for r in base[m] if r['curve'][curve_key]]
      if done:
        b = min(done, key=lambda r: (r[args.metric] is None, r[args.metric] or 0))
        pts = b['curve'][curve_key]
        if curve_key == 'train':  # running mean over ~1% of the run
          wlen = max(1, len(pts) // 100)
          ys = [mean([v for _, _, v in pts[max(0, j - wlen):j + 1]]) for j in range(len(pts))]
        else:
          ys = [v for _, _, v in pts]
        ax[1, col].plot([t / 1e9 for _, t, _ in pts], ys, color=color, lw=1, label=label)
        if b[pre_key] is not None:
          x = (b['val_precooldown_tokens'] if curve_key == 'val' else b['tokens_precooldown'])
          pre_x.append(x / 1e9)
          pre_y.append(b[pre_key])
          ax[1, col].plot(x / 1e9, b[pre_key], 'D', color='gold', mec='k', zorder=5)
    fit = huber_fit(pre_x, pre_y)
    if fit is not None:
      a0, b0, c0 = fit
      fits[curve_key] = fit
      xs = [min(pre_x) * 0.5 * (max(pre_x) * 3 / (min(pre_x) * 0.5)) ** (j / 100) for j in range(101)]
      ax[1, col].plot(xs, [a0 + b0 * x ** (-c0) for x in xs], '--', color='darkgoldenrod',
                      label=f'fit {a0:.3f} + {b0:.3f} T^-{c0:.2f}')
  titles = [('Final validation NLL (fixed noise)', 'Train ELBO (last 250M tokens)'),
            ('Validation NLL (in-run subset)', 'Train ELBO')]
  for i in range(2):
    for j in range(2):
      ax[i, j].set_title(titles[i][j])
      ax[i, j].set_xscale('log')
      ax[i, j].grid(alpha=0.3)
    ax[0, i].set_xlabel('Frank-Wolfe step size beta')
    ax[1, i].set_xlabel('Tokens (B)')
  ax[0, 1].legend(fontsize=7)
  ax[1, 1].legend(fontsize=7)
  fig.tight_layout()
  fig.savefig(out / 'bst_scion.png', dpi=150)
  for key, (a0, b0, c0) in fits.items():
    print(f'{key} pre-warmdown fit: {a0:.4f} + {b0:.4f} * T^-{c0:.3f} (T in B tokens)')
  print(f'Wrote {out / "summary.csv"} and {out / "bst_scion.png"}')


if __name__ == '__main__':
  main()
