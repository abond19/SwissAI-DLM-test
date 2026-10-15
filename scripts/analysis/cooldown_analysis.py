#!/usr/bin/env python3
"""Analysis for submit_cooldown_experiment.py.

Reads every run's run_meta.json, final_val_nll.json (eval_final_val_nll.py) and
train_curve.csv under the given root and:
  * writes <out>/summary.csv;
  * prints, per seed, the final fixed-noise validation NLL of every tail
    (rows) at every peak learning rate (columns), each arm's best peak LR, and
    the paired difference to the linear tail with the same rho and peak LR.
    Arms of one (peak LR, seed) share data, prefix and noise, so these
    differences are not blurred by seed noise;
  * draws <out>/cooldown.png: validation NLL against peak LR per arm, train
    ELBO and in-run validation NLL over the whole run (trunk prefix + tail) for
    each arm at its best peak LR, zooms on the cooldown region, and the LR
    multipliers.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def arm_label(r):
  if r['kind'] == 'trunk':
    return 'trunk (pre-cooldown)'
  label = f'rho={r["rho"]:g} {r["shape"]}'
  if r['shape'] == 'reciprocal':
    label += f' k0={r["rec_k0_frac"]:g}M'
  if r['shape'] == 'cosine' and r['rho'] >= 1 - r['warmup_frac'] - 1e-9:
    label = 'classic cosine (rho=0.9)'
  return label


def read_curves(path):
  """(train, val): lists of (step, value) for train/elbo and in-run val/nll."""
  train, val = [], []
  if path.exists():
    with open(path, newline='') as f:
      for r in csv.DictReader(f):
        point = (int(r['step']), float(r['value']))
        if r['kind'] == 'train':
          train.append(point)
        elif r['name'] == 'val/nll':
          val.append(point)
  return train, val


def spliced(run, trunk, key):
  """Whole-run curve: the trunk's prefix up to the switch step, then the tail."""
  prefix = [c for c in (trunk[key] if trunk else []) if c[0] <= run['start_step']]
  return prefix + [c for c in run[key] if c[0] > run['start_step']]


def smooth(curve, frac=0.01):
  w = max(1, int(len(curve) * frac))
  out, total = [], 0.0
  for i, (_, v) in enumerate(curve):
    total += v
    if i >= w:
      total -= curve[i - w][1]
    out.append(total / min(i + 1, w))
  return [s for s, _ in curve], out


def multiplier(it, r):
  k, rho = r['total_steps'], r['rho']
  warm = int(round(r['warmup_frac'] * k))
  cool = int(round(rho * k))
  start = k - cool
  if it < warm:
    return (it + 1) / (warm + 1)
  if it < start:
    return 1.0
  j, frac = it - start, (it - start) / max(1, cool)
  if r['shape'] == 'cosine':
    return 0.5 * (1 + math.cos(math.pi * frac))
  if r['shape'] == 'sqrt':
    return 1 - math.sqrt(frac)
  if r['shape'] == 'reciprocal':
    k0 = max(1.0, r['rec_k0_frac'] * cool)
    return k0 / (k0 + j) * (1 - frac)
  return 1 - frac


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('root')
  p.add_argument('--out', type=Path, default=None)
  p.add_argument('--train-window', type=float, default=250e6,
                 help='tokens averaged for the final train ELBO')
  args = p.parse_args()
  root = Path(args.root)
  out = args.out or root / 'analysis'
  out.mkdir(parents=True, exist_ok=True)

  runs = []
  for meta in sorted(root.glob('*/run_meta.json')):
    r = json.loads(meta.read_text())
    final = meta.parent / 'final_val_nll.json'
    r['final_val'] = json.loads(final.read_text()).get('val/nll') if final.exists() else None
    r['curve'], r['val_curve'] = read_curves(meta.parent / 'train_curve.csv')
    window = max(1, int(args.train_window // (r['gbs'] * r['seq_len'])))
    tail = [v for s, v in r['curve'] if s > r['max_steps'] - window]
    r['train_final'] = sum(tail) / len(tail) if tail else None
    r['arm'] = arm_label(r)
    runs.append(r)
  with open(out / 'summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['name', 'arm', 'kind', 'eta_mult', 'eta', 'seed', 'rho',
                                      'shape', 'rec_k0_frac', 'max_steps', 'final_val',
                                      'train_final'], extrasaction='ignore')
    w.writeheader()
    w.writerows(runs)

  by_seed = defaultdict(list)
  for r in runs:
    by_seed[r['seed']].append(r)
  best = {}
  for seed, rs in sorted(by_seed.items()):
    etas = sorted({r['eta_mult'] for r in rs})
    table = defaultdict(dict)
    for r in rs:
      table[r['arm']][r['eta_mult']] = r
    print(f'seed {seed}: final validation NLL (fixed noise); [paired difference to the '
          f'linear tail with the same rho and peak LR]')
    print(f'{"arm":<30}' + ''.join(f'{"eta x" + format(e, "g"):>22}' for e in etas)
          + f'{"best":>16}')
    for arm in sorted(table, key=lambda a: (a.startswith('trunk'), a)):
      cells = []
      for e in etas:
        r = table[arm].get(e)
        if r is None or r['final_val'] is None:
          cells.append(f'{"-":>22}')
          continue
        ref = None
        if r['kind'] == 'tail' and r['shape'] != 'linear':
          ref = table.get(f'rho={r["rho"]:g} linear', {}).get(e)
        diff = (f' [{r["final_val"] - ref["final_val"]:+.4f}]'
                if ref and ref['final_val'] is not None else '')
        cells.append(f'{r["final_val"]:.4f}{diff}'.rjust(22))
      done = [r for r in table[arm].values() if r['final_val'] is not None]
      if done:
        b = min(done, key=lambda r: r['final_val'])
        best[(seed, arm)] = b
        edge = ' (edge)' if b['eta_mult'] in (etas[0], etas[-1]) and len(etas) > 1 else ''
        cells.append(f'{b["final_val"]:.4f}@x{b["eta_mult"]:g}{edge}'.rjust(16))
      print(f'{arm:<30}' + ''.join(cells))
    print()
  print('"(edge)": the best peak LR is at the end of the grid; extend it before concluding.')

  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    print('matplotlib not available; skipped the figure.')
    return
  seed = sorted(by_seed)[0]
  rs = [r for r in by_seed[seed] if r['kind'] == 'tail']
  trunks = {t['name']: t for t in by_seed[seed] if t['kind'] == 'trunk'}
  arms = sorted({r['arm'] for r in rs})
  colors = {a: plt.get_cmap('tab10')(i % 10) for i, a in enumerate(arms)}
  fig, ax = plt.subplots(2, 3, figsize=(18, 9))
  zoom_from = None
  for arm in arms:
    color = colors[arm]
    ls = '--' if 'rho=0.2' in arm else '-'
    pts = sorted((r['eta'], r['final_val']) for r in rs if r['arm'] == arm and r['final_val'])
    if pts:
      ax[0, 0].plot(*zip(*pts), 'o' + ls, color=color, label=arm)
    b = best.get((seed, arm)) or next((r for r in rs if r['arm'] == arm), None)
    if b is None:
      continue
    k = b['total_steps']
    grid = list(range(0, k, max(1, k // 400)))
    ax[1, 2].plot(grid, [multiplier(i, b) for i in grid], ls, color=color, lw=1.2, label=arm)
    trunk = trunks.get(b.get('trunk'))
    label = f'{arm} @x{b["eta_mult"]:g}'
    train = spliced(b, trunk, 'curve')
    if train:
      xs, ys = smooth(train)
      ax[0, 1].plot(xs, ys, ls, color=color, lw=1, label=label)
      ax[0, 2].plot(xs, ys, ls, color=color, lw=1, label=label)
    val = spliced(b, trunk, 'val_curve')
    if val:
      ax[1, 0].plot(*zip(*val), 'o' + ls, color=color, ms=3, lw=1, label=label)
      ax[1, 1].plot(*zip(*val), 'o' + ls, color=color, ms=3, lw=1, label=label)
    zoom_from = min(zoom_from or k, min(r['start_step'] for r in rs if r['arm'] != 'classic cosine (rho=0.9)')
                    if any(r['arm'] != 'classic cosine (rho=0.9)' for r in rs) else k // 2)
  ax[0, 0].set_xscale('log')
  ax[0, 0].set(title='Final validation NLL (full set, fixed noise)', xlabel='peak learning rate')
  ax[0, 1].set(title='Train ELBO, whole run (1% running mean)', xlabel='step', yscale='log')
  ax[0, 2].set(title='Train ELBO, cooldown region', xlabel='step')
  ax[1, 0].set(title='Validation NLL, whole run (in-run 10% subset, fixed noise)',
               xlabel='step', yscale='log')
  ax[1, 1].set(title='Validation NLL, cooldown region', xlabel='step')
  ax[1, 2].set(title='Learning-rate multiplier', xlabel='step')
  if zoom_from is not None:
    for a in (ax[0, 2], ax[1, 1]):
      a.set_xlim(left=zoom_from * 0.95)
      a.autoscale(axis='y', tight=False)
      lines = [l for l in a.get_lines()]
      ys = [y for l in lines for x, y in zip(l.get_xdata(), l.get_ydata()) if x >= zoom_from]
      if ys:
        pad = 0.05 * (max(ys) - min(ys) or 1e-3)
        a.set_ylim(min(ys) - pad, max(ys) + pad)
  for a in ax.flat:
    a.grid(alpha=0.3)
  ax[0, 0].legend(fontsize=7)
  ax[0, 2].legend(fontsize=6)
  ax[1, 1].legend(fontsize=6)
  fig.tight_layout()
  fig.savefig(out / 'cooldown.png', dpi=150)
  print(f'Wrote {out / "summary.csv"} and {out / "cooldown.png"}')


if __name__ == '__main__':
  main()
