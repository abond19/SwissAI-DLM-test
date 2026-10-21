#!/usr/bin/env python3
"""Figures for the SCION step-1 (base-model) tuning runs.

Reads the same run directories as bst_scion_analysis.py (run_meta.json,
final_val_nll.json, train_curve.csv), keeps the base budget (multiplier 1), and
writes two PNGs to <root>/analysis/:

  step1_overview.png   (a) final validation NLL vs beta, one line per batch size;
                       (b)-(d) radii / momentum / warmup as the change from the
                       baseline, with the seed-noise band; (e) every one-change
                       run ranked by its change; (f) the baseline's seed spread.
  step1_curves.png     train ELBO of each one-change run minus the baseline's,
                       step by step.  The runs share data and noise, so this shows
                       when a change starts to help or hurt.

The baseline is the run at --base-gbs / --base-beta with the default momentum,
radii, no warmup and --base-seed.  Lower is better everywhere.
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bst_scion_analysis import load_runs, summarize  # noqa: E402

# Validated categorical palette (fixed order), slots 1-8.
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GROUPS = ['embed', 'matrix', 'bias', 'norm']
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'


def classify(r, base):
  """-> (category, label, x) for a run differing from the baseline in one way."""
  if r['gbs'] != base['gbs'] or not math.isclose(r['beta'], base['beta']):
    return None
  diffs = []
  if not math.isclose(float(r['momentum']), float(base['momentum'])):
    diffs.append('momentum')
  if r['radii'] != base['radii']:
    diffs.append('radii')
  if float(r.get('warmup_frac', 0) or 0) != 0:
    diffs.append('warmup')
  if r['seed'] != base['seed']:
    diffs.append('seed')
  if len(diffs) != 1:
    return None
  kind = diffs[0]
  if kind == 'radii':
    a = [float(x) for x in r['radii'].split(':')]
    b = [float(x) for x in base['radii'].split(':')]
    changed = [i for i in range(4) if not math.isclose(a[i], b[i])]
    if len(changed) == 1:
      i = changed[0]
      return ('radii', GROUPS[i], a[i] / b[i])
    label = '+'.join(f'{GROUPS[i]} x{a[i] / b[i]:g}' for i in changed)
    return ('radii-combined', label, None)
  if kind == 'momentum':
    return ('momentum', f'alpha {float(r["momentum"]):g}', float(r['momentum']))
  if kind == 'warmup':
    return ('warmup', f'warmup {100 * float(r["warmup_frac"]):g}%', float(r['warmup_frac']))
  return ('seed', f'seed {r["seed"]}', r['seed'])


def style(ax, title, xlabel, ylabel):
  ax.set_title(title, loc='left', fontsize=10, color=INK)
  ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
  ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
  ax.grid(color=GRID, lw=0.8)
  ax.tick_params(colors=MUTED, labelsize=8)
  for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
  for s in ('left', 'bottom'):
    ax.spines[s].set_color(GRID)


def band(ax, noise):
  ax.axhspan(-2 * noise, 2 * noise, color='#cfcdc6', alpha=0.45, lw=0, zorder=0)
  ax.axhline(0, color=MUTED, lw=1, zorder=1)


def smooth(vals, window):
  out, total = [], 0.0
  for i, v in enumerate(vals):
    total += v
    if i >= window:
      total -= vals[i - window]
    out.append(total / min(i + 1, window))
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('root')
  p.add_argument('--base-gbs', type=int, default=256)
  p.add_argument('--base-beta', type=float, default=6e-5)
  p.add_argument('--base-momentum', type=float, default=0.06)
  p.add_argument('--base-radii', default='3500:336:90:6.8')
  p.add_argument('--base-seed', type=int, default=4)
  p.add_argument('--noise', type=float, default=None,
                 help='seed std of the final NLL; default: from the baseline seeds, else 0.003')
  p.add_argument('--out', type=Path, default=None)
  args = p.parse_args()
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  out = args.out or Path(args.root) / 'analysis'
  out.mkdir(parents=True, exist_ok=True)
  runs = [r for r in load_runs([args.root]) if float(r['multiplier']) == 1.0]
  for r in runs:
    summarize(r, 250e6, 100)
  base = next((r for r in runs if r['gbs'] == args.base_gbs
               and math.isclose(r['beta'], args.base_beta)
               and math.isclose(float(r['momentum']), args.base_momentum)
               and r['radii'] == args.base_radii and r['seed'] == args.base_seed
               and float(r.get('warmup_frac', 0) or 0) == 0), None)
  if base is None or base['final_val'] is None:
    sys.exit('Baseline run not found or not evaluated; check the --base-* arguments.')
  b0 = base['final_val']

  seeds = [r for r in runs if r['final_val'] is not None
           and (classify(r, base) or ('',))[0] == 'seed']
  seed_vals = [b0] + [r['final_val'] for r in seeds]
  if args.noise is not None:
    noise, noise_src = args.noise, 'given'
  elif len(seed_vals) >= 2:
    m = sum(seed_vals) / len(seed_vals)
    noise = math.sqrt(sum((v - m) ** 2 for v in seed_vals) / (len(seed_vals) - 1))
    noise_src = f'from {len(seed_vals)} seeds'
  else:
    noise, noise_src = 0.003, 'assumed (no seed runs yet)'

  one_change = []
  for r in runs:
    c = classify(r, base)
    if c and r['final_val'] is not None:
      one_change.append((c, r))

  fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
  fig.patch.set_facecolor('white')

  # (a) beta x batch size, default everything else
  by_gbs = defaultdict(list)
  for r in runs:
    if (r['final_val'] is not None and r['radii'] == args.base_radii and r['seed'] == args.base_seed
        and math.isclose(float(r['momentum']), args.base_momentum)
        and float(r.get('warmup_frac', 0) or 0) == 0):
      by_gbs[r['gbs']].append((r['beta'], r['final_val']))
  a = ax[0, 0]
  for i, gbs in enumerate(sorted(by_gbs)):
    pts = sorted(by_gbs[gbs])
    a.plot(*zip(*pts), '-o', color=SLOTS[i], lw=2, ms=6, label=f'GBS {gbs}')
    best = min(pts, key=lambda q: q[1])
    a.plot(*best, 'D', color=SLOTS[i], ms=10, mec='white', mew=2, zorder=5)
  a.set_xscale('log')
  betas = sorted({b for pts in by_gbs.values() for b, _ in pts})
  a.set_xticks(betas, [f'{b * 1e5:g}' for b in betas], fontsize=7)
  a.minorticks_off()
  style(a, '(a) Final validation NLL vs step size, per batch size', 'Frank-Wolfe step size beta (x 1e-5)',
        'final val NLL (fixed noise)')
  a.legend(fontsize=8, frameon=False)

  # (b) radii: one line per group, change vs radius multiplier
  a = ax[0, 1]
  band(a, noise)
  for i, g in enumerate(GROUPS):
    pts = sorted([(x, r['final_val'] - b0) for (cat, lab, x), r in one_change
                  if cat == 'radii' and lab == g] + [(1.0, 0.0)])
    if len(pts) > 1:
      a.plot(*zip(*pts), '-o', color=SLOTS[i], lw=2, ms=7, label=g)
  for (cat, lab, _), r in one_change:
    if cat == 'radii-combined':
      a.plot(1.0, r['final_val'] - b0, '*', color=SLOTS[4], ms=14, mec='white', mew=1.5)
      a.annotate(lab, (1.0, r['final_val'] - b0), xytext=(6, 0), textcoords='offset points',
                 fontsize=7, color=INK, va='center')
  a.set_xscale('log', base=2)
  xs = sorted({x for (c, _, x), _r in one_change if c == 'radii'} | {1.0})
  a.set_xticks(xs, [f'{x:g}x' for x in xs])
  a.minorticks_off()
  style(a, '(b) Radii: change from baseline (one group at a time; star = combined)',
        'radius multiplier (x default)', 'val NLL - baseline')
  a.legend(fontsize=8, frameon=False, ncol=2)

  # (c) momentum, (d) warmup
  for a, cat, title, xlabel, x0, color in (
      (ax[0, 2], 'momentum', '(c) Momentum', 'momentum alpha (new-gradient weight)',
       args.base_momentum, SLOTS[5]),
      (ax[1, 0], 'warmup', '(d) Warmup', 'warmup (fraction of the run)', 0.0, SLOTS[6])):
    band(a, noise)
    pts = sorted([(x, r['final_val'] - b0) for (c, lab, x), r in one_change if c == cat]
                 + [(x0, 0.0)])
    a.plot(*zip(*pts), '-o', color=color, lw=2, ms=7)
    if cat == 'momentum':
      a.set_xscale('log', base=2)
      a.set_xticks([x for x, _ in pts], [f'{x:g}' for x, _ in pts])
      a.minorticks_off()
    style(a, f'{title}: change from baseline', xlabel, 'val NLL - baseline')

  # (e) ranking of every one-change run
  a = ax[1, 1]
  ranked = sorted(((r['final_val'] - b0, c, f'radius {lab} x{x:g}' if c == 'radii' else lab)
                   for (c, lab, x), r in one_change if c != 'seed'), key=lambda t: t[0])
  colors = {'radii': SLOTS[0], 'radii-combined': SLOTS[4], 'momentum': SLOTS[5],
            'warmup': SLOTS[6]}
  names = []
  for i, (d, c, lab) in enumerate(ranked):
    names.append(lab)
    a.barh(i, d, color=colors.get(c, MUTED), height=0.7)
  a.axvspan(-2 * noise, 2 * noise, color='#cfcdc6', alpha=0.45, lw=0, zorder=0)
  a.axvline(0, color=MUTED, lw=1)
  a.set_yticks(range(len(names)), names, fontsize=7)
  a.invert_yaxis()
  style(a, '(e) All one-change runs, best first', 'val NLL - baseline (negative = better)', '')

  # (f) seed spread at the baseline
  a = ax[1, 2]
  a.axhspan(b0 - 2 * noise, b0 + 2 * noise, color='#cfcdc6', alpha=0.45, lw=0)
  a.plot([args.base_seed], [b0], 'o', color=SLOTS[7], ms=9)
  for r in seeds:
    a.plot([r['seed']], [r['final_val']], 'o', color=SLOTS[7], ms=9)
  a.set_xticks(sorted({args.base_seed} | {r['seed'] for r in seeds}))
  style(a, f'(f) Baseline across seeds: noise std {noise:.4f} ({noise_src})', 'seed',
        'final val NLL')

  fig.suptitle(f'SCION step 1 at T0 = 2B tokens. Baseline: GBS {args.base_gbs}, beta '
               f'{args.base_beta:g}, alpha {args.base_momentum:g}, radii {args.base_radii} '
               f'(val NLL {b0:.4f}). Gray band: +-2 seed std.', fontsize=11, color=INK, x=0.01,
               ha='left')
  fig.tight_layout(rect=(0, 0, 1, 0.96))
  fig.savefig(out / 'step1_overview.png', dpi=150)

  # Paired difference curves
  base_curve = dict((s, v) for s, _, v in base['curve']['train'])
  cats = [('radii', 'Radii (one group)'), ('radii-combined', 'Radii (combined)'),
          ('momentum', 'Momentum'), ('warmup', 'Warmup')]
  fig, axes = plt.subplots(1, len(cats), figsize=(18, 4.6), sharey=True)
  for a, (cat, title) in zip(axes, cats):
    a.axhline(0, color=MUTED, lw=1)
    i = 0
    for (c, lab, x), r in one_change:
      if c != cat or not r['curve']['train']:
        continue
      pts = [(s, v - base_curve[s]) for s, _, v in r['curve']['train'] if s in base_curve]
      if not pts:
        continue
      window = max(1, len(pts) // 50)
      ys = smooth([v for _, v in pts], window)
      label = f'{lab} x{x:g}' if cat == 'radii' else lab
      color = SLOTS[GROUPS.index(lab)] if cat == 'radii' else SLOTS[i % len(SLOTS)]
      ls = '-' if (cat != 'radii' or x > 1) else '--'
      a.plot([s for s, _ in pts], ys, ls, color=color, lw=1.6, label=label)
      i += 1
    style(a, title, 'step', 'train ELBO - baseline (2% running mean)')
    if a.get_lines()[1:]:
      a.legend(fontsize=7, frameon=False)
  fig.suptitle('Train ELBO minus baseline during training (same data and noise; below 0 = '
               'better). Radii: solid = larger, dashed = smaller.', fontsize=11, color=INK,
               x=0.01, ha='left')
  fig.tight_layout(rect=(0, 0, 1, 0.92))
  fig.savefig(out / 'step1_curves.png', dpi=150)
  print(f'Wrote {out / "step1_overview.png"} and {out / "step1_curves.png"}')


if __name__ == '__main__':
  main()
