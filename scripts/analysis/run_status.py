#!/usr/bin/env python3
"""Status of every run under one or more experiment output roots.

For each run directory (one with run_meta.json, as written by the submit_*.py
launchers) it reports one of:

  evaluated   training finished (DONE) and final_val_nll.json exists
  done        training finished; not evaluated yet (run run_eval_final_val_nll.sh)
  running     its job is running now
  pending     its job is queued (PD), e.g. waiting for nodes or a dependency
  failed      not finished, not queued; the last Slurm state says why
  stopped     trained partway, no job and no Slurm record (e.g. run off the cluster)
  not started no job in the queue and no Slurm record (e.g. never submitted)

plus the last step reached (from train_curve.csv) against max_steps, the last
Slurm state and job id, and for failed runs the last error line of the log.
Standard library only.  Slurm columns need squeue/sacct, so run it on a login
node; elsewhere they are left blank.

  python3 scripts/analysis/run_status.py $SCRATCH/SwissAI-DLM-data/outputs/bst_scion/scion_bst_L12H768_T0_2B
  python3 scripts/analysis/run_status.py <root> --only failed      # just the problem runs
"""

import argparse
import getpass
import glob
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def sh(cmd):
  try:
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True, timeout=60)
    return out.stdout if out.returncode == 0 else ''
  except (OSError, subprocess.TimeoutExpired):
    return ''


def queue_states():
  """job name -> (state, job id) for this user's queued/running jobs."""
  states = {}
  for line in sh(['squeue', '-u', getpass.getuser(), '--noheader', '--format=%j|%t|%i']).splitlines():
    name, state, job = (line.split('|') + ['', ''])[:3]
    states[name.strip()] = (state.strip(), job.strip())
  return states


def last_slurm_states(days):
  """job name -> (state, job id) of the most recent job with that name."""
  out = sh(['sacct', '-u', getpass.getuser(), '-X', '--noheader', '--parsable2',
            f'--starttime=now-{days}days', '--format=JobName%200,State,JobID'])
  states = {}
  for line in out.splitlines():  # chronological, so later lines win
    parts = line.split('|')
    if len(parts) >= 3:
      states[parts[0]] = (parts[1].split()[0], parts[2])
  return states


def last_step(run_dir):
  path = run_dir / 'train_curve.csv'
  if not path.exists():
    return None
  last = None
  with open(path) as f:
    for line in f:
      if line.startswith('train,'):
        last = line
  return int(last.split(',')[1]) if last else None


def last_error(log_dir, name, job):
  if not job:
    return ''
  for path in glob.glob(str(log_dir / f'{name}_{job}.err')):
    with open(path, errors='replace') as f:
      lines = [l.strip() for l in f if l.strip()]
    for l in reversed(lines):
      if any(k in l for k in ('Error', 'error', 'Traceback', 'CANCELLED', 'Killed', 'OOM')):
        return l[:140]
    return lines[-1][:140] if lines else ''
  return ''


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('roots', nargs='+')
  p.add_argument('--only', default=None,
                 help='show only this status (evaluated, done, running, pending, failed, not started)')
  p.add_argument('--days', type=int, default=14, help='how far back to look in sacct')
  args = p.parse_args()

  queued = queue_states()
  history = last_slurm_states(args.days)
  rows = []
  for root in args.roots:
    for meta_path in sorted(Path(root).glob('*/run_meta.json')):
      run_dir = meta_path.parent
      meta = json.loads(meta_path.read_text())
      name = meta.get('name', run_dir.name)
      max_steps = meta.get('max_steps')
      step = last_step(run_dir)
      done = (run_dir / 'DONE').exists()
      evaluated = (run_dir / 'final_val_nll.json').exists()
      q_state, q_job = queued.get(name, ('', ''))
      h_state, h_job = history.get(name, ('', ''))
      if done and evaluated:
        status = 'evaluated'
      elif done:
        status = 'done'
      elif q_state == 'R':
        status = 'running'
      elif q_state:
        status = 'pending'
      elif h_state:
        status = 'failed'
      elif step:
        status = 'stopped'  # trained partway, no job now (or no Slurm record here)
      else:
        status = 'not started'
      log_dir = REPO_ROOT / 'logs' / meta.get('experiment', Path(root).name)
      err = last_error(log_dir, name, h_job) if status == 'failed' else ''
      rows.append(dict(status=status, name=name, step=step, max_steps=max_steps,
                       slurm=q_state or h_state, job=q_job or h_job, err=err))

  shown = [r for r in rows if args.only in (None, r['status'])]
  width = max([len(r['name']) for r in shown] + [10])
  print(f'{"status":<12} {"run":<{width}} {"steps":>13} {"slurm":<12} {"job":>9}')
  for r in shown:
    steps = f'{r["step"] if r["step"] is not None else "-"}/{r["max_steps"] or "?"}'
    print(f'{r["status"]:<12} {r["name"]:<{width}} {steps:>13} {r["slurm"]:<12} {r["job"]:>9}')
    if r['err']:
      print(f'{"":<12} -> {r["err"]}')
  counts = Counter(r['status'] for r in rows)
  print('\n' + ', '.join(f'{k}: {counts[k]}' for k in
                         ('evaluated', 'done', 'running', 'pending', 'failed', 'stopped',
                          'not started')
                         if counts[k]) + f'  (total {len(rows)})')
  if counts['failed'] or counts['stopped']:
    print('Failed runs resume when resubmitted with the same launcher command '
          '(finished and queued runs are skipped).')
  if counts['done']:
    print('Done but not evaluated: run scripts/analysis/run_eval_final_val_nll.sh on the root.')


if __name__ == '__main__':
  main()
