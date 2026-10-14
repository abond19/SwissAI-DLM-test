#!/usr/bin/env bash
set -euo pipefail

# Step 1 of the SCION BST experiment: re-tune the base model (L12/H768,
# T0 = 2B tokens) with fp32 master weights.  Dry run unless --submit is passed.
#
#   scripts/training/submit_scion_bst_step1.sh <stage> [--submit] [extra launcher args]
#
# Stages, in order.  Later stages start from the best point so far, given by
# BEST_GBS, BEST_BETA, BEST_MOMENTUM and BEST_RADII (embed:matrix:bias:norm).
#   grid      beta x batch: the full beta grid at GBS 256, 5 betas at 128 and 512 (18 runs)
#   radii     each radius at 1/2x and 2x, one at a time (8 runs)
#   momentum  alpha at 1/2x and 2x (2 runs)
#   refine    5 betas around BEST_BETA, if radii or momentum moved (5 runs)
#   seeds     seeds 5 and 6 at the best point, for the noise band (2 runs)
# Rank with scripts/analysis/eval_final_val_nll.py + bst_scion_analysis.py.

STAGE="${1:?usage: $0 grid|radii|momentum|refine|seeds [--submit] [launcher args]}"
shift
LAUNCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_scion_bst.py"

# Shared beta grid (also used at every budget in step 2): ratio ~1.25-1.33,
# reaching down to the T^(-1/3) prediction at 15x (0.41 x 6e-5 = 2.4e-5).
BETA_GRID="${BETA_GRID:-2e-5 3e-5 4e-5 5e-5 6e-5 7.5e-5 9e-5 1.2e-4}"
BEST_GBS="${BEST_GBS:-256}"
BEST_BETA="${BEST_BETA:-6e-5}"
BEST_MOMENTUM="${BEST_MOMENTUM:-0.06}"
BEST_RADII="${BEST_RADII:-3500:336:90:6.8}"

half_double() {  # $1 = embed:matrix:bias:norm -> the 8 one-at-a-time variants
  python3 - "$1" <<'EOF'
import sys
r = [float(x) for x in sys.argv[1].split(':')]
out = []
for i in range(4):
  for f in (0.5, 2.0):
    v = list(r)
    v[i] *= f
    out.append(':'.join(f'{x:g}' for x in v))
print(' '.join(out))
EOF
}

case "${STAGE}" in
  grid)
    python3 "${LAUNCH}" --multipliers 1 --batch-sizes 256 --betas ${BETA_GRID} \
      --momenta "${BEST_MOMENTUM}" --radii "${BEST_RADII}" "$@"
    python3 "${LAUNCH}" --multipliers 1 --batch-sizes 128 512 \
      --betas 3e-5 4.5e-5 6e-5 7.5e-5 9e-5 \
      --momenta "${BEST_MOMENTUM}" --radii "${BEST_RADII}" "$@"
    ;;
  radii)
    python3 "${LAUNCH}" --multipliers 1 --batch-sizes "${BEST_GBS}" --betas "${BEST_BETA}" \
      --momenta "${BEST_MOMENTUM}" --radii $(half_double "${BEST_RADII}") "$@"
    ;;
  momentum)
    MOMENTA=$(python3 -c "m=${BEST_MOMENTUM}; print(f'{m/2:g} {m*2:g}')")
    python3 "${LAUNCH}" --multipliers 1 --batch-sizes "${BEST_GBS}" --betas "${BEST_BETA}" \
      --momenta ${MOMENTA} --radii "${BEST_RADII}" "$@"
    ;;
  refine)
    BETAS=$(python3 -c "b=${BEST_BETA}; print(' '.join(f'{b*f:.3g}' for f in (0.67, 0.8, 1.0, 1.25, 1.5)))")
    python3 "${LAUNCH}" --multipliers 1 --batch-sizes "${BEST_GBS}" --betas ${BETAS} \
      --momenta "${BEST_MOMENTUM}" --radii "${BEST_RADII}" "$@"
    ;;
  seeds)
    python3 "${LAUNCH}" --multipliers 1 --batch-sizes "${BEST_GBS}" --betas "${BEST_BETA}" \
      --momenta "${BEST_MOMENTUM}" --radii "${BEST_RADII}" --seeds 5 6 "$@"
    ;;
  *)
    echo "Unknown stage '${STAGE}' (grid|radii|momentum|refine|seeds)." >&2
    exit 1
    ;;
esac
