#!/usr/bin/env bash
set -euo pipefail

# Step 2 of the SCION BST experiment: larger token budgets T = m * T0 with the
# BST batch B = B_base * m^(2/3), on the same beta grid as step 1.  Dry run
# unless --submit is passed.
#
#   scripts/training/submit_scion_bst_step2.sh "<multipliers>" [--submit] [extra launcher args]
#
# Use the step-1 optimum: B_BASE, BEST_MOMENTUM, BEST_RADII.  BETAS defaults to
# the full step-1 grid.  To save compute at the expensive budgets (8x and up are
# ~85% of step 2), run a 3-point bracket first, e.g.
#   BETAS="2.5e-5 4e-5 6e-5" scripts/training/submit_scion_bst_step2.sh "8 12 15" --submit
# (base beta, the T^(-1/3) prediction at 15x, and their midpoint), then let
# bst_scion_analysis.py --suggest name the next betas until each minimum is
# bracketed.

MULTIPLIERS="${1:?usage: $0 \"<multipliers>\" [--submit] [launcher args]}"
shift
LAUNCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_scion_bst.py"

BETAS="${BETAS:-2e-5 3e-5 4e-5 5e-5 6e-5 7.5e-5 9e-5 1.2e-4}"
B_BASE="${B_BASE:-256}"
BEST_MOMENTUM="${BEST_MOMENTUM:-0.06}"
BEST_RADII="${BEST_RADII:-3500:336:90:6.8}"

python3 "${LAUNCH}" --multipliers ${MULTIPLIERS} --b-base "${B_BASE}" --betas ${BETAS} \
  --momenta "${BEST_MOMENTUM}" --radii "${BEST_RADII}" "$@"
