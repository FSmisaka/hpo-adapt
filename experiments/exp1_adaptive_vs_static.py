"""Experiment 1 — Claim 1: when the initial setup is flawed, the adaptive
closed loop beats equally-budgeted static HPO (180 runs).

Design
------
6 methods x 3 tasks x 10 seeds under an IDENTICAL total budget of 120
trials per run. Static baselines spend the whole budget in one phase with
a fixed setup; adaptive variants run phases of 40 trials with diagnosis
and setup revision in between.

The tasks' initial setups (generated automatically by the
"default-centered range" convention) each truncate the optimum:

- synthetic_mlp: optimal lr = 0.45 lies OUTSIDE the initial [1e-3, 1e-1];
- branin: the initial x2 interval misses two of the three global optima;
- rf_digits: the initial setup already sits near the CV-accuracy plateau
  (nothing to repair -> the claim here is "do no harm + observability").

Expected (10 seeds): guided adaptation 1.000±0.000 vs static-tpe
0.755±0.003 on synthetic_mlp (+32% relative); 0.498±0.198 vs 0.896±0.010
on Branin (global minimum 0.398); all methods within noise on rf_digits.

Run
---
    make exp1                                        # full matrix (~10 min)
    python experiments/exp1_adaptive_vs_static.py    # equivalent
    python experiments/exp1_adaptive_vs_static.py --smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.common import (
    ADAPT_RANDOM_ACTION,
    ADAPT_RL_DEFAULT,
    ADAPT_RL_GUIDED,
    ADAPT_RULE_ONLY,
    STATIC_RANDOM,
    STATIC_TPE,
    apply_smoke,
    base_parser,
    execute,
    single_unit,
)

METHODS = [
    STATIC_RANDOM,
    STATIC_TPE,
    ADAPT_RANDOM_ACTION,
    ADAPT_RULE_ONLY,
    ADAPT_RL_DEFAULT,
    ADAPT_RL_GUIDED,
]

TASKS = ["synthetic_mlp", "branin", "rf_digits"]


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--tasks", nargs="*", default=TASKS, choices=TASKS)
    args = parser.parse_args()
    if args.smoke:
        apply_smoke(args, tasks=["synthetic_mlp", "branin"])

    units = [
        single_unit("main", task, method, seed, args.budget, args.tag)
        for task in args.tasks
        for method in METHODS
        for seed in range(args.seeds)
    ]
    execute(units, tag=args.tag, jobs=args.jobs, summary_name="exp1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
