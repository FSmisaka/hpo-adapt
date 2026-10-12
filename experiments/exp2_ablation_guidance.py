"""Experiment 2 — Claim 2 (ablation): the superiority comes from
"diagnosis-guided" decisions, not from "just moving something".

Design
------
The SAME closed loop, with the decision maker swapped between four
policies (3 tasks x 10 seeds, equal 120-trial budget):

- adapt-random-action: uniform random action selection (unguided control);
- adapt-rule-only:     always follow the rule engine's recommendation;
- adapt-rl-default:   Q-learning with spec-default epsilon 1.0 -> 0.05
                      (behaves like the random control within a single
                      episode: only ~2 decisions per episode, so the
                      Q-table never gets exploited — its value shows up
                      in experiment 3, across episodes);
- adapt-rl-guided:    Q-learning with epsilon_0 = 0.2 (guided).

Expected (10 seeds): the value of guidance shows up in VARIANCE —
rule-only / rl-guided reach 1.000±0.000 on synthetic_mlp (worst seed =
best seed) while the random control gets 0.873±0.134 (~6x spread); on
Branin the ordering holds (0.498 < 0.596 < 0.776). Honest caveat: the
random control's MEAN on synthetic_mlp also beats static-tpe (7 actions,
nearly half target learning_rate — blind wiggling sometimes repairs it by
luck); guidance's contribution is turning "occasionally repaired" into
"always repaired".

Run
---
    make exp2                                      # full matrix (~7 min)
    python experiments/exp2_ablation_guidance.py   # equivalent
    python experiments/exp2_ablation_guidance.py --smoke
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
    apply_smoke,
    base_parser,
    execute,
    single_unit,
)

METHODS = [
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
    execute(units, tag=args.tag, jobs=args.jobs, summary_name="exp2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
