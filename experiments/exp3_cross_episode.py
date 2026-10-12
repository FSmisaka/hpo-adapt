"""Experiment 3 — Claim 3: the RL controller gets stronger as experience
accumulates (60 groups x 20 episodes).

Design
------
The framework's intended usage: for each (task, seed), run 20 CONSECUTIVE
episodes of the closed loop with the Q-table and the SQLite Agent Memory
shared across episodes. Tasks: synthetic_mlp + branin (the real-ML
rf_digits task is too expensive for 20-episode sweeps); methods:
static-tpe (flat reference), adapt-random-action (control) and adapt-rl
(epsilon 1.0 -> 0.05, decay 0.95 — the controller that can learn).

Expected (10 seeds): adapt-rl's per-episode best improves with episode
index — on synthetic_mlp from 0.892 (episodes 0-4 mean) to 0.949
(episodes 15-19), while the random control stays flat (0.867 -> 0.855);
on Branin (minimize) 0.734 -> 0.524 while the random control drifts the
wrong way (0.768 -> 0.819). Honest caveat: the curves are noisy (each
episode only offers ~2 decisions, so the learning granularity is coarse).

Run
---
    make exp3                                     # full matrix (~10 min)
    python experiments/exp3_cross_episode.py      # equivalent
    python experiments/exp3_cross_episode.py --smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.common import (
    ADAPT_RANDOM_ACTION,
    ADAPT_RL,
    STATIC_TPE,
    apply_smoke,
    base_parser,
    execute,
    group_unit,
)

METHODS = [
    STATIC_TPE,
    ADAPT_RANDOM_ACTION,
    ADAPT_RL,
]

TASKS = ["synthetic_mlp", "branin"]


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--episodes", type=int, default=20, help="episodes per (task, seed) group")
    args = parser.parse_args()
    if args.smoke:
        apply_smoke(args, tasks=TASKS, episodes=3)

    units = [
        group_unit("episodes", task, method, seed, args.budget, args.tag, args.episodes)
        for task in TASKS
        for method in METHODS
        for seed in range(args.seeds)
    ]
    execute(units, tag=args.tag, jobs=args.jobs, summary_name="exp3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
