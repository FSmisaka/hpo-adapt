"""Experiment 4 — Claim 4: the observe-decide layer is cheap enough to be
"almost free" (overhead measurement on the real-ML task).

Design
------
All 6 methods x 10 seeds on rf_digits ONLY — the one task where every
trial genuinely trains models (RandomForest + 3-fold CV), so wall-clock
means something. The pipeline's ``timings`` attribute splits each run into
HPO time (model training + Optuna) versus adapt time (diagnostics, rule
engine, Q-updates, memory writes, agent execution); this experiment
reports the adapt share.

Expected (10 seeds): the adaptive methods' observe-decide layer costs
~1.4% of wall-clock on average (worst case ~1.7% over all adaptive runs);
the remainder is real model training. This is the "closed-loop
automation is nearly free" claim.

Run
---
    make exp4                              # full matrix (~5 min)
    python experiments/exp4_overhead.py    # equivalent
    python experiments/exp4_overhead.py --smoke
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
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
    RESULTS_ROOT,
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

TASK = "rf_digits"


def main() -> int:
    parser = base_parser(__doc__)
    args = parser.parse_args()
    if args.smoke:
        apply_smoke(args, tasks=[TASK])

    units = [
        single_unit("main", TASK, method, seed, args.budget, args.tag)
        for method in METHODS
        for seed in range(args.seeds)
    ]
    rows = execute(units, tag=args.tag, jobs=args.jobs, summary_name="exp4")

    # ---------------------------------------------------------- aggregation
    shares: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        total = r["total_time"] or 1e-9
        shares[r["method"]].append(r["adapt_time"] / total)

    out_path = RESULTS_ROOT / args.tag / "overhead_summary.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["task", "method", "adapt_share_mean", "adapt_share_max"]
        )
        writer.writeheader()
        for method in sorted(shares):
            vals = shares[method]
            writer.writerow(
                {
                    "task": TASK,
                    "method": method,
                    "adapt_share_mean": statistics.mean(vals),
                    "adapt_share_max": max(vals),
                }
            )
    print(f"[exp4] wrote {out_path}")

    adaptive = [v for m, vals in shares.items() if m.startswith("adapt-") for v in vals]
    if adaptive:
        print(
            f"[exp4] observe+decide layer on {TASK}: "
            f"{statistics.mean(adaptive) * 100:.1f}% of wall-clock on average "
            f"(worst case {max(adaptive) * 100:.1f}%, {len(adaptive)} adaptive runs)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
