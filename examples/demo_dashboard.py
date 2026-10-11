"""Demo: plug a real, compute-heavy ML task into hpo-adapt (~5-6 min run).

Task: sklearn HistGradientBoosting on a 15k-sample synthetic classification
dataset. One trial = one full training + holdout evaluation (1-7 s
depending on boosting rounds and tree size), so the closed loop runs for
about five to six minutes — long enough to watch the dashboard update live.

The search space is deliberately calibrated so the adaptation story is
visible and every round earns a real gain:

    - the task is genuinely hard (5 clusters/class, label noise), so more
      boosting rounds (``max_iter``) and larger trees (``max_leaf_nodes``)
      keep paying off — both optima lie far ABOVE the default-centered
      initial ranges [20, 150] and [10, 37];
    - round 0's good trials therefore pile up against the upper boundaries
      -> the Diagnostic Engine sees boundary concentration, and the RL
      controller shifts/expands the ranges round after round, unlocking the
      next accuracy step each time;
    - ``learning_rate`` is pinned near its known-good value (a practitioner
      who already knows the lr ballpark), so capacity — not noise — drives
      the diagnosis.

The wall time is real compute (bigger configurations genuinely train
longer), not padding.

The task contract is the same as for any user task:

    1. train_evaluate(params) -> float   (fit once, return holdout accuracy)
    2. a declarative search space        (bounds = hard adaptation limits;
                                          default = center of the initial
                                          compact range)

Run:
    python examples/demo_dashboard.py                  # ~5-6 min, 4 rounds
    python examples/demo_dashboard.py --fresh          # empty agent memory
    python examples/demo_dashboard.py --fast           # ~1 min smoke run
"""

import argparse
import sys
import warnings
from pathlib import Path

import hpo_adapt as ha
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

_DATASETS: dict[int, tuple] = {}
_N_SAMPLES = 15000


def _data(n_samples: int):
    """Deterministic dataset (built once per size, cached across trials)."""
    global _DATASETS
    if n_samples not in _DATASETS:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=100,
            n_informative=50,
            n_redundant=25,
            n_clusters_per_class=5,
            class_sep=0.7,
            flip_y=0.06,
            random_state=7,
        )
        _DATASETS[n_samples] = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=0
        )
    return _DATASETS[n_samples]


def train_evaluate(params: dict) -> float:
    """One trial: fit a gradient-boosted ensemble under `params`."""
    Xtr, Xte, ytr, yte = _data(_N_SAMPLES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = HistGradientBoostingClassifier(
            max_iter=int(params["max_iter"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            learning_rate=float(params["learning_rate"]),
            early_stopping=False,  # deterministic budget: max_iter rounds
            random_state=0,
        )
        model.fit(Xtr, ytr)
    return float(model.score(Xte, yte))


def main() -> int:
    global _N_SAMPLES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-trials", type=int, default=120)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--trials-per-phase", type=int, default=30)
    parser.add_argument("--samples", type=int, default=None, help="dataset size (default 15000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-path", default="examples/hgb_demo_memory.sqlite")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--fast", action="store_true", help="small dataset + budget, ~1 min")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    if args.fast:
        _N_SAMPLES = 6000
        args.total_trials = 60
        args.max_rounds = 2
        args.trials_per_phase = 30  # >= importance threshold so adaptation triggers
    if args.samples:
        _N_SAMPLES = args.samples
    if args.fresh:
        Path(args.memory_path).unlink(missing_ok=True)

    run = ha.run(
        train_evaluate,
        space=dict(
            learning_rate=ha.Float(0.08, 0.12, log=True, default=0.1),
            max_iter=ha.Int(20, 300, default=80),
            max_leaf_nodes=ha.Int(10, 64, default=24),
        ),
        direction="maximize",
        total_trials=args.total_trials,
        max_rounds=args.max_rounds,
        trials_per_phase=args.trials_per_phase,
        task_name="hgb_demo",
        model_family="sklearn.HistGradientBoosting",
        dataset=f"make_classification_{_N_SAMPLES}x100",
        objective_name="holdout_accuracy",
        seed=args.seed,
        memory_path=args.memory_path,
        dashboard=not args.no_dashboard,
        open_browser=args.open_browser,
    )

    print(f"\nbest holdout accuracy = {run.best_value:.4f}")
    print(f"best params           = {run.best_params}")
    print(f"memory kept at        = {Path(args.memory_path).resolve()}")
    if run.url:
        run.wait()  # keep the dashboard reachable until you press Enter
    return 0


if __name__ == "__main__":
    sys.exit(main())
