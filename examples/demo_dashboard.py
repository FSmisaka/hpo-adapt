"""Demo: plug a real ML task into hpo-adapt with ~10 lines of code.

The task (the required "basic task format"):

    1. train_evaluate(params) -> float   (SVC(RBF) + 3-fold CV on sklearn digits)
    2. a declarative search space        (C, gamma as log floats, kernel choices)

Bounds are the hard constraints hpo-adapt may adapt within; ``default``
values center the initial (automatically derived) search space. Here the
optimal C (~6-10) sits at the upper edge of the default-centered initial
range, so the closed loop should discover the boundary concentration and
expand the range — watch it happen live in the dashboard.

Run:
    python examples/demo_dashboard.py
    python examples/demo_dashboard.py --total-trials 60 --max-rounds 3   # faster
    python examples/demo_dashboard.py --fresh                           # fresh memory
"""

import argparse
import sys
from pathlib import Path

import hpo_adapt as ha
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_X, _Y = load_digits(return_X_y=True)


def train_evaluate(params: dict) -> float:
    """One trial: train an RBF-SVM under `params`, return mean 3-fold CV accuracy."""
    model = make_pipeline(
        StandardScaler(),
        SVC(
            C=float(params["C"]),
            gamma=float(params["gamma"]),
            kernel=str(params["kernel"]),
            random_state=0,
        ),
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    return float(cross_val_score(model, _X, _Y, cv=cv, n_jobs=-1).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-trials", type=int, default=90)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--trials-per-phase", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-path", default="examples/svc_digits_memory.sqlite")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    if args.fresh:
        Path(args.memory_path).unlink(missing_ok=True)

    run = ha.run(
        train_evaluate,
        space=dict(
            C=ha.Float(0.01, 100.0, log=True, default=1.0),
            gamma=ha.Float(1e-6, 1.0, log=True, default=1e-3),
            kernel=ha.Choice(["rbf", "poly", "sigmoid"], default="rbf"),
        ),
        direction="maximize",
        total_trials=args.total_trials,
        max_rounds=args.max_rounds,
        trials_per_phase=args.trials_per_phase,
        task_name="svc_digits",
        model_family="SVC",
        dataset="sklearn_digits_1797x64",
        objective_name="cv_accuracy",
        seed=args.seed,
        memory_path=args.memory_path,
        dashboard=not args.no_dashboard,
        open_browser=args.open_browser,
    )

    print(f"\nbest CV accuracy = {run.best_value:.4f}")
    print(f"best params      = {run.best_params}")
    print(f"memory kept at   = {Path(args.memory_path).resolve()}")
    if run.url:
        run.wait()  # keep the dashboard reachable until you press Enter
    return 0


if __name__ == "__main__":
    sys.exit(main())
