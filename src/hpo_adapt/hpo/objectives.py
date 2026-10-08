"""Objective function registry for the Classical HPO module.

Objectives are pure deterministic functions ``f(params: dict) -> float`` so the
full closed loop is runnable and testable without external resources. Real
model-training objectives can be registered with ``register_objective``.
"""

import hashlib
from typing import Any, Callable, Union

import numpy as np

Objective = Callable[[dict[str, Any]], float]

_REGISTRY: dict[str, Objective] = {}


def register_objective(name: str, fn: Objective) -> None:
    _REGISTRY[name] = fn


def get_objective(name: str) -> Objective:
    if name not in _REGISTRY:
        raise KeyError(f"objective '{name}' not registered; available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_objectives() -> list[str]:
    return sorted(_REGISTRY)


def _noise(params: dict[str, Any], salt: str = "") -> float:
    """Deterministic pseudo-noise in [-1, 1] derived from the parameters."""
    payload = repr(sorted(params.items())) + salt
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0


def _synthetic_mlp(params: dict[str, Any]) -> float:
    """Synthetic MLP-style accuracy (maximize).

    Ground truth optimum: learning_rate = 0.45, batch_size = 32,
    optimizer_type = 'sgd', dropout = 0.05. The optimal learning rate lies
    above the initial range [1e-3, 1e-1] a default-centered setup derives
    (task default 1e-2, sgd-family), but within detection reach: good trials
    concentrate at the upper boundary, which produces the boundary
    concentration that triggers search-space expansion.
    """
    lr = float(params.get("learning_rate", 1e-3))
    batch = float(params.get("batch_size", 64))
    opt = str(params.get("optimizer_type", "adam"))
    drop = float(params.get("dropout", 0.1))

    if drop > 0.85:
        raise RuntimeError("training diverged: dropout too high")  # exercises FAIL trials
    if batch >= 448:
        raise ValueError("batch too large for memory")  # exercises FAIL trials

    def _score(value: float, target: float, width: float) -> float:
        return float(np.exp(-((np.log(value / target) / width) ** 2)))

    lr_score = _score(lr, 0.45, 1.6)
    batch_score = float(np.exp(-(((np.log(batch / 32.0)) / 1.2) ** 2)))
    opt_score = 1.0 if opt == "sgd" else 0.62
    drop_score = float(np.exp(-(((drop - 0.05) / 0.35) ** 2)))

    value = (
        0.42 * lr_score + 0.22 * batch_score + 0.16 * opt_score + 0.20 * drop_score
    )
    return float(np.clip(value + 0.01 * _noise(params), 0.0, 1.0))


def _branin(params: dict[str, Any]) -> float:
    """Branin function (minimize) over x1 in [-5, 10], x2 in [0, 15]."""
    x1 = float(params["x1"])
    x2 = float(params["x2"])
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5.0 / np.pi
    d = 6.0
    e = 10.0
    f = 1.0 / (8 * np.pi)
    val = a * (x2 - b * x1**2 + c * x1 - d) ** 2 + e * (1 - f) * np.cos(x1) + e
    return float(val + 0.01 * _noise(params, salt="branin"))


def _digits_data() -> tuple[Any, Any]:
    global _DIGITS_DATA
    if _DIGITS_DATA is None:
        from sklearn.datasets import load_digits

        _DIGITS_DATA = load_digits(return_X_y=True)
    return _DIGITS_DATA


_DIGITS_DATA: tuple[Any, Any] | None = None


def _rf_digits_cv(params: dict[str, Any]) -> float:
    """3-fold CV accuracy of a RandomForest on sklearn digits (maximize).

    Real-ML objective with a categorical hyperparameter; registered at import
    time so the closed loop is runnable end-to-end on a real task. Inner
    parallelism can be limited via HPO_ADAPT_INNER_JOBS (useful when an outer
    process pool parallelizes experiments).
    """
    import os

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    n_jobs = int(os.environ.get("HPO_ADAPT_INNER_JOBS", "-1"))
    model = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        max_features=str(params["max_features"]),
        random_state=0,
        n_jobs=n_jobs,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    return float(cross_val_score(model, *_digits_data(), cv=cv, n_jobs=n_jobs).mean())


register_objective("synthetic_mlp", _synthetic_mlp)
register_objective("branin", _branin)
register_objective("rf_digits_cv", _rf_digits_cv)
