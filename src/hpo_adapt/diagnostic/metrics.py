"""Deterministic preprocessing and low-level metric helpers for the Diagnostic Engine."""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from hpo_adapt.schemas import (
    Distribution,
    HyperparameterSpec,
    ObjectiveDirection,
    TrialLog,
    TrialState,
)


@dataclass(frozen=True)
class PreparedData:
    """Deterministic preprocessing result (spec.md: 预处理).

    All objective-related metrics are computed on the internally maximized
    series y_tilde = y (maximize) / -y (minimize).
    """

    completed: list[TrialLog] = field(default_factory=list)  # in execution order
    y_tilde: np.ndarray = field(default_factory=lambda: np.empty(0))
    best_series: np.ndarray = field(default_factory=lambda: np.empty(0))  # B_t running max
    num_failed: int = 0
    num_pruned: int = 0
    failure_rate: Optional[float] = None


def prepare(trials: list[TrialLog], direction: ObjectiveDirection) -> PreparedData:
    completed = [t for t in trials if t.state == TrialState.COMPLETE and t.value is not None]
    failed = sum(1 for t in trials if t.state == TrialState.FAIL)
    pruned = sum(1 for t in trials if t.state == TrialState.PRUNED)
    total = len(trials)
    vals: list[float] = []
    for t in completed:
        assert t.value is not None
        vals.append(float(t.value))
    values = np.asarray(vals, dtype=float)
    y_tilde = values if direction == ObjectiveDirection.MAXIMIZE else -values
    best_series = np.maximum.accumulate(y_tilde) if y_tilde.size else np.empty(0)
    return PreparedData(
        completed=completed,
        y_tilde=y_tilde,
        best_series=best_series,
        num_failed=failed,
        num_pruned=pruned,
        failure_rate=(failed + pruned) / total if total else None,
    )


def good_indices(y_tilde: np.ndarray, quantile: float) -> list[int]:
    """Indices of the top-q% completed trials by y_tilde (deterministic order)."""
    n = y_tilde.size
    if n == 0:
        return []
    n_good = max(1, int(round(quantile * n)))
    order = np.argsort(-y_tilde, kind="stable")  # stable -> deterministic under ties
    return [int(i) for i in order[:n_good]]


class ParamTransform:
    """Unit-interval normalization for numerical parameters (log-space aware)."""

    def __init__(self, spec: HyperparameterSpec) -> None:
        self.log = spec.distribution == Distribution.LOG_UNIFORM
        self.low = float(spec.low or 0.0)
        self.high = float(spec.high or 1.0)

    def _t(self, x: float) -> float:
        return float(np.log(x)) if self.log else x

    def to_unit(self, x: float) -> float:
        span = self._t(self.high) - self._t(self.low)
        if span <= 0:
            return 0.0
        return (self._t(float(x)) - self._t(self.low)) / span

    def from_unit(self, z: float) -> float:
        z = min(max(z, 0.0), 1.0)
        val = self._t(self.low) + z * (self._t(self.high) - self._t(self.low))
        return float(np.exp(val)) if self.log else val


def param_vector(trials: list[TrialLog], name: str) -> list[Any]:
    return [t.params.get(name) for t in trials]


def numeric_values(trials: list[TrialLog], spec: HyperparameterSpec) -> np.ndarray:
    return np.array([float(t.params[spec.name]) for t in trials], dtype=float)


def unit_values(trials: list[TrialLog], spec: HyperparameterSpec) -> np.ndarray:
    tf = ParamTransform(spec)
    return np.array([tf.to_unit(v) for v in numeric_values(trials, spec)], dtype=float)


def is_numeric(spec: HyperparameterSpec) -> bool:
    return spec.type in ("float", "int")


def boundary_mask(z: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower_mask, upper_mask) for normalized values z in [0, 1]."""
    return z < threshold, z > (1.0 - threshold)


def mixed_pairwise_distance(
    trials: list[TrialLog], search_space: list[HyperparameterSpec]
) -> float:
    """Mean pairwise mixed-type distance d(x_i, x_k) (spec.md: Configuration Diversity)."""
    varying = [s for s in search_space if len({repr(t.params.get(s.name)) for t in trials}) > 1]
    if not varying or len(trials) < 2:
        return 0.0
    feature_rows: list[np.ndarray] = []
    for spec in varying:
        if is_numeric(spec):
            feature_rows.append(unit_values(trials, spec))
        else:
            cats = sorted({repr(v) for v in param_vector(trials, spec.name)})
            idx = {c: i for i, c in enumerate(cats)}
            feature_rows.append(
                np.array([idx[repr(v)] / max(len(cats) - 1, 1) for v in param_vector(trials, spec.name)])
            )
    feats = np.stack(feature_rows, axis=1)  # (n, m) numeric-encoded
    n = feats.shape[0]
    total = 0.0
    pairs = 0
    for i in range(n):
        for k in range(i + 1, n):
            d = float(np.mean(np.abs(feats[i] - feats[k])))
            total += d
            pairs += 1
    return total / pairs if pairs else 0.0


def bin_occupancy(z: np.ndarray, bins: int) -> float:
    """Fraction of occupied bins for normalized values z in [0, 1]."""
    if z.size == 0 or bins <= 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(z, edges[1:-1], right=False), 0, bins - 1)
    return float(len(set(int(i) for i in indices)) / bins)
