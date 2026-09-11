"""Hyperparameter importance: fANOVA (global) + LPI (local), spec.md Search Space Status.

Deterministic surrogate-based implementation:

- global_importance (fANOVA-style): fit a RandomForest surrogate on
  (normalized configs, y_tilde); for each parameter, compute the variance of
  its marginal-mean curve f_j(x) = E_{other params}[f(x, others)] over a
  grid, and normalize main-effect variances to sum to 1
  (fraction of total main-effect variance).
- local_importance (LPI): for the top configurations, vary one parameter
  over a grid while holding the others fixed and measure the variance of the
  predicted objective; averaged over top configs and normalized to sum to 1.

No fake importance is produced when trials are insufficient: the caller is
responsible for gating on ``min_trials_for_importance``.
"""

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from hpo_adapt.diagnostic.metrics import ParamTransform, is_numeric
from hpo_adapt.schemas import DiagnosticConfig, HyperparameterSpec, TrialLog


class _SurrogateFeatures:
    """Feature encoding: numerical params -> unit interval; categorical -> one-hot."""

    def __init__(self, search_space: list[HyperparameterSpec]) -> None:
        self.search_space = search_space
        self.transforms: dict[str, ParamTransform] = {}
        self.categories: dict[str, list[Any]] = {}
        for spec in search_space:
            if is_numeric(spec):
                self.transforms[spec.name] = ParamTransform(spec)
            else:
                self.categories[spec.name] = list(dict.fromkeys(spec.choices or []))
        self.numeric_names = [s.name for s in search_space if is_numeric(s)]
        self.categorical_names = [s.name for s in search_space if not is_numeric(s)]

    def encode(self, params: dict[str, Any]) -> np.ndarray:
        parts: list[float] = []
        for name in self.numeric_names:
            spec = next(s for s in self.search_space if s.name == name)
            raw = params.get(name)
            if raw is None:
                parts.append(0.5)
            else:
                parts.append(self.transforms[name].to_unit(float(raw)))
        for name in self.categorical_names:
            cats = self.categories[name]
            onehot = [0.0] * len(cats)
            raw = params.get(name)
            if raw in cats:
                onehot[cats.index(raw)] = 1.0
            elif raw is not None:
                onehot = [1.0 / len(cats)] * len(cats)  # unknown category -> uniform
            parts.extend(onehot)
        return np.array(parts, dtype=float)

    def set_feature(self, base: np.ndarray, name: str, value_unit_or_category: Any) -> np.ndarray:
        """Return a copy of ``base`` with parameter ``name`` set to a grid value."""
        vec = base.copy()
        if name in self.transforms:
            offset = self.numeric_names.index(name)
            vec[offset] = float(np.clip(value_unit_or_category, 0.0, 1.0))
        else:
            cats = self.categories[name]
            offset = len(self.numeric_names) + sum(
                len(self.categories[n]) for n in self.categorical_names[: self.categorical_names.index(name)]
            )
            vec[offset : offset + len(cats)] = 0.0
            vec[offset + cats.index(value_unit_or_category)] = 1.0
        return vec

    def grid_values(self, name: str, grid_size: int) -> list[Any]:
        if name in self.transforms:
            return list(np.linspace(0.0, 1.0, grid_size))
        return list(self.categories[name])


def _fit_surrogate(
    features: _SurrogateFeatures,
    trials: list[TrialLog],
    y_tilde: np.ndarray,
    config: DiagnosticConfig,
) -> RandomForestRegressor:
    x = np.stack([features.encode(t.params) for t in trials], axis=0)
    model = RandomForestRegressor(
        n_estimators=config.surrogate_n_estimators,
        random_state=config.surrogate_random_state,
    )
    model.fit(x, y_tilde)
    return model


def _main_effect_variance(
    model: RandomForestRegressor,
    features: _SurrogateFeatures,
    trials: list[TrialLog],
    name: str,
    config: DiagnosticConfig,
) -> float:
    """Variance over the grid of the marginal-mean curve f_j(x) = E_others[f]."""
    refs = np.stack([features.encode(t.params) for t in trials], axis=0)
    step = max(1, len(trials) // 200)  # cap references deterministically
    refs = refs[::step]
    grid = features.grid_values(name, config.lpi_grid_size)
    curve = np.array(
        [float(np.mean(model.predict(np.stack([features.set_feature(r, name, v) for r in refs], axis=0))) ) for v in grid]
    )
    return float(np.var(curve))


def compute_importance(
    trials: list[TrialLog],
    y_tilde: np.ndarray,
    search_space: list[HyperparameterSpec],
    config: DiagnosticConfig,
) -> dict[str, dict[str, float]]:
    """Return {param: {global_importance, local_importance}} normalized to sum 1 each."""
    varying = [s for s in search_space if len({repr(t.params.get(s.name)) for t in trials}) > 1]
    result = {s.name: {"global_importance": 0.0, "local_importance": 0.0} for s in search_space}
    if not varying or len(trials) < 2:
        return result

    features = _SurrogateFeatures(varying)
    model = _fit_surrogate(features, trials, y_tilde, config)

    global_vars = {
        spec.name: max(_main_effect_variance(model, features, trials, spec.name, config), 0.0)
        for spec in varying
    }
    total_global = sum(global_vars.values())
    if total_global > 0:
        for name, var in global_vars.items():
            result[name]["global_importance"] = var / total_global

    # LPI: variance of predictions when varying one parameter at top configs.
    n_top = min(config.lpi_top_configs, len(trials))
    top_order = np.argsort(-y_tilde, kind="stable")[:n_top]
    lpi_vars = {spec.name: 0.0 for spec in varying}
    for idx in top_order:
        base = features.encode(trials[int(idx)].params)
        for spec in varying:
            grid = features.grid_values(spec.name, config.lpi_grid_size)
            preds = model.predict(
                np.stack([features.set_feature(base, spec.name, v) for v in grid], axis=0)
            )
            lpi_vars[spec.name] += float(np.var(preds))
    for name in lpi_vars:
        lpi_vars[name] /= n_top
    total_lpi = sum(lpi_vars.values())
    if total_lpi > 0:
        for name, var in lpi_vars.items():
            result[name]["local_importance"] = var / total_lpi
    return result
