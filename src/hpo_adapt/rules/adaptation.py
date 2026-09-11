"""Rule-based Adaptation (spec.md 模块三 — 特别注意 pattern table).

Maps DiagnosticState -> (Diagnosis, Recommended Action) with the 7
diagnostic patterns from the spec, and ranks candidate hyperparameters with

    CandidateScore_j = alpha * Imp_j + beta * BC_j^good + gamma * (1 - WidthRatio_j)

(defaults alpha = 0.5, beta = 0.3, gamma = 0.2), computed only for
hyperparameters whose importance status is VALID.

The RL Controller (Stage 7) uses these recommendations to break ties between
unseen actions; the rule engine itself never modifies any Setup.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from hpo_adapt.diagnostic.metrics import ParamTransform
from hpo_adapt.levels import (
    Level,
    LevelConfig,
    WidthLevel,
    boundary_level,
    importance_level,
    width_level,
)
from hpo_adapt.schemas import (
    ActionType,
    DiagnosticState,
    MetricStatus,
    RLAction,
    Setup,
)

DIAGNOSIS_NAMES = {
    "truncation": "SEARCH_SPACE_TRUNCATION",
    "promising_region": "PROMISING_REGION_IDENTIFIED",
    "low_value": "LOW_VALUE_HYPERPARAMETER",
    "premature": "POSSIBLE_PREMATURE_CONVERGENCE",
    "insufficient_exploration": "INSUFFICIENT_EXPLORATION",
    "more_budget": "MORE_BUDGET_VALUABLE",
    "low_benefit": "LOW_EXPECTED_BENEFIT",
    "none": "NO_ACTION_RECOMMENDED",
}


class Candidate(BaseModel):
    """Candidate hyperparameter context (spec.md 模块三 State)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    score: float
    importance: Optional[float] = None
    boundary_concentration: Optional[float] = None
    width_ratio: Optional[float] = None
    importance_level: Optional[str] = None
    boundary_level: Optional[str] = None
    width_level: Optional[str] = None


class RuleMatch(BaseModel):
    """One matched diagnostic pattern."""

    model_config = ConfigDict(extra="forbid")

    diagnosis: str
    action: ActionType
    target: Optional[str] = None
    parameters: dict = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict = Field(default_factory=dict)


class CandidateWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: float = 0.5  # importance weight
    beta: float = 0.3  # good-trial boundary concentration weight
    gamma: float = 0.2  # (1 - width ratio) weight


class RuleEngine:
    """Deterministic DiagnosticState -> Diagnosis + Recommended RLAction."""

    def __init__(
        self,
        weights: CandidateWeights | None = None,
        level_config: LevelConfig | None = None,
        shift_center_threshold: float = 0.10,
    ) -> None:
        self.weights = weights or CandidateWeights()
        self.levels = level_config or LevelConfig()
        # Pattern 2: only SHIFT when the good region is off-center by more than
        # this fraction of the current interval (in transformed space);
        # otherwise SHRINK around the already-centered good region.
        self.shift_center_threshold = shift_center_threshold

    # ------------------------------------------------------------- candidates
    def rank_candidates(self, state: DiagnosticState) -> list[Candidate]:
        w = self.weights
        candidates: list[Candidate] = []
        for name, entry in state.search_space.importance.items():
            if entry.status != MetricStatus.VALID or entry.global_importance is None:
                continue
            bc = state.search_space.boundary_concentration.get(name)
            bc_good = bc.good if bc is not None and bc.status == MetricStatus.VALID else 0.0
            gr = state.search_space.good_region.get(name)
            width_ratio = gr.width_ratio if gr is not None and gr.status == MetricStatus.VALID else 1.0
            score = (
                w.alpha * entry.global_importance
                + w.beta * (bc_good or 0.0)
                + w.gamma * (1.0 - (width_ratio if width_ratio is not None else 1.0))
            )
            candidates.append(
                Candidate(
                    name=name,
                    score=round(score, 6),
                    importance=entry.global_importance,
                    boundary_concentration=bc_good,
                    width_ratio=width_ratio,
                    importance_level=importance_level(entry.global_importance, self.levels),
                    boundary_level=boundary_level(bc_good, self.levels) if bc_good is not None else None,
                    width_level=width_level(width_ratio, self.levels) if width_ratio is not None else None,
                )
            )
        candidates.sort(key=lambda c: (-c.score, c.name))
        return candidates

    # -------------------------------------------------------------- matching
    def _best_candidate(self, state: DiagnosticState) -> Optional[Candidate]:
        ranked = self.rank_candidates(state)
        return ranked[0] if ranked else None

    def _lowest_importance(self, state: DiagnosticState) -> Optional[tuple[str, float]]:
        worst: Optional[tuple[str, float]] = None
        for name, entry in state.search_space.importance.items():
            if entry.status != MetricStatus.VALID or entry.global_importance is None:
                continue
            if worst is None or entry.global_importance < worst[1]:
                worst = (name, entry.global_importance)
        return worst

    def _shift_needed(
        self, setup: Optional[Setup], name: str, good_region
    ) -> bool:
        """True when the good region is off-center enough to justify SHIFT.

        Falls back to True (old behavior) when the current interval or the
        good region is unavailable.
        """
        if (
            setup is None
            or good_region is None
            or good_region.status != MetricStatus.VALID
            or good_region.lower is None
            or good_region.upper is None
        ):
            return True
        spec = setup.hyperparameter(name)
        if spec is None or spec.low is None or spec.high is None:
            return True
        t = ParamTransform(spec)
        center_unit = (t.to_unit(good_region.lower) + t.to_unit(good_region.upper)) / 2.0
        return abs(center_unit - 0.5) > self.shift_center_threshold

    def _was_low_importance(
        self, previous_state: DiagnosticState, name: str, level_config: LevelConfig
    ) -> bool:
        entry = previous_state.search_space.importance.get(name)
        if entry is None or entry.status != MetricStatus.VALID or entry.global_importance is None:
            return False
        return importance_level(entry.global_importance, level_config) == Level.LOW

    def evaluate(
        self,
        state: DiagnosticState,
        setup: Optional[Setup] = None,
        previous_state: Optional[DiagnosticState] = None,
    ) -> list[RuleMatch]:
        """Evaluate the 7 spec patterns in table order."""
        matches: list[RuleMatch] = []
        opt = state.optimization
        ss = state.search_space
        bud = state.budget
        lc = self.levels

        stag = opt.stagnation.value if opt.stagnation.status == MetricStatus.VALID else None
        ri = opt.relative_improvement.value if opt.relative_improvement.status == MetricStatus.VALID else None
        mg = opt.marginal_gain.value if opt.marginal_gain.status == MetricStatus.VALID else None
        div = ss.configuration_diversity.ratio if ss.configuration_diversity.status == MetricStatus.VALID else None
        cov = ss.coverage.overall if ss.coverage.status == MetricStatus.VALID else None
        br = bud.budget_remaining.value if bud.budget_remaining.status == MetricStatus.VALID else None
        bc_consumed = (
            bud.budget_consumed.value if bud.budget_consumed.status == MetricStatus.VALID else None
        )

        # 1. stagnation + high good-boundary concentration + high importance
        for name, bc in ss.boundary_concentration.items():
            if bc.status != MetricStatus.VALID or bc.good is None:
                continue
            imp_entry = ss.importance.get(name)
            imp = (
                imp_entry.global_importance
                if imp_entry is not None and imp_entry.status == MetricStatus.VALID
                else None
            )
            if (
                stag is True
                and boundary_level(bc.good, lc) == Level.HIGH
                and importance_level(imp, lc) == Level.HIGH
            ):
                direction = "upper" if (bc.upper or 0.0) >= (bc.lower or 0.0) else "lower"
                matches.append(
                    RuleMatch(
                        diagnosis=DIAGNOSIS_NAMES["truncation"],
                        action=ActionType.EXPAND_SEARCH_SPACE,
                        target=name,
                        parameters={"direction": direction},
                        confidence=round((1.0 + (imp or 0.0) + bc.good) / 3.0, 4),
                        evidence={
                            "stagnation": stag,
                            "importance": imp,
                            "good_boundary_concentration": bc.good,
                        },
                    )
                )
                break  # primary pattern; one target is enough

        # 2. high importance + narrow good region -> SHIFT (off-center) or SHRINK (centered)
        best = self._best_candidate(state)
        if best is not None:
            if (
                best.importance_level == Level.HIGH.value
                and best.width_level == WidthLevel.NARROW.value
            ):
                gr = ss.good_region.get(best.name)
                params: dict = {}
                if gr is not None and gr.status == MetricStatus.VALID:
                    params["target_region"] = [gr.lower, gr.upper]
                shift = self._shift_needed(setup, best.name, gr)
                matches.append(
                    RuleMatch(
                        diagnosis=DIAGNOSIS_NAMES["promising_region"],
                        action=(
                            ActionType.SHIFT_SEARCH_REGION
                            if shift
                            else ActionType.SHRINK_SEARCH_SPACE
                        ),
                        target=best.name,
                        parameters=params,
                        confidence=round((best.importance or 0.0) + (1.0 - (best.width_ratio or 1.0))) / 2
                        if best.importance is not None
                        else 0.5,
                        evidence={
                            "importance": best.importance,
                            "width_ratio": best.width_ratio,
                            "good_region_off_center": shift,
                        },
                    )
                )

        # 3. low importance for MULTIPLE rounds (spec wording) -> FREEZE
        worst = self._lowest_importance(state)
        if worst is not None and importance_level(worst[1], lc) == Level.LOW:
            persistent = previous_state is not None and self._was_low_importance(
                previous_state, worst[0], lc
            )
            if persistent:
                matches.append(
                    RuleMatch(
                        diagnosis=DIAGNOSIS_NAMES["low_value"],
                        action=ActionType.FREEZE_HYPERPARAMETER,
                        target=worst[0],
                        parameters={},
                        confidence=round(1.0 - worst[1], 4),
                        evidence={"importance": worst[1], "previous_importance_low": True},
                    )
                )

        # 4. stagnation + very low diversity
        if stag is True and div is not None and div < lc.diversity_low:
            matches.append(
                RuleMatch(
                    diagnosis=DIAGNOSIS_NAMES["premature"],
                    action=ActionType.INCREASE_EXPLORATION,
                    target="optimizer",
                    parameters={},
                    confidence=round((1.0 + (1.0 - div)) / 2.0, 4),
                    evidence={"stagnation": stag, "diversity_ratio": div},
                )
            )

        # 5. stagnation + low coverage
        if stag is True and cov is not None and cov < lc.coverage_low:
            matches.append(
                RuleMatch(
                    diagnosis=DIAGNOSIS_NAMES["insufficient_exploration"],
                    action=ActionType.INCREASE_EXPLORATION,
                    target="optimizer",
                    parameters={},
                    confidence=round((1.0 + (1.0 - cov)) / 2.0, 4),
                    evidence={"stagnation": stag, "coverage": cov},
                )
            )

        # 6. clear improvement + budget nearly exhausted
        if (
            ri is not None
            and ri >= lc.improvement_high
            and br is not None
            and br < lc.budget_low
        ):
            matches.append(
                RuleMatch(
                    diagnosis=DIAGNOSIS_NAMES["more_budget"],
                    action=ActionType.INCREASE_BUDGET,
                    target="budget",
                    parameters={},
                    confidence=round(min(ri / max(lc.improvement_high, 1e-12), 1.0), 4),
                    evidence={"relative_improvement": ri, "budget_remaining": br},
                )
            )

        # 7. stagnation + very low marginal gain + budget consumed high
        if (
            stag is True
            and mg is not None
            and mg < lc.marginal_gain_low
            and bc_consumed is not None
            and bc_consumed >= lc.budget_consumed_high
        ):
            matches.append(
                RuleMatch(
                    diagnosis=DIAGNOSIS_NAMES["low_benefit"],
                    action=ActionType.STOP,
                    target=None,
                    parameters={},
                    confidence=round((1.0 + bc_consumed) / 2.0, 4),
                    evidence={"stagnation": stag, "marginal_gain": mg, "budget_consumed": bc_consumed},
                )
            )

        return matches

    def primary(
        self,
        state: DiagnosticState,
        setup: Optional[Setup] = None,
        previous_state: Optional[DiagnosticState] = None,
    ) -> Optional[RuleMatch]:
        matches = self.evaluate(state, setup, previous_state)
        return matches[0] if matches else None

    def recommend(
        self,
        state: DiagnosticState,
        setup: Setup | None = None,
        previous_state: Optional[DiagnosticState] = None,
    ) -> Optional[RLAction]:
        """Recommended RLAction for the primary matched pattern (may be None)."""
        match = self.primary(state, setup, previous_state)
        if match is None:
            return None
        return RLAction(
            action=match.action,
            target=match.target,
            parameters=dict(match.parameters),
            confidence=match.confidence,
            q_value=None,
            reason=f"rule-based recommendation: {match.diagnosis}",
        )
