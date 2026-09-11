"""Rule-based Adaptation (spec.md 模块三 pattern table)."""

from hpo_adapt.rules.adaptation import (
    DIAGNOSIS_NAMES,
    Candidate,
    CandidateWeights,
    RuleEngine,
    RuleMatch,
)

__all__ = [
    "DIAGNOSIS_NAMES",
    "Candidate",
    "CandidateWeights",
    "RuleEngine",
    "RuleMatch",
]
