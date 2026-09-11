"""Experience: one complete adaptation step, written to Memory (spec.md core data object 6).

S_t -> HPO -> D_t -> A_t -> S_{t+1} -> Outcome
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: str
    model_family: str
    dataset_signature: str


class Diagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary: str
    confidence: float = Field(ge=0.0, le=1.0)


class ActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    target: Optional[str] = None


class Outcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    best_before: Optional[float] = None
    best_after: Optional[float] = None
    relative_improvement: Optional[float] = None
    budget_consumed: int = 0


class Experience(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experience_id: str
    task_signature: TaskSignature
    setup_before: dict[str, Any]
    diagnostic_state: dict[str, Any]
    diagnosis: Diagnosis
    action: ActionRecord
    setup_after: dict[str, Any]
    outcome: Outcome = Field(default_factory=Outcome)
    reward: Optional[float] = None
    success: bool = False
    rationale: str = ""
    created_at: str = ""
