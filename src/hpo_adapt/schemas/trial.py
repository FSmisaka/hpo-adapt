"""TrialLog: single trial record produced by HPO (spec.md core data object 3)."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TrialState(str, Enum):
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAIL = "FAIL"


class TrialLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: int
    params: dict[str, Any]
    value: Optional[float] = None
    state: TrialState
    duration: Optional[float] = None
    timestamp: Optional[float] = None
