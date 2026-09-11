"""Classical HPO module (spec.md: 其它模块和说明 — Classical HPO 模块).

Executes the HPO algorithm chosen by the Setup agent and produces TrialLogs.
"""

from .objectives import get_objective, list_objectives, register_objective
from .runner import HPOResult, run_hpo

__all__ = ["HPOResult", "get_objective", "list_objectives", "register_objective", "run_hpo"]
