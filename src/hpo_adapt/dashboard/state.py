"""RunRecorder: event sink maintaining the dashboard's JSON document.

The document shape (all JSON-serializable)::

    {
      "schema": 1,
      "status": "running" | "finished" | "error",
      "error": str | None,
      "started_at": float, "updated_at": float,
      "task": {...TaskSpec summary...},
      "run": {total_budget, max_rounds, backend, seed},
      "initial_setup": {setup, how},
      "rounds": [
        {
          "round_index", "setup_id", "phase", "optimizer", "phase_budget",
          "search_space", "frozen", "global_trials_before",
          "state": "running"|"diagnosing"|"deciding"|"done",
          "trials": [{trial_id, params, value, state}],
          "trials_used", "warm_started", "phase_best", "cumulative_best",
          "diagnostic", "diagnosis", "action", "invalid_actions",
          "changes", "new_setup_id", "terminated", "reward"
        }
      ],
      "trials": [flat global trial list with round_index],
      "summary": {...episode_end payload...}
    }

Thread-safety: the pipeline thread writes, HTTP threads read; every access
goes through ``self.lock``.
"""

import threading
import time
from typing import Any

from hpo_adapt.events import EventPayload


class RunStatus:
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"


class RunRecorder:
    """Immutable-by-copy view builder over the live event stream."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._doc: dict[str, Any] = {
            "schema": 1,
            "status": RunStatus.RUNNING,
            "error": None,
            "started_at": time.time(),
            "updated_at": time.time(),
            "task": None,
            "run": {},
            "initial_setup": None,
            "rounds": [],
            "trials": [],
            "summary": None,
        }

    # ------------------------------------------------------------ event sink
    def __call__(self, event_type: str, payload: EventPayload) -> None:
        handler = getattr(self, f"_on_{event_type}", None)
        if handler is None:
            return
        with self.lock:
            handler(payload)
            self._doc["updated_at"] = time.time()

    # --------------------------------------------------------------- readers
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return _deep_copy(self._doc)

    # --------------------------------------------------------------- writers
    def _round(self, round_index: int) -> dict[str, Any]:
        rounds = self._doc["rounds"]
        for r in rounds:
            if r["round_index"] == round_index:
                return r
        raise KeyError(f"round {round_index} not started")

    def _fail(self, message: str) -> None:
        self._doc["status"] = RunStatus.ERROR
        self._doc["error"] = message

    def _on_run_start(self, p: EventPayload) -> None:
        self._doc["task"] = p["task"]
        self._doc["run"] = {
            "total_budget": p["total_budget"],
            "max_rounds": p["max_rounds"],
            "backend": p["agent_backend"],
            "seed": p["seed"],
        }

    def _on_initial_setup(self, p: EventPayload) -> None:
        self._doc["initial_setup"] = {
            "setup": p["setup"],
            "how": p.get("how", ""),
            "memory_prior": p.get("memory_prior"),
        }

    def _on_round_start(self, p: EventPayload) -> None:
        self._doc["rounds"].append(
            {
                "round_index": p["round_index"],
                "setup_id": p["setup_id"],
                "phase": p["phase"],
                "search_space": p["search_space"],
                "frozen": p["frozen"],
                "optimizer": p["optimizer"],
                "phase_budget": p["phase_budget"],
                "global_trials_before": p["global_trials_before"],
                "state": "running",
                "trials": [],
                "trials_used": None,
                "warm_started": None,
                "phase_best": None,
                "cumulative_best": None,
                "diagnostic": None,
                "diagnosis": None,
                "action": None,
                "invalid_actions": [],
                "changes": None,
                "new_setup_id": None,
                "terminated": None,
                "reward": None,
            }
        )

    def _on_trial(self, p: EventPayload) -> None:
        trial = {
            "trial_id": p["trial_id"],
            "params": p["params"],
            "value": p["value"],
            "state": p["state"],
            "round_index": p["round_index"],
        }
        self._doc["trials"].append(trial)
        try:
            self._round(p["round_index"])["trials"].append(trial)
        except KeyError:
            pass

    def _on_round_end(self, p: EventPayload) -> None:
        r = self._round(p["round_index"])
        r["state"] = "diagnosing"
        r["trials_used"] = p["trials_used"]
        r["warm_started"] = p["warm_started"]
        r["phase_best"] = p["phase_best"]
        r["cumulative_best"] = p["cumulative_best"]

    def _on_diagnostic(self, p: EventPayload) -> None:
        r = self._round(p["round_index"])
        r["diagnostic"] = p["diagnostic_state"]
        r["state"] = "deciding"

    def _on_reward(self, p: EventPayload) -> None:
        try:
            self._round(p["round_index"])["reward"] = {
                "value": p["reward"],
                "relative_improvement": p.get("relative_improvement"),
                "terminal": bool(p.get("terminal", False)),
            }
        except KeyError:
            pass

    def _on_action(self, p: EventPayload) -> None:
        r = self._round(p["round_index"])
        r["action"] = p["action"]
        r["diagnosis"] = p.get("diagnosis")

    def _on_action_invalid(self, p: EventPayload) -> None:
        try:
            self._round(p["round_index"])["invalid_actions"].append(p)
        except KeyError:
            pass

    def _on_revision(self, p: EventPayload) -> None:
        r = self._round(p["round_index"])
        r["changes"] = p["changes"]
        r["new_setup_id"] = p["new_setup_id"]
        r["terminated"] = bool(p.get("terminated"))
        r["state"] = "done"

    def _on_episode_end(self, p: EventPayload) -> None:
        self._doc["summary"] = p
        self._doc["status"] = RunStatus.FINISHED
        for r in self._doc["rounds"]:
            if r["state"] != "done":
                r["state"] = "done"

    # error handling is driven by the API layer, not by pipeline events
    def mark_error(self, message: str) -> None:
        with self.lock:
            self._fail(message)
            self._doc["updated_at"] = time.time()


def _deep_copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    return obj
