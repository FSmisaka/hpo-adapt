"""Live dashboard: a wandb-style local web UI for the closed loop.

- ``RunRecorder``: an event sink that maintains a JSON-able document of the
  whole run (task, rounds, trials, diagnostics, actions, rewards).
- ``DashboardServer``: a stdlib ThreadingHTTPServer serving the single-page
  app plus a ``/api/state`` JSON endpoint (polling; no extra dependencies).

The server runs as a daemon thread inside the user's process and shuts down
automatically when the process exits.
"""

from hpo_adapt.dashboard.state import RunRecorder, RunStatus

__all__ = ["RunRecorder", "RunStatus", "DashboardServer"]


def __getattr__(name: str):  # lazy import avoids a server-import cost in tests
    if name == "DashboardServer":
        from hpo_adapt.dashboard.server import DashboardServer

        return DashboardServer
    raise AttributeError(name)
