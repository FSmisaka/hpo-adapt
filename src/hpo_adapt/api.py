"""One-function user API: turn any ML task into an adaptive-HPO closed loop.

The task contract (the "basic task format" every ML task must satisfy):

1. an objective function  ``params: dict -> float``  that trains + evaluates
   one model under the given hyperparameters and returns the metric to
   optimize (raise ``RuntimeError``/``ValueError`` for a failed trial), and
2. a declarative description of the hyperparameters (``Float`` / ``Int`` /
   ``Choice``), the optimization direction and a trial budget.

Everything else — the initial setup (search space, optimizer, per-phase
budget), the diagnostic analysis, the RL-controlled setup interventions and
the multi-round iteration — is derived automatically:

    import hpo_adapt as ha

    def train_evaluate(params: dict) -> float:
        ...                                       # your existing ML code
        return cv_accuracy

    run = ha.run(
        train_evaluate,
        space=dict(
            n_estimators=ha.Int(10, 500, default=100),
            max_depth=ha.Int(2, 32, default=8),
            max_features=ha.Choice(["sqrt", "log2"], default="sqrt"),
        ),
        direction="maximize",
        total_trials=90,
    )
    print(run.best_value, run.best_params)

``ha.run`` prints a local dashboard URL in the terminal; open it in a browser
to watch setups being selected, diagnosed, revised and re-run in real time.
"""

import random
import string
import time
import webbrowser
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from hpo_adapt.dashboard import RunRecorder
from hpo_adapt.dashboard.server import DashboardServer
from hpo_adapt.events import EventSink
from hpo_adapt.hpo.objectives import register_objective
from hpo_adapt.memory import MemoryStore
from hpo_adapt.pipeline import AdaptationPipeline, PipelineConfig
from hpo_adapt.rl import QController, QLearningConfig
from hpo_adapt.schemas import HyperparameterMetadata, ObjectiveDirection, ResourceConstraints, TaskSpec

Objective_fn = Callable[[dict[str, Any]], float]

__all__ = ["run", "Float", "Int", "Choice", "RunResult"]


# ---------------------------------------------------------------- space DSL
def Float(  # noqa: N802 (wandb-style DSL name)
    low: float,
    high: float,
    *,
    log: bool = False,
    default: Optional[float] = None,
    description: str = "",
) -> HyperparameterMetadata:
    """Continuous hyperparameter in [low, high] (log-scaled if ``log=True``)."""
    return HyperparameterMetadata(
        name="", type="float", low=float(low), high=float(high),
        log=bool(log), default=default, description=description,
    )


def Int(  # noqa: N802
    low: int,
    high: int,
    *,
    default: Optional[int] = None,
    description: str = "",
) -> HyperparameterMetadata:
    """Integer hyperparameter in [low, high]."""
    return HyperparameterMetadata(
        name="", type="int", low=float(low), high=float(high),
        default=default, description=description,
    )


def Choice(  # noqa: N802
    choices: list[Any],
    *,
    default: Any = None,
    description: str = "",
) -> HyperparameterMetadata:
    """Categorical hyperparameter over ``choices``."""
    return HyperparameterMetadata(
        name="", type="categorical", choices=list(choices),
        default=default, description=description,
    )


def _normalize_space(
    space: Union[dict[str, Any], list[HyperparameterMetadata]],
) -> list[HyperparameterMetadata]:
    metas: list[HyperparameterMetadata] = []
    if isinstance(space, dict):
        for name, value in space.items():
            if isinstance(value, HyperparameterMetadata):
                meta = value.model_copy(update={"name": name})
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                meta = HyperparameterMetadata(
                    name=name, type="float", low=float(value[0]), high=float(value[1])
                )
            else:
                raise TypeError(
                    f"space['{name}'] must be ha.Float/ha.Int/ha.Choice or a "
                    f"(low, high) tuple, got {value!r}"
                )
            metas.append(meta)
    else:
        for value in space:
            if not isinstance(value, HyperparameterMetadata) or not value.name:
                raise TypeError(
                    "space lists must contain named HyperparameterMetadata "
                    "(use ha.Float/ha.Int/ha.Choice inside a dict for the simple form)"
                )
            metas.append(value)
    if not metas:
        raise ValueError("space must describe at least one hyperparameter")
    return metas


# -------------------------------------------------------------- objective
def _wrap_objective(objective: Objective_fn) -> Objective_fn:
    def wrapped(params: dict[str, Any]) -> float:
        try:
            value = objective(params)
            return float(value)
        except (RuntimeError, ValueError):
            raise  # treated as failed trials by the HPO runner
        except Exception as exc:
            raise RuntimeError(f"user objective raised {exc!r}") from exc

    return wrapped


# ----------------------------------------------------------------- results
@dataclass
class RunResult:
    """Return value of ``hpo_adapt.run``."""

    pipeline_result: Any  # PipelineResult (kept untyped to avoid import cycle)
    task: TaskSpec
    url: Optional[str]
    memory_path: str

    @property
    def best_value(self) -> Optional[float]:
        """Best objective value in the task's original direction."""
        return self.pipeline_result.best_value_raw

    @property
    def best_params(self) -> Optional[dict[str, Any]]:
        return self.pipeline_result.best_params

    @property
    def rounds(self) -> int:
        return len(self.pipeline_result.steps)

    def wait(self, prompt: str = "Press Enter to close the dashboard and exit ...") -> None:
        """Block until the user presses Enter (keeps the dashboard reachable)."""
        try:
            input(prompt)
        except (EOFError, OSError):
            pass


# ----------------------------------------------------------- terminal view
class _TerminalView:
    """wandb-style console progress, driven by the same events as the dashboard."""

    def __init__(self, quiet: bool, url: Optional[str]) -> None:
        self.quiet = quiet
        self.url = url
        self._round_line: Optional[int] = None
        self._failures = 0

    def _print(self, text: str) -> None:
        if not self.quiet:
            print(text, flush=True)

    def banner(self, task: TaskSpec, total_trials: int, max_rounds: int, memory: str) -> None:
        if self.quiet:
            return
        width = 60

        def row(text: str) -> str:
            return f"│ {text:<{width}}│"

        box = [
            f"┌{'─' * width}┐",
            row("hpo-adapt · adaptive HPO closed loop"),
            row(f"   task      : {task.task_id}"),
            row(f"   objective : {task.objective_name} ({task.direction.value})"),
            row(f"   budget    : {total_trials} trials · up to {max_rounds} rounds"),
            row(f"   memory    : {memory}"),
            f"└{'─' * width}┘",
        ]
        if self.url:
            box += [
                "",
                f"  ▶ Dashboard live at  {self.url}",
                "    Open this URL in your browser to watch the run in real time.",
            ]
        print("\n".join(box), flush=True)

    def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.quiet:
            return
        if event_type == "round_start":
            self._round_line = payload["round_index"]
            self._print(
                f"round {payload['round_index']}  {payload['setup_id']}  "
                f"{payload['optimizer']['name']:<6s}  searching ({payload['phase_budget']} trials) ..."
            )
        elif event_type == "trial":
            if payload["state"] == "FAIL":
                self._failures += 1
        elif event_type == "round_end":
            best = payload.get("cumulative_best")
            arrow = "↳"
            self._print(
                f"  {arrow} {payload['trials_used']} trials  "
                f"phase best={_fmt(payload.get('phase_best'))}  "
                f"cumulative best={_fmt(best)}"
                + (f"  ({self._failures} failed)" if self._failures else "")
            )
            self._failures = 0
        elif event_type == "action":
            action = payload["action"]
            diagnosis = (payload.get("diagnosis") or {}).get("primary", "")
            self._print(
                f"  ↳ ACTION {action['action']} @ {action.get('candidate_hyperparameter') or '-'}"
                f"  [{diagnosis}]"
            )
        elif event_type == "revision":
            for change in payload.get("changes") or []:
                old, new = change.get("old"), change.get("new")
                self._print(
                    f"     {change['hyperparameter']}: "
                    f"{_fmt_side(old)} → {_fmt_side(new)}"
                )

    def finish(self, result: Any, elapsed: float) -> None:
        if self.quiet:
            return
        print(
            f"\n✓ finished in {elapsed:.1f}s · best={_fmt(result.best_value_raw)} · "
            f"params={result.best_params}\n"
            f"  rounds={len(result.steps)}  trials={result.total_trials_used}  "
            f"experiences={result.experiences_recorded}  q-states={result.q_table_size}",
            flush=True,
        )
        if self.url:
            print(f"  dashboard (final state) still served at {self.url} while this process lives\n")


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_side(side: Optional[dict[str, Any]]) -> str:
    if side is None:
        return "∅"
    if "choices" in side:
        return str(side["choices"])
    if "value" in side:
        return _fmt(side["value"])
    if "max_trials" in side:
        return f"{int(side['max_trials'])} trials"
    low, high = side.get("low"), side.get("high")
    if low is not None and high is not None:
        return f"[{low:.4g}, {high:.4g}]"
    if "name" in side:
        config = {k: v for k, v in side.items() if k != "name"}
        return str(side["name"]) + (f" {config}" if config else "")
    return str(side)


# ------------------------------------------------------------------- run()
def run(
    objective: Objective_fn,
    space: Union[dict[str, Any], list[HyperparameterMetadata]],
    *,
    direction: Union[str, ObjectiveDirection] = "maximize",
    total_trials: int = 120,
    max_rounds: int = 5,
    trials_per_phase: Optional[int] = None,
    task_name: Optional[str] = None,
    task_type: str = "classification",
    description: Optional[str] = None,
    model_family: Optional[str] = None,
    dataset: Optional[str] = None,
    objective_name: str = "objective",
    allowed_optimizers: Optional[list[str]] = None,
    seed: int = 0,
    epsilon_start: float = 0.0,
    memory_path: Optional[str] = None,
    backend: str = "heuristic",
    dashboard: bool = True,
    dashboard_host: str = "127.0.0.1",
    dashboard_port: Optional[int] = None,
    open_browser: bool = False,
    quiet: bool = False,
) -> RunResult:
    """Wrap an ML task into the adaptive-HPO closed loop and run it.

    Parameters
    ----------
    objective:
        ``params -> float``; train + evaluate one model, return the metric.
    space:
        dict of ``name -> ha.Float(...) | ha.Int(...) | ha.Choice(...)``.
        Bounds are hard constraints for every adapted setup; ``default``
        values center the initial search space.
    direction, total_trials, max_rounds:
        Objective direction, global trial budget, and round cap.
    dashboard:
        Serve the local web dashboard and print its URL.
    """
    metas = _normalize_space(space)
    task_id = _slugify(task_name or getattr(objective, "__name__", None) or "ml_task")
    suffix = "".join(random.Random(seed).choice(string.ascii_lowercase + string.digits) for _ in range(6))
    registry_key = f"user::{task_id}::{suffix}"
    register_objective(registry_key, _wrap_objective(objective))

    phase_cap = trials_per_phase or max(5, min(50, total_trials // 3))
    task = TaskSpec(
        task_id=f"{task_id}_{suffix}",
        task_type=task_type,
        description=description or f"User task '{task_id}' plugged in via hpo_adapt.run().",
        model_family=model_family or _guess_family(objective),
        dataset_signature=dataset or "unspecified",
        objective_name=objective_name,
        direction=ObjectiveDirection(direction),
        objective=registry_key,
        available_hyperparameters=metas,
        resource_constraints=ResourceConstraints(
            max_total_trials=total_trials, max_trials_per_phase=phase_cap
        ),
        allowed_optimizers=allowed_optimizers or ["RANDOM", "TPE"],
    )

    recorder = RunRecorder()
    server: Optional[DashboardServer] = None
    url: Optional[str] = None
    if dashboard:
        server = DashboardServer(recorder, dashboard_host, dashboard_port).start()
        url = server.url
        if open_browser:
            webbrowser.open(url)
    memory = MemoryStore(memory_path or ":memory:")
    view = _TerminalView(quiet=quiet, url=url)
    if dashboard and server is not None and server.fallback_from is not None:
        print(
            f"  ⚠ port {server.fallback_from} is busy — an older run may still be alive;\n"
            f"    THIS run's dashboard is on the URL below (close the old run to free the port)\n",
            flush=True,
        )
    view.banner(task, total_trials, max_rounds, memory_path or ":memory:")

    def _dispatch(event_type: str, payload: dict[str, Any]) -> None:
        recorder(event_type, payload)
        view(event_type, payload)

    sink = EventSink(handler=_dispatch)
    pipeline = AdaptationPipeline(
        task,
        PipelineConfig(
            max_rounds=max_rounds, agent_backend=backend,
            memory_path=memory_path or ":memory:", seed=seed,
        ),
        controller=QController(config=QLearningConfig(seed=seed, epsilon_start=epsilon_start)),
        memory=memory,
        event_sink=sink,
    )

    t0 = time.perf_counter()
    try:
        result = pipeline.run()
    except Exception as exc:
        recorder.mark_error(repr(exc))
        if not quiet:
            print(f"\n✕ run failed: {exc!r}\n  dashboard kept the partial state at {url}\n", flush=True)
        raise
    view.finish(result, time.perf_counter() - t0)
    return RunResult(pipeline_result=result, task=task, url=url, memory_path=memory_path or ":memory:")


def _slugify(name: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()
    return slug or "ml_task"


def _guess_family(objective: Objective_fn) -> str:
    module = getattr(objective, "__module__", "") or ""
    for part in reversed(module.split(".")):
        if part and part not in {"__main__", "builtins"}:
            return part
    return "custom"
