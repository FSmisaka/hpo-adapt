"""One-command experiment runner for the HPO-Adapt evaluation.

Phases
------
main     Single-episode comparison: {static-random, static-tpe,
         adapt-random-action, adapt-rule-only, adapt-rl-default,
         adapt-rl-guided} x tasks x seeds, equal TOTAL trial budget.
episodes Cross-episode learning: consecutive episodes share the Q-table
         controller and the SQLite memory (the framework's intended usage),
         showing whether adaptation improves with accumulated experience.

Outputs (under experiments/results/<tag>/)
------------------------------------------
runs/<task>__<method>__s<seed>.json   full per-run record (steps, trials,
                                      diagnostics, actions, timings)
summary.csv                          one row per run

Usage
-----
python experiments/run_all.py --smoke              # fast sanity check
python experiments/run_all.py --phase main         # main table
python experiments/run_all.py --phase episodes     # learning curves
python experiments/run_all.py --phase all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "experiments" / "results"

BUDGET = 120  # total trials per run/episode — identical for every method


# --------------------------------------------------------------------- methods
@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str  # "static" | "adapt"
    optimizer: Optional[str] = None  # forced optimizer for static baselines
    epsilon_start: Optional[float] = None
    epsilon_end: Optional[float] = None
    epsilon_decay: Optional[float] = None


MAIN_METHODS = [
    MethodSpec("static-random", "static", optimizer="RANDOM"),
    MethodSpec("static-tpe", "static", optimizer="TPE"),
    MethodSpec("adapt-random-action", "adapt", epsilon_start=1.0, epsilon_end=1.0),
    MethodSpec("adapt-rule-only", "adapt", epsilon_start=0.0, epsilon_end=0.0),
    MethodSpec("adapt-rl-default", "adapt", epsilon_start=1.0, epsilon_end=0.05),
    MethodSpec("adapt-rl-guided", "adapt", epsilon_start=0.2, epsilon_end=0.05),
]

EPISODE_METHODS = [
    MethodSpec("static-tpe", "static", optimizer="TPE"),
    MethodSpec("adapt-random-action", "adapt", epsilon_start=1.0, epsilon_end=1.0),
    MethodSpec("adapt-rl", "adapt", epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.95),
]

TASK_BUILDERS = ("synthetic_mlp", "branin", "rf_digits")


def build_task(name: str, total: int, trials_per_phase: Optional[int] = None):
    from hpo_adapt.pipeline import branin_task, rf_digits_task, synthetic_mlp_task

    if name == "synthetic_mlp":
        return synthetic_mlp_task(total) if trials_per_phase is None else _with_phase(
            synthetic_mlp_task(total), trials_per_phase
        )
    if name == "branin":
        return branin_task(total) if trials_per_phase is None else _with_phase(
            branin_task(total), trials_per_phase
        )
    if name == "rf_digits":
        return rf_digits_task(total) if trials_per_phase is None else _with_phase(
            rf_digits_task(total), trials_per_phase
        )
    raise ValueError(name)


def _with_phase(task, trials_per_phase: int):
    task.resource_constraints.max_trials_per_phase = trials_per_phase
    return task


class ForcedOptimizerInitialAgent:
    """Static baselines: force a fixed optimizer in the initial setup."""

    def __init__(self, base, optimizer: str) -> None:
        self.base = base
        self.optimizer = optimizer

    def generate(self, task, memory_context=None):
        from hpo_adapt.schemas import OptimizerSpec

        setup = self.base.generate(task, memory_context)
        setup = setup.model_copy(deep=True)
        setup.optimizer = OptimizerSpec(name=self.optimizer, config={"n_startup_trials": 10})
        return setup


# ----------------------------------------------------------------------- runs
def _controller(method: MethodSpec, seed: int):
    from hpo_adapt.rl import QController, QLearningConfig

    cfg = dict(seed=seed)
    if method.epsilon_start is not None:
        cfg["epsilon_start"] = method.epsilon_start
    if method.epsilon_end is not None:
        cfg["epsilon_end"] = method.epsilon_end
    if method.epsilon_decay is not None:
        cfg["epsilon_decay"] = method.epsilon_decay
    return QController(config=QLearningConfig(**cfg))


def run_single(rec: dict) -> dict:
    """One experimental unit: (task, method, seed) — single episode."""
    os.environ.setdefault("HPO_ADAPT_INNER_JOBS", "1")
    from hpo_adapt.pipeline import AdaptationPipeline, PipelineConfig

    method = MethodSpec(**rec["method"])
    if method.kind == "static":
        # A static baseline spends the WHOLE budget in one phase with a fixed
        # setup (per-phase cap lifted); max_rounds=1 => no adaptation at all.
        task = build_task(rec["task"], rec["budget"], trials_per_phase=rec["budget"])
        max_rounds = 1
    else:
        task = build_task(rec["task"], rec["budget"])
        max_rounds = 6

    pipeline = AdaptationPipeline(
        task,
        PipelineConfig(max_rounds=max_rounds, seed=rec["seed"], memory_path=":memory:"),
        controller=_controller(method, rec["seed"]),
    )
    if method.kind == "static":
        from hpo_adapt.agents.initial_setup import HeuristicInitialSetupAgent

        pipeline.initial_agent = ForcedOptimizerInitialAgent(
            HeuristicInitialSetupAgent(), method.optimizer or "TPE"
        )

    result = pipeline.run()
    return {
        **rec,
        "result": result.model_dump(mode="json"),
        "timings": pipeline.timings,
    }


def run_group(rec: dict) -> dict:
    """One experimental unit for the episodes phase: N consecutive episodes
    sharing the QController and a persistent SQLite memory."""
    os.environ.setdefault("HPO_ADAPT_INNER_JOBS", "1")
    from hpo_adapt.memory import MemoryStore
    from hpo_adapt.pipeline import AdaptationPipeline, PipelineConfig

    method = MethodSpec(**rec["method"])
    controller = _controller(method, rec["seed"])
    memory_path = str(RESULTS_ROOT / rec["tag"] / "work" / f"{rec['task']}__{method.name}__s{rec['seed']}.sqlite")
    Path(memory_path).parent.mkdir(parents=True, exist_ok=True)
    Path(memory_path).unlink(missing_ok=True)  # every group starts fresh

    episodes_out = []
    if method.kind == "static":
        task = build_task(rec["task"], rec["budget"], trials_per_phase=rec["budget"])
        max_rounds = 1
    else:
        task = build_task(rec["task"], rec["budget"])
        max_rounds = 6

    memory = MemoryStore(memory_path)
    for ep in range(rec["episodes"]):
        seed_ep = rec["seed"] * 1000 + ep
        pipeline = AdaptationPipeline(
            task,
            PipelineConfig(max_rounds=max_rounds, seed=seed_ep, memory_path=memory_path),
            controller=controller,  # shared across episodes (Q-table + epsilon persist)
            memory=memory,
        )
        if method.kind == "static":
            from hpo_adapt.agents.initial_setup import HeuristicInitialSetupAgent

            pipeline.initial_agent = ForcedOptimizerInitialAgent(
                HeuristicInitialSetupAgent(), method.optimizer or "TPE"
            )
        result = pipeline.run()
        episodes_out.append(
            {"episode": ep, "result": result.model_dump(mode="json"), "timings": pipeline.timings}
        )
    return {**rec, "episodes_data": episodes_out}


def _worker_init() -> None:
    os.environ["HPO_ADAPT_INNER_JOBS"] = "1"


# ----------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["main", "episodes", "all"], default="all")
    parser.add_argument("--tasks", nargs="*", default=["synthetic_mlp", "branin", "rf_digits"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--tag", default=None, help="results subdirectory tag")
    parser.add_argument("--smoke", action="store_true", help="tiny sanity matrix")
    args = parser.parse_args()

    if args.smoke:
        args.phase = "all"
        args.tasks = ["synthetic_mlp", "branin"]
        args.seeds = 1
        args.budget = 60
        args.episodes = 3
        args.tag = "smoke"
        args.jobs = 2

    tag = args.tag or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = RESULTS_ROOT / tag
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    units: list[dict] = []
    if args.phase in ("main", "all"):
        for task in args.tasks:
            for method in MAIN_METHODS:
                for seed in range(args.seeds):
                    units.append(
                        {
                            "phase": "main",
                            "tag": tag,
                            "task": task,
                            "method": method.__dict__,
                            "method_name": method.name,
                            "seed": seed,
                            "budget": args.budget,
                            "episodes": 1,
                            "fn": "single",
                        }
                    )
    if args.phase in ("episodes", "all"):
        for task in args.tasks:
            if task == "rf_digits":
                continue  # real-ML task is too expensive for 20-episode sweeps
            for method in EPISODE_METHODS:
                for seed in range(args.seeds):
                    units.append(
                        {
                            "phase": "episodes",
                            "tag": tag,
                            "task": task,
                            "method": method.__dict__,
                            "method_name": method.name,
                            "seed": seed,
                            "budget": args.budget,
                            "episodes": args.episodes,
                            "fn": "group",
                        }
                    )

    print(f"[run_all] tag={tag} units={len(units)} jobs={args.jobs}")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_worker_init) as pool:
        futures = {}
        for rec in units:
            fn = run_single if rec["fn"] == "single" else run_group
            futures[pool.submit(fn, rec)] = rec
        done = 0
        for fut in as_completed(futures):
            rec = futures[fut]
            payload = fut.result()
            fname = f"{rec['phase']}__{rec['task']}__{rec['method_name']}__s{rec['seed']}.json"
            (runs_dir / fname).write_text(json.dumps(payload, indent=1))
            if rec["fn"] == "single":
                res = payload["result"]
                rows.append(
                    {
                        "phase": "main",
                        "task": rec["task"],
                        "method": rec["method_name"],
                        "seed": rec["seed"],
                        "best_raw": res["best_value_raw"],
                        "trials_used": res["total_trials_used"],
                        "rounds": len(res["steps"]),
                        "actions": ";".join(
                            (s.get("action") or {}).get("action", "NONE")
                            for s in res["steps"]
                        ),
                        "hpo_time": payload["timings"]["hpo"],
                        "adapt_time": payload["timings"]["adapt"],
                        "total_time": payload["timings"]["total"],
                        "q_states": res["q_table_size"],
                        "experiences": res["experiences_recorded"],
                    }
                )
            else:
                for ep in payload["episodes_data"]:
                    res = ep["result"]
                    rows.append(
                        {
                            "phase": "episodes",
                            "task": rec["task"],
                            "method": rec["method_name"],
                            "seed": rec["seed"],
                            "episode": ep["episode"],
                            "best_raw": res["best_value_raw"],
                            "trials_used": res["total_trials_used"],
                            "rounds": len(res["steps"]),
                            "actions": ";".join(
                                (s.get("action") or {}).get("action", "NONE")
                                for s in res["steps"]
                            ),
                            "hpo_time": ep["timings"]["hpo"],
                            "adapt_time": ep["timings"]["adapt"],
                            "total_time": ep["timings"]["total"],
                            "q_states": res["q_table_size"],
                            "experiences": res["experiences_recorded"],
                        }
                    )
            done += 1
            print(f"[run_all] {done}/{len(units)}  {fname}")

    fieldnames = sorted({k for r in rows for k in r})
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[run_all] wrote {out_dir / 'summary.csv'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
