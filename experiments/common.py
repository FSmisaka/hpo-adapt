"""Shared infrastructure for the four HPO-Adapt experiments (claims 1-4).

Each claim lives in its own experiment file and reuses the method specs,
runners and parallel harness defined here:

    exp1_adaptive_vs_static.py   Claim 1: adaptive loop vs equal-budget static HPO
    exp2_ablation_guidance.py   Claim 2: guidance (not random wiggling) drives the gain
    exp3_cross_episode.py       Claim 3: the RL controller improves across episodes
    exp4_overhead.py            Claim 4: the observe-decide layer is nearly free

All experiments write into ``experiments/results/<tag>/`` with the layout::

    runs/<phase>__<task>__<method>__s<seed>.json   full per-run record
    summary_<experiment>.csv                        one row per run

``make_figures.py`` then aggregates ``runs/*.json`` into the paper's figure
data (``figs/*.csv``), independent of how many experiments have been run.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULTS_ROOT = ROOT / "experiments" / "results"

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


STATIC_RANDOM = MethodSpec("static-random", "static", optimizer="RANDOM")
STATIC_TPE = MethodSpec("static-tpe", "static", optimizer="TPE")
ADAPT_RANDOM_ACTION = MethodSpec("adapt-random-action", "adapt", epsilon_start=1.0, epsilon_end=1.0)
ADAPT_RULE_ONLY = MethodSpec("adapt-rule-only", "adapt", epsilon_start=0.0, epsilon_end=0.0)
ADAPT_RL_DEFAULT = MethodSpec("adapt-rl-default", "adapt", epsilon_start=1.0, epsilon_end=0.05)
ADAPT_RL_GUIDED = MethodSpec("adapt-rl-guided", "adapt", epsilon_start=0.2, epsilon_end=0.05)
ADAPT_RL = MethodSpec("adapt-rl", "adapt", epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.95)

TASK_BUILDERS = ("synthetic_mlp", "branin", "rf_digits")


def build_task(name: str, total: int, trials_per_phase: Optional[int] = None):
    from hpo_adapt.pipeline import branin_task, rf_digits_task, synthetic_mlp_task

    if name == "synthetic_mlp":
        task = synthetic_mlp_task(total)
    elif name == "branin":
        task = branin_task(total)
    elif name == "rf_digits":
        task = rf_digits_task(total)
    else:
        raise ValueError(name)
    if trials_per_phase is not None:
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


def make_controller(method: MethodSpec, seed: int):
    from hpo_adapt.rl import QController, QLearningConfig

    cfg: dict[str, Any] = dict(seed=seed)
    if method.epsilon_start is not None:
        cfg["epsilon_start"] = method.epsilon_start
    if method.epsilon_end is not None:
        cfg["epsilon_end"] = method.epsilon_end
    if method.epsilon_decay is not None:
        cfg["epsilon_decay"] = method.epsilon_decay
    return QController(config=QLearningConfig(**cfg))


def _build_pipeline(rec: dict, method: MethodSpec):
    """Assemble the pipeline for one experimental unit (task, method, seed)."""
    from hpo_adapt.pipeline import AdaptationPipeline, PipelineConfig

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
        controller=make_controller(method, rec["seed"]),
    )
    if method.kind == "static":
        from hpo_adapt.agents.initial_setup import HeuristicInitialSetupAgent

        pipeline.initial_agent = ForcedOptimizerInitialAgent(
            HeuristicInitialSetupAgent(), method.optimizer or "TPE"
        )
    return pipeline


# ----------------------------------------------------------------------- runs
def run_single(rec: dict) -> dict:
    """One experimental unit: (task, method, seed) — single episode."""
    os.environ.setdefault("HPO_ADAPT_INNER_JOBS", "1")
    method = MethodSpec(**rec["method"])
    pipeline = _build_pipeline(rec, method)
    result = pipeline.run()
    return {**rec, "result": result.model_dump(mode="json"), "timings": pipeline.timings}


def run_group(rec: dict) -> dict:
    """One experimental unit for the cross-episode phase: N consecutive
    episodes sharing the QController and a persistent SQLite memory."""
    os.environ.setdefault("HPO_ADAPT_INNER_JOBS", "1")
    from hpo_adapt.memory import MemoryStore
    from hpo_adapt.pipeline import AdaptationPipeline, PipelineConfig

    method = MethodSpec(**rec["method"])
    controller = make_controller(method, rec["seed"])
    memory_path = str(
        RESULTS_ROOT / rec["tag"] / "work" / f"{rec['task']}__{method.name}__s{rec['seed']}.sqlite"
    )
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


# ------------------------------------------------------------------- harness
def _worker_init() -> None:
    os.environ["HPO_ADAPT_INNER_JOBS"] = "1"


def _summary_rows(payload: dict) -> list[dict]:
    """One CSV row per episode (single runs have exactly one episode)."""
    if payload.get("result") is not None:
        episodes = [{"episode": None, "result": payload["result"], "timings": payload["timings"]}]
    else:
        episodes = payload["episodes_data"]
    rows = []
    for ep in episodes:
        res = ep["result"]
        rows.append(
            {
                "task": payload["task"],
                "method": payload["method_name"],
                "seed": payload["seed"],
                "episode": ep["episode"],
                "best_raw": res["best_value_raw"],
                "trials_used": res["total_trials_used"],
                "rounds": len(res["steps"]),
                "actions": ";".join(
                    (s.get("action") or {}).get("action", "NONE") for s in res["steps"]
                ),
                "hpo_time": ep["timings"]["hpo"],
                "adapt_time": ep["timings"]["adapt"],
                "total_time": ep["timings"]["total"],
                "q_states": res["q_table_size"],
                "experiences": res["experiences_recorded"],
            }
        )
    return rows


def execute(units: list[dict], *, tag: str, jobs: int, summary_name: str) -> list[dict]:
    """Run all experimental units in parallel, persist JSON + summary CSV.

    Every unit is a dict as produced by :func:`single_unit` /
    :func:`group_unit`; ``fn`` selects the runner. Returns the summary rows.
    """
    out_dir = RESULTS_ROOT / tag
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{summary_name}] tag={tag} units={len(units)} jobs={jobs}")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init) as pool:
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
            rows.extend(_summary_rows(payload))
            done += 1
            print(f"[{summary_name}] {done}/{len(units)}  {fname}")

    if rows:
        fieldnames = sorted({k for r in rows for k in r})
        with open(out_dir / f"summary_{summary_name}.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[{summary_name}] wrote {out_dir / f'summary_{summary_name}.csv'} ({len(rows)} rows)")
    return rows


def single_unit(phase: str, task: str, method: MethodSpec, seed: int, budget: int, tag: str) -> dict:
    return {
        "phase": phase,
        "tag": tag,
        "task": task,
        "method": method.__dict__,
        "method_name": method.name,
        "seed": seed,
        "budget": budget,
        "episodes": 1,
        "fn": "single",
    }


def group_unit(phase: str, task: str, method: MethodSpec, seed: int, budget: int, tag: str, episodes: int) -> dict:
    return {
        "phase": phase,
        "tag": tag,
        "task": task,
        "method": method.__dict__,
        "method_name": method.name,
        "seed": seed,
        "budget": budget,
        "episodes": episodes,
        "fn": "group",
    }


# ------------------------------------------------------------------- parsers
def base_parser(description: str):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds (0..n-1)")
    parser.add_argument("--budget", type=int, default=BUDGET, help="total trials per run")
    parser.add_argument(
        "--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1), help="parallel workers"
    )
    parser.add_argument("--tag", default="full", help="results subdirectory tag")
    parser.add_argument("--smoke", action="store_true", help="tiny sanity matrix (tag=smoke)")
    return parser


def apply_smoke(args, *, tasks: list[str], episodes: Optional[int] = None) -> None:
    args.tasks = tasks
    args.seeds = 1
    args.budget = 60
    if episodes is not None:
        args.episodes = episodes
    args.tag = "smoke"
    args.jobs = 2


def timestamped_tag(tag: str) -> str:
    return tag or time.strftime("run_%Y%m%d_%H%M%S")
