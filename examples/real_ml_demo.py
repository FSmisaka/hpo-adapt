"""Real-ML-task demo: RandomForest on sklearn digits, with full observability.

This is the template for plugging in YOUR OWN task. Three steps:

  1. Register an objective function  (params dict -> objective value)
  2. Build a TaskSpec                (available hyperparameters + constraints)
  3. Run AdaptationPipeline          (the closed loop)

The script prints, for every round:
  - the Setup's search space actually used,
  - the DiagnosticState metrics (optimization / importance / boundary
    concentration / good region / diversity / coverage / budget),
  - the RL action and the changes the Revised Setup Agent applied,
  - the (one-phase-delayed) reward,
and afterwards dumps Agent Memory (experiences + semantic_rules) and the
learned Q-table.

Run:
    python examples/real_ml_demo.py                  # 3 rounds x 30 trials (~2-4 min)
    python examples/real_ml_demo.py --epsilon-start 1.0   # RL exploration mode
    python examples/real_ml_demo.py --max-rounds 4

The memory database survives across runs (rf_digits_memory.sqlite) unless
--fresh is passed, so historical Experience accumulates like in production.
"""

import argparse
import json
import sys
from pathlib import Path

from hpo_adapt.memory import MemoryStore
from hpo_adapt.pipeline import AdaptationPipeline, PipelineConfig, rf_digits_task
from hpo_adapt.rl import QController, QLearningConfig

# --------------------------------------------------------------------------
# Step 1 (register the objective) and Step 2 (describe the task) now live in
# the package itself: hpo_adapt.hpo.objectives registers `rf_digits_cv` and
# hpo_adapt.pipeline exposes `rf_digits_task`. This script only wires up the
# run and prints every observable of the closed loop.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Pretty printers for the observables
# --------------------------------------------------------------------------


def _f(v, nd=4):
    return "-" if v is None else round(float(v), nd)


def _metric(m):
    if not isinstance(m, dict) or m.get("status") != "VALID":
        return f"INSUFFICIENT_DATA({m.get('status') if isinstance(m, dict) else m})"
    return _f(m.get("value"))


def print_setup(title: str, search_space: list[dict], frozen: dict | None = None) -> None:
    print(f"\n  [{title}]")
    for spec in search_space:
        if spec["type"] == "categorical":
            rng = f"choices={spec.get('choices')}"
        else:
            rng = f"[{spec.get('low')}, {spec.get('high')}] dist={spec.get('distribution')}"
        print(f"    {spec['name']:<20s} {rng}")
    if frozen:
        print(f"    frozen: {frozen}")


def print_diagnostic(ds: dict) -> None:
    opt, ss, bud = ds["optimization"], ds["search_space"], ds["budget"]
    print("  -- Optimization Status --")
    print(
        f"     best_objective={_metric(opt['best_objective'])}  "
        f"recent_improvement={_metric(opt['recent_improvement'])}  "
        f"relative={_metric(opt['relative_improvement'])}"
    )
    stag = opt["stagnation"]
    print(
        f"     stagnation={stag.get('value') if stag.get('status') == 'VALID' else 'INSUFFICIENT_DATA'}  "
        f"marginal_gain={_metric(opt['marginal_gain'])}"
    )
    print("  -- Search Space Status (per hyperparameter) --")
    for name, imp in ss["importance"].items():
        if imp.get("status") != "VALID":
            print(f"     {name:<20s} importance=INSUFFICIENT_DATA")
            continue
        bc = ss["boundary_concentration"].get(name, {})
        gr = ss["good_region"].get(name, {})
        line = (
            f"     {name:<20s} imp(global)={_f(imp.get('global_importance'))} "
            f"imp(LPI)={_f(imp.get('local_importance'))} rank={imp.get('rank')}"
        )
        if bc.get("status") == "VALID":
            line += (
                f" | BC all={_f(bc.get('all'))} good={_f(bc.get('good'))}"
                f" (upper={_f(bc.get('upper'))}/lower={_f(bc.get('lower'))})"
            )
        if gr.get("status") == "VALID":
            line += (
                f" | good_region mean={_f(gr.get('mean'))} "
                f"[{_f(gr.get('lower'))}, {_f(gr.get('upper'))}] width_ratio={_f(gr.get('width_ratio'))}"
            )
        print(line)
    div = ss["configuration_diversity"]
    print("  -- Diversity / Coverage / Budget --")
    if div.get("status") == "VALID":
        print(
            f"     diversity recent={_f(div.get('recent'))} global={_f(div.get('global'))} "
            f"ratio={_f(div.get('ratio'))}"
        )
    cov = ss["coverage"]
    if cov.get("status") == "VALID":
        print(f"     coverage overall={_f(cov.get('overall'))} per_hp={ {k: _f(v) for k, v in cov.get('per_hyperparameter', {}).items()} }")
    print(
        f"     budget consumed={_metric(bud['budget_consumed'])} "
        f"remaining={_metric(bud['budget_remaining'])} "
        f"(global {bud.get('completed_trials')}/{bud.get('total_trials')} trials)"
    )


def print_memory(store: MemoryStore) -> None:
    rows = store.conn.execute("SELECT * FROM experiences ORDER BY created_at").fetchall()
    print(f"\n==================== Agent Memory: {len(rows)} persisted experiences ====================")
    for row in rows:
        action = json.loads(row["action_json"])
        diag = json.loads(row["diagnosis_json"])
        outcome = json.loads(row["outcome_json"])
        print(
            f"  {row['experience_id']}  diagnosis={diag['primary']}  "
            f"action={action['type']}@{action.get('target')}  "
            f"rel_improvement={_f(outcome.get('relative_improvement'))}  "
            f"reward={_f(row['reward'])}  success={bool(row['success'])}"
        )
    print(f"  ---- semantic_rules (aggregated from EVERY recorded outcome) ----")
    for rule in store.rules():
        print(
            f"  {rule['condition']} -> {rule['recommended_action']}: "
            f"success_rate={_f(rule['success_rate'], 2)} n={rule['sample_count']} "
            f"avg_reward={_f(rule['average_reward'])}"
        )


def print_qtable(controller: QController) -> None:
    print(f"\n==================== Learned Q-table ({len(controller.q_table)} states) ====================")
    for key, row in controller.q_table.items():
        best = max(row, key=row.get)
        nonzero = {a: round(v, 3) for a, v in row.items() if abs(v) > 1e-9}
        print(f"  state={'|'.join(key)}")
        print(f"    best={best}  nonzero={nonzero or 'all zero (unvisited)'}")


# --------------------------------------------------------------------------
# Step 3: run the closed loop and watch everything.
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--trials-per-phase", type=int, default=30)
    parser.add_argument("--max-total-trials", type=int, default=90)
    parser.add_argument("--epsilon-start", type=float, default=0.0,
                        help="0 = greedy (interpretable demo), 1.0 = full RL exploration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-path", default="rf_digits_memory.sqlite")
    parser.add_argument("--fresh", action="store_true", help="delete the memory DB before running")
    args = parser.parse_args()

    if args.fresh and args.memory_path != ":memory:":
        Path(args.memory_path).unlink(missing_ok=True)

    task = rf_digits_task(args.max_total_trials, args.trials_per_phase)
    memory = MemoryStore(args.memory_path)
    controller = QController(
        config=QLearningConfig(seed=args.seed, epsilon_start=args.epsilon_start)
    )
    pipeline = AdaptationPipeline(
        task,
        PipelineConfig(max_rounds=args.max_rounds, seed=args.seed),
        controller=controller,
        memory=memory,
    )

    print(f"Task: {task.task_id} | objective={task.objective_name} ({task.direction.value})")
    print(f"budget: {args.trials_per_phase} trials/phase, global cap {args.max_total_trials}, "
          f"rounds<={args.max_rounds}, epsilon_start={args.epsilon_start}")

    result = pipeline.run()

    print("\n==================== Round-by-round adaptation ====================")
    for i, step in enumerate(result.steps):
        print(f"\n########## Round {step.round_index}  (setup {step.setup_id}) ##########")
        print_setup("Search space used this round", step.search_space)
        if step.diagnostic_state:
            print("  -- DiagnosticState --")
            print_diagnostic(step.diagnostic_state)
        act = step.action or {}
        print(
            f"  -- RLAction: {act.get('action')}@{act.get('candidate_hyperparameter')} "
            f"q={act.get('q_value')} conf={act.get('confidence')}  diagnosis={step.diagnosis}"
        )
        for ch in step.changes:
            print(f"     change: {ch['hyperparameter']}\n       old={ch['old']}\n       new={ch['new']}")
        print(
            f"  -- outcome: trials={step.trials_used} phase_best={_f(step.phase_best)} "
            f"cumulative_best={_f(step.cumulative_best)}"
        )
        print(
            f"     reward={_f(step.reward)}"
            + ("   (closes the PREVIOUS round's action — one-phase delay)" if i > 0 else "")
        )
        if step.terminated:
            print("     episode TERMINATED by STOP action")

    print("\n==================== Final result ====================")
    print(
        f"best CV accuracy (raw space) = {_f(result.best_value_raw)}\n"
        f"best params = {result.best_params}\n"
        f"total trials used = {result.total_trials_used}/{args.max_total_trials}"
    )

    print_memory(memory)
    print_qtable(controller)
    print(
        f"\nMemory DB kept at: {Path(args.memory_path).resolve()}\n"
        f"Inspect it directly with: sqlite3 {args.memory_path} 'SELECT experience_id, reward, success FROM experiences;'"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
