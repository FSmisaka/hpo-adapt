# HPO-Adapt: Diagnostic-Driven Adaptive HPO Setup Control

Implementation of `spec.md`: combining **trajectory analysis** (CAVE-style diagnosis) with
**agent-driven setup adaptation**, controlled by **tabular Q-Learning**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Quick start: wrap any ML task in ~10 lines (wandb-style)

Any ML task that satisfies the basic task contract — an objective
`train_evaluate(params) -> float` plus a declarative search space — becomes a
self-optimizing closed loop with a live local dashboard:

```python
import hpo_adapt as ha

def train_evaluate(params):        # 1. your existing training / evaluation code
    ...
    return cv_accuracy             #    the metric to maximize (or minimize)

run = ha.run(                      # 2. plug it in — everything else is automatic
    train_evaluate,
    space=dict(
        C=ha.Float(0.01, 100.0, log=True, default=1.0),
        gamma=ha.Float(1e-6, 1.0, log=True, default=1e-3),
        kernel=ha.Choice(["rbf", "poly"], default="rbf"),
    ),
    direction="maximize",
    total_trials=90,
)
print(run.best_value, run.best_params)
```

Running this prints a local URL in the terminal:

```
┌────────────────────────────────────────────────────────────┐
│ hpo-adapt · adaptive HPO closed loop                        │
│   task      : svc_digits_abc123                             │
│   objective : cv_accuracy (maximize)                        │
│   budget    : 90 trials · up to 5 rounds                   │
│   memory    : examples/svc_digits_memory.sqlite             │
└────────────────────────────────────────────────────────────┘

  ▶ Dashboard live at  http://127.0.0.1:8710
    Open this URL in your browser to watch the run in real time.
```

The dashboard shows, round by round: how the initial setup (search space /
optimizer / per-phase budget) was derived, every trial, the DiagnosticState
(importance, boundary concentration, good region, stagnation, ...), which RL
action was selected and why, the concrete setup changes old → new, and the
reward each intervention earned. Zero extra dependencies — the server is
Python stdlib and the page is a single self-contained HTML file.

**Live demo** (RandomForest-free variant: RBF-SVM on sklearn digits):

```bash
python examples/demo_dashboard.py            # terminal prints the dashboard URL
python examples/demo_dashboard.py --fresh    # start with an empty agent memory
```

## The closed loop

```
Task ──▶ Initial Setup Agent ──▶ S0 ──▶ Classical HPO ──▶ TrialLogs ──▶ Diagnostic Engine
                                                              │
              Memory ◀── Experience ◀── Outcome ◀── Revised Setup Agent ◀── RL Controller
                 │                     (executes A_t)              (selects A_t: what to do)
                 ▼
        (prior knowledge for the next round)
```

- **Diagnostic Engine = Observe** (deterministic Python, no LLM in the metrics)
- **RL Controller = Decide** (tabular Q-Learning over 7 high-level interventions)
- **Revised Setup Agent = Execute** (turns the intervention into a concrete new Setup)

## Modules (implementation order as in spec)

| Stage | Module | Path |
|---|---|---|
| 0 | Structured schemas + Tool Validator | `src/hpo_adapt/schemas/`, `src/hpo_adapt/validator.py` |
| 1 | Classical HPO (Optuna) | `src/hpo_adapt/hpo/` |
| 2 | Diagnostic Engine (fANOVA + LPI, boundary concentration, good region, diversity, coverage, budget) | `src/hpo_adapt/diagnostic/` |
| 3 | Initial Setup Agent (heuristic + LLM backends) | `src/hpo_adapt/agents/initial_setup.py` |
| 4 | Rule-based Adaptation (7 diagnostic patterns + CandidateScore ranking) | `src/hpo_adapt/rules/adaptation.py` |
| 5 | Revised Setup Agent (executes the 7 actions) | `src/hpo_adapt/agents/revised_setup.py` |
| 6 | Agent Memory (SQLite: `experiences` + `semantic_rules`) | `src/hpo_adapt/memory/store.py` |
| 7 | RL Controller (tabular Q-Learning) | `src/hpo_adapt/rl/` |
| 8 | Closed-loop pipeline + CLI | `src/hpo_adapt/pipeline.py` |
| 9 | One-function user API (`ha.run`) | `src/hpo_adapt/api.py` |
| 10 | Live dashboard (stdlib server + single-page app) | `src/hpo_adapt/dashboard/` |

## Core data objects

`TaskSpec`, `Setup` (search space / optimizer / budget / evaluation), `TrialLog`,
`DiagnosticState` (every metric carries `VALID` / `INSUFFICIENT_DATA`), `RLAction`
(`A_t = (ActionType, Target, Parameters)`), `Experience`.

All inter-module communication goes through these pydantic schemas, and **every Setup
must pass `validate_setup` before HPO is allowed to run it** (the validator rejects
invented hyperparameters, invalid ranges, unknown optimizers, direction mismatches, ...).

## Action space (first version, exactly the 7 from spec)

`EXPAND_SEARCH_SPACE`, `SHIFT_SEARCH_REGION`, `SHRINK_SEARCH_SPACE`,
`FREEZE_HYPERPARAMETER`, `INCREASE_EXPLORATION`, `INCREASE_BUDGET`, `STOP`

Reward: `r = clip(Δ − λ·C − r_invalid, −r_max, r_max)` with
`Δ = (B_{t+1} − B_t)/(|B_t|+ε)`, `C = N_phase/N_total`.
Q-Learning defaults: `α=0.1, γ=0.9, ε: 1.0 → 0.05 (decay 0.995)`.

## Install & run

```bash
pip install -e ".[dev]"
pytest                                   # 147 tests
hpo-adapt --task synthetic_mlp --max-rounds 4 --max-total-trials 120 --seed 3
hpo-adapt --task branin --max-rounds 3                   # minimize-direction example
hpo-adapt --task rf_digits                               # real-ML task (RandomForest + CV)
hpo-adapt --task synthetic_mlp --dashboard               # ...with the live local dashboard
hpo-adapt --task synthetic_mlp --json                    # machine-readable result
```

Example output:

```
Task: task_synth_001 (maximize)  backend=heuristic
round 0: setup=setup_000 trials=40 phase_best=0.584 ... action=INCREASE_EXPLORATION@optimizer reward=-0.033
round 1: setup=setup_001 trials=40 ... action=INCREASE_EXPLORATION@optimizer reward=-0.033
round 2: setup=setup_002 trials=40 ... action=INCREASE_EXPLORATION@optimizer reward=None
total_trials=120 best(raw)=0.584 best_params={...} q_states=3 experiences=2
```

Note: with ε-greedy exploration (ε starts at 1.0) some seeds randomly select `STOP`
early — that is standard ε-greedy behavior over the 7-action space; change `--seed`
to see different adaptation trajectories.

### Python API

```python
from hpo_adapt.pipeline import AdaptationPipeline, PipelineConfig, synthetic_mlp_task

task = synthetic_mlp_task(max_total_trials=120)
result = AdaptationPipeline(task, PipelineConfig(max_rounds=4)).run()
print(result.best_value_raw, result.best_params)
```

The `ha.run()` wrapper above builds the `TaskSpec`, registers the objective,
starts the dashboard and prints the URL; `AdaptationPipeline` remains the
fully configurable entry point (`event_sink=` accepts any
`(event_type, payload) -> None` observer for custom live tooling).

## Real ML task example (with full observability)

```bash
python examples/real_ml_demo.py --fresh     # RandomForest on sklearn digits, 3 rounds
hpo-adapt --task rf_digits                  # same task through the CLI
```

The example is the template for plugging in your own task (register objective ->
build `TaskSpec` -> run `AdaptationPipeline`) and prints, round by round: the search
space actually used, all DiagnosticState metrics, the RL action and the applied
changes, plus the final Agent Memory contents (`experiences` + `semantic_rules`) and
the learned Q-table. The memory SQLite file is kept on disk so you can inspect it
directly (`sqlite3 rf_digits_memory.sqlite ...`). See the Chinese walkthrough in
[README_CN.md](README_CN.md) for a metric-by-metric reading guide.

## Reproducible experiments & paper

Everything behind the tool paper lives in `experiments/` and `paper/`:

```bash
python experiments/run_all.py --phase all --tag full   # ~20 min, 240 runs (7 workers)
python experiments/make_figures.py --tag full          # tidy CSVs per figure + case study
```

`experiments/results/full/figs/` then contains the exact data for every
figure/table (`anytime.csv`, `main_summary.csv`, `episodes_summary.csv`,
`case_study.json`, `overhead_summary.csv`); figure specifications are in
`paper/FIGURES.md`, and `paper/paper.pdf` is the compiled 4-page tool paper
(LaTeX sources in `paper/`).

## LLM backends (optional)

Both Setup Agents have deterministic heuristic backends (default) and LLM backends that
use the verbatim prompts from `spec.md`. To enable the LLM backend (no keys are ever
hardcoded):

```bash
export OPENAI_API_KEY=...            # required
export OPENAI_BASE_URL=...           # optional (OpenAI-compatible endpoints)
export OPENAI_MODEL=gpt-4o-mini      # optional
hpo-adapt --backend llm ...
```

LLM outputs are parsed and passed through the same Tool Validator; anything that
violates the schema or the task constraints is rejected.

## Conventions & notes

- **Internal maximized space**: the Diagnostic Engine and rewards operate on
  `ỹ = y (maximize) / −y (minimize)` per spec; `PipelineResult.best_value_raw` converts
  back to the original task direction.
- **Cumulative diagnostics**: each round's DiagnosticState is computed over the whole
  episode trajectory so far (CAVE-style), so importance becomes available as soon as
  30 cumulative trials exist — not 30 trials within a single phase.
- **Warm-started phases**: every new phase's Optuna study is seeded with all past
  completed trials compatible with the current search space (no re-evaluation, no
  budget consumption), so setup revisions inherit all previous knowledge.
- **Evidence-gated STOP**: `STOP` is only selectable when stagnation is observed AND at
  least half the global budget is consumed — ε-greedy exploration can no longer
  terminate an episode prematurely.
- **Insufficient data**: metrics below their minimum trial counts (5 / 20 / 30) return
  `INSUFFICIENT_DATA` instead of fabricated values; hyperparameter-targeted actions are
  unavailable while importance is not `VALID`.
- **Invalid actions** (e.g., expanding a categorical parameter) never consume budget;
  they receive the fixed penalty reward and are retried within the round. When no
  further phase can evaluate an action (budget exhausted or `max_rounds` reached), the
  loop closes without selecting one — every executed action receives an outcome, a
  reward, and a Q-update.
- **Optimizers**: RANDOM, TPE (all tasks) and CMA-ES (purely numerical tasks such as
  `branin`; the validator rejects it for tasks with categorical parameters).
- **Memory writes** happen only after a complete adaptation step, and (by default) only
  successful interventions and useful negative experiences are persisted —
  `semantic_rules` aggregates every recorded outcome regardless.
- **Seeding**: `--seed` / `PipelineConfig.seed` controls every stochastic source — HPO
  sampling (via `evaluation.seed`), the RL controller, and the surrogate forests; a
  given seed + fresh memory reproduces a run exactly.
- **Observability**: `StepRecord` stores each round's search space, full
  DiagnosticState, trial log projection, applied changes and (one-phase-delayed) reward;
  `pipeline.timings` breaks wall-clock time into HPO vs. observe+decide layers
  (`experiments/` builds all paper tables from these).
