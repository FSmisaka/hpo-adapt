# HPO-Adapt: Diagnostic-Driven Adaptive HPO Setup Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The performance of hyperparameter optimization (HPO) is largely decided
**before** the first trial runs: the initial *setup* — search space,
optimizer, budget — fixes what can be found at all. **HPO-Adapt** closes
the loop: while the search is running, a deterministic **Diagnostic
Engine** watches the trial trajectory, an RL **Controller** decides which
of seven high-level interventions to apply, and a **Revised Setup Agent**
executes it as a validated new setup. Every round is observable on a live
local dashboard.

```
Task ──▶ Initial Setup Agent ──▶ S0 ──▶ Classical HPO ──▶ TrialLogs ──▶ Diagnostic Engine
                                                               │
               Memory ◀── Experience ◀── Outcome ◀── Revised Setup Agent ◀── RL Controller
                  │                     (executes A_t)              (selects A_t: what to do)
                  ▼
         (prior knowledge for the next round)
```

- **Diagnostic Engine = Observe** — deterministic Python, no LLM in the metrics
- **RL Controller = Decide** — tabular Q-Learning over 7 interventions
- **Revised Setup Agent = Execute** — turns the intervention into a concrete, validated new Setup

## 1. Quick start

```bash
make setup          # create .venv and install hpo-adapt + dev dependencies
source .venv/bin/activate
make demo           # live demo: real training task + dashboard (~5-6 min)
```

`make demo` runs [examples/demo_dashboard.py](examples/demo_dashboard.py):
HistGradientBoosting on a hard synthetic dataset, 120 real training runs
organized into 4 adaptive rounds. The terminal prints a dashboard URL —
open it in a browser to watch, round by round: the setup actually used,
every trial, all diagnostic metrics (importance, boundary concentration,
good region, stagnation, ...), which action the controller picked and why,
the concrete setup changes (old → new), and the reward each intervention
earned. Zero extra dependencies: stdlib server + a single self-contained
HTML page.

Want it on your own task? The whole integration is ~10 lines, because the
search space you provide is only an **initial guess** — the loop repairs
it from evidence:

```python
import hpo_adapt as ha

def train_evaluate(params):        # 1. your existing training / evaluation code
    ...                            #    (raise RuntimeError/ValueError on failure)
    return holdout_accuracy        #    the metric to maximize (or minimize)

run = ha.run(                      # 2. plug it in — everything else is automatic
    train_evaluate,
    space=dict(
        max_iter=ha.Int(20, 300, default=80),
        max_leaf_nodes=ha.Int(10, 64, default=24),
        learning_rate=ha.Float(0.08, 0.12, log=True, default=0.1),
    ),
    direction="maximize",
    total_trials=120,
)
print(run.best_value, run.best_params)
```

> The `space` bounds do **not** require domain knowledge: give a rough,
> default-centered range and the tool will shift/expand/freeze it based on
> the experimental trajectory. Bounds only act as hard safety limits.

A detailed Chinese walkthrough lives in [examples/README.md](examples/README.md).

## 2. The core API

| Entry point | What it does |
|---|---|
| `ha.run(objective, space, ...)` | One-call wrapper: builds the `TaskSpec`, registers the objective, starts the dashboard, runs the closed loop. Covers 90% of use cases. |
| `ha.Float / ha.Int / ha.Choice` | Space DSL: `low/high` are hard adaptation limits, `default` centers the initial compact range, `log=True` for log-scale parameters. |
| `AdaptationPipeline(task, PipelineConfig())` | Fully configurable lower-level entry point (custom controller, memory, `event_sink=` observer) — what the experiments build on. |
| `hpo-adapt` CLI | Built-in tasks: `hpo-adapt --task synthetic_mlp \| branin \| rf_digits [--dashboard] [--json]`. |

Key `ha.run` options: `direction`, `total_trials`, `max_rounds`,
`trials_per_phase`, `memory_path` (SQLite file → experience persists
across runs), `seed` (fixes every stochastic source), `dashboard`,
`open_browser`. The returned `RunResult` exposes `best_value`,
`best_params`, `rounds`, and `wait()`.

**Action space** (7 interventions): `EXPAND_SEARCH_SPACE`,
`SHIFT_SEARCH_REGION`, `SHRINK_SEARCH_SPACE`, `FREEZE_HYPERPARAMETER`,
`INCREASE_EXPLORATION`, `INCREASE_BUDGET`, `STOP` (evidence-gated).
Reward: `r = clip(Δ − λ·C − r_invalid, −r_max, r_max)` with
`Δ = (B_{t+1} − B_t)/(|B_t|+ε)`.

**Design guarantees**: every setup passes `validate_setup` before HPO runs
it; metrics below their minimum trial counts return `INSUFFICIENT_DATA`
instead of fabricated values; new phases are warm-started with all past
compatible trials (no re-evaluation, no budget cost); both setup agents
have deterministic heuristic backends (default, used in all experiments)
and optional LLM backends (`--backend llm`, any OpenAI-compatible
endpoint, outputs still schema-validated).

## 3. Experiments & reproduction

Four experiments back four claims (all under an identical 120-trial
budget, 10 seeds, heuristic backends, one command each):

| Claim | Experiment | Result (10 seeds) |
|---|---|---|
| **C1.** With a flawed initial setup, the adaptive loop beats equal-budget static HPO | `make exp1` — 6 methods × 3 tasks (180 runs) | synthetic_mlp: guided adapt **1.000±0.000** vs static-tpe 0.755±0.003 (+32% rel.); Branin: **0.498±0.198** vs 0.896±0.010 (global min 0.398); rf_digits: tied at the CV plateau — value = observability, "do no harm" |
| **C2.** The gain comes from diagnosis *guidance*, not random wiggling (ablation) | `make exp2` — 4 deciders × 3 tasks (120 runs) | variance: guided 1.000**±0.000** (worst seed = best seed) vs random-action 0.873**±0.134** (~6× spread); Branin ordering 0.498 < 0.596 < 0.776 |
| **C3.** The RL controller improves with accumulated experience | `make exp3` — 60 groups × 20 episodes, shared Q-table + memory | synthetic: per-episode best 0.892 → 0.949 (random control flat 0.867 → 0.855); Branin: 0.734 → 0.524 (control drifts 0.768 → 0.819) |
| **C4.** The observe–decide layer is nearly free | `make exp4` — overhead on the real-ML task | diagnostics + RL + memory + agents = **1.4%** of wall-clock on rf_digits (worst case 1.7%) |

Reproduce everything:

```bash
make setup && source .venv/bin/activate
make experiments     # exp1 + exp2 + exp3 + exp4 (~30 min total, N-1 workers)
make figures         # tidy per-figure CSVs + PNG previews under experiments/results/full/figs/
make smoke           # 2-minute sanity matrix first, if you like
make test            # 149 unit tests
```

Results land in `experiments/results/full/` (`runs/*.json` per run,
`summary_exp*.csv` per experiment); `make figures` produces
`anytime.csv`, `main_summary.csv`, `episodes_summary.csv`,
`actions.csv`, `overhead_summary.csv`, `case_study.json` (+ PNG
previews). Experiment sources: [experiments/](experiments/) — one file
per claim, shared harness in `common.py`.

## Project layout

```
src/hpo_adapt/     the package (schemas, hpo, diagnostic, agents, rules, rl, memory, pipeline, api, dashboard)
examples/          demo_dashboard.py + Chinese usage guide
experiments/       exp1..exp4 (one per claim), common.py, make_figures.py
tests/             149 unit tests
```

## License

MIT — see [LICENSE](LICENSE).
