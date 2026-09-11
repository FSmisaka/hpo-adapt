# HPO-Adapt: Diagnostic-Driven Adaptive HPO Setup Control

Implementation of `spec.md`: combining **trajectory analysis** (CAVE-style diagnosis) with
**agent-driven setup adaptation**, controlled by **tabular Q-Learning**.

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
pytest                                   # 109 tests
hpo-adapt --task synthetic_mlp --max-rounds 4 --max-total-trials 120 --seed 3
hpo-adapt --task branin --max-rounds 3                   # minimize-direction example
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

## Real ML task example (with full observability)

```bash
python examples/real_ml_demo.py --fresh     # RandomForest on sklearn digits, 3 rounds
```

The example is the template for plugging in your own task (register objective ->
build `TaskSpec` -> run `AdaptationPipeline`) and prints, round by round: the search
space actually used, all DiagnosticState metrics, the RL action and the applied
changes, plus the final Agent Memory contents (`experiences` + `semantic_rules`) and
the learned Q-table. The memory SQLite file is kept on disk so you can inspect it
directly (`sqlite3 rf_digits_memory.sqlite ...`). See the Chinese walkthrough in
[README_CN.md](README_CN.md) for a metric-by-metric reading guide.

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
- **Insufficient data**: metrics below their minimum trial counts (5 / 20 / 30) return
  `INSUFFICIENT_DATA` instead of fabricated values; hyperparameter-targeted actions are
  unavailable while importance is not `VALID`.
- **Invalid actions** (e.g., expanding a categorical parameter) never consume budget;
  they receive the fixed penalty reward and are retried within the round.
- **Memory writes** happen only after a complete adaptation step, and (by default) only
  successful interventions and useful negative experiences are persisted —
  `semantic_rules` aggregates every recorded outcome regardless.
- Deterministic: fixed seeds everywhere (HPO samplers, surrogate forests, RL RNG);
  `PipelineResult` is reproducible for a given seed + fresh memory.
