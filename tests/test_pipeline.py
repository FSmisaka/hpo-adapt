"""Stage 8 tests: full closed-loop pipeline.

Task -> Initial Setup -> HPO -> Diagnostic -> RL Action -> Revised Setup
      -> HPO -> Memory
"""

import json

import pytest

from hpo_adapt.memory import MemoryStore
from hpo_adapt.pipeline import (
    AdaptationPipeline,
    PipelineConfig,
    branin_task,
    main,
    synthetic_mlp_task,
)
from hpo_adapt.rl import QController, QLearningConfig


def make_pipeline(tmp_path, task, epsilon_start=0.0, max_rounds=3, total=60, seed=0):
    config = PipelineConfig(
        max_rounds=max_rounds,
        memory_path=str(tmp_path / "memory.sqlite"),
        seed=seed,
    )
    controller = QController(
        config=QLearningConfig(seed=seed, epsilon_start=epsilon_start)
    )
    return AdaptationPipeline(task, config, controller=controller)


class TestClosedLoop:
    def test_full_closed_loop_synthetic(self, tmp_path):
        pipeline = make_pipeline(tmp_path, synthetic_mlp_task(max_total_trials=60))
        result = pipeline.run()

        assert len(result.steps) >= 2, "closed loop must run at least one adaptation"
        assert result.total_trials_used <= 60
        assert result.best_value is not None and result.best_value > 0
        assert result.q_table_size >= 1
        assert result.experiences_recorded >= 1

        # every step has a diagnosis and an executed (valid) action
        for step in result.steps:
            assert step.diagnosis is not None
            assert step.action is not None
            assert step.trials_used > 0
            # observability: per-round search-space snapshot + full diagnostic state
            assert step.search_space and step.search_space[0]["name"]
            assert step.diagnostic_state is not None
            assert "optimization" in step.diagnostic_state
            assert "search_space" in step.diagnostic_state
            # budget diagnostics reflect the GLOBAL budget, not the phase budget
            budget = step.diagnostic_state["budget"]
            assert budget["total_trials"] == 60
            assert budget["budget_consumed"]["value"] == pytest.approx(
                budget["completed_trials"] / 60, abs=1e-9
            )
        # the first phase cannot have consumed the whole global budget
        first_budget = result.steps[0].diagnostic_state["budget"]
        assert first_budget["budget_consumed"]["value"] < 1.0

        # the loop actually revised the setup at least once
        adapted = [s for s in result.steps if s.setup_after_id and s.setup_after_id != s.setup_id]
        assert adapted

    def test_closed_loop_is_deterministic(self, tmp_path):
        r1 = make_pipeline(tmp_path / "a", synthetic_mlp_task(max_total_trials=60)).run()
        r2 = make_pipeline(tmp_path / "b", synthetic_mlp_task(max_total_trials=60)).run()
        assert r1.model_dump() == r2.model_dump()

    def test_stop_terminates_episode(self, tmp_path):
        from hpo_adapt.rl import ALL_ACTIONS
        from hpo_adapt.schemas import ActionType, DiagnosticState, RLAction, Setup, TaskSpec

        # Force STOP selection via a stub controller (epsilon=0, empty rules).
        class StopController(QController):
            def select_action(self, state, setup, rule_recommendation=None):
                action = RLAction(action=ActionType.STOP, target=None, confidence=1.0, q_value=0.0)
                self.observe(state, setup)
                return action

        task = synthetic_mlp_task(max_total_trials=20)
        config = PipelineConfig(max_rounds=3, memory_path=str(tmp_path / "m.sqlite"))
        pipeline = AdaptationPipeline(task, config, controller=StopController())
        result = pipeline.run()
        assert result.terminated is True
        assert result.steps[0].terminated is True
        assert result.steps[0].action["action"] == "STOP"

    def test_minimize_direction_branin(self, tmp_path):
        task = branin_task(max_total_trials=60)
        result = make_pipeline(tmp_path, task, max_rounds=2, total=60).run()
        assert result.best_value_raw is not None
        assert result.best_value == pytest.approx(-result.best_value_raw, abs=1e-12)
        assert result.best_value_raw < 30.0

    def test_memory_persists_to_sqlite_file(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "memory.sqlite"  # also exercises dir creation
        config = PipelineConfig(max_rounds=2, memory_path=str(path), seed=0)
        controller = QController(config=QLearningConfig(seed=0, epsilon_start=0.0))
        result = AdaptationPipeline(
            synthetic_mlp_task(max_total_trials=40), config, controller=controller
        ).run()
        assert path.exists()
        store = MemoryStore(path)
        assert store.experience_count() >= 1
        assert result.experiences_recorded >= 1

    def test_qtable_saved(self, tmp_path):
        qpath = tmp_path / "q.json"
        config = PipelineConfig(
            max_rounds=2, memory_path=str(tmp_path / "m.sqlite"), qtable_path=str(qpath)
        )
        result = AdaptationPipeline(synthetic_mlp_task(max_total_trials=40), config).run()
        assert qpath.exists()
        saved = json.loads(qpath.read_text())
        assert sum(len(v) for v in saved["q_table"].values()) > 0
        assert result.q_table_size >= 1


class TestCLI:
    def test_main_json_output(self, capsys):
        code = main(
            [
                "--task", "synthetic_mlp",
                "--max-rounds", "2",
                "--max-total-trials", "40",
                "--seed", "7",
                "--json",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert "steps" in payload
        assert "best_value_raw" in payload
        assert payload["total_trials_used"] <= 40

    def test_main_human_output(self, capsys):
        code = main(
            [
                "--task", "branin",
                "--max-rounds", "2",
                "--max-total-trials", "40",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "round 0" in out
        assert "Task:" in out

    def test_console_script_installed(self):
        import subprocess

        proc = subprocess.run(
            ["hpo-adapt", "--task", "branin", "--max-rounds", "1", "--max-total-trials", "15"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert "round 0" in proc.stdout
