"""Stage 6 tests: Memory (SQLite experiences + semantic_rules)."""

from hpo_adapt.memory import MemoryConfig, MemoryStore, condition_key, new_experience_id
from hpo_adapt.schemas import (
    BoundaryConcentrationEntry,
    Diagnosis,
    DiagnosticState,
    GoodRegionEntry,
    ImportanceEntry,
    MetricStatus,
    ActionRecord,
    Experience,
    Outcome,
    TaskSignature,
)


def sig(task_type="classification", model="XGBoost", data="ds_v1") -> TaskSignature:
    return TaskSignature(task_type=task_type, model_family=model, dataset_signature=data)


def state(stagnation=True, imp=0.78, bc=0.91, width=0.2) -> DiagnosticState:
    from hpo_adapt.schemas.diagnostic import BoolMetric, OptimizationStatus

    ds = DiagnosticState(optimization=OptimizationStatus(stagnation=BoolMetric.valid(stagnation)))
    ds.search_space.importance = {
        "learning_rate": ImportanceEntry(
            global_importance=imp, local_importance=imp, rank=1, status=MetricStatus.VALID
        )
    }
    ds.search_space.boundary_concentration = {
        "learning_rate": BoundaryConcentrationEntry(
            all=0.4, good=bc, upper=bc, lower=0.05, status=MetricStatus.VALID
        )
    }
    ds.search_space.good_region = {
        "learning_rate": GoodRegionEntry(
            mean=0.008, lower=0.006, upper=0.01, width_ratio=width, status=MetricStatus.VALID
        )
    }
    return ds


def experience(
    *,
    task=sig(),
    diag=Diagnosis(primary="SEARCH_SPACE_TRUNCATION", confidence=0.91),
    action_type="EXPAND_SEARCH_SPACE",
    target="learning_rate",
    reward=0.015,
    success=True,
    rel_improvement=0.0175,
    diagnostic=None,
) -> Experience:
    return Experience(
        experience_id=new_experience_id(),
        task_signature=task,
        setup_before={"setup_id": "setup_001"},
        diagnostic_state=(diagnostic or state()).model_dump(),
        diagnosis=diag,
        action={"type": action_type, "target": target},
        setup_after={"setup_id": "setup_002"},
        outcome=Outcome(
            best_before=0.912, best_after=0.928, relative_improvement=rel_improvement, budget_consumed=20
        ),
        reward=reward,
        success=success,
        rationale="test rationale",
        created_at="2026-01-01T00:00:00+00:00",
    )


class TestWritePolicy:
    def test_successful_experience_persisted(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        assert store.record(experience()) is True
        assert store.experience_count() == 1

    def test_mediocre_experience_not_persisted_but_rule_updated(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        exp = experience(success=False, reward=0.0, rel_improvement=0.0)
        assert store.record(exp) is False
        assert store.experience_count() == 0
        assert store.rule_count() == 1  # aggregation still happened

    def test_negative_experience_persisted(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        exp = experience(success=False, reward=-0.2, rel_improvement=-0.01)
        assert store.record(exp) is True

    def test_persist_all_override(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite", MemoryConfig(persist_all=True))
        assert store.record(experience(success=False, reward=0.0, rel_improvement=0.0)) is True


class TestSemanticRules:
    def test_rule_aggregation(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        store.record(experience(success=True, reward=0.1))
        store.record(experience(success=True, reward=0.3))
        store.record(experience(success=False, reward=-0.1))
        rules = store.rules()
        assert len(rules) == 1
        rule = rules[0]
        assert rule["recommended_action"] == "EXPAND_SEARCH_SPACE"
        assert rule["sample_count"] == 3
        assert abs(rule["success_rate"] - 2 / 3) < 1e-9
        assert abs(rule["average_reward"] - 0.1) < 1e-9
        cond = rule["condition"]
        assert cond["stagnation"] is True
        assert cond["importance"] == "high"
        assert cond["boundary_concentration"] == "high"

    def test_different_conditions_create_separate_rules(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        store.record(experience(diagnostic=state(stagnation=True)))
        store.record(experience(diagnostic=state(stagnation=False)))
        assert store.rule_count() == 2

    def test_condition_key_structure(self):
        key = condition_key(state())
        assert set(key) == {"stagnation", "importance", "boundary_concentration"}


class TestRetrieval:
    def test_exact_task_match_ranks_first(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        store.record(experience(task=sig()))
        store.record(experience(task=sig(task_type="regression", model="MLP", data="ds_v2")))
        results = store.retrieve(sig())
        assert results[0].task_signature.model_family == "XGBoost"

    def test_diagnosis_similarity(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        store.record(experience(diagnostic=state(stagnation=True, imp=0.78)))
        store.record(experience(diagnostic=state(stagnation=False, imp=0.05)))
        results = store.retrieve(sig(), diagnostic_state=state(stagnation=True, imp=0.7))
        # matching stagnation + high importance should outrank the mismatched one
        assert results[0].diagnostic_state["optimization"]["stagnation"]["value"] is True

    def test_action_filter(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        store.record(experience(action_type="EXPAND_SEARCH_SPACE"))
        store.record(experience(action_type="INCREASE_BUDGET"))
        results = store.retrieve(sig(), action_type="INCREASE_BUDGET")
        assert results[0].action.type == "INCREASE_BUDGET"

    def test_top_k(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        for i in range(8):
            store.record(experience(action_type=f"A_{i}"))
        assert len(store.retrieve(sig(), top_k=3)) == 3

    def test_format_context_empty(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        assert store.format_context(sig()) == ""

    def test_format_context(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        store.record(experience())
        context = store.format_context(sig(), state())
        assert "EXPAND_SEARCH_SPACE" in context
        assert "success=True" in context
