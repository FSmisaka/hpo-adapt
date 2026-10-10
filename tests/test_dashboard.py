"""Tests for the live dashboard: recorder document + HTTP server + events."""

import json
import urllib.request

import pytest

from hpo_adapt.dashboard import RunRecorder
from hpo_adapt.dashboard.server import DashboardServer
from hpo_adapt.events import EventSink
from hpo_adapt.pipeline import AdaptationPipeline, PipelineConfig, synthetic_mlp_task


@pytest.fixture
def completed_doc():
    """Recorder document of a real (tiny) closed-loop run."""
    recorder = RunRecorder()
    pipeline = AdaptationPipeline(
        synthetic_mlp_task(max_total_trials=15),
        PipelineConfig(max_rounds=2, seed=1),
        event_sink=EventSink(handler=recorder),
    )
    pipeline.run()
    return recorder.snapshot()


class TestRunRecorder:
    def test_document_shape(self, completed_doc):
        assert completed_doc["status"] == "finished"
        assert completed_doc["task"]["task_id"] == "task_synth_001"
        assert completed_doc["summary"] is not None
        assert completed_doc["initial_setup"]["setup"]["setup_id"] == "setup_000"

    def test_rounds_and_trials_consistent(self, completed_doc):
        rounds = completed_doc["rounds"]
        assert len(rounds) >= 1
        assert all(r["search_space"] for r in rounds)
        n_by_round = sum(len(r["trials"]) for r in rounds)
        assert n_by_round == len(completed_doc["trials"])
        for t in completed_doc["trials"]:
            assert {"trial_id", "params", "value", "state", "round_index"} <= set(t)

    def test_every_round_has_diagnostic(self, completed_doc):
        for r in completed_doc["rounds"]:
            assert r["diagnostic"] is not None
            assert "optimization" in r["diagnostic"]

    def test_unknown_events_ignored(self):
        rec = RunRecorder()
        rec("something_new", {"whatever": 1})  # must not raise
        assert rec.snapshot()["status"] == "running"

    def test_error_marking(self):
        rec = RunRecorder()
        rec.mark_error("boom")
        doc = rec.snapshot()
        assert doc["status"] == "error" and doc["error"] == "boom"


class TestDashboardServer:
    def test_serves_app_and_state(self):
        rec = RunRecorder()
        rec("run_start", {
            "task": {"task_id": "t", "direction": "maximize"},
            "total_budget": 5, "max_rounds": 2, "agent_backend": "heuristic", "seed": 0,
        })
        server = DashboardServer(rec, "127.0.0.1", 0).start()
        try:
            with urllib.request.urlopen(server.url + "/") as res:
                body = res.read().decode()
                assert res.status == 200 and "<svg" not in body[:0]
                assert "hpo-adapt" in body
                assert "api/state" in body
            with urllib.request.urlopen(server.url + "/api/state") as res:
                state = json.loads(res.read())
                assert state["task"]["task_id"] == "t"
            with urllib.request.urlopen(server.url + "/api/health") as res:
                assert json.loads(res.read()) == {"ok": True}
        finally:
            server.stop()

    def test_preferred_port_fallback(self):
        busy = DashboardServer(RunRecorder(), "127.0.0.1", 0).start()
        try:
            other = DashboardServer(RunRecorder(), "127.0.0.1", busy.port).start()
            try:
                assert other.port != busy.port  # fell back to another port
            finally:
                other.stop()
        finally:
            busy.stop()


class TestPipelineEvents:
    def test_event_sequence_ordering(self):
        events: list[str] = []
        pipeline = AdaptationPipeline(
            synthetic_mlp_task(max_total_trials=15),
            PipelineConfig(max_rounds=2, seed=1),
            event_sink=EventSink(handler=lambda t, p: events.append(t)),
        )
        pipeline.run()

        assert events[0] == "run_start"
        assert events[1] == "initial_setup"
        assert events[-1] == "episode_end"
        assert events.count("round_start") == events.count("round_end")
        assert events.count("trial") == 10  # phase cap 5 (15//3) x 2 rounds
        # each round: trials after round_start and before round_end
        first_rs = events.index("round_start")
        first_re = events.index("round_end")
        assert events[first_rs + 1] == "trial"
        assert events[first_re - 1] == "trial"
        # diagnostic follows every round_end
        for i, e in enumerate(events):
            if e == "round_end":
                assert events[i + 1] == "diagnostic"
        # sink errors never propagated
        assert pipeline.events.errors == []

    def test_sink_exceptions_do_not_break_the_run(self):
        def hostile(event_type, payload):
            raise RuntimeError("observer exploded")

        pipeline = AdaptationPipeline(
            synthetic_mlp_task(max_total_trials=8),
            PipelineConfig(max_rounds=1, seed=0),
            event_sink=EventSink(handler=hostile),
        )
        result = pipeline.run()
        assert result.best_value is not None
        assert len(pipeline.events.errors) > 0
