"""Agent Memory (spec.md: 其它模块和说明 — Agent Memory).

SQLite-backed store with two tables:

1. ``experiences``  — one row per COMPLETE adaptation step
   (S_t -> HPO -> D_t -> A_t -> S_{t+1} -> Outcome), written only after the
   step finishes, and only when worth long-term storage (successful
   interventions and useful negative experiences).
2. ``semantic_rules`` — aggregated (condition -> recommended action) stats:
   success_rate, sample_count, average_reward.

Retrieval (first version, no embeddings): exact task/model match,
diagnosis-type match, hyperparameter/action match, then success/reward
ordering, combined as

    Similarity = l1 * TaskSimilarity + l2 * DiagnosisSimilarity + l3 * ModelSimilarity
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from hpo_adapt.levels import LevelConfig, diagnostic_signature
from hpo_adapt.schemas import (
    Diagnosis,
    DiagnosticState,
    Experience,
    MetricStatus,
    TaskSignature,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    experience_id TEXT PRIMARY KEY,
    task_signature TEXT NOT NULL,
    setup_before_json TEXT NOT NULL,
    diagnostic_state_json TEXT NOT NULL,
    diagnosis_json TEXT NOT NULL,
    action_json TEXT NOT NULL,
    setup_after_json TEXT NOT NULL,
    outcome_json TEXT NOT NULL,
    reward REAL,
    success INTEGER NOT NULL,
    rationale TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS semantic_rules (
    rule_id TEXT PRIMARY KEY,
    condition_json TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    success_rate REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    average_reward REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (condition_json, recommended_action)
);
"""


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lambda_task: float = 0.4
    lambda_diagnosis: float = 0.4
    lambda_model: float = 0.2
    persist_all: bool = False  # spec: prefer successful + useful negative experiences
    top_k: int = 5


def new_experience_id() -> str:
    return f"exp_{uuid.uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def condition_key(state: DiagnosticState, cfg: Optional[LevelConfig] = None) -> dict[str, Any]:
    """Semantic-rule condition (spec.md example): stagnation / importance / boundary_concentration."""
    sig = diagnostic_signature(state, cfg)
    return {
        "stagnation": sig.get("stagnation"),
        "importance": sig.get("importance"),
        "boundary_concentration": sig.get("boundary_concentration"),
    }


def should_persist(experience: Experience, config: MemoryConfig) -> bool:
    """Write policy: successful interventions and useful negative experiences."""
    if config.persist_all:
        return True
    if experience.success:
        return True
    outcome = experience.outcome
    if experience.reward is not None and experience.reward < 0:
        return True
    if outcome.relative_improvement is not None and outcome.relative_improvement < 0:
        return True
    return False


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:", config: MemoryConfig | None = None) -> None:
        self.path = str(path)
        self.config = config or MemoryConfig()
        if self.path != ":memory:":
            parent = Path(self.path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ write
    def record(self, experience: Experience) -> bool:
        """Record one complete adaptation step.

        Always updates ``semantic_rules``; persists the raw experience row
        only when ``should_persist`` accepts it. Returns True if persisted.
        """
        self._update_rules(experience)
        if not should_persist(experience, self.config):
            self.conn.commit()
            return False
        self.conn.execute(
            """
            INSERT OR REPLACE INTO experiences (
                experience_id, task_signature, setup_before_json, diagnostic_state_json,
                diagnosis_json, action_json, setup_after_json, outcome_json,
                reward, success, rationale, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experience.experience_id,
                json.dumps(experience.task_signature.model_dump()),
                json.dumps(experience.setup_before),
                json.dumps(experience.diagnostic_state),
                json.dumps(experience.diagnosis.model_dump()),
                json.dumps(experience.action.model_dump()),
                json.dumps(experience.setup_after),
                json.dumps(experience.outcome.model_dump()),
                experience.reward,
                int(experience.success),
                experience.rationale,
                experience.created_at or _now(),
            ),
        )
        self.conn.commit()
        return True

    def _update_rules(self, experience: Experience) -> None:
        try:
            state = DiagnosticState.model_validate(experience.diagnostic_state)
        except Exception:
            return
        condition = condition_key(state)
        cond_json = json.dumps(condition, sort_keys=True)
        rule_id = f"rule_{abs(hash(cond_json + experience.action.type)) % 10**10}"
        now = _now()
        row = self.conn.execute(
            "SELECT * FROM semantic_rules WHERE condition_json = ? AND recommended_action = ?",
            (cond_json, experience.action.type),
        ).fetchone()
        reward = experience.reward if experience.reward is not None else 0.0
        if row is None:
            self.conn.execute(
                """
                INSERT INTO semantic_rules (rule_id, condition_json, recommended_action,
                                            success_rate, sample_count, average_reward,
                                            created_at, updated_at)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    rule_id,
                    cond_json,
                    experience.action.type,
                    1.0 if experience.success else 0.0,
                    reward,
                    now,
                    now,
                ),
            )
        else:
            n = row["sample_count"]
            self.conn.execute(
                """
                UPDATE semantic_rules
                SET success_rate = ?, sample_count = ?, average_reward = ?, updated_at = ?
                WHERE rule_id = ?
                """,
                (
                    (row["success_rate"] * n + (1.0 if experience.success else 0.0)) / (n + 1),
                    n + 1,
                    (row["average_reward"] * n + reward) / (n + 1),
                    now,
                    row["rule_id"],
                ),
            )
        self.conn.commit()

    # --------------------------------------------------------------- retrieve
    def _similarity(self, exp: Experience, task: TaskSignature, diag: Optional[dict]) -> float:
        cfg = self.config
        ts = exp.task_signature
        task_sim = (
            (ts.task_type == task.task_type) + (ts.dataset_signature == task.dataset_signature)
        ) / 2.0
        model_sim = 1.0 if ts.model_family == task.model_family else 0.0
        diag_sim = 0.0
        if diag is not None:
            try:
                state = DiagnosticState.model_validate(exp.diagnostic_state)
                stored = diagnostic_signature(state)
                keys = [k for k in diag if diag.get(k) is not None]
                if keys:
                    matches = sum(1 for k in keys if stored.get(k) == diag.get(k))
                    diag_sim = matches / len(keys)
            except Exception:
                diag_sim = 0.0
        return cfg.lambda_task * task_sim + cfg.lambda_diagnosis * diag_sim + cfg.lambda_model * model_sim

    def retrieve(
        self,
        task_signature: TaskSignature,
        diagnostic_state: Optional[DiagnosticState] = None,
        action_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[Experience]:
        """Retrieve similar past experiences, ranked by similarity/success/reward."""
        diag = diagnostic_signature(diagnostic_state) if diagnostic_state is not None else None
        rows = self.conn.execute("SELECT * FROM experiences").fetchall()
        scored: list[tuple[float, float, float, Experience]] = []
        for row in rows:
            exp = _row_to_experience(row)
            sim = self._similarity(exp, task_signature, diag)
            if action_type is not None and exp.action.type != action_type:
                sim -= 0.5  # hyperparameter/action mismatch penalty
            scored.append((sim, float(exp.success), exp.reward or 0.0, exp))
        scored.sort(key=lambda t: (-t[0], -t[1], -t[2]))
        k = top_k if top_k is not None else self.config.top_k
        return [exp for _, _, _, exp in scored[:k]]

    def rules(self, condition: Optional[dict] = None) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM semantic_rules ORDER BY success_rate DESC, sample_count DESC"
        ).fetchall()
        result = []
        for row in rows:
            cond = json.loads(row["condition_json"])
            if condition is not None and not all(cond.get(k) == v for k, v in condition.items()):
                continue
            result.append(
                {
                    "rule_id": row["rule_id"],
                    "condition": cond,
                    "recommended_action": row["recommended_action"],
                    "success_rate": row["success_rate"],
                    "sample_count": row["sample_count"],
                    "average_reward": row["average_reward"],
                }
            )
        return result

    def experience_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0])

    def rule_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM semantic_rules").fetchone()[0])

    # ----------------------------------------------------------------- context
    def format_context(
        self,
        task_signature: TaskSignature,
        diagnostic_state: Optional[DiagnosticState] = None,
    ) -> str:
        """Human-readable retrieval context for the Setup Agents."""
        experiences = self.retrieve(task_signature, diagnostic_state)
        if not experiences and not self.rules():
            return ""
        lines: list[str] = []
        if experiences:
            lines.append("Similar past experiences:")
            for exp in experiences:
                outcome = exp.outcome
                lines.append(
                    f"- diagnosis={exp.diagnosis.primary}; action={exp.action.type}"
                    f"(target={exp.action.target}); relative_improvement="
                    f"{outcome.relative_improvement if outcome.relative_improvement is not None else 'n/a'};"
                    f" success={exp.success}"
                )
        rules = self.rules()
        if rules:
            lines.append("Semantic rules:")
            for rule in rules[:3]:
                lines.append(
                    f"- condition={rule['condition']} -> {rule['recommended_action']} "
                    f"(success_rate={rule['success_rate']:.2f}, n={rule['sample_count']})"
                )
        return "\n".join(lines)


def _row_to_experience(row: sqlite3.Row) -> Experience:
    return Experience(
        experience_id=row["experience_id"],
        task_signature=TaskSignature.model_validate(json.loads(row["task_signature"])),
        setup_before=json.loads(row["setup_before_json"]),
        diagnostic_state=json.loads(row["diagnostic_state_json"]),
        diagnosis=Diagnosis.model_validate(json.loads(row["diagnosis_json"])),
        action=json.loads(row["action_json"]),
        setup_after=json.loads(row["setup_after_json"]),
        outcome=json.loads(row["outcome_json"]),
        reward=row["reward"],
        success=bool(row["success"]),
        rationale=row["rationale"] or "",
        created_at=row["created_at"],
    )
