"""Reward for one adaptation step (spec.md 模块三 — Reward).

    Delta_t = (B_{t+1} - B_t) / (|B_t| + epsilon_y)
    C_t     = N_phase / N_total_budget
    r_t     = clip(Delta_t - lambda * C_t - r_invalid, -r_max, r_max)

NOTE: all objective values must be given in the internal maximized space
(minimize tasks: pass negated values), consistent with the Diagnostic Engine.
"""

from pydantic import BaseModel, ConfigDict


class RewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lambda_cost: float = 0.1
    invalid_action_penalty: float = 0.2
    reward_clip: float = 1.0
    epsilon_y: float = 1e-8


def compute_reward(
    best_before: float,
    best_after: float,
    n_phase_trials: int,
    total_budget: int,
    config: RewardConfig | None = None,
    invalid: bool = False,
) -> float:
    cfg = config or RewardConfig()
    if invalid:
        # Invalid action: no HPO phase happened; fixed penalty only.
        return float(max(-cfg.reward_clip, min(cfg.reward_clip, -cfg.invalid_action_penalty)))
    b_before = float(best_before)
    b_after = float(best_after)
    delta = (b_after - b_before) / (abs(b_before) + cfg.epsilon_y)
    cost = n_phase_trials / total_budget if total_budget > 0 else 0.0
    reward = delta - cfg.lambda_cost * cost
    return float(max(-cfg.reward_clip, min(cfg.reward_clip, reward)))
