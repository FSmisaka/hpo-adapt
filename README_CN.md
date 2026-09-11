# HPO-Adapt：诊断驱动的自适应 HPO Setup 控制

本项目是 `spec.md` 的完整实现：把 **HPO 轨迹分析**（CAVE / DeepCAVE 一系的诊断思路）与
**Agent 驱动的 setup 生成**结合，并由 **表格型 Q-Learning** 控制何时执行哪种 setup 干预。

> 英文版说明见 [README.md](README.md)。

## 研究问题

HPO 的效果高度依赖初始 setup（Search Space / Optimizer / Budget）。Search Space 太小可能
漏掉最优区域，太大又会降低搜索效率。本项目让系统在 HPO 运行过程中**自己诊断、自己修订setup**：

$$
S_0 \rightarrow HPO \rightarrow D_0 \rightarrow A_0 \rightarrow S_1 \rightarrow HPO \rightarrow D_1 \rightarrow A_1 \rightarrow \cdots
$$

## 完整闭环

```
Task ──▶ Initial Setup Agent ──▶ S0 ──▶ Classical HPO ──▶ TrialLogs ──▶ Diagnostic Engine
              ▲                                                              │
              │                                                              ▼
        Agent Memory ◀── Experience ◀── Outcome ◀── Revised Setup Agent ◀── RL Controller
        （提供先验/历史经验）            （执行 A_t）            （只决定"做什么" A_t）
```

三个模块的分工（spec 核心设计）：

- **Diagnostic Engine = Observe**：确定性 Python 模块，绝不让 LLM 参与指标计算
- **RL Controller = Decide**：表格 Q-Learning，只从 7 种高层干预中选动作
- **Revised Setup Agent = Execute**：把干预落实为具体的新 Setup（改多少、改成多少）

## 模块一览（实现顺序与 spec 一致）

| 阶段 | 模块 | 路径 |
|---|---|---|
| 0 | 结构化 Schema + Tool Validator | `src/hpo_adapt/schemas/`、`src/hpo_adapt/validator.py` |
| 1 | Classical HPO（Optuna） | `src/hpo_adapt/hpo/` |
| 2 | Diagnostic Engine（fANOVA+LPI、boundary concentration、good region、diversity、coverage、budget） | `src/hpo_adapt/diagnostic/` |
| 3 | Initial Setup Agent（heuristic + LLM 双后端） | `src/hpo_adapt/agents/initial_setup.py` |
| 4 | Rule-based Adaptation（7 条诊断模式表 + CandidateScore 候选排序） | `src/hpo_adapt/rules/adaptation.py` |
| 5 | Revised Setup Agent（执行 7 种 action） | `src/hpo_adapt/agents/revised_setup.py` |
| 6 | Agent Memory（SQLite：`experiences` + `semantic_rules` 两张表） | `src/hpo_adapt/memory/store.py` |
| 7 | RL Controller（表格 Q-Learning） | `src/hpo_adapt/rl/` |
| 8 | 闭环 Pipeline + CLI | `src/hpo_adapt/pipeline.py` |

## 核心数据对象

`TaskSpec`、`Setup`（Search Space / Optimizer / Budget / Evaluation）、`TrialLog`、
`DiagnosticState`（每个指标都带 `VALID` / `INSUFFICIENT_DATA` 状态）、`RLAction`
（`A_t = (ActionType, Target, Parameters)`）、`Experience`。

所有模块间通信都走这些 pydantic Schema；**任何 Setup 必须先通过 `validate_setup`才能被 HPO 执行**（验证器会拒绝：编造不存在的超参、非法范围、未知 optimizer、方向不一致等）。

## 动作空间（第一版，严格为 spec 中的 7 种）

`EXPAND_SEARCH_SPACE`、`SHIFT_SEARCH_REGION`、`SHRINK_SEARCH_SPACE`、
`FREEZE_HYPERPARAMETER`、`INCREASE_EXPLORATION`、`INCREASE_BUDGET`、`STOP`

奖励：`r = clip(Δ − λ·C − r_invalid, −r_max, r_max)`，其中
`Δ = (B_{t+1} − B_t)/(|B_t|+ε)`，`C = N_phase/N_total`。
Q-Learning 默认参数：`α=0.1, γ=0.9, ε: 1.0 → 0.05（衰减 0.995）`。

## 安装与运行

```bash
pip install -e ".[dev]"
pytest                                   # 109 个测试
hpo-adapt --task synthetic_mlp --max-rounds 4 --max-total-trials 120 --seed 3
hpo-adapt --task branin --max-rounds 3                   # minimize 方向示例
hpo-adapt --task synthetic_mlp --json                    # 机器可读结果
```

示例输出：

```
Task: task_synth_001 (maximize)  backend=heuristic
round 0: setup=setup_000 trials=40 phase_best=0.584 ... action=INCREASE_EXPLORATION@optimizer reward=-0.033
round 1: setup=setup_001 trials=40 ... action=INCREASE_EXPLORATION@optimizer reward=-0.033
round 2: setup=setup_002 trials=40 ... action=INCREASE_EXPLORATION@optimizer reward=None
total_trials=120 best(raw)=0.584 best_params={...} q_states=3 experiences=2
```

说明：ε-greedy 探索初期 ε=1.0，个别 seed 会随机抽到 `STOP` 提前结束——这是 7 动作
空间上教科书式 ε-greedy 的正常行为，换 `--seed` 可以看到不同的 adaptation 轨迹。

## 在真实 ML 任务上运行（推荐入口）

```bash
python examples/real_ml_demo.py            # sklearn 手写数字 + 随机森林，3 轮闭环
```

该示例演示如何接入你自己的任务（三步：注册目标函数 → 写 `TaskSpec` 元数据 →
跑 `AdaptationPipeline`），并逐轮打印 Search Space 变化、DiagnosticState 各指标、
Memory 中写入的 Experience 与 semantic_rules、Q-table。详见脚本内注释与 README.md。

## LLM 后端（可选）

两个 Setup Agent 都有确定性 heuristic 后端（默认）和 LLM 后端（使用 spec 原文
prompt）。启用 LLM 后端（绝不硬编码任何 key）：

```bash
export OPENAI_API_KEY=...            # 必需
export OPENAI_BASE_URL=...           # 可选（任何 OpenAI 兼容端点）
export OPENAI_MODEL=gpt-4o-mini      # 可选
hpo-adapt --backend llm ...
```

LLM 输出会经过同样的 Tool Validator：凡是违反 Schema 或任务约束的输出一律被拒。

## 约定与注意事项

- **内部统一 maximize 空间**：Diagnostic Engine 与 reward 都在
  `ỹ = y（maximize）/ −y（minimize）` 空间上计算（spec 预处理要求）；
  `PipelineResult.best_value_raw` 会换回任务原始方向。
- **数据不足不造假**：低于最小 trial 数（5 / 20 / 30）的指标返回
  `INSUFFICIENT_DATA` 而不是伪造值；importance 不为 `VALID` 时，
  针对超参的 action 不可选。
- **非法 action**（如对 categorical 超参做 EXPAND）不消耗预算，只吃固定惩罚
  reward，并在本轮内重试。
- **Memory 写入时机**：只在一个完整 adaptation step 结束后写入；默认只长期保存
  成功干预和有价值的失败经验，`semantic_rules` 则聚合每一次结果。
- **确定性**：所有随机源（HPO sampler、surrogate 随机森林、RL RNG）固定种子；
  相同 seed + 干净 Memory ⇒ 完全可复现。
