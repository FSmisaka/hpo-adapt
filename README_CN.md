# HPO-Adapt：诊断驱动的自适应 HPO Setup 控制

本项目是 `spec.md` 的完整实现：把 **HPO 轨迹分析**（CAVE / DeepCAVE 一系的诊断思路）与
**Agent 驱动的 setup 生成**结合，并由 **表格型 Q-Learning** 控制何时执行哪种 setup 干预。

> 英文版说明见 [README.md](README.md)。

## 快速上手：~10 行代码把任意 ML 任务变成自适应 HPO（wandb 式体验）

任务契约（"基本任务格式"）：一个目标函数 `train_evaluate(params) -> float`（训练 + 评估，
返回要优化的指标）+ 一份声明式搜索空间。满足即可一行接入：

```python
import hpo_adapt as ha

def train_evaluate(params):        # 1. 你已有的训练/评估代码
    ...
    return cv_accuracy

run = ha.run(                      # 2. 接入后：初始 setup 自动生成、多轮自动迭代
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

运行后终端会打印本地链接：

```
┌────────────────────────────────────────────────────────────┐
│ hpo-adapt · adaptive HPO closed loop                        │
│   task      : svc_digits_abc123                             │
│   objective : cv_accuracy (maximize)                        │
│   budget    : 90 trials · up to 5 rounds                   │
└────────────────────────────────────────────────────────────┘

  ▶ Dashboard live at  http://127.0.0.1:8710
    在浏览器打开这个链接，实时观看整个闭环的运作。
```

前端页面逐轮展示：初始 setup（搜索空间 / optimizer / 每轮预算）是如何自动推导的、每个
trial 的结果、DiagnosticState 全部指标（importance、boundary concentration、good
region、stagnation……）、RL 选择了哪个 action 以及依据、setup 修改的具体内容（旧值 →
新值与理由）、每个干预获得的 reward。零额外依赖（标准库 HTTP 服务 + 单文件前端）。

**完整 demo**（RBF-SVM on sklearn digits，真实 ML 任务）：

```bash
python examples/demo_dashboard.py            # 终端会打印 dashboard 链接
python examples/demo_dashboard.py --fresh    # 清空 Agent Memory 后重新跑
```

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
| 9 | 一行接入的用户 API（`ha.run`） | `src/hpo_adapt/api.py` |
| 10 | 实时 Dashboard（标准库服务 + 单页前端） | `src/hpo_adapt/dashboard/` |

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
pytest                                   # 126 个测试
hpo-adapt --task synthetic_mlp --max-rounds 4 --max-total-trials 120 --seed 3
hpo-adapt --task branin --max-rounds 3                   # minimize 方向示例
hpo-adapt --task rf_digits                               # 真实 ML 任务（随机森林 + 交叉验证）
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
- **累计诊断**：每轮的 DiagnosticState 基于**本 episode 至今的全部轨迹**计算
  （CAVE 式），因此 importance 在累计满 30 个 trial 后即可用，而不是要求单个
  phase 内凑满 30 个。
- **Warm-start**：每个新 phase 的 Optuna study 会注入与当前搜索空间兼容的全部
  历史 trial（不重复评估、不消耗预算），setup 修订不丢失任何已有知识。
- **STOP 证据门控**：只有观测到 stagnation 且全局预算消耗过半时 STOP 才可选，
  ε-greedy 探索不会再造成 episode 被随机提前终止。
- **数据不足不造假**：低于最小 trial 数（5 / 20 / 30）的指标返回
  `INSUFFICIENT_DATA` 而不是伪造值；importance 不为 `VALID` 时，
  针对超参的 action 不可选。
- **非法 action**（如对 categorical 超参做 EXPAND）不消耗预算，只吃固定惩罚
  reward，并在本轮内重试。当预算耗尽或到达 `max_rounds`、下一 phase 无法再
  评估 action 时，闭环直接收尾而不选新 action——每个被执行的 action 都有
  outcome、reward 和 Q-update。
- **Optimizer**：RANDOM、TPE（全部任务）与 CMA-ES（仅纯数值任务如 `branin`；
  含 categorical 的任务会被 validator 拒绝）。
- **Memory 写入时机**：只在一个完整 adaptation step 结束后写入；默认只长期保存
  成功干预和有价值的失败经验，`semantic_rules` 则聚合每一次结果。
- **种子**：`--seed` / `PipelineConfig.seed` 控制全部随机源——HPO 采样
  （`evaluation.seed`）、RL 控制器、surrogate 随机森林；相同 seed + 干净 Memory
  ⇒ 完全可复现。
- **可观测性**：`StepRecord` 逐轮记录搜索空间、完整 DiagnosticState、trial
  投影、应用的修改与（延迟一轮的）reward；`pipeline.timings` 把墙钟时间拆成
  HPO 层 vs 观察+决策层（`experiments/` 由此生成论文全部表格）。
