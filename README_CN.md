# HPO-Adapt：诊断驱动的自适应 HPO Setup 控制

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

初始 **setup**（搜索空间、optimizer、预算）框定了能搜到什么，很大程度决定了
 HPO 的效果。如果没有好的 setup，HPO 没有办法搜到好的超参数。为此我们构建
 **HPO-Adapt** 尝试解决这个问题：搜索进行中，确定性的
**诊断引擎**持续分析 trial 轨迹，RL **控制器**从七种高层干预中决策，**Setup 修订
Agent** 把干预落实为通过校验的新 setup。全程可在本地实时 dashboard 上观测。

> English version: [README.md](README.md)

```
Task ──▶ Initial Setup Agent ──▶ S0 ──▶ Classical HPO ──▶ TrialLogs ──▶ Diagnostic Engine
               ▲                                                              │
               │                                                              ▼
         Agent Memory ◀── Experience ◀── Outcome ◀── Revised Setup Agent ◀── RL Controller
         （提供先验/历史经验）            （执行 A_t）            （只决定"做什么" A_t）
```

三个模块的分工：

- **Diagnostic Engine = Observe**：确定性 Python，绝不让 LLM 参与指标计算
- **RL Controller = Decide**：表格 Q-Learning，只从 7 种干预中选动作
- **Revised Setup Agent = Execute**：把干预落实为具体的新 Setup

## 1. Quick start

```bash
conda activate your_conda_env  # 可选
make setup                     # 创建 .venv 并安装 hpo-adapt + 开发依赖
source .venv/bin/activate
make demo                      # 实时 demo：真实训练任务 + dashboard（约 5-6 分钟）
```

`make demo` 运行 [examples/demo_dashboard.py](examples/demo_dashboard.py)：
HistGradientBoosting + 高难度合成数据集，120 次真实训练组成 4 个自适应轮次。终端
会打印 dashboard URL，打开浏览器可逐轮看到实际使用的 setup、每个 trial、全部
诊断指标（importance、boundary concentration、good region、stagnation……）、
控制器选了哪个 action 及依据、setup 修改的具体内容（旧值 → 新值）、每个干预获得
的 reward。

接入到自己的任务上：

```python
import hpo_adapt as ha

def train_evaluate(params):        # 1. 你已有的训练/评估代码（失败抛 RuntimeError）
    ...
    return holdout_accuracy        #    要优化的指标

run = ha.run(                      # 2. 接入后：初始 setup 自动生成、多轮自动迭代
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

> `space` 的范围**不需要领域知识**：随手给一个以默认值为中心的大致范围即可，
> 工具会根据实验轨迹自动 shift / expand / freeze。边界只作为硬性安全上限。

详细的中文上手指南见 [examples/README.md](examples/README.md)。

## 2. 核心 API

| 入口 | 作用 |
|---|---|
| `ha.run(objective, space, ...)` | 自动构建 `TaskSpec`、注册目标函数、启动 dashboard、运行闭环。 |
| `ha.Float / ha.Int / ha.Choice` | 空间 DSL：`low/high` 是硬性适配边界，`default` 决定初始紧凑区间中心，`log=True` 按对数尺度。 |
| `AdaptationPipeline(task, PipelineConfig())` | 可配置的底层入口（自定义控制器、记忆、`event_sink=` 观察者） |
| `hpo-adapt` CLI | 内置任务：`hpo-adapt --task synthetic_mlp \| branin \| rf_digits [--dashboard] [--json]`。 |

`ha.run` 常用参数：`direction`、`total_trials`、`max_rounds`、`trials_per_phase`、
`memory_path`（SQLite 文件 → 经验跨 run 保留）、`seed`（固定全部随机源）、
`dashboard`、`open_browser`。返回值 `RunResult`：`best_value`、`best_params`、
`rounds`、`wait()`。

**动作空间**（7 种干预）：`EXPAND_SEARCH_SPACE`、`SHIFT_SEARCH_REGION`、
`SHRINK_SEARCH_SPACE`、`FREEZE_HYPERPARAMETER`、`INCREASE_EXPLORATION`、
`INCREASE_BUDGET`、`STOP`（证据门控）。奖励：
`r = clip(Δ − λ·C − r_invalid, −r_max, r_max)`，其中 `Δ = (B_{t+1} − B_t)/(|B_t|+ε)`。

**设计保证**：任何 Setup 必须先通过 `validate_setup` 才会被执行；低于最小 trial
数的指标返回 `INSUFFICIENT_DATA` 而非伪造值；新 phase 用全部历史兼容 trial
warm-start（不重评、不耗预算）；两个 Setup Agent 均有确定性 heuristic 后端
（默认，全部实验基于它）与可选 LLM 后端（`--backend llm`，任何 OpenAI 兼容端点，
输出同样过校验器）。

## 3. 实验与复现

四个实验对应四个主张：

| 主张 | 实验 | 结果（10 seeds） |
|---|---|---|
| **C1.** 初始 setup 有缺陷时，自适应闭环优于等预算静态 HPO | `make exp1` — 6 方法 × 3 任务（180 runs） | synthetic_mlp：引导式 adapt **1.000±0.000** vs static-tpe 0.755±0.003（相对 +32%）；Branin：**0.498±0.198** vs 0.896±0.010（全局最优 0.398）；rf_digits：CV 精度平台打平——价值在可观测性，"不伤害" |
| **C2.** 优越性来自诊断引导而非随机修改（消融） | `make exp2` — 4 种决策器 × 3 任务（120 runs） | 方差：引导式 1.000**±0.000**（最差 seed = 最好 seed）vs 随机干预 0.873**±0.134**（约 6 倍方差）；Branin 排序 0.498 < 0.596 < 0.776 |
| **C3.** RL 控制器随经验积累变强 | `make exp3` — 60 组 × 20 episodes，共享 Q-table + memory | synthetic：逐 episode 最优 0.892 → 0.949（随机对照持平 0.867 → 0.855）；Branin：0.734 → 0.524（对照反而 0.768 → 0.819） |
| **C4.** 观察-决策层足够便宜 | `make exp4` — 真实 ML 任务开销测量 | 诊断 + RL + Memory + Agent 在 rf_digits 上仅占 **1.4%** 墙钟时间（最差 1.7%） |

实验复现：

```bash
make setup && source .venv/bin/activate
make experiments     # exp1 + exp2 + exp3 + exp4（共约 30 分钟，N-1 进程并行）
make figures         # 生成每张图的数据 CSV + PNG 预览，落在 experiments/results/full/figs/
make smoke           # 先跑 2 分钟的冒烟矩阵验证环境
make test            # 149 个单元测试
```

结果写入 `experiments/results/full/`（每个 run 一个 `runs/*.json`，每个实验一份
`summary_exp*.csv`）；`make figures` 产出 `anytime.csv`、`main_summary.csv`、
`episodes_summary.csv`、`actions.csv`、`overhead_summary.csv`、`case_study.json`
（+ PNG 预览）。实验源码见 [experiments/](experiments/)，每个实验主张一个文件，
共享执行框架在 `common.py`。

## 项目结构

```
src/hpo_adapt/     包本体（schemas、hpo、diagnostic、agents、rules、rl、memory、pipeline、api、dashboard）
examples/          demo_dashboard.py + 中文使用指南
experiments/       exp1..exp4（每主张一个文件）、common.py、make_figures.py
tests/             149 个单元测试
```

## License

MIT——见 [LICENSE](LICENSE)。
