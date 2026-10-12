# hpo_adapt 使用指南

`hpo_adapt` 的目标：**把已有的 ML 训练代码，变成一个会自我诊断、自我修订 search space 的多轮自动超参数寻优系统，并附带一个实时 dashboard**。

不需要理解诊断指标、不需要理解 RL、不需要写循环控制代码。

---

## 1. 安装

```bash
# 方式一：Makefile 一键配置 .venv 并安装（推荐）
make setup
source .venv/bin/activate

# 方式二：手动安装（任意虚拟环境中）
pip install -e .            # 包本体
pip install -e ".[dev]"     # 含 pytest / mypy / matplotlib 等开发依赖
```

验证安装：

```bash
python -c "import hpo_adapt as ha; print(ha.__version__)"
make test                   # 运行全部单元测试（149 个）
```

---

## 2. 把 ML 任务接入 hpo_adapt

以本目录的 `demo_dashboard.py` 为例（HistGradientBoosting + 合成数据集，约 5–6 分钟）：

```bash
make demo                                 # 一键运行
python examples/demo_dashboard.py         # 等价的手动命令
python examples/demo_dashboard.py --fast  # 约 1 分钟的快速体验
```

接入你自己的任务只需三步：

**第一步：写一个目标函数** `train_evaluate(params) -> float`——封装原本的训练代码。接收一个超参数字典，训练一次、评估一次，返回要优化的指标（如 holdout accuracy）。训练失败就抛 `RuntimeError` / `ValueError`，hpo_adapt 会把它记为失败的 trial。

**第二步：声明超参数范围**——用 `ha.Float / ha.Int / ha.Choice` 描述每个超参数：

```python
space = dict(
    learning_rate=ha.Float(0.08, 0.12, log=True, default=0.1),
    max_iter=ha.Int(20, 300, default=80),
    max_leaf_nodes=ha.Int(10, 64, default=24),
)
```

**第三步：调用 `ha.run`**——初始 setup 生成、多轮 HPO、诊断分析、setup 修订：

```python
run = ha.run(
    train_evaluate,            # 第一步的函数
    space=space,               # 第二步的范围
    direction="maximize",      # 或 "minimize"
    total_trials=120,          # 总 trial 预算
    max_rounds=4,              # 最多几轮"搜索→诊断→修订"
    task_name="my_task",
)
print(run.best_value, run.best_params)
```

运行后终端会打印 `▶ Dashboard live at http://127.0.0.1:xxxx`，浏览器打开即可实时看到：每轮实际使用的 search space、全部 trial、诊断指标（importance / boundary concentration / good region / stagnation…）、RL 选择了哪个干预动作、setup 修改的具体内容（旧值 → 新值）以及每个干预获得的 reward。

### 关于第二步的 `space`：不需要领域知识，可以相对随意地指定

这是使用本工具最重要的心智模型：**`space` 只是初始猜测，不是最终答案**。

- 区间给小了、给偏了都没关系——诊断引擎会从实验轨迹中发现"好 trial 堆在边界上"这类信号，然后自动 shift / expand 搜索区间去修复它（demo 里 `max_iter`、`max_leaf_nodes` 的区间就是被一轮轮推上去的）；
- `default` 值决定初始区间的中心，`low` / `high` 只是硬性安全边界（修订永远不超过它）；
- 你唯一要保证的是边界安全（比如 learning_rate 不会取到负数），而不是准确。

---

## 3. 基本方法

### 3.1 `ha.run(...)`

入口方法，全部参数见下表：

| 参数 | 说明 |
|---|---|
| `objective` | 你的训练函数 `params -> float` |
| `space` | `dict` 形式的超参数声明（`ha.Float / ha.Int / ha.Choice`） |
| `direction` | `"maximize"` / `"minimize"` |
| `total_trials` | 总预算（trial 数），静态基线与自适应方法共享同一预算口径 |
| `max_rounds` | 闭环轮数上限（每轮 = 一次搜索 phase + 诊断 + 修订） |
| `trials_per_phase` | 每轮 trial 数（默认自动取 `total_trials // 3`） |
| `memory_path` | Agent Memory 的 SQLite 路径；跨 run 保留时传文件名，否则用内存 |
| `seed` | 固定全部随机源（HPO 采样 / RL / surrogate），保证可复现 |
| `dashboard` | 是否启动本地 dashboard（默认开） |
| `open_browser` | 是否自动打开浏览器（默认关） |

返回值 `run`：`run.best_value`（原始方向的最优指标）、`run.best_params`、`run.rounds`、`run.wait()`（按回车前保持 dashboard 在线）。

### 3.2 空间 DSL：`ha.Float / ha.Int / ha.Choice`

```python
ha.Float(low, high, log=True, default=..., description=...)  # 连续参数（log=True 按对数尺度搜索）
ha.Int(low, high, default=...)                               # 整数参数
ha.Choice([...], default=...)                                # 类别参数
```

`default` 建议填上（初始区间以它为中心）；`description` 可选，会进入任务元数据供 setup agent 参考。

### 3.3 `AdaptationPipeline` —— 可配置的底层入口

`ha.run` 的下层是 `hpo_adapt.pipeline.AdaptationPipeline`，需要自定义控制器、记忆、事件回调时用它：

```python
from hpo_adapt.pipeline import AdaptationPipeline, PipelineConfig, synthetic_mlp_task

task = synthetic_mlp_task(max_total_trials=120)
result = AdaptationPipeline(task, PipelineConfig(max_rounds=4, seed=0)).run()
print(result.best_value_raw, result.best_params)
```

它接受自定义 `controller`（RL 控制器）、`memory`（SQLite 记忆）、`event_sink`（任意 `(event_type, payload)` 观察者，可接自己的可视化）。

### 3.4 CLI

```bash
hpo-adapt --task synthetic_mlp --max-rounds 4 --seed 3   # 内置合成任务
hpo-adapt --task branin                                  # minimize 方向示例
hpo-adapt --task rf_digits --dashboard                   # 真实 ML 任务 + dashboard
hpo-adapt --task synthetic_mlp --json                    # 机器可读输出
```

### 3.5 可观测产物

- **Dashboard**：运行时实时观测（见上文）；
- **Agent Memory**（`memory_path` 指定文件时）：SQLite 中的 `experiences`（每次干预的完整上下文与结果）和 `semantic_rules`（按"诊断 → 动作"聚合的成功率统计），跨 run 累积；
- **`PipelineResult`**：逐轮 `StepRecord`（search space 快照、完整诊断状态、动作、reward），以及 `pipeline.timings`（HPO 时间 vs 观察-决策层时间的拆分）。
