# 选题依据

## 背景：autoML（HPO）

### 基本的功能

AutoML旨在自动化机器学习流程，其中 **Hyperparameter Optimization（HPO）**负责自动寻找模型超参数的优良配置。[引用论文：Hutter et al., An Efficient Approach for Assessing Hyperparameter Importance, ICML 2014.]

### 实际是一个比较复杂的实验过程

HPO 并非简单的“一次优化”，而是一个持续进行的【配置—评估—反馈—再配置】的实验过程。传统 HPO 通常需要预先确定：Search Space、Optimizer、Budget等。[引用论文：Tornede et al., AutoML in the Age of Large Language Models: Current Challenges, Future Opportunities and Risks, TMLR 2024.]

### 研究问题

setup 决定了 HPO 能搜索什么。Search Space 过小可能遗漏最优区域；Search Space 过大导致搜索效率下降；Optimizer 和 Budget 设置不当则可能导致搜索效率低下。因此，一个好的 setup 方法是很重要的。

## 现有研究

### 路线一：人主导的 HPO 分析框架

- 思路：Human Setup $\rightarrow$ HPO $\rightarrow$ Trajectory Analysis $\rightarrow$ Human Diagnosis

- 典型研究：1. CAVE [引用论文：Eggensperger et al., AutoML: State of the Art and Open Challenges, 2021]、2. DeepCAVE [引用论文：Segel et al., DeepCAVE: A Visualization and Analysis Tool for Automated Machine Learning, JMLR 2025]

- 优点：提供 trajectory、configuration footprint、importance、partial dependence 等分析，增强了 HPO 的透明性、可解释性与可理解性。[引用论文：Trust in AutoML: Exploring Information Needs for Establishing Trust in Automated Machine Learning Systems, IUI 2020]

- 不足：setup 高度依赖人工经验

### 路线二：Agent 驱动的 HPO

- 思路：LLM Agent $\rightarrow$ Setup $\rightarrow$ HPO

- 典型研究：1. AgentHPO [引用论文：Liu et al., AgentHPO: Large Language Model Agent for Hyper-Parameter Optimization, 2025]、2. AutoML-Agent [引用论文：Trirat et al., AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML, ICML 2025]

- 优点：利用 LLM 的推理与工具调用能力，降低人工 setup 成本；

- 不足：1. 现有方法仍受到任务描述、搜索边界、资源预算、工具接口等外部约束；2. 更倾向于让 LLM 直接参与/替代传统 HPO，而非专门解决 setup adaptation。但最新研究表明 LLM 仍然无法代替传统 HPO 方法 [引用论文：Ferreira F, Wobbe L, Krishnakumar A, et al. Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch[J]. arXiv preprint arXiv:2603.24647, 2026]

## 研究想法

### 核心思想

将两条现有路线相结合，形成下面的研究架构（信息流）：

Task、Code、Agent Memory $\xrightarrow{Initial Setup Agent}$ 初始 Setup: Search Space、Optimizer、Budget $\xrightarrow{Classical HPO 方法}$ Trial Logs $\xrightarrow{Diagnostic Engine}$ Diagostic State $\xrightarrow{RL Controller}$ Action $\xrightarrow{Revised Setup Agent}$ 新 Setup: Search Space、Optimizer、Budget $\rightarrow \dots$

下面将详尽地阐述这条研究路线。

### 全局数据流与核心数据对象

整个系统围绕以下 6 个核心数据对象运行：

1. `TaskSpec`：任务描述、模型、数据集、目标函数和资源约束。
2. `Setup`：当前 HPO setup，包括 Search Space、Optimizer、Budget、Evaluation。
3. `TrialLog`：HPO 产生的单次 trial 记录。
4. `DiagnosticState`：Diagnostic Engine 从 Trial Logs 提取出的结构化状态。
5. `RLAction`：RL Controller 选择的 setup intervention。
6. `Experience`：一次完整 adaptation step 的历史经验，用于 Memory。

核心数据流：

$$
\boxed{
TaskSpec
\rightarrow
Setup
\rightarrow
TrialLog
\rightarrow
DiagnosticState
\rightarrow
RLAction
\rightarrow
Setup
}
$$

完整闭环：

$$
\boxed{
S_0
\rightarrow
HPO
\rightarrow
D_0
\rightarrow
A_0
\rightarrow
S_1
\rightarrow
HPO
\rightarrow
D_1
\rightarrow
A_1
\rightarrow
\cdots
}
$$

其中：

- $S_t$：第 $t$ 轮 HPO Setup；
- $D_t$：第 $t$ 轮 Diagnostic State；
- $A_t$：第 $t$ 轮 RL Action。

# 开发方案

## 模块一：Initial Setup Agent

> **实现注意：不要要求模型输出或暴露完整 CoT（Chain-of-Thought）。**  
> Prompt 中只要求模型输出结构化结果、关键依据（rationale）、confidence 和 assumptions，不要求隐藏思维过程。

与后面的 Revised Setup Agent 不同，本 Agent 只负责**第一次 HPO setup**。此时尚不存在 HPO trajectory，因此不能进行 trajectory-based diagnosis。

Initial Setup Agent 使用：

$$
\text{Task Information}+\text{Available Metadata}+\text{Prior Knowledge}+\text{Memory}
$$

必要时增加：

$$
+\text{Web Search}
$$

生成初始：

$$
S_0=
\{
SearchSpace,\ Optimizer,\ Budget,\ Evaluation
\}
$$

### Initial Setup Agent Prompt

```text
You are an HPO Initial Setup Agent.

Your goal is to construct a reasonable initial setup for hyperparameter
optimization for the given machine learning task.

You are NOT responsible for executing HPO. You only construct the setup.

INPUTS
------
1. Task description
2. Model / training code description
3. Dataset information
4. Optimization objective and direction
5. Computational constraints
6. Available hyperparameter metadata
7. Relevant prior knowledge retrieved from memory
8. Optional web-search results

YOUR TASK
---------

Step 1. Identify the tunable hyperparameters that are actually available
in the provided model/training code.

Step 2. Construct an initial search space.

For each hyperparameter specify:
- name
- type
- range / choices
- distribution
- default value
- optional conditional dependency
- brief rationale
- confidence in [0, 1]

Step 3. Select one HPO optimizer from the allowed optimizer list.

Step 4. Determine an initial HPO budget under the given computational constraints.

Step 5. Specify the evaluation protocol.

REQUIREMENTS
------------

- Never invent hyperparameters that do not exist in the provided system.
- Prefer a compact and meaningful search space.
- Use log-scale distributions for parameters spanning multiple orders of
  magnitude when appropriate.
- Do not unnecessarily include low-impact or purely deterministic parameters.
- All generated configurations must satisfy the provided constraints.
- Explicitly state assumptions.
- Provide concise rationales, not hidden chain-of-thought reasoning.

OUTPUT
------

Return ONLY a JSON object following the provided SetupSchema.

The output must contain:
- search_space
- optimizer
- budget
- evaluation
- assumptions
- confidence

The JSON must be valid and machine-readable.
```

## 模块二：Diagnostic Engine

### 模块目标

Diagnostic Engine 的目标不是生成大量可视化，而是从 HPO Trial Logs 中提取：

$$
\boxed{\text{能够直接支持下一轮 Setup Decision 的结构化 Diagnostic State}}
$$

核心信息分为：

```text
Diagnostic State
├── Optimization Status
│   ├── best_objective
│   ├── recent_improvement
│   ├── stagnation
│   └── marginal_gain
│
├── Search Space Status
│   ├── importance
│   ├── boundary_concentration
│   ├── good_region
│   ├── configuration_diversity
│   └── coverage
│
└── Budget Status
    ├── budget_consumed
    ├── budget_remaining
    └── marginal_gain_per_budget
```

Diagnostic Engine 必须是**确定性的 Python 模块**，不能让 LLM 自己从 raw logs 计算这些指标。


### 模块输入

Diagnostic Engine 接收：

```python
TrialLog
Setup
ObjectiveDirection
DiagnosticConfig
```

其中 Trial Log 至少包含：

```python
{
    "trial_id": int,
    "params": dict[str, Any],
    "value": float | None,
    "state": "COMPLETE" | "PRUNED" | "FAIL",
    "duration": float | None,
    "timestamp": float | None
}
```

Optuna 的 Study/Trial 接口本身能够提供 trial history、parameter values、objective value 和 trial state 等信息，可直接作为这一层的数据来源。

### 预处理

所有指标统一内部转换为 **maximize** 方向。

定义：

$$
\tilde y_i=
\begin{cases}
y_i,& \text{maximize}\\
-y_i,& \text{minimize}
\end{cases}
$$

后续所有 objective-related metric 均基于 $\tilde y_i$ 计算。

只使用 $state_i=\text{COMPLETE}$ 的 trial 计算 objective、importance、good region 等统计量。

对于 FAIL / PRUNED：

- 保留在原始 Trial Logs 中；
- 不参与默认的 objective-based analysis；
- 单独统计 failure rate，作为后续扩展指标。

### 数据充分性与 Metric Status

Diagnostic Engine 不得在数据不足时产生伪造或不可靠的指标。

全局定义最小 trial 数：

```yaml
min_trials_for_basic_metrics: 5
min_trials_for_importance: 30
min_trials_for_good_region: 20
min_trials_for_stagnation: 20
```

每个 Diagnostic Metric 都必须返回 VALID 或 INSUFFICIENT_DATA，例如
```json
{
  "global": 0.78,
  "local": 0.72,
  "rank": 1,
  "status": "VALID"
}
```

### Optimization Status

#### Best Objective

截至第 $t$ 个 completed trial 的最佳目标：

$$
\boxed{
B_t=\max_{i\leq t}\tilde y_i
}
$$

其中 $t$ 按 completed trial 的执行顺序计算。

- 用途：作为所有 optimization progress 的基准。

#### Recent Improvement

设 analysis window 为 $w$。

定义：

$$
\boxed{
I_t=B_t-B_{t-w}
}
$$

以及归一化 improvement：

$$
\boxed{
RI_t=
\frac{B_t-B_{t-w}}
{|B_{t-w}|+\epsilon_y}
}
$$

默认：

```yaml
analysis_window: 20
epsilon_y: 1e-8
```

- 用途：衡量最近阶段的 optimization progress。

#### Stagnation

Stagnation 不由“搜索区域变小”定义，而首先由**目标函数长期没有改善**定义。

定义：

$$
\boxed{
Stagnation_t=
\mathbb{1}
\left(
RI_t<\epsilon_{stag}
\right)
}
$$

当 $t<w$ 时：

```text
stagnation = false
```

默认：

```yaml
stagnation_relative_threshold: 0.001
```

即最近 $w$ 个 completed trials 的最佳值相对提升小于 0.1% 时，认为当前阶段存在 stagnation。

- 用途：Stagnation 是进一步 diagnosis 的触发信号，而不是直接等价于某一种 failure mode。例如：

$$
Stagnation
+
Low\ Diversity
$$

才进一步判断：

$$
Possible\ Premature\ Convergence
$$

#### Marginal Gain

定义：

$$
\boxed{
MG_t=
\frac{B_t-B_{t-w}}
{w}
}
$$

即最近 $w$ 个 trial 的平均 best-objective improvement。

- 用途：用于衡量继续消耗 trial budget 的收益。

### Search Space Status

#### Hyperparameter Importance

第一版使用 $\boxed{\text{fANOVA + LPI}}$

DeepCAVE 当前也采用 fANOVA 和 LPI 作为 hyperparameter importance 的核心方法。

对每个 hyperparameter $x_j$ 输出：

```python
{
    "global_importance": float,
    "local_importance": float,
    "rank": int
}
```

其中：

- `global_importance`：fANOVA importance；
- `local_importance`：LPI；
- `rank`：按照 global importance 排序。

- 用途：回答

$$
\boxed{\text{Which hyperparameters are worth modifying?}}
$$

例如 $Importance(lr)=0.78$，而 $Importance(dropout)=0.03$，则下一轮应优先考虑 $lr$，而不是大幅修改 dropout。

- 实现要求

    如果 trial 数不足以可靠建立 surrogate：

    ```text
    importance_status = "insufficient_data"
    ```

    不要产生伪造的 importance。

#### Boundary Concentration

仅对 numerical / ordinal hyperparameters 计算。

将 $x_j\in[l_j,u_j]$ 归一化：

$$
z_{ij}
=
\frac{x_{ij}-l_j}{u_j-l_j}
$$

设 boundary threshold $\delta=0.05$

定义全部 trial 的 boundary concentration：

$$
\boxed{
BC_j=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}
\left(
z_{ij}<\delta
\lor
z_{ij}>1-\delta
\right)
}
$$

更重要的是 top-$q$% good trials 的 boundary concentration：

$$
\boxed{
BC_j^{good}
=
\frac{
\#\{i\in G:
z_{ij}<\delta
\lor
z_{ij}>1-\delta
\}
}{
|G|
}
}
$$

其中 $G=\text{top }q\%\text{ trials}$

默认：

```yaml
good_trial_quantile: 0.10
boundary_threshold: 0.05
```

- 用途：检测

$$
\boxed{\text{Potential Search-Space Truncation}}
$$

例如：

```text
importance(lr) = 0.81
good_boundary_concentration(lr) = 0.92
```

意味着大量优秀 configuration 位于 $lr$ 的搜索边界附近，应重点考虑扩大该边界。

- 注意：

    $$
    BC_j^{good}\uparrow
    $$

    **本身不能说明 search space 有问题**。

    必须结合：

    - high importance
    - stagnation / continued improvement
    - boundary direction

    共同判断。

#### Good Region

选择 top $q$% completed trials：

$$
G=
Top_q(\{\tilde y_i\})
$$

对于 numerical hyperparameter $x_j$，计算：

$$
\boxed{
\mu_j^{good}
=
\frac{1}{|G|}
\sum_{i\in G}x_{ij}
}
$$

以及：

$$
\boxed{
[q_{j,low}^{good},q_{j,high}^{good}]
=
Quantile_{[q_l,q_h]}(x_j|G)
}
$$

默认：

```yaml
good_region_quantiles: [0.10, 0.90]
```

输出：

```python
{
    "mean": ...,
    "lower": ...,
    "upper": ...,
    "width_ratio": ...
}
```

其中：

$$
\boxed{
WidthRatio_j=
\frac{
q_{j,high}^{good}-q_{j,low}^{good}
}{
u_j-l_j
}
}
$$

对于 categorical parameter，输出：

```python
{
    "category_frequency": {...},
    "dominant_category": ...,
    "dominant_ratio": ...
}
```

- 用途：回答

> 优秀 configuration 实际集中在哪里？

因此支持 $\boxed{Shift}$ 和 $\boxed{Shrink}$ 两种 action。

#### Configuration Diversity

首先将 configuration 转换到统一 normalized representation：

- Numerical / Integer

    $$
    z_{ij}\in[0,1]
    $$

- Categorical

    $$
    d(x_{ij},x_{kj})
    =
    \mathbb{1}(x_{ij}\neq x_{kj})
    $$

两个 configuration 的 mixed-type distance：

$$
\boxed{
d(x_i,x_k)
=
\frac{1}{m}
\sum_{j=1}^{m}d_j(x_{ij},x_{kj})
}
$$

对最近 $w$ 个 completed trials：

$$
\boxed{
Div_t=
\frac{2}{w(w-1)}
\sum_{i<k}d(x_i,x_k)
}
$$

同时记录：

$$
DivRatio_t=
\frac{Div_t}{Div_{global}+\epsilon}
$$

- 用途:检测 search behavior 是否发生 collapse。

- 注意：

    $$
    \boxed{
    Low\ Diversity\neq Bad
    }
    $$

    只有：

    $$
    Low\ Diversity
    +
    Stagnation
    $$

    才应判断为 potential premature convergence。

#### Search-space Coverage

对于每个 numerical hyperparameter，将搜索空间划分为 $K$ 个 bins。

默认：

```yaml
coverage_bins: 10
```

定义：

$$
\boxed{
Coverage_j=
\frac{
N_{occupied\ bins,j}
}{
K
}
}
$$

整体 coverage：

$$
\boxed{
Coverage=
\frac{1}{m}
\sum_{j=1}^{m}Coverage_j
}
$$

对于 log-scaled parameter，在 log-space 中进行 binning。

- 用途：区分

$$
\text{充分探索后收敛}
$$

和：

$$
\text{尚未充分探索就陷入局部区域}
$$

### Budget Status

Budget 不应仅由 Trial Logs 决定，而应由：

$$
\boxed{
Initial\ Setup + Trial\ Logs
}
$$

共同得到。

#### Budget Consumed

如果采用 trial budget：

$$
N_{budget}=\text{maximum number of trials}
$$

当前 completed trials：

$$
N_t
$$

定义：

$$
\boxed{
BudgetConsumed_t=
\frac{N_t}{N_{budget}}
}
$$

以及：

$$
\boxed{
BudgetRemaining_t=
1-BudgetConsumed_t
}
$$

#### Marginal Gain per Budget

若按照 trial 数度量：

$$
\boxed{
MGB_t=
\frac{
B_t-B_{t-w}
}{
w
}
}
$$

若存在真实运行时间信息，则额外计算：

$$
\boxed{
MGBTime_t=
\frac{
B_t-B_{t-w}
}{
T_t-T_{t-w}+\epsilon_t
}
}
$$

- 用途：支持

```text
continue
increase budget
stop
```

等决策

### Diagnostic State 最终 Schema

Diagnostic Engine 必须输出统一 JSON：

```json
{
  "optimization": {
    "best_objective": 0.923,
    "recent_improvement": 0.001,
    "relative_improvement": 0.0011,
    "stagnation": true,
    "marginal_gain": 0.00005
  },

  "search_space": {
    "importance": {
      "learning_rate": {
        "global": 0.78,
        "local": 0.72,
        "rank": 1
      }
    },

    "boundary_concentration": {
      "learning_rate": {
        "all": 0.43,
        "good": 0.91,
        "upper": 0.86,
        "lower": 0.05
      }
    },

    "good_region": {
      "learning_rate": {
        "mean": 0.008,
        "lower": 0.006,
        "upper": 0.010,
        "width_ratio": 0.40
      }
    },

    "configuration_diversity": {
      "recent": 0.081,
      "global": 0.243,
      "ratio": 0.333
    },

    "coverage": {
      "overall": 0.42,
      "per_hyperparameter": {}
    }
  },

  "budget": {
    "total_trials": 100,
    "completed_trials": 80,
    "budget_consumed": 0.80,
    "budget_remaining": 0.20,
    "marginal_gain_per_trial": 0.00005
  },

  "meta": {
    "num_complete_trials": 80,
    "num_failed_trials": 2,
    "num_pruned_trials": 18,
    "analysis_window": 20,
    "data_sufficient": true
  }
}
```

## 模块三：RL Controller

### 核心定位

RL Controller 不负责直接修改整个 Search Space。它负责学习：

$$
\boxed{
Diagnostic\ State
\rightarrow
Setup\ Intervention
}
$$

即当前 HPO 处于什么状态时，哪一种 setup intervention 最值得执行？

LLM 仍负责 reasoning 和具体 setup 修改。

因此：

$$
\boxed{
Diagnostic\ Engine = Observe
}
$$

$$
\boxed{
RL\ Controller = Decide
}
$$

$$
\boxed{
Revised\ Setup\ Agent = Execute
}
$$

### RL Environment

一次完整的 HPO adaptation episode：

$$
S_t
\rightarrow
HPO
\rightarrow
D_t
\rightarrow
A_t
\rightarrow
S_{t+1}
$$

其中：

- $S_t$：当前 HPO setup；
- $D_t$：Diagnostic State；
- $A_t$：RL Controller 选择的 intervention；
- $S_{t+1}$：Revised Setup Agent 执行 intervention 后的新 setup。

RL transition：

$$
\boxed{
D_t\xrightarrow{A_t}D_{t+1}
}
$$

### State

第一版不要直接把所有浮点数送进 tabular RL。

使用离散化后的 state representation。

Global state：

```text
stagnation:
    0 / 1

recent_improvement:
    low / medium / high

configuration_diversity:
    low / medium / high

coverage:
    low / medium / high

budget_remaining:
    low / medium / high

marginal_gain:
    low / medium / high
```

同时建立 Candidate Hyperparameter Context。

选择 TopK 个最值得干预的 hyperparameters。

Candidate ranking：

$$
CandidateScore_j
=
\alpha Imp_j
+
\beta BC_j^{good}
+
\gamma (1-WidthRatio_j)
$$

其中：

- $Imp_j$：global hyperparameter importance；
- $BC_j^{good}$：top-$q$% good trials 的 boundary concentration；
- $WidthRatio_j$：good region 的相对宽度。

因此：

- importance 越高，越值得干预；
- good trials 越靠近 boundary，越值得干预；
- good region 越集中，越值得干预。


默认：

$$
\alpha=0.5,\quad
\beta=0.3,\quad
\gamma=0.2
$$

仅对 `status = VALID` 的 hyperparameter 计算 CandidateScore

对每个 candidate 记录：

```text
importance:
    low / medium / high

boundary_concentration:
    low / medium / high

good_region_width:
    narrow / medium / wide
```

因此 RL state 可以表示成：

$$
\boxed{
s_t=(GlobalState_t,\ CandidateContext_t)
}
$$

### Action Space

第一版只允许以下 7 种 high-level actions：

```text
1. EXPAND_SEARCH_SPACE
2. SHIFT_SEARCH_REGION
3. SHRINK_SEARCH_SPACE
4. FREEZE_HYPERPARAMETER
5. INCREASE_EXPLORATION
6. INCREASE_BUDGET
7. STOP
```

其中：

- EXPAND_SEARCH_SPACE
    
    扩大某个 numerical hyperparameter 的有效范围。  
    target = numerical hyperparameter  
    parameters.direction = lower / upper / both

- SHIFT_SEARCH_REGION

    把 search space 向 good region 移动。  
    target = hyperparameter  
    parameters.target_region = [lower, upper]

- SHRINK_SEARCH_SPACE

    将搜索空间收缩到已经表现良好的区域。  
    target = hyperparameter  
    parameters.target_region = [lower, upper]

- FREEZE_HYPERPARAMETER

    固定长期低 importance 的 hyperparameter。  
    target = hyperparameter  
    parameters.value = ...

- INCREASE_EXPLORATION

    增加 optimizer exploration，或者切换到更具有探索性的 optimizer。  
    target = optimizer  
    parameters = optimizer-specific configuration


- INCREASE_BUDGET

    增加剩余 trial / time budget。  
    target = budget  
    parameters.additional_trials = ...


- STOP

    终止当前 HPO adaptation episode。  
    target = null  
    parameters = {}

### RL Action Schema

RL Action 不仅包含 Action Type，还必须包含具体作用对象。

定义：

$$
\boxed{
A_t=
(
ActionType_t,
Target_t,
Parameters_t
)
}
$$

其中：

- `ActionType`：7 个预定义 high-level actions 之一；
- `Target`：需要修改的 hyperparameter 或 optimizer；
- `Parameters`：执行 action 所需的具体参数。

统一 JSON Schema：

```json
{
  "action": "EXPAND_SEARCH_SPACE",
  "target": "learning_rate",
  "parameters": {
    "direction": "upper"
  },
  "confidence": 0.87,
  "q_value": 0.63
}

### Reward

一次 intervention 后重新执行一个 HPO phase。

定义 intervention 前后的 best objective：

$$
B_t,\quad B_{t+1}
$$

则 relative improvement：

$$
\boxed{
\Delta_t=
\frac{
B_{t+1}-B_t
}{
|B_t|+\epsilon_y
}
}
$$

定义 intervention cost：

$$
C_t=
\frac{
N_{\text{phase}}
}{
N_{\text{total budget}}
}
$$

reward：

$$
\boxed{
r_t=
\operatorname{clip}
\left(
\Delta_t
-\lambda C_t
-r_{\text{invalid}},
-r_{\max},
r_{\max}
\right)
}
$$

其中：

- $\lambda$：budget cost coefficient；
- $r_{\text{invalid}}$：非法 action penalty；
- $r_{\max}$：reward clipping threshold。

默认：

```yaml
lambda_cost: 0.1
invalid_action_penalty: 0.2
reward_clip: 1.0
```

### RL Algorithm

第一版使用 Tabular Q-Learning。不直接训练 LLM。

更新公式：

$$
\boxed{
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
\left[
r_t+
\gamma\max_{a'}Q(s_{t+1},a')
-Q(s_t,a_t)
\right]
}
$$

其中：

- $\alpha$：learning rate；
- $\gamma$：discount factor。

默认：

```yaml
learning_rate: 0.1
discount_factor: 0.9
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay: 0.995
```

Action selection：

$$
a_t=
\begin{cases}
\text{random action},&p<\epsilon\\
\arg\max_aQ(s_t,a),&p\geq\epsilon
\end{cases}
$$

### 为什么只使用 Q-Learning

不要直接使用 PPO / GRPO / RLHF 等大型算法。

当前 action space 很小，而 HPO adaptation 本身已经是一个昂贵 environment。

Tabular Q-Learning 的优势：

- 不需要训练 LLM；
- 不需要 differentiable environment；
- action space 小；
- 容易记录和解释：
  
$$
Q(s,a)
$$

直接表示在这种 HPO 状态下采取这个 action 的历史收益。

### RL Controller 输出

```json
{
  "action": "EXPAND_SEARCH_SPACE",
  "candidate_hyperparameter": "learning_rate",
  "confidence": 0.87,
  "q_value": 0.63
}
```

注意，RL Controller 只确定 **what to do**，不负责决定具体把 learning rate 改成多少。

具体数值修改交给 Revised Setup Agent。

### 特别注意

Agent/RL 根据以下逻辑进行 action selection。

| Diagnostic Pattern | Diagnosis | Recommended Action |
|---|---|---|
| stagnation + high good-boundary concentration + high importance | Search-space truncation | EXPAND_SEARCH_SPACE |
| high importance + narrow good region | Promising region identified | SHIFT / SHRINK |
| low importance for multiple rounds | Low-value hyperparameter | FREEZE_HYPERPARAMETER |
| stagnation + very low diversity | Possible premature convergence | INCREASE_EXPLORATION |
| stagnation + low coverage | Insufficient exploration | INCREASE_EXPLORATION |
| clear improvement + budget nearly exhausted | More budget may be valuable | INCREASE_BUDGET |
| stagnation + very low marginal gain + budget consumed high | Low expected benefit | STOP |

## 模块四：Revised Setup Agent

### 模块定位

Revised Setup Agent 接收：

```text
Current Setup
+
Diagnostic State
+
RL Action
+
Relevant Memory
```

输出：

$$
\boxed{S_{t+1}}
$$

即下一轮：

```text
Search Space
Optimizer
Budget
Evaluation
```

### Revised Setup Agent Prompt

```text
You are the Revised HPO Setup Agent.

Your task is to revise the current HPO setup based on structured
diagnostic evidence and a setup intervention selected by an RL controller.

You must NOT execute HPO yourself.

INPUTS
------

1. Current Setup
2. Diagnostic State
3. RL-selected Action
4. Relevant historical experiences retrieved from Memory
5. Allowed optimizers and setup constraints

CURRENT SETUP
-------------
{{CURRENT_SETUP}}

DIAGNOSTIC STATE
----------------
{{DIAGNOSTIC_STATE}}

RL ACTION
---------
{{RL_ACTION}}

RELEVANT MEMORY
---------------
{{MEMORY}}

YOUR TASK
---------

1. Interpret the diagnostic evidence.

2. Determine which component of the current setup should be changed.

3. Apply the requested intervention conservatively.

4. Preserve all unchanged setup components whenever possible.

5. Produce the revised setup.

ACTION DEFINITIONS
------------------

EXPAND_SEARCH_SPACE:
- Identify the target hyperparameter.
- Expand its range toward the side indicated by the good-trial
  distribution.
- Do not expand unrelated parameters.

SHIFT_SEARCH_REGION:
- Move the search interval toward the empirically observed good region.
- Preserve sufficient exploration range.

SHRINK_SEARCH_SPACE:
- Narrow the range around the observed good region.
- Only do this when evidence indicates that focusing the search is
  beneficial.

FREEZE_HYPERPARAMETER:
- Fix a low-importance hyperparameter to a reasonable value.
- Prefer its default value or the value supported by high-performing
  configurations.

INCREASE_EXPLORATION:
- Modify the optimizer or optimizer-specific exploration settings.
- Only select another optimizer from the allowed optimizer list.

INCREASE_BUDGET:
- Increase the remaining trial/time budget within the global resource
  constraint.

STOP:
- Return the current setup and mark the optimization process as terminated.

RULES
-----

- Never introduce hyperparameters that do not exist.
- Never violate hard resource constraints.
- Never create an invalid conditional search space.
- Never change unrelated setup components without evidence.
- Do not blindly trust a single diagnostic metric.
- Prefer changes supported by multiple diagnostic signals.
- Preserve the original setup in the output for auditability.
- Provide concise rationale and confidence, not hidden chain-of-thought.

OUTPUT
------

Return ONLY a valid JSON object following RevisedSetupSchema.
```

### Revised Setup 输出 Schema

```json
{
  "base_setup_id": "setup_001",
  "action": "EXPAND_SEARCH_SPACE",

  "changes": [
    {
      "hyperparameter": "learning_rate",
      "old": {
        "low": 1e-5,
        "high": 1e-2,
        "log": true
      },
      "new": {
        "low": 1e-5,
        "high": 1e-1,
        "log": true
      },
      "reason": "High importance and strong concentration of good trials near the upper boundary."
    }
  ],

  "new_setup": {
    "search_space": {},
    "optimizer": "TPE",
    "budget": {},
    "evaluation": {}
  },

  "confidence": 0.87,
  "assumptions": []
}
```

## 其它模块和说明

### Tool Validator

一些模块的输出有特定的格式要求，例如 Initial Setup Agent 应当输出 JSON，并且必须包括 Search Space、HPO、Budget 等 Keys。这个时候就需要一个 Tool Validator 去检查它们的输出是否符合格式。

### Classical HPO 模块

这个模块需要根据前面 Agent 设计的 HPO 算法，去实际调用它，完成超参数的搜索。

### Agent Memory

#### 第一次 Setup

在 Initial Setup Agent 启动前：

$$
Task
\rightarrow
Memory Retrieval
\rightarrow
Initial Setup Agent
$$

读取：

- 相似 task 的 setup；
- 相似 model 的 hyperparameter；
- 历史成功 setup；
- 历史失败 setup。

作用：

$$
\boxed{\text{提供 Prior}}
$$

#### 后续 Setup

在 Revised Setup Agent 执行前：

$$
Diagnostic\ State
\rightarrow
Memory\ Retrieval
\rightarrow
Revised\ Setup\ Agent
$$

重点检索：

$$
\boxed{
Similar\ Diagnosis
\rightarrow
Historical\ Action
\rightarrow
Historical\ Outcome
}
$$

例如：

```text
Current Diagnosis:
stagnation = true
learning_rate importance = high
boundary_concentration = 0.91

Retrieved Experience:
same diagnosis
→ expand learning_rate
→ improvement = +1.8%
→ success = true
```

#### 什么时候写入 Memory？

不要每个 trial 都写。只在一个完整 adaptation step 结束之后写入：

$$
\boxed{
S_t
\rightarrow
HPO
\rightarrow
D_t
\rightarrow
A_t
\rightarrow
S_{t+1}
\rightarrow
Outcome
}
$$

形成一个完整 Experience。


#### Memory Schema

推荐使用 SQLite，而不是一开始引入复杂 Vector Database。

每一个 Experience：

```json
{
  "experience_id": "exp_001",

  "task_signature": {
    "task_type": "classification",
    "model_family": "XGBoost",
    "dataset_signature": "..."
  },

  "setup_before": {},

  "diagnostic_state": {},

  "diagnosis": {
    "primary": "SEARCH_SPACE_TRUNCATION",
    "confidence": 0.91
  },

  "action": {
    "type": "EXPAND_SEARCH_SPACE",
    "target": "learning_rate"
  },

  "setup_after": {},

  "outcome": {
    "best_before": 0.912,
    "best_after": 0.928,
    "relative_improvement": 0.0175,
    "budget_consumed": 20
  },

  "reward": 0.015,

  "success": true,

  "rationale": "High-importance parameter concentrated near upper boundary."
}
```

#### Memory 数据库设计

至少设计两个表。

1. experiences

```text
experience_id
task_signature
setup_before_json
diagnostic_state_json
diagnosis_json
action_json
setup_after_json
outcome_json
reward
success
created_at
```

2. semantic_rules

用于保存从多次 Experience 中总结得到的经验：

```text
rule_id
condition_json
recommended_action
success_rate
sample_count
average_reward
created_at
updated_at
```

例如：

```json
{
  "condition": {
    "stagnation": true,
    "importance": "high",
    "boundary_concentration": "high"
  },
  "recommended_action": "EXPAND_SEARCH_SPACE",
  "success_rate": 0.73,
  "sample_count": 22
}
```

#### Memory Retrieval

检索 query：

$$
\boxed{
q=
TaskSignature
+
DiagnosticSignature
}
$$

优先按照：

$$
Similarity=
\lambda_1 TaskSimilarity
+
\lambda_2 DiagnosisSimilarity
+
\lambda_3 ModelSimilarity
$$

排序。

第一版不要求实现复杂 embedding retrieval。

优先实现：

1. exact task/model match；
2. diagnosis type match；
3. hyperparameter/action match；
4. success/reward 排序。

之后可以增加 embedding-based semantic retrieval。

#### Memory 写入规则

不是所有 Experience 都值得长期保存。

优先保存：

$$
\boxed{
\text{successful intervention}
}
$$

以及：

$$
\boxed{
\text{unexpected failure / useful negative experience}
}
$$

例如：

```text
Action:
EXPAND_SEARCH_SPACE

Outcome:
performance decreased significantly

→ 保存，因为 Agent 未来应该避免类似 intervention。
```

#### Memory 与 RL 的关系

Memory 和 RL 不重复。

Memory 回答：

> **过去类似情况发生过什么？**

RL 回答：

> **根据历史 reward，当前应该选择什么 action？**

因此：

$$
\boxed{
Memory = Experience Retrieval
}
$$

$$
\boxed{
RL = Policy Learning
}
$$