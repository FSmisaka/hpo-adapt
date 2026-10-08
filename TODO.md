# HPO-Adapt 收尾 + Tool Paper 一周执行计划

> 目标：① 完成工具包（达到 spec.md / README.md 承诺）；② 写一篇 ~4 页英文 **Tool Paper**（学术界对"介绍软件工具包"论文的叫法：*tool paper / demo paper / system demonstration*；ML 领域对口专栏是 JMLR MLOSS track，4 页篇幅对标各会议 demo/short track）。
> 约束：仅 heuristic 后端做实验；总工期 ~7 天；质量优先、时间盒管理。

---

## 本次会话状态（2026-10-04 晚）

**已完成：Phase A 全部（A7 按计划砍掉）、Phase B 全部（B4/B5 按计划砍掉）、Phase C（C1/C2/C4/C5 完成；C3 按计划只报 mean±std）、Phase D 论文完整初稿（`paper/paper.tex` 已编译为 4 页正文 + 1 页参考文献的 `paper.pdf`，图由你手绘，需求见 `paper/FIGURES.md`）。**

- 正式实验：240 runs 全部完成，`experiments/results/full/`（`summary.csv` + `figs/` 全套图表数据 + `case_study.json`）。
- 实验过程中发现并修复了 4 个真问题（详见下方"实现与计划不同之处"）。
- **待办（下次会话/你本人）**：Phase E1 fresh-venv 复现性检查、E3 论文终审、E4（可砍）、E5 git tag；以及按 `paper/FIGURES.md` 绘制 4 张图并替换 `paper.tex` 里的 `[FIGURE: ...]` 占位框（替换后需重新编译并微调分页）。

### 实现与计划不同之处（诚实记录）

1. **A3 的收尾方式**：不是"按 STOP 语义收尾"，而是`has_next_phase == False` 时**不选 action**、只记录诊断后 break（更干净：没有 outcome 的 action 根本不该产生）。
2. **STOP 证据门控（新增修复）**：ε-greedy 会随机抽中 STOP 导致 20/120 trials 提前终止（smoke 实测），严重影响 adapt 方法。现在 STOP 仅在 `stagnation == true 且预算消耗 ≥ 50%` 时可选（与 spec 模式表第 7 行一致）。为此新增 `test_stop_gated_*` 测试。
3. **累计诊断 + warm-start（两项核心修复）**：原实现每轮只看当轮 20 个 trial（importance 永远 INSUFFICIENT，超参 action 永远不可用）；且每 phase 新建 study 丢掉 TPE 历史。现在 Diagnostic Engine 分析累计轨迹，且新 phase 注入历史 trial（不重评、不耗预算）。修复后 synthetic_mlp 上 guided adapt 达到 1.000 vs static-tpe 0.755。
4. **synthetic_mlp 任务校准**：原 metadata `default=1e-3` 使初始 lr 区间 [1e-4,1e-2] 内信号低于噪声（±0.01），任何方法都无法检测截断；改为 sgd 家族标准默认 1e-2（初始区间 [1e-3,1e-1]，GT 0.45 仍在区间外但在可检测范围内）。
5. **Divergence 距离公式**、**seed 传播**、**CMA-ES** 均按计划完成；`cmaes` 加入依赖。
6. **rf_digits 实验结论是"打平"而非"胜出"**（所有方法 ≈0.977，初始 setup 已接近该数据集的 CV 平台）；论文按真实结果如实撰写，未粉饰。

---

## 0. 现状基线（2026-10-04 审计结论，不要重复做）

- 代码库已**高保真实现 spec.md**：6 个核心数据对象、Diagnostic Engine 全部指标（含 VALID/INSUFFICIENT_DATA 门控）、7 action 表格 Q-Learning（默认参数与 spec 完全一致）、双 prompt（与 spec 逐字一致）、SQLite Memory（experiences + semantic_rules）、Tool Validator、闭环 Pipeline + CLI。
- **114 个测试全部通过**（README 写的 109 已过期）。
- 已知缺口（详见下方 Phase A）：diversity 距离公式与 spec 有偏差、`--seed` 未传播到 HPO 采样、最后一轮 pending action 不会闭合、optimizer 只有 RANDOM/TPE、无 LICENSE、web-search 输入未实现（spec 标注"必要时"，可声明 out of scope）。

---

## 1. 总排期（7 天）

| 天 | 阶段 | 产出 |
|---|---|---|
| D1 | Phase A 工具收尾 | 全部 P0 修复合入，测试数更新 |
| D2 | Phase B 实验基建 | `experiments/` 一键跑通小矩阵 |
| D3 | Phase C 正式实验 | 全量 CSV + 图表生成完毕 |
| D4 | Phase D1 论文骨架 | System + Evaluation 初稿 + 图表嵌入 |
| D5 | Phase D2 论文成文 | Intro/Related/Abstract/Conclusion 初稿 |
| D6 | 打磨 | 全文修订、复现性检查（fresh install）、README 更新 |
| D7 | 缓冲 | 意外处理、终审、打包提交 |

---

## Phase A：工具包收尾（D1，全部 P0）

- [x] **A1. 修复 diversity 距离公式对齐 spec**（`src/hpo_adapt/diagnostic/metrics.py:110-136`）
  - categorical 距离改为 spec 的二值不匹配指示 `1(x_ij≠x_kj)`（现为 ordinal 编码 `idx/(n-1)`）；均值分母改为全部 m 个超参（现仅对 varying 参数取均值）。同步更新 `tests/test_diagnostic.py` 对应用例。
  - 验收：`pytest` 全绿；新旧公式在同一测试数据上给出文档化说明（commit message 里记录偏差原因）。
- [x] **A2. `--seed` 传播到 HPO 采样**（`src/hpo_adapt/agents/initial_setup.py:198` 硬编码 `evaluation.seed=0`；`src/hpo_adapt/pipeline.py` 需把 `PipelineConfig.seed` 注入 Setup.evaluation.seed）
  - 验收：不同 `--seed` 产生不同 trial 序列，同 seed 完全复现；补充/修改测试。
- [x] **A3. 闭合最后一轮 pending action 边界情况**（`src/hpo_adapt/pipeline.py:163-289`）
  - 方案：当剩余预算不足以再跑一个 phase 时，不再选择 action，按 STOP 语义收尾（terminal Q-update + 写 Experience）。
  - 验收：任何 ending 路径（STOP / 预算耗尽 / max_rounds）下每个已执行 action 都有 outcome、reward、Q-update、Experience，补测试。
- [x] **A4. 增加 CMA-ES optimizer**（`src/hpo_adapt/hpo/runner.py:42-49` 映射 `CmaEsSampler`；各 task builder 的 allowed list 加入 CMAES）
  - 意义：`INCREASE_EXPLORATION` 的"切换 optimizer"分支才真正可被实验触发。
  - 验收：validator 接受 CMAES；runner 冒烟测试通过；CMA-ES 仅对连续参数 task 启用（含 categorical 的 task 自动回退 TPE，写清规则）。
- [x] **A5. 添加 LICENSE（MIT）** —— tool paper 的 availability 硬要求。
- [x] **A6. README/README_CN 更新**：测试数（当前 114）、CMA-ES、seed 行为、LICENSE 徽章。
- [~] A7.（本会话按计划砍掉）（P2，可砍）Initial Setup Agent 的 web-search 输入：仅加一个可选 `web_context` 参数透传进 prompt，不做真实搜索集成；论文中声明 out of scope。

## Phase B：实验基建（D2）

- [x] **B1. 把 `rf_digits_cv` 从 example 提升为内置任务**（`examples/real_ml_demo.py:49-66` → `src/hpo_adapt/hpo/objectives.py` + `pipeline.py` task builder + CLI 选项）
- [x] **B2. 新增实验目录与一键脚本**
  - `experiments/run_all.py`：扫 任务 × 方法 × seed 矩阵，输出 `experiments/results/*.csv`（每 run 一行：final best、各 budget 分位点 best、rounds、action 计数、诊断耗时占比、invalid 次数、Q 状态数）。
  - `experiments/make_figures.py`：从 CSV 生成 `experiments/figures/*.pdf`。
  - `matplotlib` 加入 dev 依赖；脚本支持 `--smoke`（1 seed 小预算冒烟）。
  - 验收：`python experiments/run_all.py --smoke` 在本机 <10 分钟出结果；`make_figures.py` 一键出全部图；论文里每个数字都能追溯到 CSV。
- [x] **B3. 实验矩阵（P0 版）**

  | 维度 | 取值 |
  |---|---|
  | 任务 | `synthetic_mlp`（GT 最优在初始范围外，专测 EXPAND）、`branin`（minimize）、`rf_digits`（真实 ML，RF + 3-fold CV） |
  | 方法 | `static-random`、`static-tpe`（固定初始 setup 用完全部预算，无 adaptation）、`adapt-full`（完整闭环）、`adapt-rule-only`（消融：只用规则表选 action）、`adapt-random-action`（消融：随机 action，无 RL 无规则） |
  | seeds | 10 个（D1 按单 run 时长定：单任务全 10 seeds 需 <1.5h，否则降 trials 或减 seeds） |
  | 预算 | 每 run 总 trial 数与 static 完全一致（如 120），保证公平 |

- [~] B4.（本会话按计划砍掉）（P1）第六个方法 `adapt-no-memory`（每 run 全新 memory，量化 within-run memory 贡献）
- [~] B5.（本会话按计划砍掉）（P2，可砍）第四个任务 `svm_rbf_digits`

## Phase C：正式实验（D3）

- [x] **C1. 跑全量矩阵**，CSV 归档（文件名带日期与 git commit hash，保证可追溯）。
- [x] **C2. 图表生成**（论文用）：
  - Fig-anytime：各任务 best-so-far vs trials 曲线（mean ± std 阴影），adapt vs static；
  - Fig-case：`synthetic_mlp` 单 episode 案例——lr 搜索区间逐轮演化（标出 GT 最优 0.45）+ 每轮 action + best 曲线（"钱图"）；
  - Fig-action/Q：action 频率分布 + Q 值/ε 随轮次演化（证明 controller 在学）；
  - Tab-main：各任务 final best（mean±std over seeds）+ rounds + 开销占比；
  - Tab-ablation：full vs rule-only vs random-action。
- [~] **C3.（本会话按计划砍掉） 统计**（P1）：Wilcoxon signed-rank，adapt-full vs static-tpe 每任务一测；报 p 值。
- [x] **C4. 开销测量**：Diagnostic Engine + RL + Memory 的 wall-clock 占总时间比例（tool paper 必答："观察-决策"层便宜吗）。
- [x] **C5. 案例素材**：从 `rf_digits` run 里导出一步完整诊断→action→修订记录（README 已有逐指标解读，论文案例直接取材）。

## Phase D：论文（D4–D5，LaTeX 放 `paper/`）

格式：4 页正文 + 参考文献不限；先用通用 article 单栏撰写（若后续确定投稿 venue 再换模板）。

章节与内容映射（含页预算）：

| 章节 | 页预算 | 内容要点 |
|---|---|---|
| Abstract | ~0.1 | 一段式：setup 问题 → 闭环 Observe/Decide/Execute → 结果一句话 → 开源可用 |
| 1 Introduction | ~0.7 | HPO setup 决定搜索上限；路线一（CAVE/DeepCAVE，人在环）与路线二（AgentHPO/AutoML-Agent，替代而非修订 setup）之间的空档；贡献 bullets：工具、7-action 闭环、可复现实验、开源 |
| 2 Related Work | ~0.4 | CAVE/DeepCAVE；LLM-agent HPO；经典 adaptive/resampling HPO 一句带过；定位：deterministic diagnosis + RL decision + agent execution |
| 3 System | ~1.2 | 架构图（Fig-arch，重画 README 闭环图）；6 数据对象一笔带过；Observe（指标表 + VALID/INSUFFICIENT_DATA）；Decide（7 action 表 + Q-learning + reward 公式）；Execute（保守修订 + validator）；Memory（两表 + 写入策略）；heuristic/LLM 双后端说明；CLI/API 快速上手代码块 |
| 4 Evaluation | ~1.2 | 设置（矩阵/预算/seeds）→ Fig-anytime + Tab-main → Tab-ablation → Fig-case 案例走读 → 开销 → 可复现性（114+ tests、全 seed 固定、一键脚本） |
| 5 Conclusion & Availability | ~0.15 | MIT、安装一行命令、复现命令一行 |

- [x] D1a. `paper/paper.tex` 骨架 + Fig-arch 架构图（TikZ 或 draw.io 导出 PDF）
- [x] D1b. System 节初稿（素材直接来自 spec/README，最快写）
- [x] D1c. Evaluation 节初稿（全部数字来自 C 阶段 CSV，禁止手填）
- [x] D2a. Intro + Related + Abstract + Conclusion
- [x] D2b. 参考文献列表（spec 中已给出的：Hutter'14 fANOVA、CAVE、DeepCAVE JMLR'25、AgentHPO、AutoML-Agent、Tornede TMLR'24、LLM-vs-classical-HPO arXiv'26 等，补 Optuna）
- [x] D2c. 全文自查清单：每个 claim 有出处/数字、无过度声称（不声称 beat SOTA，声称"透明、可复现的自适应 setup 控制"）、4 页限制内、图表自包含（caption 可独立读懂）

## Phase E：打磨与交付（D6–D7）

- [ ] E1. 复现性检查：fresh venv `pip install -e ".[dev]"` → `pytest` → `python experiments/run_all.py --smoke` 全通
- [ ] E2. README 增补论文实验复现入口（一条命令）
- [ ] E3. 论文通读修订 ≥2 轮（术语一致性：setup / intervention / adaptation episode）
- [~] E4.（本会话按计划砍掉）（P2）terminal 录制或 run-log 截图（demo paper 惯例素材）
- [ ] E5. 打包：git tag `v0.2.0`，提交课程材料

---

## 削减线（时间不够时按序砍）

1. P2 全砍：A7 web-search、B5 第四任务、E4 录制、Zenodo DOI
2. C3 统计检验降级为只报 mean±std
3. B4 no-memory 消融砍掉（保留 rule-only 与 random-action 两个消融）
4. seeds 10 → 5（最后手段）

**不可砍**：A1–A6、B1–B3、C1–C2、C4、D 全部、E1/E3。

## 风险与对策

| 风险 | 对策 |
|---|---|
| adapt-full 在部分任务上赢不了 static-tpe | 工具论文的卖点不是 SOTA，是"闭环可观察、可解释、可复现"。诚实报告 + 用 Fig-case 展示机制价值（如 synthetic_mlp 上 EXPAND 找到界外最优）；在 paper 中明确讨论局限 |
| 全量实验跑不完 | 先 `--smoke` 定价单 run 时长，再定 seeds/tasks；优先保 3 任务 × 5 方法 × 10 seeds |
| CMA-ES 与 categorical 参数不兼容 | A4 中按任务类型自动启用/回退；论文 Evaluation 里写明 |
| 论文页数超 4 | 优先压缩 Related Work 与 System 细节（指标表瘦身），Evaluation 不动 |
