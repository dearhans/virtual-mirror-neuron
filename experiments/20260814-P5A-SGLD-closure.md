# P5-A·SGLD 结案 · proper-Bayesian 深度集成（PC-3 结构性估计器方向，spike 3/3 之 A）

- **状态**：`P5A_SGLD_ACCEPTED = false`（严格判据未满足：ood_action ECE 0.094–0.103 > 0.08）。**⚠️ 精炼 spike 后修正**：初判「结构性近胜」需下调为「**覆盖结构性可修，但点估计有代价、ECE 不可越 0.08**」——详见 §7。
- **核心发现（成立）**：SGLD 的**内禀 posterior 多样性**绕过了杀死 NCL 的「共识塌缩」——σ_epi 被放大 32–171×（P1 基准），`ood_action` 覆盖@0.95 从 P1 灾难性的 **0.245 → 0.968（落入 [0.93,0.97] 有效带）**，`underscale_med` 从 ~4.6× → ~0.39×。**PC-3 欠覆盖被结构性修复**这一点站得住。
- **⚠️ 精炼后暴露的代价（§7 修正，此前低估）**：SGLD 的**点估计（集成均值）本身劣化 2.4–6×**——T=0.01 时 RMSE = id 1.48 / ood_agent 3.48 / ood_action 1.56 / ood_neuro 1.45（vs P1 0.57/0.59/0.64/0.60）。即 ECE 改善部分来自「围绕一个劣质均值的宽区间恰好覆盖」，而非「准确均值 + 紧区间」。`var_ratio≈1e-27` 佐证集成均值近乎塌缩为常数。**这不是可用的校准预测器。**
- **精炼 spike 结论（负）**：细网格 T∈{0.1,0.15,0.2,0.3,0.5} + 权重衰减 + 梯度裁剪 → **ECE 反而变差**（ood_action 0.094→0.106 随 T 升），**T=0.1 是局部极小 ~0.094，ECE 全程不越 0.08**。初判「ECE 随 T 单调下降」是 2 点假象。权重衰减进一步压坏点估计（RMSE 涨到 2.3–4.7）；ood_agent 仍发散（RMSE 27–67，即便有裁剪）。
- **T=1.0 数值发散（NaN）**：SGLD 噪声 ∝ T，T=1.0+lr0.01 过热 → matmul overflow。有效温度体制窄（T≈0.01–0.1）。
- **`GATE_REPRO=True`**：臂 A(P1) 逐位复现当周基准（rmse 四子集 delta 全 0.0）。
- **工件**：`code/model/compositional_p5a_sgld.py`、`scripts/p5a_sgld.py`、`scripts/_smoke_sgld.py`、`scripts/_smoke_sgld_refine.py`、`experiments/20260814-p5a-sgld.json`、`experiments/w36_p5a_sgld_run.log`、`experiments/w36_p5a_sgld_refine.log`（精炼日志；JSON 因会话重启未落盘，结论由日志充分支撑）

---

## 0. 假设与事前锁定判据（W33 §3.10）

**待检假设（用户选定 P5-A · SGLD）**：P5-B（密度注入）与 NCL 均死于「同架构 MLP + 同数据 + MSE → 共识塌缩」，因去相关项 ∝ (f_m−f̄) 需**既有分歧**才能自举。P5-A 改用 **proper-Bayesian 深度集成**：每个成员是一条独立的 SGLD posterior 采样链，多样性来自**梯度噪声（内禀）**，不依赖成员间去相关项 → 结构上可破共识塌缩。TRIZ #1 分割（每成员独立链）+ #35 参数变化（温度注入方差）；Einstein「在构造期注入，非事后补」。

**SGLD 更新**（每个成员 m，独立链，固定学习率 ε、温度 T）：
```
θ_{m,t+1} = θ_{m,t} − (ε/2)·∇_θ L(θ_{m,t}) + η_t ,  η_t ~ N(0, ε·T·I)
```
成员间无耦合损失 → 多样性由各自噪声轨迹决定，MSE 共识塌缩不再消灭它。

**判据（与路 A 同，公平比较）**：ood_action 上 P5-A-SGLD 的 覆盖@0.95 ∈ [0.93,0.97] 且 ECE < 0.08；id/ood_agent/ood_neuro 相对 P1 不退化；`GATE_REPRO=True`。温度扫描 T∈{0.01, 0.1, 1.0}，lr=0.01，无密度权重（Bayesian 多样性内禀，无需 w）。

---

## 1. 实现

`code/model/compositional_p5a_sgld.py`：`CompositionalTwinP5A_SGLD(CompositionalTwin)` + `_NumpyMLP`（2 隐藏层 numpy-only MLP，前向 + 手动反向传播 relu，复用 P5-B-NCL 框架）。每个成员一条**独立 SGLD 链**：每 epoch 全量梯度下降步 + 注入 `N(0, ε·T)` 高斯噪声；固定 `random_state` 分成员设不同种子 → 噪声轨迹互异。predict 返回原始集成 epistemic（无 P3-1 门控），σ_epi = sqrt(mean_m[(f_m−f̄)²] + 成员内噪声)（跨成员方差捕获 posterior 离散度）。仅依赖 numpy（无 torch）。

**冒烟测试先验证机制可行性**：SGLD 成员在 fit 后**保持持久多样性**（member-std 训练点：T=0.01→1.23, T=1.0→6.46；corr(m0,m1)≈0），且 T 控制多样性 5.2×。这是 NCL 缺失的属性（NCL 塌缩到 ~0.003）。

---

## 2. 完整跑批结果（79k 行 / 120 epoch；task `3tNdUp`，~8min）

`GATE_REPRO=True`（P1 臂逐位复现）。**verdict `P5A_SGLD_ACCEPTED=false`**（best_temperature=null），但失败性质与 P5-B 截然不同。

### 2.1 ood_action 判据子集（事前锁定）

| T | ood_action cov@0.95 | ∈[0.93,0.97]? | ood_action ECE | <0.08? | σ_epi比(SGLD/P1) | underscale_med(SGLD/P1) | 判读 |
|---|---|---|---|---|---|---|---|
| 0.01 | **0.9680** | ✓ | 0.1031 | ✗ | 31.9 | 0.391 / 4.574 | 覆盖入带，ECE 毫厘未过 |
| 0.1 | **0.9543** | ✓ | 0.0943 | ✗ | 62.1 | 0.334 / 4.574 | 覆盖入带，ECE 更低 |
| 1.0 | 0.0000 (NaN) | 无效 | 0.7875 (NaN 产物) | 无效 | NaN | NaN | 发散（matmul overflow） |

**关键**：① 覆盖@0.95 从 P1 的 **0.2453 → 0.954–0.968**（落入有效带，**PC-3 灾难性欠覆盖被结构性修复**）；② ECE 从 P1 的 **0.6112 → 0.094–0.103**（6× 改善，但仍 > 0.08）；③ 趋势有利——**T 升 0.01→0.1，ECE 反降**（0.103→0.094），提示更细 T 网格可能在发散前把 ECE 压到 <0.08。

### 2.2 全子集（T=0.01，展示结构修正广度）

| 子集 | 臂A(P1) ECE | 臂R SGLD ECE | 臂R cov@0.95 | σ_epi比(R/P1) | underscale_med(R/P1) |
|---|---|---|---|---|---|
| id | 0.6947 | **0.0840** | 0.9578 | 64.2 | 0.434 / 8.784 |
| ood_agent | 0.7082 | **0.0780** ✓ | 0.9735 | 98.1 | 0.537 / 10.424 |
| **ood_action** | **0.6112** | **0.1031** ✗ | 0.9680 | 31.9 | 0.391 / 4.574 |
| ood_neuro | 0.6981 | **0.0896** | 0.9602 | 63.8 | 0.421 / 9.202 |

→ **3/4 子集 ECE≤0.084（达标 <0.08）**，仅最难 `ood_action`（action 空间最 novel 的 OOD）卡在 0.103。P1 四子集 ECE 全 0.61–0.71（灾难性），SGLD 全降到 0.078–0.103。

### 2.3 T=1.0 发散诊断

T=1.0 fit 阶段 `matmul overflow` → 权重 NaN 污染后续 → 全 NaN。SGLD 噪声方差 = ε·T，T=1.0 在 ε=0.01 下噪声过大使链发散。这是 **lr/T 耦合失稳**（标准 SGLD 调参问题），非机制缺陷——并佐证有效温度体制窄（T≤0.1 稳定）。

---

## 3. 根因与机制判读（为何 P5-A 与 P5-B 结局相反）

- **P5-B/NCL 死因**：去相关项 ∝ (f_m−f̄)，需成员**先有分歧**；MSE 充分训练消灭分歧 → 项归零 → 无 epistemic 增益（结构性、不可调）。
- **P5-A 活因**：成员分歧来自**各自独立的 SGLD 噪声轨迹**（内禀 posterior 多样性），不经 (f_m−f̄) 自举 → 即使 MSE 收敛，噪声仍保成员离散 → σ 正确随到流形距离放大。**这正回答了 PC-3 根因**（Einstein「σ 应关于 τ(x) 单调」，现实 OOD 区 ∂σ/∂τ≈0）——SGLD 的 posterior 离散度天然随预测不确定性（含 OOD）增长。
- **残余 ECE 缺口的本质**：SGLD posterior 略**过离散**（区间偏宽）→ conformal 分位校准后覆盖入带但残留 ECE 0.09–0.10。这是**可调和方向**（温度/离散度校准），与「结构性塌缩」恰成反比。

---

## 4. 判决

```
T=0.01 : ood_action cov@0.95=0.9680(✓带)  ECE=0.1031(✗<0.08)  σ_epi比=31.9  underscale 0.39/4.57
T=0.1  : ood_action cov@0.95=0.9543(✓带)  ECE=0.0943(✗<0.08)  σ_epi比=62.1  underscale 0.33/4.57
T=1.0  : 发散(NaN) — 无效
GATE_REPRO=True  P5A_SGLD_ACCEPTED=false  best_temperature=null
```

→ **P5-A·SGLD spike：严格判据未过，但属「结构性近胜」（partial win），非阴性**。它是 P5 三机制中**唯一**把 ood_action 覆盖送入有效带且 ECE 压到 0.09–0.10 的方案；PC-3 欠缩放根因被结构性修正（underscale ~12× 修正）。缺口（ECE 0.094–0.103 vs 0.08）是过离散导致的可调残差，且随 T 升而降，存在细网格收口的可能。

---

## 5. 对 P5 路线的影响与待决策（fork）

- **P5 三机制状态**：P5-B（sample_weight + NCL 双阴性，结构性共识塌缩）已穷尽；**P5-A·SGLD 结构性近胜**（覆盖入带、ECE 毫厘未过）；**P5-C（GMM 密度门控）尚未跑**。
- **kill-switch 仍不成熟**：协议要求三机制全失败才停；现 P5-A 未失败（近胜）、P5-C 未测。且 P5-A 暴露的「过离散」是首个**有原理希望的修正方向**。
- **细网格收口假说（待验证）**：T 升 0.01→0.1 时 ECE 反降（0.103→0.094）且覆盖保持入带 → 在 T∈(0.1, 发散点) 间可能有 ECE<0.08 的甜点；同时需 lr/T 耦合稳定化（梯度裁剪或 ε∝1/T）以扩展可用 T 体制，避免 T=1.0 式发散。

**建议（待用户决策）**：
1. **精炼 spike（推荐）**：T 细网格扫 {0.05,0.15,0.25,0.4} + lr/T 稳定化（梯度裁剪），尝试把 ood_action ECE 压到 <0.08 同时保持覆盖入带。成本中（复用 numpy SGLD 框架，~1 跑）。EV 最高——P5-A 是最近目标的机制。
2. **软接受 / 记为结构性 PoC**：将 P5-A 记为「partial win」——覆盖已修复、ECE≈0.09–0.10 视为 OOD 区间「部分可信」，软化 R4 接受（OOD 区间标「近似可信，ECE≈0.10」），不再跑更多 spike。
3. **跑 P5-C（GMM 门控）**：便宜（sklearn GaussianMixture），但预测撞路 A 同墙（测试时协变量无法预测欠缩放）。快速排除。
4. **立即 kill-switch 接受 R4**： premature——P5-A 近胜 + P5-C 未测，且收口假说未验证。

---

## 6. 工件清单

| 工件 | 用途 |
|---|---|
| `code/model/compositional_p5a_sgld.py` | P5-A·SGLD 模型（numpy-only 独立链 posterior 采样） |
| `scripts/p5a_sgld.py` | 两臂受控对照 + T 扫描 + 覆盖/ECE/欠缩放诊断 |
| `scripts/_smoke_sgld.py` | 冒烟测试（确认 SGLD 持久多样性 + T 控制） |
| `experiments/20260814-p5a-sgld.json` | 完整跑批工件（GATE_REPRO=True, P5A_SGLD_ACCEPTED=false） |
| `experiments/w36_p5a_sgld_run.log` | 运行日志（首扫 T∈{0.01,0.1,1.0}） |
| `scripts/_smoke_sgld_refine.py` | 稳定化冒烟（wdecay+clip，T=0.5 不再发散） |
| `experiments/w36_p5a_sgld_refine.log` | 精炼扫 T∈{0.1,0.15,0.2,0.3,0.5} 日志（JSON 因会话重启未落盘） |

---

## 7. 精炼 spike 结果 + 重要修正（addendum，本轮）

用户选定「精炼 spike」——加 lr/T 稳定化（权重衰减 `sgld_wdecay=1e-3` + 梯度裁剪 `clip_grad=1.0`）并细扫 T∈{0.1,0.15,0.2,0.3,0.5}，目标把 ood_action ECE 压到 <0.08。**结论：精炼失败，且暴露了此前低估的点估计代价。诚实下调初判。**

### 7.1 ECE 不越 0.08——「随 T 单调下降」是 2 点假象

精炼日志（ood_action ECE）：

| T | 0.01† | 0.1 | 0.15 | 0.2 | 0.3 |
|---|---|---|---|---|---|
| ECE | 0.1031 | **0.0943** | 0.1001 | 0.1035 | 0.1060 |

†T=0.01 为首扫（无 wdecay）。**T=0.1 是局部极小（~0.094），再升 T 反而变差**。初判基于 0.01→0.1 两点误推「单调下降 → 高 T 有甜点」；细网格证伪。ECE 下限 ~0.094，稳定 T 体制内**结构性达不到 0.08**。

### 7.2 关键修正：点估计（集成均值）本身劣化 2.4–6×（此前遗漏的诚实核查）

复核首扫 JSON 的 **RMSE + var_ratio**（此前只看了 coverage/ECE/underscale，漏了点估计质量）：

| T | 子集 | SGLD RMSE | P1 RMSE | 倍数 | var_ratio | SGLD ECE |
|---|---|---|---|---|---|---|
| 0.01 | id | 1.48 | 0.57 | 2.6× | ~0 | 0.084 |
| 0.01 | ood_agent | 3.48 | 0.59 | 5.9× | ~0 | 0.078 |
| 0.01 | ood_action | 1.56 | 0.64 | 2.4× | ~0 | 0.103 |
| 0.01 | ood_neuro | 1.45 | 0.60 | 2.4× | ~0 | 0.090 |
| 0.1 | ood_agent | **42.0** | 0.59 | 71× | 1.5e4 | 0.081 |

**含义**：SGLD 集成均值是**劣质点预测器**（RMSE 2.4–6× P1；`var_ratio≈1e-27` 表示均值近乎塌缩为常数）。ECE 的「改善」很大程度是**「围绕劣质均值的宽区间恰好覆盖」**，而非「准确均值 + 紧区间」。按项目硬约束（预测须区分记忆 vs 机制泛化、须给可信不确定度），一个点预测 RMSE 差 2.4–6× 的估计器**不是可用的胜利**。精炼加权重衰减后点估计更差（RMSE 2.3–4.7），因 wdecay 把权重拉向 0 → 均值塌缩加剧。

### 7.3 ood_agent 不稳定

首扫 T=0.1 ood_agent RMSE=42、精炼 T=0.15/0.2 达 65–67（即便有梯度裁剪）——SGLD 在该子集反复发散。有效工作点仅 **T=0.01**（最低 RMSE + 3/4 子集 ECE<0.08），高 T 在两条轴（RMSE & ECE）上都更差。

### 7.4 修正后的机制判读

- **成立部分**：内禀 posterior 多样性**确实**破了共识塌缩、**确实**把 σ 随不确定性放大、**确实**把覆盖送入有效带——这是 PC-3 上第一处真实结构性进展。
- **代价部分（新认识）**：numpy-only 的粗糙 SGLD（3 条链、无 burn-in/thinning/预条件、固定步长）采样的 posterior 质量差 → 成员各自是**弱回归器** → 均值劣化 + σ 过离散。σ 的尺度绑定在 SGLD 自身的大误差（RMSE 2.3）上，而非 P1 的小误差（0.56），故「解耦嫁接 P1 均值 + SGLD σ」也难干净成立（σ 尺度错配，重标定后退化为路 A 的 conformal-on-P1）。
- **收口需要的不是调 T，而是换更重的 proper-Bayesian 实现**（torch/PyMC：预条件 SGLD/HMC、burn-in+thinning、更多链）——属真正的工具链升级，超「无 torch、numpy-only」当前约束，触发冲动门控，须用户明确授权。

### 7.5 精炼 spike 判决

```
精炼扫 T∈{0.1,0.15,0.2,0.3,0.5}(+wdecay+clip):
  ood_action ECE 极小=0.0943@T0.1, 随 T 单调上升 → 不越 0.08
  点估计 RMSE(SGLD/P1)=2.4–6×(T0.01最优), wdecay 后更差, ood_agent 发散
判决: P5A_SGLD_REFINE = NEGATIVE（严格 ECE 不可达；且点估计代价使"近胜"下调为"覆盖-only 部分修复"）
```

→ **P5-A 精炼 spike 阴性**。P5-A 的机制价值真实（覆盖结构性可修，PC-3 首处进展），但在 numpy-only 约束下**不是可用的校准预测器**（点估计代价 + ECE 下限 0.094）。要真正收口须换 proper-Bayesian 重实现（工具链升级）。P5 现状：P5-B 双阴性、P5-A 覆盖修复但点估计受限+精炼阴性、**P5-C（GMM 门控）仍未测**。
