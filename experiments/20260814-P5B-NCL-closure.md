# P5-B·proper NCL 结案 · 负相关性学习（PC-3 结构性估计器方向，spike 2/3）

- **状态**：`P5B_NCL_ACCEPTED = false`（sparse-w 与 uniform-w 两跑批**均阴性**，GATE_REPRO=True）
- **结论**：P5-B 的 NCL（负相关性学习）实现**正确**（已修复 member-agnostic 梯度 bug 并子集验证），但在**完整训练体制**下无法把多样性转移到测试集 epistemic σ：σ_epi 反而比 P1 更小（≈0.06×），ECE 恶化（0.775 vs P1 0.61–0.69），欠缩放加剧（72–109× vs 4–9×）。λ 至多 4 位小数扰动。
- **根因（修正）**：**MSE 共识塌缩**。NCL 去相关项 ∝ (f_m−f̄) 随成员收敛到同一解而→0；完整 79k 行 / 120 epoch 训练下成员必然收敛 → NCL 信号归零 → 无 epistemic 增益。**早期子集诊断显示的 λ 效应是欠收敛瞬态，未存活到完整训练** → 故「w 权重误指定」假设被完整跑批**证伪**；真瓶颈是共识塌缩，w 调参修不了。
- **P5-B 概念穷尽**：sample_weight 与 NCL 两变体均失败，因「同架构 MLP + 同数据 + MSE」必然收敛共识，训练期注入多样性无法在塌缩中存活。这是该集成的**结构性**性质，非调参问题。
- **工件**：`code/model/compositional_p5b_ncl.py`（修复后）、`scripts/p5b_ncl.py`（含 `--w-mode`）、`scripts/_smoke_ncl_fix.py`、`scripts/_diag_ncl.py`、`experiments/20260814-p5b-ncl-sparse.json`、`experiments/20260814-p5b-ncl-uniform.json`、`experiments/w36_p5b_ncl_sparse.log`、`experiments/w36_p5b_ncl_uniform.log`

---

## 0. 假设与事前锁定判据（W33 §3.10）

**待检假设（用户选定「投资 proper NCL 实现」）**：P5-B（密度感知多样性注入）概念成立，但 sklearn `sample_weight` 代理表达不了成员负相关。本 spike 用 custom numpy 训练循环在**损失层**直接表达 Negative Correlation Learning（Liu & Yao 1999 / Brown 2005），公平验证 P5-B 概念：

```
L_m = MSE(f_m, y) + λ · mean_i[ w_i · (f_m(x_i) − f̄(x_i)) · (f̄(x_i) − y_i) ]
f̄ = mean_m f_m ;  w_i = 局部训练支撑稀疏度（连续 kNN，高=稀疏/OOD-like）
```

第二项在**稀疏区强制成员负相关** → 集成在 OOD 天然分散 → ∂σ/∂τ>0 由构造满足（TRIZ #1 分割 + #15 动态化）。

**判据（与路 A 同，公平比较）**：ood_action 上 P5-B-NCL 的 覆盖@0.95 ∈ [0.93,0.97] 且 ECE < 0.08；id/ood_agent/ood_neuro 相对 P1 不退化（覆盖@0.95 不跌破 0.85、ECE 不爆）。`GATE_REPRO=True`（臂 A=P1 逐位复现当周基准）。

---

## 1. 实现与修复

`code/model/compositional_p5b_ncl.py`：`CompositionalTwinP5B_NCL(CompositionalTwin)` + `_NumpyMLP`（2 隐藏层 numpy-only MLP，前向 + 手动反向传播 relu）。NCL 联合训练：每 mini-batch 前向所有成员，按 NCL 梯度更新各自参数。predict 返回原始集成 epistemic（无 P3-1 门控）。仅依赖 numpy（无 torch）。

**关键修复（首次 NCL 跑批 λ 零效应 bug）**：原梯度为
```python
ncl_grad = (self.ncl_lambda / len(Xb)) * (wb[:, None] * (Fbar - Yb))   # ❌ member-agnostic
```
该梯度只依赖 `Fbar, Yb, wb`，**与成员 m 及 `Fm` 无关** → 每步把所有成员推向同一方向 → 相对差异仅由 `mse_grad` 决定（λ 无关）→ 三档 λ byte-identical。这与 P5-B `sample_weight` 破代理同源（无法注入负相关）。

**正确 NCL 交叉项梯度**（对 `P_m = λ·Σ_i w_i·(f_m−f̄)(f̄−y)` 关于 `f_m` 求导，含 member-specific 的 `(f_m−f̄)` 因子）：
```python
M = len(members)
ncl_factor = (1.0 - 1.0/M) * (Fbar - Yb) + (1.0/M) * (Fm - Fbar)
ncl_grad   = (self.ncl_lambda / len(Xb)) * (wb[:, None] * ncl_factor)
```
`(1/M)(f_m−f̄)` 项使每个成员的梯度**互异** → 强制负相关。

---

## 2. 子集诊断（小子集 3000 行；已确证梯度修复，但揭示其局限性）

为隔离「稀疏度 w」与「λ 量级」，构造三场景（60 epoch，3000 行）：

| 场景 | w 权重 | λ 扫描 | std_tail(训练点跨成员) | corr(m0,m1) | 判读 |
|---|---|---|---|---|---|
| A 默认稀疏 w | kNN 归一化 | 0.1→1.0 | 0.0031→0.0031 | −0.264→−0.264 | λ 零效应（w≈0 静音） |
| **B uniform w=1** | 全 1 | 0.1→1.0 | 0.0035→**0.0118** | −0.261→**−0.166** | **λ 强改变多样性 → 梯度修复生效** |
| C 默认稀疏 w | kNN 归一化 | 1.0→10 | 0.0031→0.0032 | −0.264→−0.264 | 10×λ 也撬不动 |

**子集诊断结论**：① 梯度修复有效（场景 B：λ 改变多样性 3.4×，成员相关松弛 −0.26→−0.17）；② 默认稀疏 w 在密集主体 ≈0 静音 NCL（场景 A/C）。→ 早期据此**误判「w 权重误指定」为瓶颈**（见 §3 修正）。

**子集诊断的致命局限（事后揭示）**：3000 行 / 60 epoch 训练**欠收敛**，`(f_m−f̄)` 尚未归零，故 NCL 信号尚存。完整训练体制下此瞬态消失（见 §3）。

---

## 3. 完整跑批结果（79k 行 / 120 epoch；决定性）

两跑批：`MPTlLA→QjJBCx`(sparse-w 重跑) 与 `zARRRZ`(uniform-w)，各 ~15min。`GATE_REPRO=True`（臂 A=P1 逐位复现）。**两跑批 verdict 均为 `P5B_NCL_ACCEPTED=false`**。

### 3.1 sparse-w（默认 kNN 归一化稀疏度）

| 子集 | 臂A(P1) ECE | 臂R λ=0.1 ECE | 臂R λ=1.0 ECE | σ_epi比(R/P1) | 欠缩放_med(R/P1) | 判读 |
|---|---|---|---|---|---|---|
| id | 0.6947 | 0.7782 | 0.7782 | 0.093 | 98.3 / 8.8 | λ 零效应，更差 |
| ood_agent | 0.7082 | 0.7729 | 0.7729 | 0.157 | 62.6 / 10.4 | λ 零效应 |
| **ood_action** | **0.6112** | **0.7754** | **0.7754** | **0.064** | 72.1 / 4.6 | **恶化，判据失败** |
| ood_neuro | 0.6981 | 0.7786 | 0.7786 | 0.092 | 104.2 / 9.2 | λ 零效应 |

三档 λ(0.1/0.3/1.0) **byte-identical** → NCL 信号被静音（w≈0 于密集主体）。

### 3.2 uniform-w（强制 w=1，机制探针上限）

| 子集 | 臂R λ=0.1 | 臂R λ=0.3 | 臂R λ=1.0 | σ_epi比(λ1.0) | 欠缩放_med(λ1.0) |
|---|---|---|---|---|---|
| id | ECE 0.7782 | 0.7783 | 0.7786 | 0.088 | 103.4 / 8.8 |
| ood_agent | 0.7729 | 0.7729 | 0.7729 | 0.158 | 62.3 / 10.4 |
| **ood_action** | **0.7755** | **0.7755** | **0.7759** | **0.061** | 75.3 / 4.6 |
| ood_neuro | 0.7786 | 0.7787 | 0.7790 | 0.087 | 109.6 / 9.2 |

**关键**：即便 uniform w=1（最大 NCL 信号）且 λ 提到 1.0，ood_action ECE **仍 0.775+**（目标 <0.08），σ_epi比 **仍 ≈0.06（比 P1 更小！）**，欠缩放 **72–109×**（比 P1 更差）。λ 仅 4 位小数扰动，且 λ=1.0 方向**反了**（σ 更小、欠缩放更重：72→75）。

### 3.3 修正后的根因（共识塌缩，非 w 权重）

完整训练体制下，sparse-w 与 uniform-w **同样失败且 λ 近似无效** → 证伪「w 权重误指定」假设。真瓶颈：

> **MSE 共识塌缩**。NCL 去相关项 ∝ (f_m − f̄)；完整训练驱动所有成员收敛到同一解 → (f_m − f̄) → 0 → NCL 梯度归零。NCL 无法从已塌缩的集成中「自举」多样性；它只能在成员已有分歧时放大分歧，而 MSE 在充分训练下消灭分歧。

这解释了为何子集诊断（欠收敛）显示 λ 效应而完整跑批（收敛）不显示。P5-B 的两种实例化（sample_weight、NCL）失败同源：**同架构 MLP + 同数据 + MSE → 共识塌缩 → 训练期多样性注入无法存活**。

---

## 4. 判决

```
sparse-w  : ood_action ECE 0.7754 (目标<0.08)  cov@0.95 饱和  σ_epi比=0.064  λ 零效应
uniform-w : ood_action ECE 0.7759 (目标<0.08)  cov@0.95 饱和  σ_epi比=0.061  λ 仅4位扰动(方向反)
GATE_REPRO=True  P5B_NCL_ACCEPTED=false  best_lambda=null
```

→ **P5-B·proper NCL spike 阴性（有效）**。梯度实现正确（已修复并验证），但 NCL 机制在完整训练体制下被共识塌缩淹没，无法把多样性转移到测试 epistemic σ。P5-B 概念（密度感知多样性注入）**穷尽**——两种实例化均失败，根因为结构性共识塌缩，非调参可解。

---

## 5. 对 P5 路线的影响与待决策（fork）

- **P5-B 已穷尽**（spike 1 sample_weight 阴性 + spike 2 NCL 阴性）。**三机制 A/B/C 尚未全失败**：P5-A（proper-Bayesian/SGLD）、P5-C（生成式/GMM 密度门控）均**未跑**。当前**不应**触发 kill-switch（需 A/B/C 三机制全失败）。
- **P5-C 预测撞墙**：GMM 密度 q(x) 是**测试时**协变量（对局部密度的平滑估计，与 kNN novelty 同源）；路 A 已证「任何测试时协变量都无法预测欠缩放」（novelty 常量、σ_epi 弱负相关）。→ P5-C 大概率阴性。
- **P5-A 是剩余最高希望但最重**：SGLD/温度化似然的 posterior 多样性是**内禀**的（不依赖 (f_m−f̄) 自举），有可能真正让 σ 在 OOD 转移。但需 custom SGLD 训练循环（比 NCL 更重，且本环境无 torch，需 numpy-only SGLD）。
- **连带工具链打通**：本 spike 已用 numpy-only custom 训练循环绕过 sklearn MLP 限制——P5-A 可复用此框架（SGLD 噪声注入同框架）。

**建议（待用户决策，serotonin 停止信号未触发）**：
1. **投资 P5-A（proper-Bayesian/SGLD）**——剩余唯一有原理希望的机制（内禀 posterior 多样性）；或
2. **试 P5-C（GMM 密度门控）**——便宜但大概率撞路 A 同墙（快速排除）；或
3. **触发 kill-switch**：若用户判定 P5-A 成本过高且 P5-C 预期阴性 → 停 P5、接受 R4 闸门（OOD 区间标「不可信」），把 P5-B 双阴性写成可发表级发现（"ID-only MSE 训练下集成共识塌缩使 OOD epistemic 校准结构性不可行"）。

---

## 6. 工件清单

| 工件 | 用途 |
|---|---|
| `code/model/compositional_p5b_ncl.py` | P5-B proper-NCL 模型（**修复后** member-specific 梯度） |
| `scripts/p5b_ncl.py` | 两臂受控对照 + λ 扫描 + `--w-mode {sparse,uniform}` 机制探针 |
| `scripts/_smoke_ncl_fix.py` | 梯度修复冒烟测试（子集确认 λ 改变多样性） |
| `scripts/_diag_ncl.py` | 三场景机制诊断（隔离 w 与 λ；揭示欠收敛局限） |
| `experiments/20260814-p5b-ncl-sparse.json` | sparse-w 跑批工件（GATE_REPRO=True, P5B_NCL_ACCEPTED=false） |
| `experiments/20260814-p5b-ncl-uniform.json` | uniform-w 机制探针工件（同上） |
| `experiments/w36_p5b_ncl_sparse.log` / `w36_p5b_ncl_uniform.log` | 运行日志 |
