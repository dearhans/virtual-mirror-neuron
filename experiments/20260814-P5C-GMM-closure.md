# P5-C·GMM 密度门控结案 · 生成式 epistemic（PC-3 结构性估计器方向，spike 3/3 之 C）

- **状态**：`P5C_GMM_ACCEPTED = false`（严格判据未满足：ood_action 覆盖@0.95=0.9989 越带上界 0.97、ECE=0.2021≫0.08）。门控在真实数据上**退化为「ID↔OOD」二值开关**——GMM 密度在 OOD 协变量上塌缩到 ~0，`density_novelty u` 对 ood_action/ood_agent 全部饱和到 1.4×10⁹ → 统一封顶 50× → conformal 重标定把粗膨胀压平成全域过宽区间 → 撞路 A 同墙。
- **核心发现（阴性，且机制比路 A 更干净）**：P5-C 与路 A 同族（测试时协变量密度门控 σ），但用 sklearn `GaussianMixture` 的**平滑参数化密度** q(x) 取代 kNN 距离 nd(x)。诚实预测（W33 §7 P0）「撞路 A 同墙」**被确认**——且证据更尖锐：**GMM 密度在高维稀疏协变量空间是二值新颖度检测器**（训练壳内 q≈训练中位，壳外 q≈0），没有 路 A 那种分级新颖度；门控因此无分级信号可学（ID 校准箱内 `gate_ratio` 全 1.0），唯一动作是 OOD 处封顶外推（统一 50×）。
- **`GATE_REPRO=True`**：臂 A(P1) 逐位复现当周基准（rmse 四子集 delta 全 0.0）。P5-C 点估计与 P1 逐位相同（标准 MLPRegressor 集成，同监督口径 → σ 合成改造的增量可干净归因）。
- **三机制全部阴性 → kill-switch 正当**：P5-A·SGLD（精炼阴性，ECE 下限 0.094 + 点估计代价）、P5-B·NCL（双阴性，共识塌缩）、P5-C·GMM（本文件，密度门控退化）。协议要求三机制全失败才停 P5 → **触发 kill-switch，接受 R4**（OOD 区间标不可信，仅点估计参与机制泛化判决）。
- **工件**：`code/model/compositional_p5c_gmm.py`、`scripts/p5c_gmm.py`、`scripts/_smoke_p5c.py`、`experiments/20260814-p5c-gmm.json`、`experiments/w36_p5c_gmm.log`

---

## 0. 假设与事前锁定判据（W33 §3.10）

**待检假设（用户选定 P5-C · GMM 门控）**：P5-C 是 P5 三机制面板最后一格，与路 A（kNN 新颖度门控）同族——在测试时按「输入协变量密度」撑大 epistemic σ，以攻克 PC-3（σ 应关于到训练流形距离 τ(x) 单调，现实 OOD 区 ∂σ/∂τ≈0 欠缩放 ~41×）。差别：用 `GaussianMixture` 拟合训练特征密度 q(x)（平滑参数化），以密度新颖度 `u(x)=log q_med − log q(x)` 取代 kNN 距离，直觉上在离散扰动空间外推比 kNN 优雅。

**诚实预测（事前，W33 §7 P0）**：PC-3 根因是 epistemic 欠缩放是**深度集成 posterior 的性质**，不是输入协变量密度的函数；路 A 已证「测试时协变量密度无法预测欠缩放，conformal 重标定压平门控」。P5-C 大概率撞同墙。协议要求三机制全测方停 P5，故以廉价（无 torch/PyMC 工具链升级）补完面板。

**判据（与路 A / P5-B / P5-A 同，事前锁定，公平比较）**：ood_action 上 P5-C 的 覆盖@0.95 ∈ [0.93,0.97] 且 ECE < 0.08；id/ood_agent/ood_neuro 相对 P1 不退化（覆盖@0.95 不跌破 0.85 且 ECE 不爆 0.25、不 2× 基线）；`GATE_REPRO=True`。

---

## 1. 实现

`code/model/compositional_p5c_gmm.py`：`CompositionalTwinP5C_GMM(CompositionalTwin)`——

- `fit`：调用 `super().fit`（**标准 MLPRegressor 集成**，与 P1 同监督口径 → 点估计增量可干净归因到 σ 合成改造；无 SGLD、无 NCL），然后额外在 `train_feats_=hstack[pa,pb,act,g]` 上拟合 `GaussianMixture(n_components=8, cov_type='diag')` 训练流形密度 q(x)；特征先列 z-score 标准化（19→217 维异质尺度，diag 协方差防奇异）。
- `fit_density_gate(X_cal, y_cal)`：与路 A `fit_novelty_gate` 同辙——在 ID 校准集上估欠缩放比 `r(u)=median|resid|/σ_epi` 作为密度新颖度 u 的函数，强制单调非减（越 OOD 越欠缩放），参考点 `u_ref=0`（训练密度中位）→ 倍率≈1；OOD 按 `cap=50` 外推防饱和（PC-3 宽↔窄矛盾）。
- `predict`：σ_pred = σ_epi · g(u)，u 由 GMM `score_samples` 在测试点即时算出（ID/OOD 测试点皆可，不依赖 ID 校准覆盖 OOD——与路 A 同前提）。

**冒烟先验证机制可行**（合成数据，`scripts/_smoke_p5c.py`）：GMM 拟合成功、门控倍率单调非减、密度新颖度 u 在 OOD（特征整体平移）显著 > ID、门控倍率随之 >1、σ 在 OOD 被撑大 → 机制与 API 正确。

---

## 2. 完整跑批结果（99k×217；task `shZV1L`，~2min）

`GATE_REPRO=True`（P1 臂逐位复现）。**verdict `P5C_GMM_ACCEPTED=false`**。

### 2.1 ood_action 判据子集（事前锁定）

| 指标 | 臂A(P1) | 臂R(P5-C GMM) | 判读 |
|---|---|---|---|
| ood_action cov@0.95 | 0.9994 | **0.9989** | ✗ 越带上界 0.97（过覆盖） |
| ood_action ECE | 0.2060 | **0.2021** | ✗ ≫0.08 |
| 覆盖@0.5 / 0.8 / 0.9 | 0.967 / 0.994 / 0.998 | 0.968 / 0.995 / 0.998 | 全域过宽（每级名义↔实测都偏宽） |
| RMSE | 0.6351 | 0.6351 | 点估计与 P1 逐位相同（σ 合成改造零增量） |
| density_novelty u_med | — | **1.41×10⁹** | OOD 密度塌缩，全部封顶 |
| gate_mult_med | — | 50.0（封顶） | 门控退化为统一 50× |

→ 覆盖不降反微升 0.9994→0.9989（仍过覆盖），ECE 仅从 0.2060→0.2021（改善 0.004，远未达 0.08）。门控把 σ 统一撑到 50× 但**未能修出 [0.93,0.97] 带 + 低 ECE**。

### 2.2 全子集（四子集，臂 A vs 臂 R）

| 子集 | n | P1 ECE | P5C ECE | P1 cov@.95 | P5C cov@.95 | P1 RMSE | P5C RMSE | gate_mult_med | u_med |
|---|---|---|---|---|---|---|---|---|---|
| id | 11953 | 0.1748 | 0.1742 | 0.9936 | 0.9932 | 0.5674 | 0.5674 | **1.0** | 0.0 |
| ood_agent | 11748 | 0.1357 | 0.1428 | 0.9717 | 0.9683 | 0.5854 | 0.5854 | **50.0** | 1.41×10⁹ |
| **ood_action** | **14212** | **0.2060** | **0.2021** | **0.9994** | **0.9989** | 0.6351 | 0.6351 | **50.0** | 1.41×10⁹ |
| ood_neuro | 5912 | 0.1697 | 0.1691 | 0.9924 | 0.9919 | 0.6031 | 0.6031 | **1.0** | 0.0 |

→ **门控是二值开关**：id / ood_neuro（训练壳内密度）gate=1.0，ood_agent / ood_action（壳外）gate=50.0。无中间档。

### 2.3 欠缩放诊断（门控前原始集成 epistemic，median|resid|/σ_epi）

| 子集 | median|r|/σ_epi | p95|r|/σ_epi | 判读 |
|---|---|---|---|
| id | 9.62 | 43.38 | 欠缩放 ~10× |
| ood_agent | 18.76 | 107.73 | 欠缩放 ~19× |
| ood_action | 8.23 | 36.79 | 欠缩放 ~8× |
| ood_neuro | 10.08 | 46.09 | 欠缩放 ~10× |

→ 欠缩放在 OOD 仍 ~8–19×（全局 ~41× 的逐子集版本），**未被门控修正**：门控把 σ 撑 50× 后，conformal 重标定 q（41.26→43.68，仅 +6%）把区间整体重缩 → 残差/σ 回到校准口径 → ECE 不变。

### 2.4 GMM 门控拟合结果（关键证据）

```
gmm_ok=True  n_components=8  cov_type='diag'  standardize=True  logq_med=362.75
gate_u_centers = [-869.1, -22.9, -13.4, -8.0, 7.4, 22.3, 52.1, 117.3]   (ID 校准箱内 u 范围)
gate_ratio     = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]               (ID 校准内欠缩放与密度无关！)
r_ref=9.157  cap=50.0
```

→ **ID 校准箱内 `gate_ratio` 全 1.0**：欠缩放比在 ID 密度范围内是**平坦的**（不随密度变化）。门控在 ID 内**无分级信号可学**，唯一动作是 OOD 处（u>117 校准箱上界）按 cap 外推——而 OOD 点 u≈1.4×10⁹ 远超 117 → 全部统一封顶 50×。这正是「二值开关」的数值根因。

---

## 3. 根因与机制判读（为何 P5-C 撞路 A 同墙）

- **墙在机制层，不在平滑度**：路 A（kNN 距离）的 u 有分级范围，但其欠缩放比作为 u 的函数也不转移到 OOD（conformal 压平）。P5-C（GMM 密度）在 217 维稀疏协变量上更极端——GMM 密度在训练壳外**塌缩到 ~0**（8 个对角成分极尖），u 对 OOD 点饱和到 1.4×10⁹ → 门控连分级都丢了，直接二值。两者同阴性**证实墙是 epistemic 欠缩放的本质**：它是深度集成 posterior 的性质（OOD 上 confidently-wrong），**不是输入协变量密度 q(x) 的函数**——任何「按测试时密度撑 σ」的门控都只能做 in/out 检测，无法预测 posterior 的真实欠缩放，conformal 重标定必然压平。
- **conformal 为何压平**：门控把 OOD σ 撑 50×，但 OOD 残差/σ_epi 仅 ~8–19×；若 conformal 用 OOD 残差定 q，区间应入带。但 conformal `q` 在 **ID 校准集**上定（qR=43.68 vs P1 41.26，仅 +6%），对「统一 50× 膨胀」整体重缩 → 每级名义区间都过宽（0.5 级实测 0.968）→ 过覆盖 + ECE 0.20。门控的膨胀被校准层完全吸收，对 OOD 校准零贡献。
- **P5-C 比路 A 信息量更大的一点**：它表明即便换「平滑参数化密度」，高维稀疏密度估计也只能给二值新颖度——**问题不是 kNN 太粗糙或 GMM 太生硬，是「用协变量密度门控 epistemic」这条路本身在 217 维不可行**。这把路 A 的阴性从「门控回归量选择」升级为「回归量类选择」的判定。

---

## 4. 判决

```
ood_action: cov@0.95=0.9989(✗ 越带 0.97, 过覆盖)  ECE=0.2021(✗<0.08)
            RMSE=0.6351(=P1, 点估计零增量)  gate_mult=50.0(封顶)  u=1.41e9(饱和)
GATE_REPRO=True  P5C_GMM_ACCEPTED=false
gate_ratio(校准内)全 1.0 → 门控在 ID 无信号；OOD 统一封顶 → conformal 压平 → 撞路 A 同墙
```

→ **P5-C·GMM spike：阴性**（门控退化 + 撞墙，非「近胜」）。P5 三机制全部阴性。

---

## 5. 对 P5 路线的影响与 kill-switch 决策

- **P5 三机制状态（全部阴性）**：
  - **P5-A·SGLD**：覆盖结构性可修（PC-3 首处进展）但点估计代价 2.4–6× + ECE 下限 0.094（精炼阴性）。
  - **P5-B·NCL**：双阴性（sample_weight + NCL 共识塌缩，结构性）。
  - **P5-C·GMM**：密度门控退化二值开关，撞路 A 同墙（本文件，阴性）。
- **kill-switch 正当（协议触发）**：协议要求三机制全失败才停 P5。现 P5A_ACCEPTED/P5B_ACCEPTED/P5C_ACCEPTED 全 false → **触发 kill-switch，接受 R4**：OOD 区间标不可信，仅点估计（rmse / var_ratio）参与机制泛化判决（详见 W33 §4.3 R4 闸门）。
- **科学收益（诚实保留）**：P5-A 证明「覆盖可由 proper-Bayesian 后验多样性结构性修复」是真实方向（非共识塌缩类死路）；P5-C 把路 A 的阴性从「门控回归量」升级为「回归量类」判定（协变量密度门控 epistemic 在 217 维不可行）。这收窄了未来正确方向：**换 proper-Bayesian 重实现**（torch/PyMC：预条件 SGLD/HMC、burn-in+thinning、更多链）——属真正的工具链升级，超当前「numpy-only 无 torch」约束，触发冲动门控，须用户明确授权。
- **结论**：当前估计器家族（含三机制所有 spike）**无法产出可信 OOD 不确定度**；R4 接受后，本周报 OOD 区间标「不可信 / 不标」，机制泛化判决仅基于点估计 + OOD 子集 RMSE 不退化。proper-Bayesian 重实现作为独立 future-work（工具链升级）保留，不在本周内推进。

---

## 6. 工件清单

| 工件 | 用途 |
|---|---|
| `code/model/compositional_p5c_gmm.py` | P5-C·GMM 密度门控模型（标准集成 + GaussianMixture 密度门控） |
| `scripts/p5c_gmm.py` | 两臂受控对照 + GMM 门控 + 覆盖/ECE/欠缩放/密度新颖度诊断 |
| `scripts/_smoke_p5c.py` | 冒烟测试（合成数据验证 GMM 密度门控机制不退化） |
| `experiments/20260814-p5c-gmm.json` | 完整跑批工件（GATE_REPRO=True, P5C_GMM_ACCEPTED=false） |
| `experiments/w36_p5c_gmm.log` | 运行日志（2min） |
