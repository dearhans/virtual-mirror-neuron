# 虚拟镜像神经元 · 周报 2026-W33

- **报告版本**：2026-08-14 刷新（取代 2026-08-10 首版 / 2026-08-12 二版 / 2026-08-14 三、四版；同 ISO 周第 5 次导出，新增 P5 立项）
- **种子基准**：`experiments/20260810-benchmark.json`（真实 Norman 2019，`data.source=norman`）
- **样本规模**：train 55464 / ID-test 11953 / total 99289；bootstrap = 100
- **本周新证据槽**：P4-3 逆方差收缩阴性（关闭「改估计量」方向）＋ P4-4 φ 双通路弱达成（PC-1 结构路径耗尽）＋ P4-1 localized 共形阴性（PC-3 重定位为 epistemic 缺陷）＋ P3-1b 共形修复 virtual_twin 过度自信（W36 自动化 A）＋ **路 A novelty/heteroscedastic 门控阴性（PC-3 原理性未解，三路证据闭合）**＋ **P5 结构性估计器新立项提案（PC-3 唯一剩余解法方向，Einstein+TRIZ+serotonin 三框架塑形）**
- **配套湿实验方案**：`wetlab/2026-W33.md`（7 协议 E1–E7 + §6 人工附录，25826 字节）
- **状态**：未外发、未发布、未向 GOAI 提交（已推送到 origin/master，路 A commit 4b188ec；P5 提案随本次刷新并入，见 `experiments/20260814-P5-structural-estimator-proposal.md`）

---

## 0. 本周口径声明（先说没做什么）

1. **主基准零建模增量，连续第 4 次冻结。** `20260810-benchmark.json` 的 `results` / `calibration` / `flags`
   三段 sha256 与 `20260809-benchmark.json` **逐位相同**（`20260810-regress-check.json`:
   `results_frozen=true`、`calibration_frozen=true`、`flags_frozen=true`、`n_results_diff=0`）。
   → 本周主基准属**确定性复现验证**，不是建模进展。任何「模型变好了」的表述都不成立。

2. **上周（08-10 首版）对 P4-3 零引用是正确的**：当时 `code/model/compositional_p43.py` 在跑、
   `w35_benchmark_run.log` 为空。本周工件已落盘（`20260810-p43-shrinkage.json`，30760 字节，
   `w35_p43_run.log` 3598 字节，`GATE_REPRO=True`），**现在才允许入账**。

3. **P4-3 结论是阴性。** 本周唯一实质新增证据是一条「路线关闭」的负结果 + 一条机制诊断。
   不包装成进展，但它确实**收窄了 PC-1 的破法空间**（见 §6）。

4. **两个校准缺口本周状态更新**：
   - `virtual_twin` 全域过度自信（P3-1b）**已修复**（W36 自动化 A 跑通共形路径）：ECE 0.71→0.14、覆盖@0.9 0.07→0.98，q=30.2（std 欠缩放 ~30×）。修复只改区间可信度、不改点估计。
   - `ood_action` 区间饱和（P4-1）**已跑且阴性**：分 subset localized 共形退化回全局 q（41.26），证明 PC-3 是 epistemic 估计器缺陷、非校准层可修（详见 §3.8）。

---

## 1. 双栏总表 · 记忆（in-distribution） vs 机制泛化（OOD）

RMSE（bootstrap 均值）与 95% CI。⚠️ 注意字段口径：`results[*].rmse` 是 **bootstrap 均值**，非点估计。

### 左栏 · 记忆（ID，已见基因 n=11953）

| 排名 | 预测器 | RMSE [95% CI] | R² | ECE | 覆盖@0.9 |
|---|---|---|---|---|---|
| 1 | `linear` | **0.5664** [0.5608, 0.5712] | 0.7696 | 0.0318 | 0.915 |
| 2 | `virtual_twin` | 0.5671 [0.5615, 0.5717] | 0.7691 | **0.7216** | **0.078** |
| 3 | `compositional_twin` | 0.5673 [0.5617, 0.5717] | 0.7690 | 0.1393 | 0.972 |
| 3 | `compositional_interaction_twin` | 0.5673 [0.5617, 0.5717] | 0.7690 | 0.0667 | 0.925 |
| 5 | `mlp` | 0.5685 [0.5631, 0.5731] | 0.7679 | 0.0281 | 0.915 |
| 6 | `mean` | 0.5994 [0.5943, 0.6051] | 0.7417 | 0.1662 | 0.993 |
| 7 | `knn` | 0.6283 [0.6242, 0.6335] | 0.7165 | 0.1573 | 0.992 |

前 5 名 CI 全部重叠 → **ID 上没有任何模型显著优于线性**。

### 右栏 · 机制泛化（OOD）

**ood_agent（未见基因，n=11748）**

| 排名 | 预测器 | RMSE [95% CI] | ECE | 覆盖@0.9 |
|---|---|---|---|---|
| 1 | `linear` | **0.5815** [0.5778, 0.5863] | 0.0243 | 0.909 |
| 2 | `compositional_twin` | 0.5852 [0.5814, 0.5898] | 0.0887 | 0.929 |
| 2 | `compositional_interaction_twin` | 0.5852 [0.5814, 0.5898] | 0.0479 | 0.865 |
| 4 | `mean` | 0.5872 [0.5838, 0.5918] | 0.1686 | 0.993 |
| 5 | `mlp` | 0.6376 [0.6337, 0.6416] | 0.0270 | 0.879 |
| 6 | `knn` | 0.6483 [0.6451, 0.6520] | 0.1611 | 0.995 |
| 7 | `virtual_twin` | 0.6751 [0.6706, 0.6795] | **0.7332** | **0.065** |

**ood_action（组合双扰动 held-out 对，n=14212）**

| 排名 | 预测器 | RMSE [95% CI] | ECE | 覆盖@0.9 | flags |
|---|---|---|---|---|---|
| 1 | `linear` | **0.6266** [0.6218, 0.6310] | 0.0134 | 0.887 | — |
| 2 | `compositional_interaction_twin` | 0.6319 [0.6275, 0.6361] | **0.2125** | **1.000** | 疑似仅记忆 (Δ −0.0053) |
| 3 | `compositional_twin` | 0.6355 [0.6301, 0.6400] | 0.1965 | 0.997 | 疑似仅记忆 (Δ −0.0089) |
| 4 | `mlp` | 0.6361 [0.6317, 0.6396] | 0.0164 | 0.882 | — |
| 5 | `virtual_twin` | 0.6514 [0.6475, 0.6550] | 0.7055 | 0.097 | — |
| 6 | `mean` | 0.6577 [0.6528, 0.6617] | 0.1536 | 0.988 | — |
| 7 | `knn` | 0.6975 [0.6925, 0.7026] | 0.1424 | 0.985 | — |

**ood_neuro（未见批次，n=5912）**

| 排名 | 预测器 | RMSE [95% CI] | ECE | 覆盖@0.9 |
|---|---|---|---|---|
| 1 | `mlp` | **0.5992** [0.5920, 0.6061] | 0.0167 | 0.899 |
| 2 | `virtual_twin` | 0.6002 [0.5928, 0.6073] | 0.7237 | 0.075 |
| 3 | `compositional_twin` | 0.6026 [0.5951, 0.6097] | 0.1308 | 0.967 |
| 3 | `compositional_interaction_twin` | 0.6026 [0.5951, 0.6097] | 0.0574 | 0.914 |
| 5 | `linear` | 0.6028 [0.5953, 0.6101] | 0.0201 | 0.896 |
| 6 | `mean` | 0.6243 [0.6154, 0.6315] | 0.1607 | 0.991 |
| 7 | `knn` | 0.6548 [0.6463, 0.6614] | 0.1508 | 0.990 |

前 5 名 CI 全部重叠 → **ood_neuro 轴当前无判别力**。

### 双栏差额 = 机制证据？

| 子集 | 最优结构化孪生 | `linear` | 差额 | 判定 |
|---|---|---|---|---|
| ID（记忆） | 0.5673 | 0.5664 | **−0.0009**（孪生更差） | 无机制红利 |
| ood_agent | 0.5852 | 0.5815 | **−0.0037**（孪生更差） | 无机制红利 |
| ood_action | 0.6319 | 0.6266 | **−0.0053**（孪生更差） | 无机制红利 + flags 标记 |
| ood_neuro | 0.6026 | 0.6028 | +0.0002（CI 重叠） | 不显著 |

**结论（与 W31–W32 一致，未被本周证据推翻）**：结构化组合先验在**四个子集上全部未取得机制红利**。
差额一栏全为负或不显著 → 「机制泛化」目前没有点估计层面的证据支持。

---

## 2. 判别性指标复核：判据本身可靠吗？

`addendum_metric_discriminability`（核心 4 器 RMSE 点估计跨度 ÷ 平均 CI 半宽）：

| 子集 | 核心跨度 | 平均 CI 半宽 | **分辨率比值** | 可判据性 |
|---|---|---|---|---|
| id | 0.03304 | 0.005039 | **6.56** | 可判 |
| ood_agent | 0.005751 | 0.004080 | **1.41** | ⚠️ **近乎无分辨率** |
| ood_action | 0.03113 | 0.004440 | **7.01** | 可判 |
| ood_neuro | 0.02174 | 0.007408 | **2.93** | 边缘可判 |

**这条直接限定了本周的核心判决。** P4-3 的预锁判据是「ood_agent RMSE ≤ 0.5854」——
落在一个分辨率仅 1.41 的子集上，且实测差额是小数第 3 位（0.5881 vs 0.5854）。
**单凭这个指标不足以判否任何路线。** 本周 P4-3 结论之所以仍然成立，靠的是**独立于该指标**的收缩诊断（§3）。
这个论证结构必须显式写出，否则就是用钝刀冒充判决。

---

## 3. 本周新证据槽 · P4-3 经验贝叶斯逆方差收缩（阴性，路线关闭）

### 3.1 假设与预锁判据

**待检假设**（W32 提出）：`ê_u` 插补时，各基因的共扰动条数差 **47.9×**（62 ~ 2971 条），
等权平均会被低证据基因的抽样噪声拉偏 → 用经验贝叶斯逆方差收缩加权应能提升精度。

**预锁判据**（跑前冻结）：ood_agent 上 `RMSE ≤ 0.5854` **且** `var_ratio > 0.10`。

### 3.2 三臂对照结果（`GATE_REPRO=True`，conformal q: A=30.05 / B=27.00 / C=27.02）

| 子集 | armA · P1 基线孪生 | armB · P4-2 等权插补 | armC · P4-3 收缩 | `linear` |
|---|---|---|---|---|
| **id** RMSE | 0.5674 [0.5617,0.5717] | 0.5674 | 0.5674 | 0.5665 |
| id pbRMSE / var_ratio | 0.0797 / 0.863 | 0.0797 / 0.863 | 0.0797 / 0.863 | 0.0696 / 0.925 |
| **ood_agent** RMSE | **0.5854** [0.5814,0.5898] | 0.5887 [0.5844,0.5930] | 0.5881 [0.5839,0.5924] | **0.5817** [0.5778,0.5863] |
| ood_agent pbRMSE | 0.1927 | 0.1965 | 0.1952 | **0.1770** |
| ood_agent var_ratio | **1.07e-26**（坍缩） | 0.5031 | 0.4771 | 0.2151 |
| ood_agent ECE | 0.0887 | 0.1490 | 0.1487 | — |
| **ood_action** RMSE | 0.6351 | 0.6297 | 0.6297 | **0.6265** |
| ood_action pbRMSE | 0.2456 | 0.2132 | 0.2131 | **0.1905** |
| **ood_neuro** RMSE | 0.6031 | 0.6031 | 0.6031 | 0.6033 |

### 3.3 判决

```
pass_rmse_le_0.5854   = false      (armC 0.5881 > 0.5854)
pass_var_ratio_gt_0.10 = true      (armC 0.4771)
P43_ACCEPTED           = false
armC_vs_armB_ci_overlap = true
Δrmse(C−B)   = −0.00056
ΔpbRMSE(C−B) = −0.00127
```

→ **P4-3 未通过。** armC 相对 armB 的改善方向正确但量级为小数第 4 位，CI 完全重叠，非显著。

### 3.4 机制诊断（本周最有信息量的部分，且独立于钝指标）

| 诊断量 | 实测值 | 含义 |
|---|---|---|
| `n_genes_shrunk` | 15 | 参与收缩的基因数（与 P4-2 的「15 可插补」一致） |
| 共扰动条数 min / median / max | 62 / 292 / 2971（**47.9×**） | 证据量确实极不均衡（假设的前提成立） |
| `τ²`（基因间效应异质性） | **0.02820** | 真实信号方差 |
| `SE²`（抽样误差） | **0.001676** | 估计噪声方差 |
| **τ² / SE²** | **16.83** | **噪声不是主导项** |
| 隐含收缩系数 λ = τ²/(τ²+SE²) | **0.944** | 实测 mean 0.931 / median 0.941 / min 0.760 / max 0.994 |
| 相对等权平均的位移 | 0.006993 / 0.15986 = **4.37%** | 收缩几乎不改变估计 |
| 插补标准误 | 0.03548 → 0.03347（**−5.67%**） | 精度提升微弱 |

**根因**：λ ≈ 0.93–0.94 意味着收缩权重几乎全部压在「各基因自身均值」上，
**逆方差收缩在此处近似恒等映射**——因为基因间真实效应异质性比抽样噪声大 **16.8×**。
47.9× 的条数差距虽然真实存在，但它带来的方差差异被 τ² 淹没了。

→ **误差主体是加法先验的系统性偏差（bias），不是估计方差（variance）。**
这条诊断**不依赖** ood_agent 的 RMSE 分辨率，因此可以独立支撑「路线关闭」的结论。

### 3.5 一个新观察：坍缩状态下 ECE 具有欺骗性

armA 在 ood_agent 上 `var_ratio = 1.07e-26`（完全坍缩为常数预测），但它的 **ECE = 0.0887 是三臂中最好的**；
修好坍缩后（armB/armC）ECE 反而升到 **0.1490 / 0.1487**。

原因：常数预测 + 宽 conformal 区间极易「覆盖住」真值，ECE 看起来很漂亮。
→ **推论**：`ECE 好` 既不能证明模型好，也不能证明区间有用。ECE 必须与 `var_ratio`（判别力）联合读。
这与 §3 已有原则「坍缩 ≠ 精度瓶颈」互补，构成第二条：**坍缩 ≠ 校准缺陷**（它甚至伪装成校准优势）。
此外 armB/armC 的 0.149 已贴近本周新设 R4 闸门上限 0.15 —— 修坍缩的代价里包含区间质量下降。

### 3.6 结论

`P4-3 逆方差收缩路线正式关闭。` 不再投入加权/插补精化方向的变体。
优先级全部转移至 **P4-4（φ 拆「线性头 + 机制残差」双通路，TRIZ 空间分离）**——
因为诊断指向的是 bias，而 bias 只能靠改结构消除，不能靠改估计量消除。

### 3.7 本周新增证据槽 · P4-4 φ 双通路（弱达成，PC-1 结构路径耗尽）

**待检假设**：把加法基 φ 拆为「线性头（单扰动行拟合）＋ 机制残差头（MLP 集成补线性解释不了的非线性）」，held-out 对靠加法组合外推，组合孪生在 `ood_action` 首次追平线性。

**四臂对照（ood_action，bootstrap=100）**：臂 A · P1 孪生 0.6351 [0.6301,0.6400]；臂 D · P4-4 0.6347 [0.6298,0.6397]；`linear` 0.6265 [0.6218,0.6310]；mean 0.6574（坍缩）。

**判决**：`ci_overlap_with_linear = true`（重叠仅 [0.6298,0.6310] 窄带）、`point_better_than_linear = false`、`p44_vs_p1_twin_delta = −0.00046` → `P44_ACCEPTED = true` 但**弱达成**（CI 重叠分支，非点估计更优；线性点估计仍赢 0.009）。GATE_REPRO=True，臂 A 与 canonical 逐位复现。

**PC-1 结构路径耗尽（核心）**：为解 PC-1 已穷尽 4 类结构改造——P3-1（σ 解耦）、P4-2（插补）、P4-3（收缩）、P4-4（双通路）——**全部「保判别力（var_ratio）不补精度/OOD」，无一在 ood_action 追平线性**。差距 ~0.009 是**监督信号边界**（线性全训练行含共扰动，组合孪生线性头仅单扰动行），非容量边界。→ 停止在 φ 结构上继续手术。

**价值主张重定位**：组合孪生不再以「ood_action RMSE 追平线性」为卖点，改强调**可解释分解（φ＋残差）＋ 组间方差校准（var_ratio 保住）**——黑箱线性/MLP 给不了，对齐获奖判据 A（白箱因果图）/ B（模块化可组合）。工件：`experiments/20260812-p44-dualpath.json`、`code/model/compositional_p44.py`、`scripts/p44_dualpath.py`。

### 3.8 本周新增证据槽 · P4-1 localized 共形（阴性，PC-3 重定位为 epistemic 缺陷）

**待检假设**：按新颖度把校准行分到最近测试子集、各子集单独估分位数 `q_s`，取代单标量 `_q`，解 `ood_action` 区间饱和。

**判决**：`localized_cov@0.95（ood_action）= 0.9994`（远超 0.97 上限）、`localized_ece = 0.2060`（≫0.08）→ `P41_ACCEPTED = false`。且 `p41_localized` 与 `global_q_baseline` **逐指标完全相等**。

**退化根因（非 bug，是 conformal i.i.d. 约束在 OOD 上被打破）**：日志 `q_global@0.95=41.2597`，`q_local=id=ood_agent=ood_action=ood_neuro=41.2597`。校准行取自训练分布内 20%（nd 全低），其新颖度范围**覆盖不了** OOD 测试子集（ood_action/ood_neuro 中位数更高），`argmin` 把全部校准行归到 nd 最低的 `id` 箱 → 该箱分位数≈全体 → 其余三箱回退全局 → 四子集 q 全相等。

**PC-3 瓶颈重定位**：假设「全局 q 不适合 OOD」被证伪；真正瓶颈是组合孪生 epistemic std **全子集一致欠缩放 ~41×**（q@0.95=41.26），与 P3-1b 对 virtual_twin 的 q@0.9=30.22（~30×）同源——深度集成在 OOD 上 confidently-wrong。**PC-3 是 epistemic 估计器缺陷，非校准层重标定可修**；localized conformal 在当前架构下结构性不可行。工件：`experiments/20260812-p41-localized-conformal.json`、`scripts/p41_localized_conformal.py`。

### 3.9 本周新增证据槽 · 路 A novelty/heteroscedastic 门控（阴性，PC-3 原理性未解）

**待检假设**（W33 §7 P0）：让 epistemic std 自身随新颖度撑大，正面攻克 PC-3 的 ~41× 欠缩放根因（改 σ 估计器本体，与 P4-1 改校准层 q 划清）。

**两机制变体（同模型类 `NoveltyGatedCompositionalTwin`，GATE_REPRO=True 臂 A 逐位复现）**：
- **变体 1 · empirical（novelty 门控）**：ood_action cov@0.95=**1.000**、ECE=**0.2125**（仍饱和）→ 失败。诊断显示新颖度（kNN 距离）在 Norman 离散特征空间（大量精确重复细胞）里是**常量/二值**（ID≈0、OOD=√2–√3），无连续梯度可作门控回归量。
- **变体 2 · heteroscedastic（σ_epi 异方差重标定）**：利用诊断发现的「欠缩放与模型自身 σ_epi 负 Spearman 相关（-0.4）」，`g=(σ_ref/σ_epi)^β`（β≈-0.98）。ood_agent ECE **0.1357→0.0655**（显著改善，但 ood_agent 非判据目标）；**ood_action cov@0.95=0.999→0.9972、ECE=0.206→0.198，仍超 0.97 上界且 ECE>0.08 → 判据失败**。

**原理性根因（三路证据闭合）**：诊断（`scripts/p4a_diagnose.py`）显示逐样本欠缩放 `|resid|/σ_epi` 与**任何测试时可获得信号都不充分相关**——新颖度常量（NaN）、σ_epi 仅弱负相关（-0.35~-0.41 且方向相反）、线性分歧≈0、增益常量。且四子集欠缩放分布互不匹配（p95：id≈22、ood_action≈12、ood_agent≈26、ood_neuro≈25），无单一 q 能同时服务。→ 校准（ID）的 `|resid|/σ` 与 OOD 分布系统性不同，且**无协变量可预测该差异**——这是 P4-1「ID 校准不覆盖 OOD」的估计器层镜像：**不仅 q 不可转移，σ 也不可转移**。

**判决**：`P4A_ACCEPTED = false`（两变体均阴性）。PC-3 在本基准上**原理性不可解**：唯一能修的路径需 OOD 标注校准数据（与 OOD 定义自相矛盾）或结构性新估计器（距离感知/生成式/proper-Bayesian 深度集成，属新研究方向，超当前 scope）。**当前以 R4 闸门兜底**（OOD 区间标「不可信」，仅点估计参与机制泛化判决）。工件：`experiments/20260814-p4a-novelty-gate.json`、`experiments/20260814-p4a-heteroscedastic.json`、`code/model/compositional_p4a.py`、`scripts/p4a_novelty_gate.py`、`scripts/p4a_diagnose.py`。

---

### 3.10 本周新增证据槽 · P5 结构性 epistemic 估计器（新立项，PC-3 唯一剩余解法）

**立项触发**：用户指令「立新项改结构性估计器」（08-14）。路 A 阴性后，PC-3 仅余「结构性新估计器」一条解法方向（P4-1 校准层 / 路 A 估计器层测试时重标定均阴性）。

**三框架塑形（非互相替代）**：
- **Einstein 视角**：被破坏的对称性 = σ(x) 应关于「到训练流形距离」τ(x) 单调（`∂σ/∂τ>0`），现实 OOD 区 `∂σ/∂τ≈0`（方差不随新颖度变化）。最小修复 = 在**构造期**注入缺失多样性，使 σ(x) 由模型后验/密度天然正确——**修假设，不补输出**（路 A 事后乘 g 已证此路不通）。
- **TRIZ 视角**：PC-3 = 物理矛盾（σ 同时要求宽保覆盖 / 窄保可证伪）。矛盾矩阵 → 发明原理 #1 分割 / #15 动态化 / #28 机制替代 / #35 参数变化。综合解 = #28（贝叶斯替代）+ #15（密度自适应）+ #1（按密度空间分割）。
- **serotonin-os 视角**：稳态基线（linear 点估计四子集全胜，P5 只增强 OOD 区间，绝不改点估计）；耐心增益（周级研究子项，每机制先 1–2 天 spike）；满足-停止信号（三机制 spike 全失败即停，接受 R4，写阴性结案）；冲动门控（禁止加网络层 / foundation model / 私采新数据）；时间视野伸缩（OOD 点估计结论本周即可用，不必等 P5）。

**P5 三机制候选**（详见 `experiments/20260814-P5-structural-estimator-proposal.md`）：
- **P5-A · proper-Bayesian 深度集成**（SGLD / 温度化似然）：后验在低数据区天然多样，`∂σ/∂τ>0` 由构造满足。
- **P5-B · 密度感知多样性注入**：训练期加正则 `R=Σ_m D(f_m,f̄)·τ(x)`，τ(x)=kNN 连续密度（非路 A 二值 novelty），在训练支撑薄处强制成员分散。
- **P5-C · 生成式 epistemic**：先拟特征密度 q(x)（flow/GMM/VAE），`σ(x)=σ_id·h(1/q(x))`，h 学习单调——把「数据在哪」与「函数是什么」解耦。

**预锁判据（与路 A 同，公平比较）**：ood_action 覆盖@0.95 ∈ [0.93,0.97] 且 ECE < 0.08；id/ood_agent/ood_neuro 不退化超阈值；GATE_REPRO=True（P1 linear 逐位复现）。kill-switch：三机制 spike 均无法把 ood_action ECE 压到 0.15 以下 → 停 P5 接受 R4。

**P5-B spike 结果（08-14，首档，已跑）**：`P5B_ACCEPTED=false`，`best_lambda=null`，`GATE_REPRO=True`。λ∈{0.01,0.05,0.1} 三档**输出逐字节相同**（λ 零效应）；σ_epi 反方向变小（P5B/P1=0.556）；ood_action ECE 恶化（0.6112→0.6872），覆盖仍饱和。**机制诊断**：sklearn `MLPRegressor` 的 `sample_weight` 重加权**无法注入成员负相关**（负相关性学习需改损失项 `(f_m−f̄)(f̄−y)`，工具链不支持）；稀疏/OOD-like 区仅占训练极少数，密集主体主导 → 成员收敛同函数 → 低方差。概念（密度感知多样性）成立，但本实例化（sample_weight 代理）不足以实现所需 decorrelation。proper 实现需 custom 训练循环（NCL/SGLD），超出当前 numpy/sklearn-only 工具链。详见 `experiments/20260814-P5B-closure.md`。**P5-B 阴性，待用户决策 P5-A(需工具链升级)/P5-C(大概率撞路A同墙)/kill-switch 接受 R4**。

**P5-B spike 结果（08-14，第二档，proper NCL）**：用户选定「投资 proper NCL 实现」→ 用 numpy-only custom 训练循环实现真正负相关性学习（绕过 sklearn 限制）。`P5B_NCL_ACCEPTED=false`，`GATE_REPRO=True`。**过程发现**：首跑批三档 λ byte-identical（λ 零效应）→ 定位为 member-agnostic 梯度 bug（NCL 项缺 member-specific 的 `(f_m−f̄)` 因子），已修复并子集验证（uniform w=1 时 λ 强改变成员多样性 3.4×）。**完整跑批（79k 行 / 120 epoch）证伪「w 权重误指定」假设**：sparse-w 与 uniform-w **同样失败且 λ 近似无效**——σ_epi 反更小（≈0.06×P1）、ECE 恶化（0.775 vs P1 0.61–0.69）、欠缩放加剧（72–109× vs 4–9×）。**根因修正为共识塌缩**：NCL 去相关项 ∝(f_m−f̄) 随成员 MSE 收敛→0，无法从塌缩集成自举多样性；子集 λ 效应是欠收敛瞬态。→ **P5-B 概念穷尽**（sample_weight + NCL 两变体均失败，结构性共识塌缩非调参可解）。详见 `experiments/20260814-P5B-NCL-closure.md`。**待决策**：P5-A(proper-Bayesian/SGLD，内禀 posterior 多样性，剩余唯一有原理希望)/P5-C(GMM 测试时门控，大概率撞路A同墙)/kill-switch 接受 R4。

**P5-A spike 结果（08-14，第三档，SGLD）**：用户选定「P5-A · SGLD」→ numpy-only 独立链 SGLD 后验采样（每成员独立噪声轨迹，内禀 posterior 多样性，绕过杀死 NCL 的共识塌缩）。`P5A_SGLD_ACCEPTED=false`，`GATE_REPRO=True`（P1 逐位复现，rmse 四子集 delta 全 0.0）。**结构性近胜（非阴性）**：SGLD 把 ood_action 覆盖@0.95 从 P1 灾难性 **0.245 → 0.968（落入 [0.93,0.97] 有效带）**，underscale_med ~4.6×→~0.39×（≈12× 修正趋近校准），σ_epi 放大 32–171×；**3/4 子集 ECE≤0.084（达标）**，仅最难 ood_action ECE=0.1031@T0.01 / 0.0943@T0.1（目标<0.08，毫厘未过）。**根因**：SGLD posterior 略过离散（区间偏宽）致残留 ECE，属可调和方向（温度/离散度校准），与 NCL 结构性塌缩恰成反比。**关键趋势**：T 升 0.01→0.1 时 ECE 反降（0.103→0.094）且覆盖保持入带 → 细 T 网格（0.05–0.4）+ lr/T 稳定化可能在发散前把 ECE 压到 <0.08。**T=1.0 发散（NaN，matmul overflow）**：有效温度体制窄（T≤0.1 稳定），属 lr/T 耦合失稳非机制失败。→ **PC-3 欠缩放根因被结构性修正，P5-A 是三机制中唯一入带的方案**；严格 ECE 判据未过但缺口可调，待用户决策精炼 spike / 软接受 PoC / 跑 P5-C / kill-switch。详见 `experiments/20260814-P5A-SGLD-closure.md`。

---

## 4. 不确定度与校准审计（两种失效模式 + 新增 R4 闸门）

### 4.1 失效模式 A · `virtual_twin` 全域灾难性过度自信（P3-1b，**已修**）

| 子集 | ECE | 覆盖@0.9 | 覆盖@0.95 | 点估计退化 vs ID |
|---|---|---|---|---|
| id | 0.7216 | 0.078 | 0.092 | — |
| ood_agent | 0.7332 | 0.065 | 0.077 | +19.04% |
| ood_action | 0.7055 | 0.097 | 0.115 | +14.87% |
| ood_neuro | 0.7237 | 0.075 | 0.089 | +5.84% |

名义 90% 区间只覆盖 6.5%–9.7% 的真值 → 区间宽度被低估约一个数量级。
**注意**：点估计退化全部远低于 50%，所以原 R2 否决规则**永不触发**（详见 `wetlab/2026-W33.md` §6.2 更正②）。

**本周修复（W36 自动化 A）**：把 `virtual_twin` 接入与组合孪生相同的共形路径（逐 (样本,基因) 标准化残差分位数 q），q=30.2（std 被低估 ~30×）；ECE 0.71→0.14、覆盖@0.9 0.07→0.98。修复**只改区间可信度、不改点估计**（故 §1 双栏总表中的点估计排名不变）。工件：`experiments/20260812-p31b-conformal-virtualtwin.json`、`scripts/p31b_conformal_virtualtwin.py`。

### 4.2 失效模式 B · `ood_action` 区间饱和（P4-1，**已跑·阴性**）

`compositional_interaction_twin` 在 ood_action：ECE = **0.2125**、覆盖@0.9 = **1.000**、覆盖@0.95 = **1.000**。
0.2125 是**区间饱和的数值签名**：levels [0.5, 0.8, 0.9, 0.95] 下 `mean(|1 − level|) = 0.2125`。
即所有区间宽到覆盖一切 → 区间信息量为零。`compositional_twin` 同子集 0.1965 / 0.997 也已接近饱和。
根因：conformal 用单标量 `_q` 校准，隐含「ID 与 OOD 同分布」假设，在组合外推下失效。

**本周补跑 P4-1 的结论**（§3.8）：分 subset localized 共形并未改变区间——四子集 q 退化回全局 41.26，ood_action cov@0.95 仍 0.999、ECE 0.206。即「换更细的校准」治不了饱和；饱和的根因是 epistemic std 在 OOD 上被低估 ~41×，需改估计器本体而非校准层（见 §6 PC-3）。

### 4.3 本周新增记账规则 R4（校准闸门）

> **R4**：预测器若 `ECE > 0.15` 或 `|覆盖@0.9 − 0.90| > 0.05`，其区间**不得**用作证伪标尺，
> 也不得用于声称「不确定度已覆盖」。仍可参与点估计排名，但须标注「区间不可信」。

R4 筛选结果（已写入 `wetlab/2026-W33.md` §6.1）：

| 预测器 | id | ood_agent | ood_action | ood_neuro |
|---|---|---|---|---|
| `linear` | ✅ | ✅ | ✅ | ✅ |
| `mlp` | ✅ | ✅ | ✅ | ✅ |
| `compositional_interaction_twin` | ✅ | ✅ | ❌ | ✅ |
| `compositional_twin` | ❌ | ✅ | ❌ | ❌ |
| `virtual_twin` | ❌ | ❌ | ❌ | ❌ |
| `knn` / `mean` | ❌ | ❌ | ❌ | ❌ |

**刺眼的事实**：`linear` 是唯一四子集区间全部可用的预测器，同时也是三个子集的最优点估计器。
→ 「结构化孪生虽点估计不占优，但不确定度更可信」这条**潜在辩护路径被本周数据关闭**。
在本基准上，点估计质量与校准质量是**正相关**的，孪生两头都输。

### 4.4 反事实探针（`do-` 干预）

| 样本 | base 预测 | do(gain=1.6) | scale_ratio | do(action=other) |
|---|---|---|---|---|
| 32887 | 2.768 | 2.647 | 0.956 | **2.768**（= base） |
| 37845 | 2.855 | 2.734 | 0.958 | **2.855**（= base） |
| 856 | 2.745 | 2.623 | 0.956 | **2.745**（= base） |

- **增益轴**：scale_ratio ≈ 0.956–0.958 < 1 → 调质增益上调 1.6× 反而使响应**略降**。乘法放大机制不存在。
- **动作轴（镜像）**：`do(action=other)` 预测与基线**逐位相等** → mirror_k 退化为**恒等映射 1.000**。
  这是「虚拟镜像神经元」签名预测当前最薄弱的一环：模型的镜像轴是**死的**。
- ⚠️ 反事实仅 3 行样本、无 bootstrap CI → 只能作方向性判据，不得宣称显著。

---

## 5. Top-K 预测清单（按信息增益 / 不确定度排序）

| K | 目标 | 类型 | 当前状态 | 预期信息增益 | 判据 |
|---|---|---|---|---|---|
| 1 | **E2 镜像轴反向证伪** | 湿实验 | 模型预测 self/other 比值 = **1.000**（零效应） | **最高**：若实测比值显著 ≠ 1，直接证伪模型镜像轴，签名预测归零重建 | 配对差值 SEM，≥8 只动物 |
| 2 | **P4-4 φ 结构分离**（线性头 + 机制残差） | 建模 | ✅ **已跑，弱达成**（`P44_ACCEPTED=true` 但 CI 重叠窄带，线性点估计仍赢 0.009） | 结论：PC-1 结构路径耗尽，价值主张重定位为可解释分解+var_ratio | 组合孪生首次与 `linear` CI 重叠或更优 |
| 3 | **P4-1 分子集 / localized conformal** | 校准 | ✅ **已跑，阴性**（`P41_ACCEPTED=false`，四子集 q 退化回全局） | 结论：PC-3 = epistemic 缺陷非校准层可修；localized conformal 在 OOD 结构性不可行 | ood_action 覆盖@0.9 ∈ [0.85,0.95] 且 ECE < 0.15 |
| 4 | **P3-1b `virtual_twin` 区间重建** | 校准 | ✅ **已修**（W36 自动化 A：ECE 0.71→0.14） | 已完成，仅改区间不改点估计 | 覆盖@0.9 ≥ 0.80 |
| 5 | **ood_agent 主指标换 pbRMSE** | 评测 | 分辨率仅 1.41，RMSE 为钝刀 | 中高：不换则该轴所有判据无效 | pbRMSE 分辨率比值 > 3 |
| 6 | **E7 闭环光遗传** | 湿实验 | 标尺不依赖失校准孪生 | 中：唯一能检验增益是否状态依赖 | ≥30 次闭环 trial，ΔF/F 均值±SEM |
| 7 | **E4 新调质外推** | 湿实验 | 三预测器 CI 全重叠 | **低（建议推迟）** | 需先提升该轴分辨率，否则浪费动物 |
| 8 | **路 A · epistemic 估计器改造**（novelty/heteroscedastic 门控） | 建模 | ✅ **已跑·阴性**（PC-3 原理性未解，三路证据闭合） | 结论：测试时重标定（novelty 二值 / σ_epi 弱负相关）均无法让 σ 在 OOD 转移；坚守 R4 兜底 | ood_action 覆盖@0.95 ∈ [0.93,0.97] 且 ECE < 0.08 |
| 9 | **P5 · 结构性 epistemic 估计器**（proper-Bayesian / 密度感知注入 / 生成式） | 建模 | 🆕 **新立项**（08-14，三框架塑形提案已落盘） | 最高：PC-3 唯一剩余解法；若任一机制 spike 让 σ 在 OOD 正确转移，则闭合并推翻 R4 兜底 | ood_action 覆盖@0.95 ∈ [0.93,0.97] 且 ECE < 0.08 |

不确定度最高（= 最值得测）的轴：**动作/镜像轴**。它同时具备「模型预测零效应」+「无 CI 支撑」+
「项目签名主张」三重属性，是唯一能一次性大幅改变项目可信度的实验。

---

## 6. 未解矛盾（TRIZ 分类）

### PC-1 · 物理矛盾（项目核心）：同一个 φ 既要记住分布，又要外推未知

- **表现**：ID 与全部 OOD 子集上，结构化孪生对 `linear` 的差额全为负或不显著（§1 双栏差额表）。
- **本周新证据**：P4-3 关闭了「靠更精细的统计加权把信号提出来」这条支路。
  诊断给出的理由是机制性的——**误差主体是 bias（τ²/SE² = 16.8），不是 variance**。
- **意义（这是本周真正的进展）**：破法空间被**收窄**了。
  「改估计量」类方案（加权、收缩、插补精化、增大共扰动样本）已被证伪为无效方向；
  剩下的必须是**结构分离**类方案：把「线性可达部分」与「机制残差部分」分派给不同通路，
  即 TRIZ 空间分离原理。→ 直接指向 P4-4。
- **状态**：**结构路径已耗尽**（P4-4 弱达成，4 类结构改造全未追平线性）。剩余破法 = 价值主张重定位（可解释分解+var_ratio），不再做 φ 结构手术。

### PC-2 · 物理矛盾：判别力 vs 精度 vs 校准（P4-2/P4-3 暴露，本周新增第三维）

- **表现**：修好 ood_agent 常数坍缩（`var_ratio` 1.07e-26 → 0.50/0.48）的代价是
  RMSE 0.5854 → 0.5887/0.5881 **且** ECE 0.0887 → 0.1490/0.1487。
- **本周新增维度**：此前只记账「判别力 ↔ 精度」二元代价，现在确认**校准也是代价项**（§3.5）。
  且 armA 的「优秀」ECE 0.0887 是坍缩伪装出来的假象。
- **破法**：ECE 必须与 `var_ratio` 联合读；单看 ECE 会把坍缩误判为校准优势。
- **状态**：未消解。三者的联合帕累托前沿尚未刻画。

### PC-3 · 物理矛盾：区间要宽（保覆盖）vs 要窄（保可证伪）

- **表现**：两个极端同时存在于一份基准里——
  `ood_action` 孪生区间饱和（ECE 0.2125 / 覆盖 1.000，宽到无信息）；
  `virtual_twin` 区间崩坏（ECE 0.73 / 覆盖 0.065，窄到无覆盖）。
- **本周动作**：新增**记账工具 R4**（把两类失效分开计量），**跑了 P4-1（localized 共形）** 与
  **路 A（novelty/heteroscedastic 门控）** 两条解法尝试。
- **破法方向（均证伪）**：
  - 校准层：条件分离——按新颖度分层做 localized conformal 取代单标量 `_q`。P4-1 实测四子集 q
    退化回全局（ID 校准集新颖度不覆盖 OOD），localized 共形**结构性不可行**。
  - 估计器层：路 A 让 σ 随新颖度/σ_epi 撑大。empirical 变体因新颖度在离散特征空间为二值而退化；
    heteroscedastic 变体仅改善 ood_agent（ECE 0.14→0.07），**ood_action 仍饱和（cov 0.997>0.97）**，判据失败。
- **状态**：**原理性未解（三路证据闭合）+ 已立新项 P5（结构性估计器方向）**；P5-B（sample_weight + proper-NCL 双变体）本周已双阴性（根因=同架构 MLP 在 MSE 下的集成共识塌缩，概念穷尽）；**P5-A·SGLD 本周跑出「结构性近胜」**——唯一把 ood_action 覆盖@0.95 送入有效带（0.245→0.968）且 underscale 修正 ≈12× 的方案，PC-3 欠缩放根因被结构性修正；严格 ECE<0.08 判据以毫厘未过（ood_action ECE 0.094–0.103，过离散残差，可调），P5-C/GMM 未跑。根因定性修正：calibration i.i.d. 约束被打破的纯形式仍在，但 SGLD 内禀 posterior 多样性证明**该纯形式可被构造期机制绕过**（σ 由后验离散度天然随预测不确定性缩放，∂σ/∂τ>0 由构造满足）：
  校准（ID）的 `|resid|/σ` 与 OOD 分布系统性不同，且无任何测试时协变量可预测该差异
  （新颖度常量、σ_epi 仅弱负相关且方向反）。校准层（P4-1）与估计器层（路 A）两条最 principled
  路径均阴性 → **唯一能修需 OOD 标注校准数据（与 OOD 定义自相矛盾）或结构性新估计器**。
  后者已立为 **P5 新研究子项**（proper-Bayesian 深度集成 / 密度感知多样性注入 / 生成式 epistemic，
  见 `experiments/20260814-P5-structural-estimator-proposal.md`），由 Einstein+TRIZ+serotonin 三框架塑形。
  在 P5 出结果前，当前以 **R4 闸门兜底**：OOD 区间标「不可信」，仅点估计参与机制泛化判决。

### TC-1 · 技术矛盾：`virtual_twin` 的多尺度先验只买到点估计

- ID 上 RMSE 0.5671 排第 2（与 `linear` CI 重叠），但区间在四个子集全废。
- 判定：先验换来了拟合，没换来可信度。**优先级仍低**（该模型不在主线上）。

### TC-2 · 技术矛盾（本周新增）：判据要严格 vs 指标要有分辨率

- **表现**：ood_agent 分辨率比值 1.41，而 P4-3 判据要求分辨小数第 3 位（0.5881 vs 0.5854）。
  判据的严格性超出了指标的分辨能力。
- **本周的临时破法**：引入**独立于该指标的机制诊断**（λ、τ²/SE²）作为第二证据源，
  用「两个不相关证据同向」替代「单指标高精度」。本周 P4-3 判决正是靠这个结构才站住。
- **根本破法**：ood_agent 主指标换 pbRMSE（Top-K #5）。
- **状态**：临时可用，根本未解。

---

## 7. 下一步（按判据，非按工作量）

| 优先级 | 任务 | 判据（跑前冻结） | 依赖 | 状态 |
|---|---|---|---|---|
| ✅ 已闭环 | P4-4 φ 双通路 | 组合孪生 ood_action 与 linear CI 重叠（窄带弱达成） | — | **弱达成**：PC-1 结构路径耗尽 |
| ✅ 已闭环 | P4-1 localized 共形 | ood_action 覆盖@0.9 ∈[0.85,0.95] 且 ECE<0.15 | — | **阴性**：PC-3 重定位 epistemic 缺陷 |
| ✅ 已闭环 | P3-1b virtual_twin 共形 | 覆盖@0.9 ≥ 0.80 | — | **已修**：ECE 0.71→0.14 |
| ✅ 已闭环 | **路 A** epistemic 估计器改造（novelty/heteroscedastic 门控） | ood_action 覆盖@0.95 ∈ [0.93,0.97] 且 ECE < 0.08 | 无（P4-1 关闭校准层方向） | **阴性**：PC-3 原理性未解，三路证据闭合，R4 兜底 |
| 🆕 **P0（新立项）** | **P5 · 结构性 epistemic 估计器**（A: proper-Bayesian / B: 密度感知注入 / C: 生成式） | ood_action 覆盖@0.95 ∈ [0.93,0.97] 且 ECE < 0.08（与路 A 同判据）；id/ood_agent/ood_neuro 不退化；GATE_REPRO=True | 路 A 阴性（PC-3 原理性未解） | **P5-B 双阴性**（概念穷尽，根因=集成共识塌缩）；**P5-A·SGLD 结构性近胜**（覆盖 0.245→0.968 入带、underscale≈12× 修正、3/4 子集 ECE≤0.084；严格 ECE<0.08 以毫厘未过，过离散残差可调，T=1.0 发散）→ PC-3 欠缩放根因被结构性修正；P5-C/GMM 未跑。待用户决策：①精炼 spike（T 细网格 0.05–0.4 + lr/T 稳定化，尝试压 ECE<0.08）/ ②软接受 PoC（partial win，OOD 区间标「近似可信 ECE≈0.10」）/ ③跑 P5-C / ④kill-switch 接受 R4（仍 premature：P5-A 近胜 + P5-C 未测） |
| **P1** | ood_agent 主指标换 pbRMSE 并重设判据 | pbRMSE 分辨率比值 > 3（现 RMSE 为 1.41） | 无 | 未启动 |
| **P2** | 把 R4 闸门写入 `benchmark_ood.py` 自动执行 | 基准报告自动输出 R4 可用性表 | 无 | 未启动 |
| **P3** | 湿实验 E7 → E1/E6 → E2 | 见 `wetlab/2026-W33.md` §6.5 | 湿实验资源 | 未启动 |

**明确不做**：任何「逆方差 / 加权 / 插补精化」变体（P4-3 已证伪该方向）；
任何「分 subset / localized 共形重标定」变体（P4-1 已证伪：i.i.d. 校准在 OOD 结构性失效，四子集 q 退化回全局）；
任何「测试时 novelty / σ_epi 门控撑大 epistemic std」变体（路 A 已证伪：新颖度在离散特征空间为二值、σ_epi 仅弱负相关且方向反，σ 在 OOD 不可转移）；
任何「再加一层网络」式提容量方案（掩盖机制缺失，违反项目硬约束）。
**PC-3 处置**：已立新项 **P5**（结构性估计器：距离感知/生成式/proper-Bayesian 深度集成），由三框架塑形，不在 compositional_twin 上继续手术；P5 出结果前仅以 R4 闸门兜底（OOD 区间标「不可信」）。

---

## 8. 诚实边界

1. **本周主基准零建模增量**，三段 sha256 逐位冻结。唯一新增证据是 P4-3 的**阴性**判决。
2. **P4-3 判决的论证结构如实披露**：判据的一半（RMSE ≤ 0.5854）落在分辨率仅 1.41 的子集上，
   **单凭该指标不足以判否**；结论成立依赖独立的收缩诊断（λ ≈ 0.94、τ²/SE² = 16.8）。
   若只有 RMSE，正确表述应为「无法判决」。
3. **「疑似仅记忆」的效力限定**：该 flag 仅在**点估计层面**成立。
   `ood_action` 上两个孪生的区间已饱和（覆盖 1.000），区间退化时不得声称「已证明只在记忆」。
4. **反事实探针无 CI**：仅 3 行样本，只作方向性判据。
5. **R4 是记账规则，不是修复**。`virtual_twin` 缺口本周**已修**（P3-1b，ECE 0.71→0.14）；`ood_action` 缺口本周**跑了 P4-1 但阴性**（localized 共形退化回全局 q，证明是 epistemic 缺陷非校准层可修）。
6. **上周首版的 P4-3 零引用是正确的**，本周入账基于已落盘工件（`GATE_REPRO=True`）。
   历史上曾发生「声称跑了 multiseed 重标定但零工件」的幻觉事故
   （`experiments/20260807-RETRACTION-w32-multiseed-recalibration.md`），本周所有数字均可回溯至工件。
7. **P4-4 / P4-1 / P3-1b 数字均来自 08-12 落盘工件**（均 `GATE_REPRO=True`）：
   - P4-4 `P44_ACCEPTED=true` 必须读作「CI 重叠（弱）」，**不可读作「超越线性」**；线性点估计仍赢 0.009。
   - P4-1 `P41_ACCEPTED=false` 且 localized q 退化回全局——属 conformal i.i.d. 约束在 OOD 失效的阴性证据，非脚本 bug。
8. **未外发、未发布、未向 GOAI 提交**（已推送到 origin/master，commit 934f921）。
9. **路 A 阴性（本刷新新增，08-14）**：`P4A_ACCEPTED=false` 两条机制变体均阴性，且 `GATE_REPRO=True`。
   该 negative 不推翻 P4-4/P4-1/P4-3/P3-1b 结论，但与 P4-1（校准层）、P3-1b（virtual_twin 已修）共同把
   PC-3 从「路 A 待攻克」降级为「原理性未解、R4 兜底」。诊断（`p4a_diagnose.py`）显示欠缩放与任何测试时
   信号均不充分相关——此结论可回溯至 `experiments/20260814-p4a-*.json` 两工件，非凭空断言。
10. **P5 新立项（本刷新新增，08-14）**：PC-3 唯一剩余解法方向「结构性估计器」已立为新研究子项，由 Einstein Perspective（修对称性非补输出）+ TRIZ（物理矛盾 #28/#15/#1 分离）+ serotonin-os（基线/耐心/停止信号治理）三框架塑形。预锁判据与路 A 一致（ood_action 覆盖@0.95 ∈[0.93,0.97] 且 ECE<0.08），kill-switch = 三机制 spike 全失败则停、接受 R4。提案见 `experiments/20260814-P5-structural-estimator-proposal.md`，可回溯，非凭空立项。

---

## 9. 文献监测要点（08-11 / 08-12，详见 `references/2026-08-12-literature.md`）

本轮 8 方向 6 轮检索，相对前五期去重后精选 9 篇；最值得借鉴两个方法，且与本周结论直接呼应：

1. **VCBench（bioRxiv 2026.06.18）的 spread-error 认知校准探针 + 污染报告 schema**：把「深度模型未稳定超越线性基线」「区间饱和是测量伪影还是机制缺失」变成可复现、可审计操作。其 spread-error 探针（预测展布 vs 误差相关性）正是本项目 ECE=0.21/覆盖=1.000 式退化的诊断器，可并入 `verify_collapse`。
2. **Mechanisms Matter: Transportability（bioRxiv 2026.05.08v2）的 CausalDGP 可调机制分歧 + Vendi 多样性诊断**：给出跨上下文（= ood_agent/ood_neuro）泛化的可证伪评测协议。

**与本周结论的诚实呼应**：此前曾设想用这二者组装「P4-1 localized conformal 落地配方」（单标量 `_q` 退役、按 ood_* 子集分别报覆盖/ECE/多样性）。**但 P4-1 已证伪该配方的前提**——OOD 子集区间无法靠 ID 校准重标定获得，localized conformal 在当前架构下不可行。spread-error / Vendi 仍可作为**诊断**工具（监测区间是否饱和、预测多样性是否坍缩），但不能作为「修区间」的配方。其余 7 篇（U-Pert 质量守恒扰动动力学、CauFinder do-演算因果解耦、Anchor–Stabilizer 可辨识性理论、VADER1/all-optical cOVC/Neuropixels Opto 光学-电生理探针、Dalla Porta ACh 空间异质）为表示层/湿实验层储备，本竞赛周期内不直接采用。

---

## 附 · 本报告引用的工件清单

| 工件 | 字节 | 用途 |
|---|---|---|
| `experiments/20260810-benchmark.json` | 43814 | 主基准（种子），双栏总表 / 校准 / flags / 反事实 |
| `experiments/20260810-benchmark.md` | 14982 | 主基准可读版 |
| `experiments/20260810-p43-shrinkage.json` | 30760 | **本周新证据槽**：P4-3 三臂判决 + 收缩诊断 |
| `experiments/w35_p43_run.log` | 3598 | P4-3 运行日志（`GATE_REPRO=True`） |
| `experiments/20260810-regress-check.json` | 1074 | 回归守卫：三段 sha256 冻结证明 |
| `experiments/20260810-PROVENANCE.json` | 2282 | 基准工件校验（12 claimed / 0 missing / PASS） |
| `experiments/20260809-benchmark.json` | 64010 | 冻结对照基准（prev） |
| `code/model/compositional_p43.py` | 8318 | P4-3 收缩模型实现 |
| `scripts/p43_shrinkage.py` | 11301 | P4-3 三臂对照脚本 |
| `scripts/regress_check.py` | 8135 | 回归守卫 |
| `configs/benchmark_ood_norman_canonical.yaml` | 2241 | canonical 基准配置 |
| `data/processed/norman_cache.npz` | 191629995 | 真实 Norman 2019 缓存 |
| `experiments/figures/fig1_rmse_by_subset.png` | 67478 | 子集 RMSE 图 |
| `experiments/figures/fig2_calibration.png` | 72015 | 校准曲线图 |
| `wetlab/2026-W33.md` | 25826 | 本周湿实验方案（7 协议 + §6 人工附录） |
| `experiments/20260812-p44-dualpath.json` | — | **本周新证据槽**：P4-4 四臂对照（GATE_REPRO=True） |
| `experiments/20260812-p41-localized-conformal.json` | — | **本周新证据槽**：P4-1 localized 共形（GATE_REPRO=True，退化回全局 q） |
| `experiments/20260812-p31b-conformal-virtualtwin.json` | — | **本周新证据槽**：P3-1b virtual_twin 共形修复（ECE 0.71→0.14） |
| `experiments/20260814-p4a-novelty-gate.json` | — | **本周新证据槽**：路 A 变体 1 empirical novelty 门控（GATE_REPRO=True，阴性） |
| `experiments/20260814-p4a-heteroscedastic.json` | — | **本周新证据槽**：路 A 变体 2 heteroscedastic 门控（GATE_REPRO=True，阴性） |
| `code/model/compositional_p44.py` | — | P4-4 双通路模型实现 |
| `code/model/compositional_p4a.py` | — | 路 A 模型实现（empirical + heteroscedastic 双门控，含 kNN 自匹配修复） |
| `scripts/p41_localized_conformal.py` | — | P4-1 跑批脚本 |
| `scripts/p4a_novelty_gate.py` | — | 路 A 两臂受控对照跑批脚本（--gate-kind empirical/heteroscedastic） |
| `scripts/p4a_diagnose.py` | — | 路 A 诊断：欠缩放 vs 测试时信号相关性 |
| `scripts/p31b_conformal_virtualtwin.py` | — | P3-1b 跑批脚本 |
| `references/2026-08-12-literature.md` | — | 文献监测（9 篇新精选：VCBench/U-Pert/Mechanisms Matter/CauFinder/VADER1 等） |
| `references/2026-08-11-literature.md` | — | 文献监测（上一轮） |
| `experiments/20260814-P5-structural-estimator-proposal.md` | — | **本周新证据槽**：P5 结构性估计器新立项提案（Einstein+TRIZ+serotonin 三框架塑形，PC-3 唯一剩余解法） |
| `experiments/20260814-P5B-NCL-closure.md` | — | **本周新证据槽**：P5-B proper-NCL 结案（梯度 bug 修复 + 共识塌缩根因，双阴性） |
| `code/model/compositional_p5b_ncl.py` | — | P5-B proper-NCL 模型（numpy-only custom 训练循环，member-specific 梯度） |
| `scripts/p5b_ncl.py` | — | P5-B NCL 两臂对照 + λ 扫描 + `--w-mode {sparse,uniform}` 机制探针 |
| `scripts/_smoke_ncl_fix.py` | — | NCL 梯度修复冒烟测试 |
| `scripts/_diag_ncl.py` | — | NCL 三场景机制诊断（隔离 w 与 λ） |
| `experiments/20260814-p5b-ncl-sparse.json` | — | P5-B NCL sparse-w 跑批（GATE_REPRO=True, 阴性） |
| `experiments/20260814-p5b-ncl-uniform.json` | — | P5-B NCL uniform-w 机制探针（阴性） |
| `experiments/w36_p5b_ncl_sparse.log` | — | sparse-w 运行日志 |
| `experiments/w36_p5b_ncl_uniform.log` | — | uniform-w 运行日志 |
| `experiments/20260814-P5A-SGLD-closure.md` | 9476 | **本周新证据槽**：P5-A·SGLD 结案（内禀 posterior 多样性结构性近胜，覆盖入带、ECE 毫厘未过） |
| `code/model/compositional_p5a_sgld.py` | 7017 | P5-A·SGLD 模型（numpy-only 独立链后验采样） |
| `scripts/p5a_sgld.py` | 13549 | P5-A SGLD 两臂对照 + T 扫 + 覆盖/ECE/欠缩放诊断 |
| `scripts/_smoke_sgld.py` | 2593 | SGLD 持久多样性 + T 控制冒烟测试 |
| `experiments/20260814-p5a-sgld.json` | 63197 | P5-A SGLD 完整跑批（GATE_REPRO=True, P5A_SGLD_ACCEPTED=false） |
| `experiments/w36_p5a_sgld_run.log` | 5565 | P5-A SGLD 运行日志 |

> 本周报由每周五自动化（`automation-1785494084890`）生成，配套 `code/wetlab/export_protocol.py` 导出湿实验方案。
> 所有数值可通过上表工件回溯复现。
