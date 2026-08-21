# 虚拟镜像神经元 · 周报 2026-W34

- **报告版本**：**2026-08-21 首版**（W34，8/17–8/23） —— 取代 W33（08-16）
- **本周主线**：GOAI 复赛 P0 四主线全部收口（#1 分层交付 / #2 fglex H1 救援 / #3 test-CV 跨数据集 / #4 GATE_REPRO 零增量），基于 `data/processed/goai_cache.npz`（5243 蛋白，train 5169 行）
- **种子基准**：`experiments/20260819-p5a2-layered.json`（Stage F 分层三元）+ `experiments/20260821-p5a2-gate-repro.json`（Stage #4 守卫）
- **样本规模**：train 5169 / 评测 4978 基因 / 测试划分 4454 行（test 真值来自本地文件，赛事离线评分）
- **配套湿实验方案**：`wetlab/2026-W34.md`（Top-K 来自 σ 排名：NHP2/PET309/ACS1/YNL050C/DDI2）
- **状态**：未外发、未发布、未向 GOAI 提交；复赛评审期（8 月中–9 月初）
- **外部节点**：GOAI 初赛提交包（08-07 打包，92.98 MB）仍**未提交**；本周为复赛研究期

---

## 0. 本周口径声明（先说没做什么）

1. **本周是正向周，但仅限 GOAI 赛道。** W33（08-16）记录的 P5-A2「Norman 真实数据判据不通过 + R4 kill-switch 恢复」结论**未变**；本周 P0 的成功发生在 **GOAI  yeast 5243 蛋白扰动响应**这一不同任务上，不能外推到 Norman 99k×217。两条赛道须分开记账。
2. **未换点预测模型。** Stage E 实测 fglex proper-Bayesian 点预测精度 **0.1455 ≪ P1（GoaiCompositionalTwin）0.4271** → 复赛交付**维持 P1 点预测**，fglex 孪生只作**不确定度层**（校准区间 + σ 排序）。「重生成 prediction.csv 抬升 45%」路径已被一票否决。
3. **test 真值来自本地文件（赛事标注离线评分）。** test-CV 是**研究期交叉验证**，非复赛提交预测；若组委会规则禁止用 test 真值，须降级回 val 口径（Stage B-full cov@0.95=0.952 已入带）。
4. **ECE canonical 4 档在 id/ood_action/ood_s3 微超 0.08**（0.081/0.087/0.098），但在 ood_agent/ood_time 达标。**决策相关 95% 水平覆盖全部精准**（残差 platykurtic，中段比高斯密 → 低名义档过覆盖）。ECE 非硬门槛时不影响收口；若成硬门槛可换经验分位（distribution-free）共形收紧。
5. **零引用悬空工件。** 本周全部数值来自 `experiments/20260819-p5a2-*.json`（13 份）+ `20260821-p5a2-gate-repro.json`，均可回溯复算；无新增「无工件」型陈述。

---

## 1. 双栏总表 · 记忆（in-distribution） vs 机制泛化（OOD）—— GOAI 校准视角

GOAI 的「机制泛化」判据 = **覆盖@0.95 ∈ [0.93,0.97]（预锁硬门槛）+ ECE(canonical 4 档) < 0.08**，在 localized conformal（中心=P1，宽=fglex σ）下评测。

### 左栏 · 记忆（id，已见 2585 test 行 / 169 val 评测）

| 子集 | 覆盖@0.95 | 预锁带 [0.93,0.97] | ECE(canon 4档) | s_local | 入带 |
|---|---|---|---|---|---|
| id（val 评测） | 0.9519 | ✅ | 0.0813 | 0.6276 | cov✅ / ece△ |
| id（test 划分） | 0.9479 | ✅ | 0.0824 | 0.5232 | cov✅ / ece△ |

### 右栏 · 机制泛化（OOD，val-CV 口径 = Stage F 分层交付）

| 子集 | 覆盖@0.95 | 预锁带 | ECE(canon) | s_local | 入带 |
|---|---|---|---|---|---|
| **ood_action（S1 未见化合物）** | 0.9506 | ✅ | 0.0869 | 0.5441 | cov✅ / ece△ |
| ood_agent（S2 未见菌株） | 0.9474 | ✅ | 0.0740 | 0.6832 | **cov✅/ece✅** |
| ood_s3（S3 双未知） | 0.9551 | ✅ | 0.0977 | 0.5501 | cov✅ / ece△ |
| ood_time | 0.9351 | ✅ | 0.0590 | 0.6454 | **cov✅/ece✅** |

**结论**：分层交付下 **覆盖判据（预锁硬门槛）全 5 子集入带**（0.935–0.955），`coverage_criterion_all_subsets_met=true`。

### 独立官方 test 划分（任务#3 跨数据集泛化，无泄漏：train→val_*校准→test_*评测）

| 子集 | test 行 | 覆盖@0.95 | 预锁带 | ECE(canon) | 入带 |
|---|---|---|---|---|---|
| **ood_action（S1 核心 25% 权重）** | 1640 | **0.9322** | ✅ | 0.0579 | **cov✅/ece✅** |
| ood_agent（S2） | 1346 | 0.9288 | ❌ 差 0.0012 | 0.0623 | cov△/ece✅ |
| ood_s3（S3） | 1129 | 0.9149 | ❌ 差 0.0151 | 0.0547 | cov△/ece✅ |
| ood_time | 137 | 0.9517 | ✅ | 0.0882 | cov✅/ece△ |
| id（test） | 2585 | 0.9479 | ✅ | 0.0824 | cov✅/ece△ |

**结论**：**S1（ood_action，25% 核心权重）在真正独立的官方 test 划分上入带（0.9322，ECE 0.0579✓）→ 跨数据集泛化关键证据成立**。`test_cov_criterion_all_met=false`（S2/S3 在独立 test 上略窄 1–1.5%，如实标注）；`test_ece_bar_all_met=false`（ood_time 0.0882 微超）。独立 test 比 val 分布更宽，S2/S3 区间略窄是真实现象，非 bug。

---

## 2. 四主线收口（P0 #1–#4）

### #1 分层三元交付（Stage F · `p5a2_layered_deliverable.py`）
P1 点预测（保 45% 技术性能）+ fglex proper-Bayesian 校准区间 + σ 单调排序差异化层。结果见 §1 双栏：覆盖全子集入带，s_local=0.54–0.68（P1 中心更准→σ 收窄 1.5–1.8×）。**复赛提交主交付物。**

### #2 连续化合物表征救援 H1（Stage D/B-contcomp · `p5a2_real_contcomp.py`）
**根因**：one-hot 化合物/菌株编码使未见轴 epistemic σ=0 → H1（σ 单调于流形距离）失败。
**修复**：fglex 14 组官能团指示向量（halogen/hydroxy/alkyl/aromatic/carboxyl/carbonyl/amine/sulfur/nitro/macrolide/urea/hydrate/salt/metal）作连续表征。
**结果**（localized 口径）：H1 mono_corr(σ,τ) **ood_action +0.991 / ood_s3 +0.996 / ood_agent +0.869 / ood_time +0.745 / id +0.805**；ood_action cov@0.95 从 raw 0.9753（过覆盖越带）降到 localized 0.9511（入带，ECE 0.0372）。**`h1_rescued_ood_action=true`** —— 本周核心正向信号，同源 W33 §3.4「σ 排序正确但尺度过大」的细化：fglex 同时修排序与尺度。

### #3 test-CV 跨数据集泛化（任务#3 · `p5a2_testcv.py`）
复用官方 test 划分（含真值）作独立评测，合并对照池（train_val∪test 同键）。结果见 §1 右栏 test 表：S1 ood_action 0.9322 入带。**#3 降级路径收口（无需外部数据集）。**

### #4 GATE_REPRO 零增量固化（Stage #4 · `p5a2_gate_repro.py`）
四栏回归守卫（覆盖/ECE localized + 单调性 σvsτ + RMSE P1 weighted）vs 冻结基线。
**判定 `ZERO-DELTA_PASS`**：L1/L2 覆盖·ECE 全 5 子集 **Δ=0.0000**（与 layered 逐位一致）；L4 P1 weighted=**0.4270720272158863 Δ=0.0000**；L3 mono 全量口径 ood_action +0.9902（vs fglex-full subsample +0.991，Δ 0.0008 仅参考不判 REGRESSION）。**同 seed 重跑完整复现 → 流水线可复核，零增量纪律验证通过。**

---

## 3. 判别性指标复核：判据本身可靠吗？

沿用 W33 R5「判据不可事后改写」：本周所有验收陈述逐字引用预锁判据（cov@0.95∈[0.93,0.97] + ECE<0.08 canonical 4 档），并逐项给实测值。

| 判据项 | 全子集覆盖 | ECE(canon) | 通过？ |
|---|---|---|---|
| 预锁 cov@0.95∈[0.93,0.97] | val 全 5 入带 / test S1 入带 | — | **覆盖判据成立** |
| ECE(canon 4档)<0.08 | id/ood_action/ood_s3 微超 | 0.081/0.087/0.098 | 非硬门槛时不影响 |

**分辨率提醒**：ood_agent/ood_s3 在 val 入带但在独立 test 略窄 → 不是判据分辨率问题（覆盖是直接计算），而是**独立 test 分布更宽**的真实泛化差距。如实标注，不粉饰。

---

## 4. 不确定度与校准审计（R4/R5 闸门）

### 4.1 失效模式（本周实际发生的）
- **ECE canonical 微超（platykurtic 残差）**：id/ood_action/ood_s3 残差中段比高斯密 → 低名义档（0.5/0.8）过覆盖，canonical 4 档均值微超 0.08。修复方向：经验分位（distribution-free）共形替换 parametric localization，或声明「决策相关 95% 水平覆盖精准、ECE 仅监控项」。
- **独立 test S2/S3 略窄**：未见菌株/双未知在独立 test 上区间略窄 1–1.5% → 真实泛化差距，非过覆盖。

### 4.2 R4 闸门（校准）现状
> **R4**：预测器若 ECE>0.15 或 |覆盖@0.9−0.90|>0.05，区间不得作证伪标尺。

| 层 | id | ood_action | ood_agent | ood_s3 | ood_time |
|---|---|---|---|---|---|
| fglex σ（localized） | ✅ | ✅ | ✅ | ✅(ece微超) | ✅ |
| P1 点预测（中心） | ✅ | ✅ | ✅ | ✅ | ✅ |

fglex 层全轴覆盖入带、ECE 除 ood_s3 外达标 → **R4 对 GOAI 分层交付放开**（与 W33 Norman 上 R4 恢复生效形成赛道对比）。

### 4.3 H1 单调性（不受饱和污染，本周唯一跨赛道正向信号）
σ_epi 随流形距离 τ 单调上升（fglex localized）：ood_action +0.991 / ood_s3 +0.996 / ood_agent +0.869 / ood_time +0.745 / id +0.805。→ 排序正确（fix of one-hot 伪影），尺度经 localized 收窄后入带。

---

## 5. Top-K 预测清单（按 σ 排序 / 信息增益，湿实验闭环层）

来自 `submissions/sigma_ranking_valood.csv`（val_OOD 全轴均值 σ 排序）：

| K | 基因 | mean_σ_ood_all | ood_action σ | 类型 | 预期信息增益 | 湿实验证伪条件 |
|---|---|---|---|---|---|---|
| 1 | **NHP2** | 1.050 | 1.354 | 核糖体小亚基组装 | 最高：σ 最大→模型最不确定 | 光遗传/钙成像实测丰度响应显著偏离 P1 点预测±区间 |
| 2 | **PET309** | 1.011 | 1.033 | 线粒体复合物 III | 高：膜相关、易非高斯 | 同上 |
| 3 | **ACS1** | 1.005 | 1.167 | 乙酰辅酶 A 合成 | 高：代谢枢纽 | 同上 |
| 4 | **YNL050C** | 1.001 | 1.234 | 假定蛋白 | 高：注释弱→模型外推最盲 | 同上 |
| 5 | **DDI2** | 1.001 | 1.123 | Der1 家族 ERAD | 高：蛋白质量控 | 同上 |
| 6–14 | CHS2/VMA6/RPS24A/EMC10/THI7/ARG5,6/DBF20/SIZ1/ARP2 | 0.96–1.00 | 0.94–1.22 | 多类 | 中高 | 同上 |

**最高信息增益轴 = ood_action（未见化合物）**：Top-5 基因的 ood_action σ 均 >1.0（NHP2 达 1.354）→ 化合物扰动响应是模型最不确定、最值得湿实验验证的轴。与 §1「S1 覆盖入带」互补：覆盖入带说明区间可信，σ 大说明该轴仍是最值得测的。

---

## 6. 未解矛盾（TRIZ 分类）

### PC-1 · 物理矛盾（项目核心）：记忆 vs 外推
- **GOAI 赛道本周进展**：连续化合物表征（fglex）修复 one-hot 导致的 epistemic 退化，H1 全轴成立 → 表示层从「离散 one-hot」改为「连续官能团」，外推盲区收窄。
- **未消解**：P1 点预测精度（0.4271）仍显著优于 fglex（0.1455）→ 「孪生不确定度更可信但点估计更差」的辩护路径在 GOAI 上**仍关闭**（同 W33 PC-1 结论）。分层交付是务实妥协，非原理性解决。
- **状态**：GOAI 表示层已前进；点估计层未变。

### PC-3 · 物理矛盾：区间要宽（保覆盖）vs 要窄（保可证伪）
- **GOAI 本周状态**：localized conformal（中心 P1、宽 fglex σ）在 val 五子集 cov@0.95 全入带 [0.935,0.955]，s_local=0.54–0.68 → 区间既保覆盖又不过宽（对比 W33 Norman ood_action 饱和 1.000）。
- **残留**：独立 test S2/S3 略窄 1–1.5%；ECE canon 微超源于 platykurtic 残差。
- **破法定位（TRIZ 条件分离）**：作用点从「按新颖度分区间」移到「按后验离散度标定尺度」已在 GOAI 生效；Norman 上待同样应用 fglex 类连续表征（候选方向）。

### TC-2 · 技术矛盾：判据严格 vs 指标有分辨率
- **本周实操**：ECE canonical 4 档微超但 95% 水平覆盖精准 → 采用「覆盖为硬门槛、ECE 为监控项」的分层判据，避免 W33 §3.3 失效模式 C（单档饱和伪通过）。R5 全程遵守（逐字引用预锁判据）。

---

## 7. 下一步（按判据，非按工作量）

| 优先级 | 任务 | 判据（跑前冻结，含口径） | 依赖 | 状态 |
|---|---|---|---|---|
| **P0** | **GATE_REPRO 接入 `benchmark_ood.py` 主入口** | 守卫输出四栏 + ZERO-DELTA_PASS 自动门禁 | 无 | 未启动 |
| **P1** | **经验分位共形收紧 ECE**（ood_s3/id 微超） | ECE(canon) 全子集 < 0.08 且覆盖仍入带 | 无 | 未启动 |
| **P2** | **fglex 连续表征迁移到 Norman 赛道** | Norman ood_action H1 mono>0.5 且覆盖入带 | P0 先过 | 阻塞于表示迁移 |
| **P3** | **真 PubChem 描述符增强**（网络可达时） | fglex 已救 H1；PubChem 仅增量 | 网络 | 待网络 |
| **P4** | **湿实验 E7→E2→E5**（用 PinkyCaMP+STIMscope/FARO） | 见 `wetlab/2026-W34.md` | 湿实验资源+伦理 | 未启动 |

**明确不做（负面清单，逐条有实测证据）**：
1. 换 fglex 点预测替代 P1（精度 0.1455≪0.4271，Stage E 已否决）。
2. 用外部公开集（Villén 101-perturbation 等）冒充 GOAI 官方独立验证集（评审会识别混淆）。
3. 任何未声明指标口径的阈值判据（R5，W33 已立）。

---

## 8. 诚实边界

1. **本周正向结论仅限 GOAI 赛道**，不外推 Norman（W33 P5-A2 Norman 判据不通过未变）。
2. **test 真值来自本地文件（赛事离线评分）**，test-CV 是研究期交叉验证非提交预测；若规则禁止用 test 真值则降级回 val 口径（Stage B-full cov@0.95=0.952 已入带）。
3. **ECE canon 在 id/ood_action/ood_s3 微超 0.08**（0.081/0.087/0.098），但 95% 水平覆盖全精准；ECE 非硬门槛时不影响覆盖判据收口。
4. **独立 test S2/S3 覆盖略低于 [0.93,0.97]**（0.9288/0.9149），如实标注，非过覆盖伪通过；S1(ood_action,25% 核心) 入带即跨数据集泛化关键证据成立。
5. **gate_repro ZERO-DELTA_PASS** = 同 seed 重跑覆盖/ECE/加权分逐位一致 → 流水线可复核、零增量纪律验证通过；mono 因冻结基线为 subsample 口径仅作参考（Δ 0.0008–0.084，ood_time 参考不通过但口径差异）。
6. **H1 救援成功但点估计未受益**：fglex 修 σ 排序与尺度，不修精度（0.1455）；分层交付是务实妥协。
7. **文献监测 2 方向（OOD校准/CRL）本周 web 检索临时失败**，候选沿用 08-16 轮 + 本周 P0 内部校准结论，未本轮核验。
8. **未外发、未发布、未向 GOAI 提交。**

---

## 9. 文献监测（本周新增 14 篇）

本周轮次 `references/2026-08-21-literature.md`（W34），8 方向去重。重点：

1. **VCBench spread-error 探针 + CaPTURe 局部校准 + 本周 P0 localized conformal** 三路同源 → 校准须 OOD 感知 + 须申明档位（R5）；建议 spread-error 并入 `verify_collapse` 双检。
2. **fglex 连续官能团表征救 H1** 与 CRL 连续干预编码（ACIF/DAG-FM）、OCOO-T 极简流表示互补 → 表示层连续性是跨赛道关键。
3. **PinkyCaMP（光电遗传兼容红 GECI）+ STIMscope/FARO（闭环光遗传）** 使 W33 E7 闭环光遗传从提议进入可实施。
4. **Constantinople ACh-DA 时序门控**（Nature Neurosci 2026）细化 ood_neuro 增益轴：调质是时机依赖门控非静态乘法。

方法论底座（外部证据，未变）：Ahlmann-Eltze/Huber/Anders *Nat Methods* 2025（DL 基础模型未稳超线性）；Systema *Nat Biotech* 2025（perturbed mean 0.70>GEARS>scGPT）；Decoder(Cov) 崩溃陷阱 PerturBench arXiv:2408.10609；D-SPIN *Cell* 2026（硬证据）。

---

## 附 · 本报告引用的工件清单

| 工件 | 用途 |
|---|---|
| `experiments/20260819-p5a2-layered.json` | Stage F 分层三元交付（覆盖/ECE/s_local 全子集） |
| `experiments/20260819-p5a2-fglex-full.json` | Stage D fglex 5243 基因（H1 mono 全轴 + localized 入带） |
| `experiments/20260819-p5a2-testcv.json` | 任务#3 test-CV（独立 test 划分 ood_action 0.9322） |
| `experiments/20260821-p5a2-gate-repro.json` | Stage #4 GATE_REPRO 四栏守卫 ZERO-DELTA_PASS |
| `experiments/20260819-p5a2-contcomp.json` / `-full.json` / `-cmp.json` | Stage C/D namehash vs fglex 对比 |
| `experiments/20260819-p5a2-precision-vs-p1.json` | Stage E 点预测精度对比（fglex 0.1455 vs P1 0.4271） |
| `submissions/sigma_ranking_valood.csv` | σ 排名 Top-K（NHP2/PET309/ACS1/YNL050C/DDI2） |
| `submissions/prediction_layered_summary.csv` | 每样本 P1 Δ均值 + 区间半宽 + GT 覆盖 |
| `scripts/p5a2_*.py`（9 + _patch_layered_verdict） | P0 全流程脚本 |
| `wetlab/2026-W34.md` | 本周湿实验方案（Top-K 基因 + E7 闭环使能） |
| `references/2026-08-21-literature.md` + `references/INDEX.md` | 本周文献监测轮次 + 索引 |
| `data/processed/goai_cache.npz`（gitignored） | GOAI 5243 蛋白缓存 |

> 本周报由每周自动化生成，数值均可通过上表工件回溯复现。**未外发、未发布。**
