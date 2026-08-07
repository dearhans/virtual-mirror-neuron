# 历年获奖「虚拟细胞 / 扰动响应预测」方案系统调研报告

> **调研目的**：为 GOAI 赛道三（虚拟细胞方向）初赛方案提供可行动的优化依据
> **调研范围**：8 个与本赛题最同构的竞赛 / 基准 / 方法族
> **产出定位**：本报告只保留「能改写我方方案的结论」，所有论断均附可核查出处
> **调研日期**：2026-08-07 ｜ 原始逐项结果：`results/item_1.yaml` ~ `results/item_8.yaml`

---

## 0. 我方赛题速览（作为全篇的对照锚点）

| 维度 | GOAI 赛道三设定 |
|---|---|
| 输入 | strain(6) × compound(55) × medium(2) × temp(2) × time(6)，**纯类别 ID，无 control 观测输入** |
| 输出 | 4232 维 log2 蛋白丰度向量，**~20% 缺失** |
| 数据量 | train 5920 / val 3038 / test 4454 |
| OOD 场景 | val_chem_only(1065)、val_strain_only(1547)、val_both(269)、val_time(157) |
| 评分 | 6 模块加权：fold-change 25% ｜ 绝对保真 20% ｜ 上下文残差 20% ｜ 药物残差 20% ｜ 双重未知+时间 10% ｜ DEP 检测 5% |

**结构判定**：这是一个「极小样本 × 万维输出 × 只有类别 ID 输入 × 目标侧大量缺失 × 无对照锚点」的任务。
下文所有迁移建议都以这五个特征为筛选条件。

---

## 1. 调研对象总览

| # | 对象 | 年份/主办 | 与我方同构点 | 迁移价值 |
|---|---|---|---|---|
| 1 | Open Problems – Single-Cell Perturbations | NeurIPS 2023 / Kaggle | 几百行样本 × 18211 维输出 × 只有 (cell_type, sm_name) 两个 ID | ★★★★★ |
| 2 | Arc Virtual Cell Challenge 2025 | Arc Institute / NeurIPS 2025 | 未见扰动外推 + 官方多指标加权 + 强官方基线 | ★★★★★ |
| 3 | Kaggle MoA Prediction (lish-moa) | 2020 / Broad-LISH | 对照样本的四种正交用法；分组 CV 决定胜负 | ★★★★ |
| 4 | OpenVaccine mRNA Degradation | 2020 / Stanford Das Lab | 高维输出 + 只有部分维度被评分 + 标签带已知噪声 | ★★★★ |
| 5 | Recursion CellSignal / RxRx1 | NeurIPS 2019 | 批次级 OOD；条件归一化 vs 对抗抹除的对照实验 | ★★★★ |
| 6 | 扰动 SOTA 族（CPA/chemCPA/GEARS/Biolord/SAMS-VAE/PerturBench） | 2022–2025 | **直接给出我方架构的天花板与陷阱** | ★★★★★ |
| 7 | DREAM SCSBrC + NCI-CPTAC | 2017–2021 | 细胞系×扰动×时间的蛋白响应预测，条件轴几乎一一对应 | ★★★★★ |
| 8 | 蛋白组缺失值 / 批次校正基准族 | 2019–2025 | **与转录组竞赛的最大差异面：MNAR 左删失** | ★★★★★ |

---

## 2. 三条决定性判据（会直接改写方案架构）

### 判据 A ——「深度模型未稳定超越简单线性基线」是当前领域共识

> Ahlmann-Eltze, Huber & Anders, *Nature Methods* 2025（DOI 10.1038/s41592-025-02772-6）

- scGPT / scFoundation / GEARS 等基础模型在未见扰动预测上**均未稳定超越简单线性基线**。
- 表现最好的配置是：**线性模型 + Replogle 预训练扰动 embedding**。
- 蛋白组侧完全同构：`limma` 跨 6 种数据设置稳居前 4（Nat Commun 2024, DOI 10.1038/s41467-024-47899-w）。

**对我方的含义**：主干必须是显式统计结构，深度模型只允许出现在残差位置，并且**必须在残差指标上单独证明增益**。
把这一点写进方案，本身就是评委认可的「方法论清醒度」，而不是保守。

### 判据 B —— 可加性既是天花板也是地板

> Systema, *Nature Biotechnology* 2025（DOI 10.1038/s41587-025-02777-8）

跨三个独立基准（Ahlmann-Eltze / Systema / PerturBench）的一致结论：**当前所有深度模型学到的基本就是可加的平均效应，非可加互作项几乎没学到**。

| 方法 | 得分 |
|---|---|
| perturbed mean（条件边际均值） | **0.70** |
| GEARS | 0.65 |
| scGPT | 0.62 |
| CPA | 0.02（未见扰动上灾难性失败） |

**对我方的含义**：
1. **主干 = 设计边际 ANOVA**（可加分解），这是地板，也几乎是天花板；
2. 深度残差模块的**门控初始化必须为 0**，让它从「等价于可加基线」出发，只在有证据时才偏离；
3. 若残差项在残差指标上无显著增益 → 诚实地报告并退回可加主干，这比强行堆模型得分更高。

### 判据 C —— Decoder(Cov) 崩溃陷阱：我方默认就在陷阱里 ⚠️

> PerturBench, arXiv:2408.10609

PerturBench 做了一个完美的反面实验：构造一个 **不输入任何扰动信息、只用协变量的 Decoder(Cov) 模型**。

| 指标类型 | Decoder(Cov) 表现 |
|---|---|
| RMSE / Cosine（拟合类） | **看起来很好** |
| `rank_average`（排序类） | **≈0.47**（0.5 = 完全随机） |

即：**模型完全没学到扰动，但拟合指标漂亮得可以骗过所有人**。

**对我方的含义（最关键的一条）**：我方赛制是 **decoder-only、无 control 输入**，与 Decoder(Cov) 的设定结构一致。
必须假设自己**默认处于这个陷阱中**，并主动设计检测手段：
- 自建内部指标：`rank_average`、`top-k DEP recall`、残差相关性 —— 而非只看 RMSE/R²；
- 强制加入 **B6 崩溃探针基线**（见 §3）：去掉 compound 输入重训，若主指标几乎不掉 → 崩溃确诊。

---

## 3. 强制基线清单（Phase 0，动模型之前必须全部跑完）

来源：item_6 的 transfer_action + item_7 的「最强基线是什么都不做」。

| 编号 | 基线 | 作用 | 判据来源 |
|---|---|---|---|
| **B1** | 全局均值（所有条件同一预测） | 绝对下限 | DREAM SC2 null RMSE=29.156，**16 队中仅 4 队打赢** |
| **B2** | 条件边际均值（perturbed mean） | **真正的强基线**，Systema 中 0.70 击败所有 DL | Systema 2025 |
| **B3** | 可加 ANOVA / matching mean | 我方主干的直接候选 | Systema / PerturBench |
| **B4** | Linear（岭回归 on one-hot） | Ahlmann-Eltze 的冠军配置 | Nat Methods 2025 |
| **B5** | Latent Additive | 深度残差的最简形态 | PerturBench |
| **B6** | **DecoderOnly（去掉 compound 输入）** | **崩溃探针**：若它分数接近正式模型，说明模型没学到药物 | PerturBench Decoder(Cov) |

> **纪律**：任何正式模型必须在**每一个 OOD 分层上**同时打赢 B2 与 B3，并且与 B6 拉开显著差距，才允许进入下一阶段。

---

## 4. 逐项可迁移动作

### 4.1 Open Problems 2023（item_1）—— 小样本高维的工程胜负手

**冠军方案**：LSTM/GRU/1D-CNN 异构集成。

| 赢法 | 迁移到我方 |
|---|---|
| **target encoding**：per-context / per-perturbation 的目标均值、分位数作为输入特征 | 多轴 masked target encoding（per-strain / per-compound / per-time），**同时把「有效观测数 n」作为特征**，让模型知道该统计量有多可信 |
| **ChemBERTa on SMILES** 是唯一成功的外部预训练 | compound 侧用 ChemBERTa / MolFormer / ECFP4 + 人工 MoA 标注；strain 侧用 ESM-2 / 基因型差异 |
| 复合 loss `0.32*MSE + 0.24*MAE + 0.24*LogCosh + 0.20*BCE` | BCE 项的价值在于**「目标≈0 处仍有梯度」**——我方 DEP 检测模块（5%）正好需要这个 |
| 30% 特征 dropout | 直接可用，对抗 target encoding 过拟合 |
| 按 OOD 轴分组 CV，宁选 CV 稳定不选 LB 最高 | 建 4 套本地 CV，对齐官方 4 个 OOD 场景 |

**⚠️ 负结果（OP3 官方复盘）**：误差随 DE 基因数上升 → **模型系统性低估强扰动**。
→ 我方须加 **anti-shrinkage 机制**：对预测幅度做分层校准，避免把强响应压平。

### 4.2 Arc VCC 2025（item_2）—— 把评测公式写进 loss

| 事实 | 数字 |
|---|---|
| 三甲得分 | 34.0 / 32.8 / 32.6 |
| 第 3 名 TransPert | **零神经网络**，只落后冠军 1.4 分 |
| Arc 方 Goodarzi 评论 | 「纯端到端 NN 未赢过 hybrid，scaling 已失效」 |

**核心迁移动作 1：把官方加权公式逐项写进损失函数**

```
L = 0.25·L_fc + 0.20·L_abs + 0.20·L_ctx + 0.20·L_drug + 0.10·L_double + 0.05·L_DEP
```
- `L_fc` 用 soft-Spearman（可微排序近似）；
- `L_DEP` 用 BCE；
- **所有项都必须 mask-aware**（分母按有效观测数归一化，不是按维度数）。

**核心迁移动作 2：CV 选出的全局/分组线性缩放因子做后处理校准**（`ŷ ← α·ŷ`）。
这一招与架构无关，但在 Arc 上是三甲共有的操作。

**核心迁移动作 3**：`ŷ = f_ctx + g_drug` 的 residual/delta 学习结构；纯统计版作为保底下限锁分。

### 4.3 MoA 2020（item_3）—— 对照样本的四种正交用法 + CV 才是护城河

对照（vehicle）样本有 **四种互不冲突的用法**：

| 用法 | 说明 | 我方对应 |
|---|---|---|
| (a) 推理时硬置零 | 确定性增益 | 我方无 control → 用**伪对照锚点**替代 |
| (b) 保留在训练集作零效应锚点 | 让模型学到 baseline | 构造 pseudo-basal（跨 compound 中位数） |
| (c) **`x + ctl1 − ctl2` 噪声增强** | 3rd 名的胜负手，对所有模型有效 | **用同 strain/medium/temp 下不同 batch 的差作为噪声算子做增强** |
| (d) ctl-vs-trt KS 检验筛特征 | 特征选择 | 用于筛「真正响应的蛋白子集」 |

**其它硬结论**：
- 头部方案差距仅 ~2e-4 → **护城河是 drug_id 分组 CV，不是架构**。我方对应 **GroupKFold by compound**。
- 被「惩罚过度自信」的指标评分时，**一切保守化都是净收益**：clipping、多 seed 平均、多模型平均。
- 融合权重按**与其它模型的预测相关性**分配，而非按单模型分数（1st 明确做法）。

### 4.4 OpenVaccine 2020（item_4）—— 三层加权 mask-aware loss

这是「部分维度被评分 + 标签带噪声 + 大量位置无标签」的最佳实践模板，与我方 20% 缺失结构高度同构。

**三层权重正交叠加（不是三选一）**：

| 层 | 区分什么 | OpenVaccine 做法 | 我方对应 |
|---|---|---|---|
| 列权重 | 「被评分」vs「仅有标签」 | `[0.3, 0.3, 0.3, 0.05, 0.05]` | 按 6 模块权重给蛋白/指标加权 |
| 元素级 mask | 「可信」vs「不可信」 | NaN mask + `*_error` 阈值 | 缺失位置 mask，**分母必须按 mask 归一化** |
| 样本权重 | 「冗余」vs「独特」 | `1/sqrt(簇大小)` | 按 compound/strain 出现频次降权高频条件 |

**其它**：
- 最大单项增益（80 bps）来自**伪标签**，不是架构 → 值得排一个实验位；
- 1st 名原话「My ideas are mainly on the data side」→ **数据侧 > 架构侧**；
- 本赛是 DL 战胜线性基线的**正例**，但胜因可辨识：输入含 RNA 二级结构这一强非线性结构先验。
  → 提供了「什么时候值得上复杂模型」的判据：**先确认存在结构先验，再上 DL**。引用 Ahlmann-Eltze 时须带此限定。

### 4.5 RxRx1 / CellSignal（item_5）—— 条件归一化 > 对抗抹除

| 操作 | 效果 |
|---|---|
| **AdaBN[domain=plate]**（在最细可行域上重估 BN 统计） | val 准确率 **40% → 70%**，单招 |
| 实验设计硬约束 + 全局 LSA 指派 | 0.926 → 0.983，**收益大于任何 backbone 升级** |
| 通道级随机 gain-bias 增广 `a~N(1,.1), b~N(0,.1)` | 把批次不变性写进训练目标，比对抗训练稳定 |

**最重要的一条**：赢家用**条件归一化（把批次显式建模为条件变量）**，而不是对抗抹除。这与 item_6 的「去掉对抗判别器反而更好」完全同向。

**⚠️ 风险提示**：RxRx1 每板固定 30 个 control 提供批次锚点，主办方称这是「整个赛题可解的前提」。
**我方无 control = 失去这个锚点 = 难度质变**。必须用伪对照锚点 + missingness pattern 作特征来部分补偿，并在方案中诚实标注这一风险。

**CV 纪律**：leave-batch-out + GroupKFold；把 **「val-LB 一致性」当作流程正确性的核心 KPI**。

### 4.6 扰动 SOTA 族（item_6）—— 失败清单最有价值

| 被验证失败的想法 | 证据 |
|---|---|
| ★ 对抗式解耦（CPA adversarial classifier） | **消融后变好** |
| ★ 稀疏机制假设（SAMS-VAE 二值掩码） | **消融后变好** |
| ★ 单细胞基础模型 embedding（scGPT/scFoundation/Geneformer/UCE/scBERT） | 几乎无增益 |
| ★★ mode/posterior collapse 被拟合类指标完美掩盖 | 见判据 C |
| ★★ systematic variation 让所有方法分数被系统性高估 | Systema |
| GEARS 在真正 unseen-gene OOD 上不敌均值基线 | scPerturBench (s41592-025-02980-0) |
| 复杂模型不随数据规模变好，简单模型才会 | Ahlmann-Eltze |
| 指标选择本身就是失败源 | PerturBench metric_pitfalls |

**推荐架构（两段式，item_6 transfer_action 原文精神）**：

```
ŷ = B(设计边际 ANOVA)  +  R_θ(Latent Additive 残差, 门控初始化=0)
```

- **compound 双通道 embedding**：ID embedding（拟合已见 38 个）+ 结构/靶点 embedding（泛化未见 11 个），残差式叠加。
  依据：chemCPA 教训 + CPA 在未见扰动上 0.02 的灾难 → **化合物泛化只能靠结构**。
- **有序条件当有序变量建模**：time(6)、temp(2) 用 Biolord 式有序潜变量 / CPA 式连续缩放，**不要 one-hot**，否则时间维会出现不合理的非单调抖动。
- **HPO 目标**：`主指标 + λ·rank_average`，强制模型同时关注排序。
- **汇报范式**：采用 Ahlmann-Eltze 的「bootstrapped mean ratio vs baseline + 95% CI，CI 跨 0 就淡化结论」。这是评委最认的诚实呈现方式。

### 4.7 DREAM SCSBrC + CPTAC（item_7）—— 条件轴一一对应的最近参照

| 发现 | 数字 | 迁移 |
|---|---|---|
| 最强基线是「什么都不做」 | SC2 null RMSE=29.156，**16 队仅 4 队打赢** | Phase 0 必跑 null model |
| basal 平移校正 | 单招降误差 **10.4%** | 平移校准层 + pseudo-basal anchor |
| **条件内插 >> 个体外推** | 前三名都用「同细胞系的其它条件」推，而非「其它细胞系的同条件」 | **直接预测我方 val_strain_only(1547) 难于 val_chem_only(1065)** |
| 冠军架构 | ElasticNet + ExtraTrees + RF + LightGBM + linear 的**简单平均** | 异质 ensemble，线性模型是骨干而非陪衬 |
| 统计判定 | bootstrap-over-conditions + Bayes factor | 选型决策用统计而非单点分数 |
| 误差异质性 | cell line / marker 上高度非均匀（P<2.2e-16），treatment / time 上均匀（P=0.67/0.68） | **难度来自「个体」而非「扰动」** → 与镜像神经元的跨主体轴发现互证 |
| 跨癌种 joint learning | 共享通路参数联合训练 breast+ovarian，两边同时提升 | 支持跨 strain 共享 + strain-specific 残差的结构 |

### 4.8 蛋白组缺失 / 批次（item_8）—— 与转录组竞赛的根本差异面

**这是我方最容易做出真创新的地方，因为绝大多数虚拟细胞方案都是转录组出身，不处理 MNAR。**

| 缺失率区间 | 推荐策略 |
|---|---|
| TMT ~0.2% | 不填补最好 |
| DIA ~3% | MinDet / Impseq |
| **DDA ~17%** | MinProb / missForest —— **我方 ~20% 落在此区间：填补收益为正但风险最高** |

**核心机制判定：蛋白组缺失是 MNAR（左删失），不是 MCAR/MAR。**
- 诊断方法极简：画「per-protein 检出频率 vs 平均强度」散点，若正相关即 MNAR。
  原文：*"In proteomics this correlation is almost always strong."*
- **后果**：observed-only 训练集在丰度上是有偏样本，**直接 masked MSE 会让模型系统性高估低丰度蛋白**。

**→ 由此导出我方最强创新点候选：MNAR censored hinge loss**

对缺失位置施加**单边惩罚**：预测值若高于检测限（LOD）则罚，低于 LOD 不罚。
这把「这个蛋白在这个条件下低于检测限」从「损失」变成「可利用的监督信号」。
依据：MANNERS 证明 99% 缺失下 mask-aware 优于任何纯填补；SHIFT 证明把不完整队列纳入训练能提升完整数据上的性能。

**log2 相关的具体坑（务必写进方案的数据处理章节）**：
1. 必须在 **log2 之后**填补，不能在原始强度上填补（MS 强度近似 log-normal）；
2. **不要加伪计数**——对 log2 是灾难，伪计数大小直接决定低值区的 fold change；
3. **单侧检出蛋白（一组全有一组全无）不要报 fold change**——min-shift 填补会给出人为压低的值，产生虚高 log2FC 与假阳性 DEP，应作定性结果单列；
4. 填补后的低值是「人造的」，任何以 log2FC 排序的下游都会优先富集这些人造条目。

**严禁 mean imputation。** 批次校正在 protein level 做最稳健（Nat Commun 2025, DOI 10.1038/s41467-025-64718-y）；简单方法（Median centering / Ratio / linear regression / LOESS）反复胜出。

**ensemble 是本领域一贯答案**：DreamAI = 3 个获奖法 + 3 个 baseline（KNN/missForest/ADMIN）平均 + bagging，在低丰度蛋白上比现有工具 Pearson 相关高 15%–50%。理由是**没有任何单一 MVI 算法在跨设置时排名稳定**。

---

## 5. 虚拟镜像神经元资产的接入映射（第 6 创新点）

本项目已有的方法学资产与本次调研结论**高度互证**，可作为方案的差异化创新点完整接入。

| 镜像神经元资产 | 本次调研的独立佐证 | 在方案中的角色 |
|---|---|---|
| **五轴 OOD taxonomy** | Ahlmann-Eltze / scPerturBench 均指出「指标选择本身就是失败源」，OOD 分层是必需 | 把官方 4 个 val 场景升级为带几何刻画的 OOD 分层评测协议 |
| **跨轴几何探针（共享 PCA + 主角度）**：id↔ood_batch≈13°、id↔ood_agent≈23°、id↔ood_action≈49° | DREAM 误差异质性分析：难度来自「个体」而非「扰动」（cell line P<2.2e-16 vs treatment P=0.67） | **赛前难度预判工具**：预测 val_strain_only 与 val_chem_only 的相对难度，指导算力与建模重心分配 |
| **bootstrap CI + KNN 显著性判定** | item_6 明确推荐 Ahlmann-Eltze 的「bootstrapped mean ratio + 95% CI，CI 跨 0 就淡化」；item_7 用 bootstrap-over-conditions + Bayes factor 选型 | 直接采用为方案的统计汇报规范 |
| **「对齐 ≠ 预测优势」诚实负结果** | item_6 三大失败清单（对抗解耦 / 稀疏掩码 / 基础模型 embedding 全部消融后更好） | **指导初赛不押注预训练 embedding**，改用显式残差分解 + 统计先验 |

**轴的同构映射**（母体项目 → 本赛题）：

| 镜像神经元轴 | 几何 | GOAI 对应场景 | 预判难度 |
|---|---|---|---|
| `ood_batch`（跨批次） | ≈13°，强对齐 | 仪器/板批次效应 | 低 —— 可迁移 |
| `ood_agent`（跨主体） | ≈23°，可迁移 | `val_strain_only`(1547) | 中 |
| `ood_action`（未见动作） | ≈49°，近正交 | `val_chem_only`(1065) | **高 —— 结构缺失** |
| 两轴交叉 | — | `val_both`(269) | **最高 —— 试金石** |

> **不确定性标注**：上述几何角度来自人单细胞转录组（Replogle/Norman）的实测，迁移到酵母蛋白组尚未验证。
> 方案中应标注为「**待验证先验**」，并在初赛阶段用我方数据重算一遍主角度作为验证动作，而非直接当结论使用。

---

## 6. 对方案的最终修订建议（按优先级）

| 优先级 | 修订项 | 落点章节 |
|---|---|---|
| **P0** | 加入 B1~B6 强制基线清单，特别是 **B6 DecoderOnly 崩溃探针** | §4.2 基线对标 |
| **P0** | 架构改为两段式：**ANOVA 主干 + 门控初始化为 0 的深度残差** | §3.1.2 架构 |
| **P0** | 内部指标增加 `rank_average` / top-k DEP recall，**不能只看 RMSE/R²** | §3.1.4 稳定性 |
| **P0** | 新增创新点：**MNAR censored hinge loss** | §3.1.5 |
| **P1** | 官方 6 模块加权公式**逐项写进 loss**，全部 mask-aware | §3.1.5 |
| **P1** | compound **双通道 embedding**（ID + 结构/靶点，残差叠加） | §3.1.3 特征 |
| **P1** | time/temp 改**有序潜变量**建模，禁用 one-hot | §3.1.3 |
| **P1** | 三层加权 mask-aware loss（列权重 / 元素 mask / 样本权重）正交叠加 | §3.1.5 |
| **P1** | CV 改 **GroupKFold by compound** + 4 套对齐官方 OOD 场景 | §3.2 路线 |
| **P2** | 条件归一化（FiLM/AdaBN）校准分支替代对抗抹除 | §3.1.2 |
| **P2** | `ctl1 − ctl2` 式噪声增强（用 batch 差构造） | §3.2 |
| **P2** | anti-shrinkage 分层幅度校准（对抗 OP3 的系统性低估强扰动） | §3.1.5 |
| **P2** | 镜像神经元几何探针作为**第 6 创新点**接入，标注「待验证先验」 | §3.1.5 + §2.2 |
| **P2** | bootstrap CI + Bayes factor 作为选型与汇报规范 | §4.2 |

---

## 7. 开放待确认项（不阻塞修订，需在赛题数据/规则发布后核实）

| # | 待确认 | 影响 | 处理方式 |
|---|---|---|---|
| 1 | 官方 6 模块指标的**确切数学定义**（尤其 fold-change 与 DEP 检测） | 决定 loss 各项的具体形式 | 先按最可能定义实现，留参数化接口 |
| 2 | 我方 ~20% 缺失是 **MNAR** 还是**设计缺口 MAR**（即某些条件根本没测） | 决定 censored hinge 是否成立 | 用「检出频率 vs 平均强度」散点在真实数据上诊断，前置为 Phase 0 动作 |
| 3 | 官方 ground truth 在**缺失位置是否计分** | 决定 mask 归一化的分母口径 | 若计分则需预测缺失位置，censored hinge 更重要；若不计分则纯 mask-aware |
| 4 | 是否允许使用外部预训练化学表征（ChemBERTa 等） | 决定 compound 结构通道的实现 | 查赛规；若禁则退回 ECFP4 + 人工 MoA |

---

## 8. 一句话总结

> **这不是一个靠架构取胜的赛题，而是靠「正确的分解 + 正确的损失 + 正确的 CV + 正确的缺失机制处理」取胜的赛题。**
> 八个独立来源指向同一结论：显式可加分解做主干、深度模型只做门控残差、把评测公式写进 loss、按 compound 分组 CV、
> 全流程 mask-aware 不填补，并且**主动检测自己是否掉进了 Decoder(Cov) 崩溃陷阱**。
> 我方最可能的差异化优势不在模型规模，而在两处：**MNAR censored loss**（蛋白组特有，转录组出身的方案不会做）
> 与**跨轴几何诊断**（镜像神经元资产，可在建模前预判 OOD 难度排序）。

---

## 附录：可引用出处

| 来源 | 标识 |
|---|---|
| Ahlmann-Eltze, Huber & Anders, *Nature Methods* 2025 | DOI 10.1038/s41592-025-02772-6 |
| Systema, *Nature Biotechnology* 2025 | DOI 10.1038/s41587-025-02777-8 |
| scPerturBench, *Nature Methods* 2025 | DOI 10.1038/s41592-025-02980-0 |
| PerturBench (GSK.ai) | arXiv:2408.10609 |
| GEARS — Roohani, Huang & Leskovec | *Nature Biotechnology* 42:927-935 (2024) |
| CPA — Lotfollahi et al. | *Molecular Systems Biology* 2023 |
| chemCPA — Hetzel et al. | NeurIPS 2022 |
| Biolord — Piran et al. | *Nature Biotechnology* 2024 |
| SAMS-VAE — Bereket & Karaletsos | NeurIPS 2023 |
| OP3 benchmark — Szałata et al. | NeurIPS 2024 D&B Track |
| OpenVaccine 复盘 — Wayment-Steele et al. | *NAR* 2022 |
| SCSBrC DREAM — Gabor, Tognetti, Driessen, Tanevski et al. | *Mol Syst Biol* 17(10):e10402, PMCID PMC8522707 |
| CPTAC DREAM 冠军 — Li H, Guan Y et al. | *BMC Biology* 17:107 (2019) |
| DreamAI — Chowdhury, Ma, ... Saez-Rodriguez, Wang | NCI-CPTAC DREAM SC3, Synapse syn8228304 |
| 蛋白组 DEA workflow 基准 | *Nat Commun* 2024, DOI 10.1038/s41467-024-47899-w |
| 蛋白 level 批次校正基准 | *Nat Commun* 2025, DOI 10.1038/s41467-025-64718-y |
| Arc Virtual Cell Challenge 2025 | virtualcellchallenge.org；赛题介绍发表于 *Cell* |
| RxRx1 / CellSignal | NeurIPS 2019 Competition Track；rxrx.ai/rxrx1 |
| Kaggle MoA Prediction | kaggle.com/c/lish-moa |
| Kaggle OpenVaccine | kaggle.com/c/stanford-covid-vaccine |

> 逐项详细证据（含具体 URL、分数、消融表）见 `results/item_1.yaml` ~ `results/item_8.yaml` 的 `证据.sources` 字段。
