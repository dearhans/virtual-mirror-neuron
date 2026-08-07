# 虚拟镜像神经元 · 每周文献监测（2026-08-07）

> 增量周报。去重基准：`2026-07-31-literature.md`、`2026-08-01-literature.md`、`2026-08-02-literature.md`（已逐条比对标题/作者/DOI，重复项不再收录）。
> 本期新收录 **21 篇**。检索轮次：6 轮 Web 检索 × 8 个方向（virtual cell / mirror neuron / perturb-seq / OOD / CRL / calcium imaging / optogenetics / neuromodulator）。
> ⚠️ 部分条目来自预印本与新闻聚合源，正式引用前须核对原始 DOI 与版本号。

---

## 0. 本期最高优先级（TL;DR）

| 排序 | 论文 | 为什么对本项目是"必读" |
|---|---|---|
| ★★★ | **Empirical Comparison of Virtual Cell Models（Baseline Gap）** | 独立验证了我们 P1/P2 的核心结论，并给出反向线索：深度模型的可复现增益**恰恰在组合（非加和）扰动**上——直接指向我们把 ood_action 判定为"无判别力"可能是**基准设计而非方法**问题 |
| ★★★ | **Causal Delta Embeddings（CDE, ICLR 2026）** | 把"干预"显式建模为潜空间**差向量** δ = φ(x̃) − φ(x)，并用独立性/稀疏性/不变性三约束正则；组合 OOD（未见 物体×动作 组合）正是我们 ood_action 的同构问题 |
| ★★☆ | **PRESCRIBE（多元深度证据回归）** | 未见基因扰动的 **epistemic + aleatoric 联合不确定度**，pseudo E-distance 置信分与实际误差强相关 → 可直接补齐我们"不确定度强制"这条硬约束的模型内生版本 |
| ★★☆ | **Plausibility Is Not Prediction + CORE（Mila / Jian Tang）** | "可解释性 ≠ 预测力"的严格证伪；CORE 把预测重构为**跨相关扰动的对比任务** → 与我们"记忆 vs 机制泛化双栏"是同一诊断哲学的两种实现 |
| ★★☆ | **Joint analysis of multiply perturbed cells（sprinter + PerturbMatch）** | 不丢弃多重扰动细胞、用多元交互筛选统计量直接估 epistasis → 我们 P2 交互头缺的正是**交互信号的统计功效** |

---

## 1. Virtual Cell（虚拟细胞 / 扰动响应预测）

### 1.1 An Empirical Comparison of Virtual Cell Models: Perturbation Prediction, Representation, and the Baseline Gap
- **作者**：（Research Square 预印本，作者名单未在检索页完整披露）
- **年份/来源**：2026，Research Square 预印本 `rs-10434123/v1`（CC BY 4.0，未同行评审）
- **核心方法**：统一基准 —— 11 个方法 × 5 个模型族（专用扰动预测器 GEARS / CPA / chemCPA / scGen；单细胞基础模型嵌入 Geneformer / scGPT / scFoundation / UCE；集成型 State）× 6 类任务 × 9 个公开数据集，**固定数据切分、预处理与指标**，并刻意配置强简单基线。
- **三条结论**：
  1. **未见单基因扰动上无方法能显著击败调好的加性/线性基线**：最佳模型 State 相对 cell-mean 基线改进 26%，线性基线 19%，GEARS 22%；多个基础模型预测器**低于线性基线**。
  2. **"成功"高度依赖指标**：全基因 Pearson 相关被"预测无变化"的模型最大化，几乎无法区分好坏预测器；DE 指标与判别性指标结论完全不同。
  3. **深度模型确有可复现增益的三块阵地**：**组合（非加和）扰动**、**跨 context 迁移到未见细胞类型**、以及**表征任务**（注释/整合）。
  - 作者定性：当前领域"更接近一台有用的**虚拟显微镜**，而非虚拟模拟器"；基础模型编码的是**有组织的相关性结构**而非因果调控逻辑。
- **与本项目相关点（★★★，最高）**：
  - **正向互证**：我们 Norman 真实基准上 CompositionalTwin P1=0.636 / P2=0.632 与线性 0.627 CI 重叠，与该文"单基因未见扰动无人能显著超线性"完全一致 → 我们的 SOP 阶段 D 诊断是**领域级共识**而非本地 bug。
  - **反向警示（关键）**：该文明确指出深度模型的增益**恰在组合非加和扰动上**。而我们 P2 结论是"ood_action（双扰动 held-out）在公平基准下所有模型 ~0.63、无判别力"。二者张力 → 需回查我们的 ood_action **指标选择**是否落入该文批评的第 2 条陷阱（RMSE ≈ 全基因平均误差，与全基因 Pearson 同族，可能被"预测均值"策略最大化）。**行动项：在 ood_action 上补 DE-gene 子集指标与判别性指标（如 top-K DE 重合率、E-distance），再重判 P2 是否真的"无判别力"。**
  - **指标 SOP 升级候选**：把"全基因 RMSE"从主指标降级为辅指标，主指标改为 DE 导向 + 判别性指标双报。
- **待核**：预印本，需核对最终作者与数据切分脚本是否公开。

### 1.2 UniPert-G2CP bridges genetic and chemical screens from molecular representation to phenotype modeling
- **作者**：李一鸣（第一作者）、李敏（中南大学）、姚建华、杨帆（腾讯 AI for Life Sciences Lab）
- **年份/来源**：2026，**Cell**，DOI `10.1016/j.cell.2026.06.005`
- **核心方法**：两阶段深度学习框架。
  - **UniPert**：融合大分子（蛋白/核酸序列）与小分子（原子连接图）多模态信息，用**对比学习 + GNN + 序列比对**构建**遗传扰动与化学扰动的统一表征空间**，回答"施加了什么干预、不同干预在机制上有多相似"。
  - **G2CP**：在大规模基因扰动数据上预训练学习生物学知识，再用化学扰动数据微调做**跨域迁移**，回答"这种干预会让特定细胞发生什么变化"，保留细胞背景异质性。
  - 评估：多个大规模双域扰动数据集；**未见小分子**条件下准确预测表型响应；联合表征空间支持药物 MoA 解析与靶点关联发现。
- **与本项目相关点（★★☆）**：
  - **扰动统一表征 = 我们 φ(p) 的成熟工业版**。我们当前 φ 只吃基因 one-hot；UniPert 提供"把 GOAI 化学扰动（compound）与遗传扰动（strain）放进同一坐标系"的现成范式 —— **GOAI 赛道的适配层可直接借鉴**（GOAI 真实列含 `perturbation_no_concentration` 化合物 + `Strains` 菌株，正是双域）。
  - **跨域迁移 = OOD 的另一个轴**：他们的"未见小分子"对应我们的 ood_action；"不同细胞背景"对应 ood_agent。可作为 GOAI 提交方案文档中"科学意义(30分)"的对标引用。
  - **注意**：论文未强调不确定度，我们若引用需自行补 CI/校准。

### 1.3 VCWorld — 白箱细胞模拟器（结构化生物知识 + LLM 迭代推理）
- **作者**：Wei et al.
- **年份/来源**：2026，**ICLR 2026 接收**
- **核心方法**：把"虚拟细胞"重构为**生物世界模型**——不是直接吐出差异表达向量，而是**复现扰动诱导的信号级联**，产出可解释的**逐步推理**与显式**机制假设**。声称数据高效（data-efficient），在药物扰动基准上 SOTA，推断出的通路与已知生物学吻合。
- **与本项目相关点（★★）**：
  - 与我们"机制先验、拒绝纯黑箱"的立场同盟；"信号级联复现"可类比我们多尺度孪生的"离子通道→突触→环路"逐层传播。
  - **但必须与 §1.5 合读**：VCWorld 正是 §1.5 的主要批评对象。

### 1.4 AROMA — 知识图谱 grounding 的遗传扰动预测
- **作者**：Wang et al.
- **年份/来源**：2026，**ACL 2026 Findings**（权重与代码已开源）
- **核心方法**：构建两个生物知识图谱（gene–gene 关联 + pathway 结构）+ 大型推理数据集；整合**文本证据、图拓扑信息、蛋白序列特征**建模"扰动–靶点"依赖；两阶段训练配方，目标是"既准确又可解释"；报告在**未见细胞系上 zero-shot** 成立。
- **与本项目相关点（★★）**：
  - "图拓扑 + 序列 + 文本"三模态融合 ≈ 我们因果图上 `do-` 干预的知识先验注入方式。
  - **未见细胞系 zero-shot = ood_agent 的直接对标**，可作为我们 ood_agent 子集的外部参照方法。
  - 开源权重 → 可作为 canonical 基准里的第 8 个预测器候选（若算力允许）。

### 1.5 Plausibility Is Not Prediction: Contrastive Evidence for LLM-Based Cellular Perturbation Reasoning
- **作者**：Xinyu Yuan, Xixian Liu, Jianan Zhao, Yashi Zhang, Hongyu Guo, **Jian Tang**（Mila / UdeM / HEC / U Ottawa / NRC / CIFAR）
- **年份/来源**：2026-05-31 提交，arXiv `2606.01042`
- **核心方法（诊断 + 修复）**：
  - **诊断**：LLM 型虚拟细胞（VCWorld / PerturbQA / CellVerse）虽产出生物学上"讲得通"的解释，却**捕捉不到扰动特异性效应**——系统性高估差异表达（yes-bias）、在聚合指标上常**输给最朴素的 gene-frequency 基线**、在 per-gene 层面**坍缩到随机水平**。根因：模型学到的是"哪些基因倾向于动"（intrinsic gene response tendency），而不是"这个扰动到底做了什么"。
  - **失败机理归因**：现有方法**孤立地**评估每个 (扰动, 基因) 对，从未暴露"相关扰动在同一基因上的效应差异"，因此缺乏区分扰动特异性的**对比信号**。
  - **修复 CORE（Contrastive Organization of Relational Evidence）**：用生物医学 KG（ReasonKG，融合 5 个资源）检索**相关扰动**，把它们的结果组织成**正/负证据对**，把预测从"孤立判断"重构为"比较任务"。结果：药物扰动上 CORE-Reasoning 使 Qwen3.5-9B 聚合指标提升最多 **28.6%**；通用扰动上 CORE-Voting 把 macro-per-gene AUROC 从随机水平提到 **0.703**（4 个细胞系均值），且**校准度改善**。
- **与本项目相关点（★★☆）**：
  - **方法论同构**："intrinsic gene response tendency" ≡ 我们说的**记忆（in-distribution）**；"perturbation-specific effect" ≡ **机制泛化（OOD）**。该文用 per-gene AUROC 坍缩到 chance 来揭示这个 gap，**正是我们双栏报告的另一种实现**——而且比 RMSE 双栏更锋利（RMSE 会被"预测均值"策略掩盖，AUROC 不会）。
  - **可直接移植的指标**：把 **macro-per-gene AUROC** 加进我们的 canonical 评测表，作为"是否真的学到扰动特异性"的判别指标。这可能正是 §1.1 提醒我们缺的那类"判别性指标"。
  - **CORE 的对比组织 → P3 候选**：我们的 CompositionalTwin ψ 交互头目前只吃单基因效应；若改成"对比相关基因对的响应差"来监督，可能把 P2 那 Δ=−0.005 的微弱增益放大。
  - **校准**：CORE 明确报告 calibration 改善，与我们"不确定度强制"约束对齐。

### 1.6 Response Magnitude as a Dominant Signal for Held-Out CRISPRi Perturbation Effect Prediction
- **作者**：Mehrdad Shoeibi, Niloofar Yousefi
- **年份/来源**：2026-07-31 提交，arXiv `2608.00152`（cs.LG）
- **核心方法**：在 **Virtual Cell Challenge (VCC) 基准**上做严格 held-out 目标基因切分，追问"简单基线为何常赢"。目标量是相对非靶向对照的 log Anderson–Darling 距离。发现：该目标可由 2000 维输入的**4 个确定性标量函数**强预测；深度 MLP 编码器（有完整输入访问权）反而**坍缩到训练边际均值**，常规补救手段无效；**仅用 4 个 magnitude 标量的线性回归就超过最强的 x-only 经典模型**，而"输入 + 4 标量"的随机森林大幅超过深度编码器。两个预注册对照把增益归因于 **per-row alignment** 而非维度增加。零样本迁移到两个外部 CRISPRi 屏幕时：**magnitude-only 预测器正迁移，expression-only 预测器负迁移或不显著**。另外指出该屏幕自带的 Anderson–Darling 列度量的是**转录组范围的响应广度**，而非靶基因效应强度 —— 拿它评迁移是在评另一个结果。
- **与本项目相关点（★★☆）**：
  - **"深度模型坍缩到训练均值"是我们 P1 前 CompositionalTwin 崩塌（RMSE 2.025）的镜像现象的反面**：他们是坍缩到均值（过保守），我们是外推失真（过激进）。两者同源于**动作轴/幅度轴处理不当**。
  - **强烈建议的诊断动作**：在我们的 Norman 基准上做同款"标量幅度特征"消融 —— 若 `ood_action` 的 RMSE 也主要由**响应幅度**这个低维信号解释，那么"所有模型 ~0.63"就不是"epistasis 信号小"，而是**指标本身只在测幅度**。这是对 P2 结论的一个可证伪检验，成本很低。
  - **基准卫生警示**：数据集自带的评价列可能测的不是你以为的量 → 我们 GOAI 适配层 COLUMN_MAP 对齐时须同款警惕。

---

## 2. Mirror Neuron（镜像神经元）

### 2.1 【增补】Chatzimichail et al., Science Advances 2026 —— 执行↔观察泛化的**不对称性**
- **来源**：Science Advances 12(25), 2026-06-19, DOI `10.1126/sciadv.aed9309`（论文本体已在 2026-07-31 期收录，本次为**新增细节**，非重复条目）
- **新增关键事实（来自作者 Raos 的访谈与全文）**：
  - 群体几何对齐**最强出现在 movement 与 hold 阶段**（动作物理上最复杂的时段），而非起始或结束。
  - **不对称性（最重要）**：用**观察**期神经活动训练的统计模型，能成功预测猴子**执行**时的物理运动；反过来，用**执行**期训练的模型**泛化不好**到预测被观察的人类动作。→ 执行涉及超出"共享部分"的额外特异性神经过程。
  - 神经↔运动学是**双向**可预测的（神经→手部运动、运动→神经），且该关系**跨神经元群体、跨 agent 泛化**。
  - non-MirN 也强烈跟踪运动学 → 连续追踪运动结构是前运动皮层的**广义属性**，不是镜像神经元专利。
  - 作者自陈局限：分 session 记录后数学合并、物体身份与抓握构型未完全解耦。
- **与本项目相关点（★★★，对建模层是直接约束）**：
  - **不对称性推翻"对称共享潜空间"的朴素设定**。我们的核心抽象是"感知与动作在共享潜空间的对称性"——该文证据要求把它改成**部分重叠 + 有向不对称**：`z_obs ⊂ z_exec`（观察子空间是执行子空间的真子集），执行额外携带 `z_exec_only`。
  - **建模层可执行改动**：把镜像模块从"共享 encoder + 对称约束"改为"共享核心 + 执行侧私有残差通道"，并在损失里**只对 obs→exec 方向施加强对齐约束**，exec→obs 方向放松。这是一个可证伪的架构假设：若真实数据也呈现同款不对称，则该架构在 ood_agent（跨主体）上应优于对称版本。
  - **时间维度**：对齐强度随动作阶段变化 → 潜空间对称性应是**时变**的（phase-conditioned），而非常数。呼应我们"条件/时间分离"的 TRIZ 破法。

### 2.2 Motor resonance and inhibitory mechanisms in action observation as revealed by corticospinal excitability
- **作者**：Carlos Nieto-Doval, Aynur Ragimova, Gleb Perevoznyuk, Traian Popa, Oleg Shevtsov, Victoria Moiseeva, Matteo Feurra
- **年份/来源**：2025，**Scientific Reports**，DOI `10.1038/s41598-025-03989-3`
- **核心方法**：spTMS + MEP（FDI / ADM 两块肌肉），对比**静态照片 / 完整视频 / 视频后期（Postvideo）** 三种呈现条件，TMS 施加于运动起始（或结束）后 0 / 320 / 640 ms。
- **关键结果**：**Postvideo 条件产生最强 MEP 调制** —— 非匹配肌肉**抑制**、匹配肌肉**易化**；Photo 与 Video 条件则表现为随时间递减的皮层兴奋性（尤其非匹配肌）。作者归因于 **motor surround inhibition（运动周边抑制）**。
- **与本项目相关点（★★）**：
  - **镜像不是纯激活，而是"激活 + 侧向抑制"的对比增强结构**。我们的共享潜空间目前只建模"对齐"，缺**竞争性抑制**项。
  - **可移植先验**：在动作解码头上加 **winner-take-most / surround-inhibition 正则**（非匹配动作维度被主动压低），这在 ood_action 上应表现为"预测更锐利、不坍缩到均值"—— 恰好对治 §1.6 的均值坍缩病。
  - **时间窗结论**：动作**结束后**才是镜像信号最清晰的时刻 → 湿实验（钙成像）读出窗口应包含 offset 后 0–640 ms，而不只是 onset 对齐。**直接可写进下周湿实验方案的采样窗设计。**

### 2.3 Synchronizing Minds through Collective Predictive Coding: A Computational Model of Parent–Infant Homeostatic Co-Regulation
- **作者**：Yushi Tsubamoto, Takato Horii
- **年份/来源**：2026-05-08（v1）/ 2026-05-25（v3），arXiv `2605.07524`（cs.MA）
- **核心方法**：把 **POMDP 主动内感受推断（active interoceptive inference）** 与源自**集体预测编码（CPC）假设**的 **Metropolis–Hastings Naming Game (MHNG)** 结合。两个 agent 知识**非对称**：家长只能通过外感受信号观察婴儿、知道"动作如何改变内脏状态"但要学"婴儿在传达什么"；婴儿直接感知自身内感受状态、但要学"动作如何影响它"。二者通过一个**共享通信变量**达成调节动作，接受与否由**局部可计算的 M-H 概率**决定。在 6×6 内脏状态网格世界中，MHNG 中介的交互比单侧控制更适应性地调节了婴儿状态，两个后验**迅速对齐**。
- **最关键的发现**：**潜状态对齐远早于学到的生成矩阵收敛** —— 即"表征同步不以完全共享世界模型为前提"。
- **与本项目相关点（★★☆）**：
  - **直接击中我们的核心抽象**："自我/他者同源"不需要两侧共享同一个生成模型，只需要**潜状态在通信变量上对齐**。这为 §2.1 的"不对称性"提供了计算解释：执行侧和观察侧可以有不同的生成模型，却仍在潜空间对齐。
  - **可移植架构**：把镜像模块建成"两个非对称 encoder + 一个共享潜变量 + M-H 式接受/拒绝门控"，而不是"一个共享 encoder"。M-H 接受概率天然是**局部可计算的**，符合生物学可实现性（无需全局梯度）。
  - **不确定度天然内生**：M-H 接受率本身就是一个校准良好的置信度信号 → 可作为我们 ood_agent 子集上不确定度的**机制性**（而非事后共形）来源。
  - **可证伪预测**：若该框架对，则跨主体（ood_agent）泛化误差应与"通信变量对齐速度"相关，而与"生成模型相似度"弱相关。这是一条可写进本周预测清单的**双栏可分离**假设。

---

## 3. Perturb-seq / scRNA-seq Perturbation

### 3.1 Joint analysis of multiply perturbed cells improves statistical power and cost efficiency in Perturb-seq
- **作者**：（bioRxiv 预印本，作者名单未在检索页完整披露）
- **年份/来源**：2026-07-10，**bioRxiv** `10.64898/2026.07.10.737863v1`
- **核心方法**：
  - **不丢弃多重扰动细胞**（传统做法常只用单扰动细胞），转而联合分析，提升统计功效与成本效率。
  - **多元 sprinter 交互筛选**：把单变量 reluctant interaction modeling（R 包 `sprintr`）的筛选准则 `Î_η = sd(r) · |cor(Z_ℓ, r)|` 从残差**向量** `r ∈ R^{n×1}` 扩展到残差**矩阵** `R = [r_1,…,r_k] ∈ R^{n×k}`（k 个响应基因）。先拟合主效应 Limma-voom（88,287 细胞 × 3,022 高表达基因），得残差矩阵；构造 p=55 个主特征（45 个启动子扰动 + 5 个 NTC + 4 个技术协变量），生成 1,485 × 3,022 的筛选矩阵 S，每个交互项对所有基因取均值得到全局交互强度排序。
  - **PerturbMatch（倾向得分匹配）**：按 guide 多重度分层（n=1,2,3,4,≥5）保证处理组/对照组的 guide-per-cell 分布平衡，层内再做两阶段（加权-GLM 倾向预筛 + 最近邻匹配）平衡技术协变量，消除"NTC guide 丰度远高于任一特定扰动 guide"造成的混杂。
- **与本项目相关点（★★☆，直接可用）**：
  - **正面回应我们 P2 的核心痛点**。P2 结论是"真实 epistasis 信号量小"（交互≈−2.3，拮抗型，ψ 增益 Δ=−0.005 不显著）。但我们从未检验过**这个"小"里有多少是统计功效不足造成的**。多元 sprinter 提供了一个**与模型无关的 epistasis 强度估计器** —— 可以先跑它在 Norman 上算出"交互项全局强度排序"，得到 epistasis 的**独立上界估计**，再回头判断 ψ 交互头是不是真的到顶了。
  - **PerturbMatch 直接修我们的混杂**：我们 P1 的切分是 96 个双扰动基因对 held-out 48/48，但**没有对 guide 多重度与技术协变量做平衡**。Norman 数据里单扰动细胞（55,464 训练）与双扰动细胞（14,212 测试）在总 UMI、线粒体比例上很可能系统性不同 → 我们测到的"ood_action 误差"里混入了**批次/深度效应**，而非纯机制外推失败。**这是继 P1（基准不公平）之后的第二个基准卫生问题。**
  - **行动项（P3 候选 A）**：在 `code/data/norman_adapter.py` 里加 PerturbMatch 式分层 + 倾向匹配，重跑 canonical 基准；若 ood_action 上各模型差距被"放大"，说明之前的"全都 ~0.63"是混杂压平的结果。

---

## 4. OOD Generalization（分布外泛化）

### 4.1 Conditionally Identifiable Latent-Environment Modeling for Out-of-Distribution Recommendation (CILER)
- **作者**：Qianqian Wang, Wenwu Gong, Yunshan Li, Zhenqing Wu, Ruili Wang, Lili Yang
- **年份/来源**：2026-08-04 提交，arXiv `2608.03647`（cs.IR）
- **核心方法**：把 OOD 任务形式化为**条件可辨识风险感知（CI-RR）**。用 **user-conditioned 指数族**建模潜环境，用 **feature-indexed 多项式**刻画潜环境如何改变偏好；预测时对推断出的环境分布做边缘化。在"充分变异 + 正确设定 + 解码器正则性"三条件下，**证明环境敏感表征可辨识到给定等价类**；进一步给出**部署超额 log-risk 由环境推断误差界定**的界。实验覆盖特征/时间/地理三类偏移。
- **与本项目相关点（★★）**：
  - 领域是推荐系统，但**框架完全可移植到 ood_neuro**：把"潜环境"换成"神经调质状态 / 培养基-温度条件"，把"用户偏好"换成"扰动响应"。
  - **最有价值的是那条界**：`超额部署风险 ≤ f(环境推断误差)`。这给了我们一个**理论化的不确定度来源** —— ood_neuro 上的预测区间宽度可以由"调质状态推断的后验熵"解析给出，而非只靠 bootstrap 经验 CI。
  - **可辨识性条件"充分变异（sufficient variation）"是可检验的**：我们 ood_neuro 子集只有 5,912 个细胞、调质轴变异范围窄 → 该条件很可能**不满足**。这解释了为什么我们 ood_neuro 各模型也都 ~0.60 无差别。**行动项：先做一次 sufficient-variation 诊断，若不满足则 ood_neuro 子集需要重新设计（扩大调质状态跨度）而不是继续换模型。**

### 4.2 Additive Causal Construction for Transferable and Reconfigurable Cross-System Learning (ACC / ACC-CRL)
- **作者**：Zhizhong Fu, Wei Zhou, Zhaoyang Jiang, Yulong Lin, Yifu Hou, Xiaorong Ding, Qiang Yan, Yifan Chen
- **年份/来源**：2026-06-30 提交，arXiv `2607.02572`（cs.CV / cs.AI）
- **核心方法**：把多源融合视为**多个因果系统的组合**，识别两类病症：跨系统差异（CSD）与跨系统纠缠（CSE），二者在 OOD 下导致性能崩塌。**ACC 框架**两层：
  1. 通过**干预一致性（intervention consistency）** 建立多系统共享的因果"**锚点**"，实现**因果图可迁移性（CGT）**；
  2. 把融合过程形式化为**因果构造**，用**不确定度量化**建模构造路径的可靠性，实现**因果图可重构性（CGR）**。
  - **ACC-CRL** 是其可学习实例：content–mechanism 解耦探索跨系统联合因果内容表征、共享锚点下做响应对齐（治 CSD）；引入**结构不确定度自适应调节融合**（抑制不稳定的 CSE）。合成（ColorMNIST）+ 真实多中心医学影像（MVI 预测）验证：**OOD 显著提升且 ID 不退化**。
- **与本项目相关点（★★☆）**：
  - **名字就叫"Additive Causal Construction"** —— 与我们 CompositionalTwin 的加性 φ 是同一族归纳偏置，但他们多做了两件我们没做的事：
    1. **干预一致性锚点**：要求"同一 `do-` 操作在不同系统/context 下产生一致的因果效应方向"作为显式约束。我们的 φ 是隐式共享的，从未显式施加跨 context 一致性损失。
    2. **结构不确定度门控融合**：不是无条件相加，而是**按路径可靠性加权**。这正是我们 P2 缺的东西 —— 我们的 ψ 交互头是无条件叠加在 φ 上的，没有"什么时候该信加性、什么时候该信交互"的门控。
  - **P3 候选 B（强推荐）**：给 CompositionalTwin 加一个**不确定度门控**：`ŷ = φ(a) + φ(b) + g(u)·ψ(a,b)`，其中 `g` 由结构不确定度驱动。在 epistasis 弱的基因对上 `g→0`（退化为加性、稳如线性基线），在 epistasis 强的对上 `g→1`（吃到交互）。这直接对治我们 P2·epistasis 扫描发现的"结构化先验只在纯加法 regime 占优"—— 门控让模型**自动在两个 regime 间切换**，正是 TRIZ 物理矛盾的**条件分离**破法。
  - **ID 不退化 + OOD 提升**：正是我们双栏报告要的形状。

### 4.3 CGRL: Causal-Guided Representation Learning for Graph Out-of-Distribution Generalization
- **作者**：Bowen Lu 等（3 位作者）
- **年份/来源**：2026，arXiv `2603.24304v1`（stat.ML / cs.LG）
- **核心方法**：从节点分类本质出发构造因果图，用**后门调整（backdoor adjustment）阻断非因果路径**，理论推导出 GNN OOD 泛化的**下界**。落地为两部分：节点级因果不变性捕获 + 图后验分布重构；以及"同阶渐近损失替换原损失"的 loss replacement 策略。实验显示能缓解"OOD 下预测表征与真值标签互信息学习不稳定"的现象。
- **与本项目相关点（★）**：
  - 我们的多尺度虚拟孪生本质是**图上的 `do-` 干预**，后门调整是标准 do-算学工具，但我们目前**没有显式做后门调整** —— 训练时直接回归，等于放任非因果路径（如批次、测序深度）传递信息。与 §3.1 的 PerturbMatch 是同一问题的两种解法（一个在数据层匹配、一个在模型层调整）。
  - 优先级低于 §4.2，但"互信息学习不稳定"这个诊断量可以借来做我们训练过程的监控指标。

---

## 5. Causal Representation Learning（因果表征学习）

### 5.1 Learning Robust Intervention Representations with Delta Embeddings (CDE)
- **作者**：（ICLR 2026 接收；arXiv `2508.04492`，有 Project Page）
- **年份/来源**：2026，**ICLR 2026**
- **核心方法**：
  - **核心思想**：把"干预/动作"建模为潜空间中的**方向向量（Delta 嵌入）**：给定干预前后观测对 `(x, x̃)`，编码器 φ（ViT-DINO 骨干 + 因果投影器）分别映射后**逐元素相减**：`δ_a := φ(x̃) − φ(x)`。在理想完美反事实假设下，`δ_a = [0 … z̃_a − z_a … 0]^T` —— **只有被动作真正改变的那几维非零，共享背景在相减时自动抵消**。
  - **三条因果约束**（由 ICM 独立因果机制假设 + SMS 稀疏机制偏移假设翻译而来）：
    - **独立性**：动作表示不受场景属性与未被影响物体干扰 —— **由减法天然保证**；
    - **稀疏性**：`L1` 正则（一次干预只影响少量机制）；
    - **不变性**：有监督对比损失（同一动作跨物体/场景表示一致）。
  - **两种网络实例**：Model A 全局 CDE（取 CLS token，动作影响全局时用）；Model B **Patch-wise CDE**（逐 patch 算 Delta，按 L2 范数取 Top-K，动作只改局部时用）。两支路的 δ 都送进同一个动作分类器。
  - **两类 OOD 挑战被显式区分**：**组合偏移**（训练见过 open(door)、close(drawer)，测试要认 open(drawer)）与**系统偏移**（测试出现全新物体类别）。在 Causal Triplet 挑战上显著超越基线，并自动发现**反义动作的反平行语义结构**。
- **与本项目相关点（★★★，最高，P3 首选）**：
  - **这是我们 φ 应该长的样子**。我们当前 `φ(p)` 直接从扰动 ID 回归到响应；CDE 主张 φ 应作用在**状态对**上、干预表示由**差分**导出。对 Perturb-seq 的映射：`δ_p = φ(x_perturbed) − φ(x_control)` —— 而 Norman 数据恰好有 non-targeting control，**可以直接构造这个差分**（我们目前只是"减 null"做标量归一，没有在潜空间做结构化差分）。
  - **"减法天然保证独立性"直接对治我们的混杂问题**：§3.1 指出的批次/深度效应，在成对相减中会被**自动抵消**（同批次的 control 与 perturbed 共享背景）—— 这比 PerturbMatch 的事后匹配更优雅，且**零额外假设**。
  - **组合偏移 ≡ 我们的 ood_action**：`open(drawer)` 未见组合 ≡ 我们 held-out 的 48 个双扰动基因对。CDE 在这个设定上"显著超越基线"，而我们 P1/P2 只做到"与线性持平"。**差别很可能就在 δ 的构造方式上。**
  - **反平行语义结构 = epistasis 的几何解释**：若敲低与过表达在潜空间是反平行向量，那么双扰动的**非加和性**就可以表述为两个 δ 向量的**非平行分量**——这给了 ψ 交互头一个**几何化、可外推**的参数化（只依赖两个 δ 的夹角与模长，不依赖基因对身份），恰好满足我们 P2 时定的"可泛化到 held-out 基因对"要求。
  - **Patch-wise Top-K ≡ 稀疏机制偏移在基因维度上的实例**：一次基因扰动只显著改变少数下游基因 → Top-K 基因维度的 δ 选择，天然对应 DE 基因集。这可能同时解决 §1.1 提醒的"该用 DE 导向指标"的问题。
  - **P3 具体方案（建议列为下一个实验）**：`CompositionalDeltaTwin` —— φ 改为在 control/perturbed 配对上做潜空间差分（含 L1 稀疏 + 跨 context 对比不变性两条正则），ψ 改为只吃 `(‖δ_a‖, ‖δ_b‖, cos∠(δ_a, δ_b))` 三个几何量。冻结随机种子，在 Norman canonical 基准上跑双栏 + bootstrap CI + 共形覆盖。

---

## 6. Calcium Imaging（钙成像）

### 6.1 A sensitive orange fluorescent calcium ion indicator for imaging neural activity (OCaMP)
- **作者**：Abhi Aggarwal, Heather A. Baker, …, Alexander W. Lohman
- **年份/来源**：2026-06-22，**Nature Communications**（Open Access）
- **核心方法**：开发 **OCaMP** —— 橙色荧光钙指示剂，针对 **>1000 nm 波长成像**优化，改善深组织成像，并**扩展与现有传感器的兼容性**（多色复用）。
- **与本项目相关点（★★）**：
  - 与 08-02 期收录的 **PinkyCaMP**（mScarlet 红色 GECI，兼容蓝光）形成**光谱互补矩阵**：绿(GCaMP) / 橙(OCaMP) / 红(PinkyCaMP)。
  - **对闭环湿实验的意义**：三色可同时读出**三个尺度/三群细胞**的钙活动，而把蓝光波段整段留给光遗传致动器 → 我们"多尺度孪生"的**多尺度同时读出**从设想变成硬件可行。写湿实验方案时应把"多色 GECI 组合"列为标准配置。
  - >1000 nm 深组织 → 可覆盖更深的皮层层级，支持"跨层级预测编码"假设的直接检验。

### 6.2 Smart Dura: a functional artificial dura for multi-modal neural recording and modulation
- **作者**：S. Montalvo Vargo, N. Hong, T. Belloir, N. Stanis, J. Zhou, K. Khateeb, G. Hatanaka, Z. Ahmed, I. Kimukin, D. J. Griggs 等
- **年份/来源**：2026，**Microsystems & Nanoengineering**，DOI `10.1038/s41378-026-01166-8`
- **核心方法**：把常用的人工硬脑膜做成"功能版" —— 薄膜微加工把**微米级电极阵列（最多 256 通道）**单片集成进柔软、柔性、透明基底，机械柔顺性与原生组织匹配，**光学透明度 >98%**。在 NHP（非人灵长类）上完成体内验证：电生理记录 + 神经调控 + 结构/功能光学成像（钙成像、光遗传）三合一，覆盖大面积皮层。
- **与本项目相关点（★★☆）**：
  - **规模跨越到灵长类**：我们的镜像神经元现象学根植于猕猴 PMd/PMv（§2.1），但闭环读写工具此前主要在啮齿类。Smart Dura 让"**在猕猴前运动皮层同时做大面积钙成像 + 光遗传 `do-` 干预 + μECoG 读出**"在技术上可行。
  - **这是我们闭环验证路线的关键使能件**：预测 → 光遗传干预（`do(神经元群 X = 激活)`）→ 钙成像读出群体几何变化 → 误差回流。§2.1 的"执行/观察不对称性"假设**恰好需要在猕猴上、在群体尺度上**才能证伪。
  - **写进 wetlab/ 方案**：把 Smart Dura 列为长期（12+ 月）闭环平台候选，短期仍以啮齿类全光方案（APPC + PinkyCaMP/OCaMP）推进。

### 6.3 Combining ultra-flexible electrodes with two-photon imaging to illuminate brain-wide neural dynamics
- **作者**：Christopher Lewis, Adrian Roggenbach, Linus Meienberg, Tansel Yasar 等（共 9 位）
- **年份/来源**：2026-08-03 posted，**Research Square** `rs-10218667/v1`（Brief Communication）
- **核心方法**：慢性植入**超柔性电极**做皮层下记录，同时在小鼠新皮层做**双光子钙成像**。柔性电极即使在**陡峭插入角**下也保留光学通路，支持**数周单神经元追踪**。用该组合展示了**皮层下–皮层耦合如何随慢脑状态调制、以及在快速 ripple 事件周围的变化**。
- **与本项目相关点（★★）**：
  - **跨尺度同时观测 = 多尺度虚拟孪生的验证数据源**。我们的孪生横跨"离子通道→突触可塑性→环路"，但训练/验证数据一直是单尺度的。这类"皮层下电生理 + 皮层光学"配对数据是**跨尺度一致性约束**的天然素材。
  - **"慢脑状态"是现成的 ood_neuro 轴**：脑状态（清醒/静息/ripple）本身就是全局调质状态的代理变量，且**可在同一动物内自然遍历** → 比药理操纵更容易获得"充分变异"（正是 §4.1 CILER 可辨识性要求的条件）。
  - **数周单神经元追踪 → 纵向 ood_agent**：同一神经元跨周的漂移，可作为"未见主体"之外的第四类 OOD（未见时间点/漂移后状态）。

---

## 7. Optogenetics（光遗传）

### 7.1 Dual-channel optogenetics in yeast for multiplexed light-based control of cellular processes and pathways
- **作者**：Linus Yu Han Tan, Zhangyuan Lin, Chueh Loo Poh
- **年份/来源**：2026-05-22，**Nature Communications**（Open Access）
- **核心方法**：把**红光与蓝光**两套光遗传系统整合进 *S. cerevisiae*，实现**双通道正交光控**；演示了对木犀草素（luteolin）合成与絮凝（flocculation）的光控。
- **与本项目相关点（★★☆，GOAI 赛道直接相关）**：
  - **我们的 GOAI 数据就是酵母**（`data/raw/goai/`，列含 `Strains` / `Medium` / `Temperature` / `pert_time` / `Yeast_cell_plate`）。这篇提供了在同一酵母系统里做**双通道、时序可控 `do-` 干预**的现成工具。
  - **组合扰动的黄金实验设计**：红/蓝双通道 = 两个独立可调的 `do-` 操作，可以**连续调节各自强度**并测量响应面 → 直接测出 epistasis 的**函数形式**（而不是只有 0/1 双敲的两个点）。这正是我们 P2·epistasis 扫描在**合成数据**上做的事，这里给出**真实生物系统的对应实验**。
  - **闭环回流的可行入口**：预测 →（红光强度 a, 蓝光强度 b）组合干预 → 蛋白组/转录组读出 → 误差回流。比哺乳动物细胞快、便宜、可高通量。**建议写进本周湿实验候选清单（优先级高于 Smart Dura 的长期路线）。**

### 7.2 Rapid dynamics of dorsal raphe serotonin neurons regulate the strength of visual attention
- **作者**：Jonas Lehnert, Kuwook Cha, Arjun Krishnaswamy
- **年份/来源**：2026-03-18，**Nature Communications**（Open Access）
- **核心方法**：神经调质长期被认为参与视觉注意，但直接证据缺失。该文显示**背侧中缝核（dorsal raphe）5-HT 活动及其在皮层的释放，选择性调节小鼠对线索化视觉特征的注意强度**，强调的是**快速动力学**（rapid dynamics）而非缓慢基调调制。
- **与本项目相关点（★★）**：
  - **给 ood_neuro 加第三个调质轴**。我们此前的调质先验主要来自 08-02 期的 ACh/DA 集群（Colangelo 生物物理开放模型、ACh demix DA、ACh–DA 努力、ACh 显著性）。加入 **5-HT** 后，调质空间从 2 维（ACh, DA）扩到 3 维 → **直接改善 §4.1 CILER 要求的"充分变异"条件**，这正是我们 ood_neuro 子集目前变异不足的解法方向。
  - **"快速动力学"改变建模假设**：若 5-HT 是快速、事件锁定的，则调质不能建模为"每个 episode 一个常数标量"，而必须是**时变门控场** `m(t)`。这与 08-02 期 Colangelo"突触级门控而非全局标量"的结论合流，形成一致的建模指令：**调质 = 时变、突触级、多维的门控场**。
  - **与 §2.2 的 surround inhibition 呼应**：注意力"强度"调节 ≈ 我们在动作解码头上想加的 winner-take-most 增益 —— 5-HT 可能就是这个增益的生物学实现。可写成一条双栏预测：*操纵 5-HT 应特异性改变 ood_action 上预测的锐度（而非 in-distribution 的准确率）*。

---

## 8. Neuromodulator（神经调质）

> 本周原发性研究主要经 §7.2（5-HT / dorsal raphe）与 §6.3（慢脑状态调制）交叉覆盖，无独立新增条目。08-02 期收录的 ACh/DA 集群仍是该方向的当期主线。

---

## 9. 不确定度与校准（跨方向，本项目硬约束专栏）

### 9.1 PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation
- **作者**：Jiabei Cheng（上海交大）, Changxi Chi（西湖大学）, Jingbo Zhou（西湖大学）, Hongyi Xin（上海交大）, Jun Xia（HKUST-GZ / HKUST）
- **年份/来源**：arXiv `2510.07964`（代码：`github.com/Bunnybeibei/PRESCRIBE`）
- **核心方法**：
  - 明确把未见基因扰动预测的可靠性拆成两源：**epistemic（模型）不确定度** —— 取决于目标基因与训练集基因的相似度；**aleatoric（数据）不确定度** —— 取决于对应训练数据质量与生物学固有随机性。
  - 提出 **pseudo E-distance**：借用 E-distance（组间距离 − 组内离散度）的思想，联合建模**预测误差**与**扰动后固有离散度**。因真值分布对未见基因不可得，故用"伪"版本估计。
  - 架构：**多元深度证据回归**（Natural Posterior Network 的多元扩展）—— 不输出单一表达谱，而同时输出**转录组景观上的后验分布**与由**扰动空间学到的隐密度**导出的 evidence score。
  - 结果：置信分与经验准确率**强相关**；据此过滤不可信预测，相比可比基线取得**稳定 >3% 的准确率提升**。
- **与本项目相关点（★★☆，直接补齐硬约束）**：
  - **我们的不确定度目前是事后的**（bootstrap CI + split conformal）。PRESCRIBE 提供**模型内生**版本，且**epistemic/aleatoric 分离**——这正是我们双栏哲学在不确定度维度上的镜像：epistemic ↔ "模型没见过"（机制外推风险），aleatoric ↔ "生物本身就随机"（不可约噪声）。
  - **"置信分与准确率强相关"可直接当作机制证据**：如果我们的 CompositionalTwin 在 ood_action 上的置信分**不能**预测其误差，那就说明它没有真正的机制感知，只是在拟合边际分布 —— 这是一个比 RMSE 更难作弊的判别指标。
  - **过滤不可信预测提升 3%**：这是"选择性预测（selective prediction）"范式，对我们导出**可执行湿实验清单**极有价值 —— 只把高置信预测送去做光遗传/钙成像验证，湿实验成本直接下降。
  - **行动项**：把 evidence score / pseudo E-distance 加进 `code/benchmark_ood.py` 的报告表，与现有 bootstrap CI 并列成第三栏。

### 9.2 Keeping SCORE: interpretable uncertainty-aware classification from diffusion models for genomics
- **作者**：（bioRxiv；通讯含 YG，声明与 Biohub / Schmidt Sciences / Gilead 有咨询关系）
- **年份/来源**：2025-11-26 v1 / v2，**bioRxiv** `10.1101/2025.11.26.690838v2`
- **核心方法**：把**条件扩散模型**转化为分类/回归的概率引擎 —— 沿**随机加噪轨迹计算精确似然**，从而无需重训练、无需改架构即可获得**校准的不确定度 + 特征级归因**。验证跨度：图像识别（MNIST / 自然图像）→ 2200 万细胞图谱上的 164 类细胞类型分类（准确率匹配或超过 SOTA，且**独有后验概率估计与预测置信度**）→ **多研究 Perturb-seq 数据集上 100 个 CRISPRi 条件的遗传扰动映射**（匹配或超过判别式基线，并给出驱动每个决策的基因组特征归因）→ 蛋白序列突变稳定性回归。
- **与本项目相关点（★★）**：
  - **"无需重训练即可加不确定度"是极高性价比的工程路径**。若我们后续引入任何扩散型组件（对标 08-02 期的 PerturbDiff、08-01 期的 X-Cell），可直接套用 Keeping SCORE 获得校准输出，不必另建共形层。
  - **特征级归因 = 机制可解释性的低成本来源**，对 GOAI 评委"科学意义(30 分)"直接有用。
  - **同时在 Perturb-seq CRISPRi 上验证** → 与我们的数据模态完全对口。

### 9.3 【机会】Frontiers Research Topic: Advances in Quantifying and Communicating Uncertainty in Single-Cell Biology
- **性质**：征稿专题（非论文），**投稿截止 2026-09-18**，正在接收。
- **范围**：区分技术 vs 生物学不确定度的模型；不确定度在下游分析（DE、轨迹推断、通路分析、细胞通讯、命运映射）中的**传播**框架；**评估与校准不确定度感知方法的基准、指标与最佳实践**；概率与深度学习框架；不确定度感知的质控、决策与**实验设计**；可复现管线与社区资源。接收 Methods / Benchmark / Technology and Code / Original Research 等类型。
- **与本项目相关点（★★，非技术但战略相关）**：
  - 我们已有的资产 —— **canonical OOD 基准（含 bootstrap CI、共形覆盖、ECE、校准曲线）+ 记忆/机制双栏 SOP + 可复现脚本与随机种子** —— 与"评估与校准不确定度感知方法的基准、指标与最佳实践"这一征稿条目**高度吻合**。
  - ⚠️ 按项目约束，**本条仅作记录，不发起任何对外投稿动作**；是否投稿须先经你确认。

---

## 10. 本周总结：最值得借鉴的两个方法，及其对建模层（多尺度虚拟孪生）的启示

本周最值得借鉴的是 **Causal Delta Embeddings（CDE, ICLR 2026）** 与 **Additive Causal Construction（ACC-CRL）** 这一对互补方法，它们联手给出了我们 P2 卡点的一条具体出路。CDE 的核心洞见是：**干预不应该被编码成一个 ID，而应该被构造成潜空间里的一根差向量** `δ = φ(x̃) − φ(x)` —— 因为减法会自动抵消两侧共享的背景（批次、测序深度、细胞状态），这天然满足独立因果机制假设，无需任何额外的混杂校正；再叠加 L1 稀疏（一次干预只动少数机制）与跨 context 对比不变性两条正则，δ 就成了一个可跨物体、跨场景复用的动作表示，并在"未见 物体×动作 组合"这一**与我们 ood_action 完全同构**的任务上显著超越基线。这直接给了 CompositionalTwin 一个可执行的重构方向：把 φ 从"扰动 ID → 响应"的回归器，换成作用在 (control, perturbed) 配对上的**潜空间差分算子**——Norman 数据自带 non-targeting control，材料是现成的；更妙的是，CDE 观察到的"反义动作呈反平行几何"提示我们，双扰动的非加和性（epistasis）可以被重新参数化为**两个 δ 向量的非平行分量**，于是 ψ 交互头只需要吃 `(‖δ_a‖, ‖δ_b‖, cos∠(δ_a,δ_b))` 三个几何量，天然不依赖基因对身份、天然可外推到 held-out 基因对——这恰好是我们 P2 当初给 ψ 定的硬要求，而当时用的是"基因效应"这种更弱的特征。ACC-CRL 则补上另一半：它证明加性因果构造**不该无条件相加**，而应由**结构不确定度门控路径可靠性**。落到我们身上就是把模型改成 `ŷ = φ(a) + φ(b) + g(u)·ψ(a,b)`——在 epistasis 微弱的基因对上 `g→0`，模型退化成加性、稳得像线性基线；在 epistasis 强的对上 `g→1`，交互头才被激活。这正面回应了我们 P2·epistasis 扫描的尴尬结论（"结构化先验只在纯加法 regime 占优，richer epistasis 反被黑箱 MLP 吃掉"）：不是要在两种归纳偏置里二选一，而是**用条件分离让模型自己切换**——这就是 TRIZ 物理矛盾（既要记住加性分布、又要外推非加性未知）的标准破法，而不是"再加一层网络"。

对多尺度虚拟孪生的启示因此有三层。**第一，表示层要从"ID 索引"升级为"差分几何"**：无论是分子尺度的基因扰动、还是环路尺度的光遗传 `do-` 干预，都应统一表示为"干预前后潜状态的差向量"，这样离子通道→突触→环路的跨尺度传播就变成了 δ 在层级间的**向量变换**，而非各层各自重新学一套编码——多尺度的"多"才真正被结构统一起来。**第二，融合层要从"无条件叠加"升级为"不确定度门控"**，且这个门控应与 PRESCRIBE 的 epistemic/aleatoric 分解共用同一套后验：模型对某个组合"没见过"（epistemic 高）时收缩到保守的加性先验，"见过但本身就抖"（aleatoric 高）时则拓宽预测区间而非改变均值——这一步同时满足了我们"不确定度强制"的硬约束，并且让置信分变成一个**比 RMSE 更难作弊的机制判别指标**。**第三，也是本周最该警醒的一条**：`Empirical Comparison of Virtual Cell Models` 与 `Response Magnitude` 两篇联手指出，深度模型的真实增益**恰恰在组合非加和扰动上**，而全基因 RMSE 这类指标会被"预测均值"策略最大化、并且可能只在测响应幅度这一个低维标量。我们 P2 判定"ood_action 无判别力（所有模型 ~0.63）"很可能**是指标选错、外加 §3.1 指出的 guide 多重度混杂未平衡所致的假象**——与 P1 时"基准不公平造成崩塌假象"是同一类错误的第二次复发。因此在动手实现 δ-φ 与门控 ψ 之前，应先做两件低成本的证伪检验：**(a)** 在 ood_action 上补 DE 基因子集指标、macro-per-gene AUROC 与 E-distance，看各模型是否重新拉开差距；**(b)** 做一次"4 标量幅度特征"消融，确认 RMSE 不是只在测响应幅度。只有这两条过关，"epistasis 信号量小"才是一个可以写进结论的机制事实，而不是又一个基准伪影。

---

*生成时间：2026-08-07 · 自动化 `automation-1785494084767` · 仅写入本地文件，未外发*
