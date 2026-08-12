# 文献监测周报 · 2026-08-12

> 执行：自动化 `automation-1785494084767`（虚拟镜像神经元-文献监测）
> 范围：virtual cell / mirror neuron / perturb-seq·scRNA-seq perturbation / OOD generalization / causal representation learning / calcium imaging / optogenetics / neuromodulator
> 去重基线：相对 07-31 / 08-01 / 08-02 / 08-07 / 08-11 五期 INDEX 去重，本期新增 9 篇精选 + 1 基准动态 + 数条概念性注记（均标注来源层级）。
> ⚠️ 未外发 / 未发布，仅写文件与汇总。

---

## §0 TL;DR 优先级表（本期新精选）

| # | 方向 | 论文/来源 | 一句话价值 | 与本项目关联强度 |
|---|---|---|---|---|
| 1 | virtual cell | **VCBench**（bioRxiv 2026.06.18） | 7 维基准证明线性/近邻基线在 4/5 维度 ≥ 基础模型；附污染报告 schema + spread-error 认知校准探针 | ★★★ 直接加固「须显式超越线性基线」SOP |
| 2 | virtual cell | **U-Pert**（bioRxiv 2026.06.30） | 质量非平衡扰动动力学：联合建模转录态转移 + 细胞数动态，前向预测未见扰动 + 逆设计 | ★★★ 直击 ood_action / 扰动表型含丰度轴 |
| 3 | perturb-seq / do-演算 | **Mechanisms Matter: Transportability**（bioRxiv 2026.05.08 v2） | 因果可迁移理论 + 可调机制分歧半合成模拟器 + Vendi 多样性诊断（mode collapse 检测） | ★★★ 直击 ood_agent / ood_neuro 评测设计 |
| 4 | CRL / do-演算 | **CauFinder**（Advanced Science 2026-06-16） | 因果解耦 + do-演算信息流 + 网络控制，从观测转录组识别状态转换因果调控因子 | ★★ 因果表征 + do 演算落地 |
| 5 | calcium imaging | **VADER1**（bioRxiv 2026.06.01 / Nature Methods） | 首个红光 2P 电压成像 GEVI，与 GCaMP 双色复用，全光电压+钙成像 | ★★ 湿实验闭环读端使能 |
| 6 | optogenetics | **Neuropixels Opto**（Nature Methods, UCL+Allen） | 单硅探针集成电生理记录 + 光遗传刺激，深脑同步读写 | ★★ 湿实验闭环写端使能 |
| 7 | CRL（理论） | **Anchor–Stabilizer Theory**（preprints 202606.0188） | 可辨识性锚定理论：干预/因果机制锚作为对称性破缺 | ★★ 为 φ/ψ 分解提供可辨识性理论底座 |
| 8 | neuromodulator | **Dalla Porta et al. PNAS 2026** | ACh 受体空间异质性 → 全脑动力学；同信号因受体密度不同而效应异 | ★★ ood_neuro 调制场先验（空间异质而非全局标量） |
| 9 | optogenetics | **All-optical cOVC**（eLife 2026-07） | 细胞特异光遗传电压钳，活体 C. elegans 全光测量缝隙连接电导 | ★ 全光电压钳方法学 |

基准动态（非论文）：**Arc 2026 Virtual Cell Challenge**（8/4 预注册、8/20 正式开赛，zero-shot 多细胞系基因敲除预测）。见 §10。

---

## §1 Virtual Cell（虚拟细胞）

### 1.1 VCBench: A Multi-Dimensional Benchmark for Single-Cell Foundation Models
- **作者**：L. Weidener, M. Brkić, M. Jovanović, E. Ulgac, A. Meduri 等（AppliedScientific）
- **年份/来源**：bioRxiv 2026.06.18.733146（预印本，含 GitHub / HuggingFace 资产）
- **核心方法**：把 4 个独立虚拟细胞框架综合为 7 个能力维度（扰动响应预测、跨物种通用性、GRN 推断、模态整合、时序动态、多尺度整合、in-silico 实验）；对 5 个基础模型（Geneformer / scGPT / UCE / TranscriptFormer / Arc State）对比**预注册的线性 + 最近邻基线**；发布 **Contamination Reporting Schema**（污染报告 schema）+ **spread-error correlation probe**（展布-误差相关探针，用于认知校准）。
- **与本项目相关点**：① 结论与本项目 Norman 真实基准、Ahlmann-Eltze 2025、OCOO-T、SLIM 等**第五次互证**——「深度模型未稳定超越简单基线」在扰动预测/跨物种/GRN/时序 4/5 维度成立；② **spread-error correlation probe** 直接对应本项目 P4-1「区间饱和 / ECE=0.2125」类退化——它是检测「高置信低误差相关」的现成认知校准探针，可并入 `verify_collapse` 诊断；③ **Contamination Reporting Schema** 加固本项目「必须发布训练清单、防数据泄漏」SOP（我们的 `regress_check` / 训练 manifest 思路与之同向）；④ TranscriptFormer 唯一在跨模态 RNA→蛋白超越基线，却因谱坍缩（spectral collapse）牺牲时序——印证「架构红利≠外推红利」。

### 1.2 U-Pert: Unbalanced Perturbation Dynamics For Cell Fate Design
- **作者**：Qiangwei Peng, Yuchuan Wang, Jianzhe Li, Xinyu Wang, Yao Xiao, Peijie Zhou
- **年份/来源**：bioRxiv 2026.06.30.735555v1（预印本）
- **核心方法**：提出**质量非平衡（unbalanced）生成框架**，从非配对单细胞快照学习条件/上下文相关的扰动动力学；**联合建模转录态转移 + 细胞数动态**（增殖/凋亡/选择），支持未见扰动/上下文的前向预测，以及逆设计（筛出达成用户指定转录组或群体水平结局的遗传/药理干预）。
- **与本项目相关点**：① 直击本项目长期盲点——**扰动表型含丰度轴**（细胞杀伤/扩增/选择），而我们当前 `ê_u` 监督口径只在表达态空间，忽略群体质量变化；ood_action 若只评表达位移，会被「平衡传输强行把差异塞进跨细胞匹配」污染；② 逆设计能力 ↔ 本项目「湿实验闭环回流」目标（预测→实验→误差回流）；③ 跨上下文（context）前向预测 ↔ ood_neuro / ood_agent 的跨主体外推。

---

## §2 Mirror Neuron（镜像神经元）

> 本期**无新增原发性研究**。2026-07-25 Science Advances 的「执行↔观察群体运动学编码」（Chatzimichail eaed9309，Raos 组）已于 08-07 / 08-11 索引；本期多数为该文媒体延续（PsyPost、科技媒体）与理论性回顾，无新实验数据。

**概念性注记（低置信、博客/综述源，仅作方法论启发，非引用级）**：mental-momentum 综述把镜像神经元重述为 **「Learned Matching 假说 + 预测编码」** 框架——前运动皮层作为生成模型，自顶向下预测观察到的动作感觉后果，预测误差最小化即「镜像」；并引用 Heyes 的「认知小工具」说（镜像性来自联想学习而非硬编码）。对本项目**自/他同源悖论**的启示：把镜像性建模为**预测编码单元在共享潜空间的对称性 + 自顶向下预测误差最小化**，而非静态表征匹配——与本项目「感知↔动作共享潜空间」设定一致，可补充进孪生因果图的动作观测分支。

---

## §3 Perturb-seq / scRNA-seq Perturbation

### 3.1 Mechanisms Matter: Transportability of Cellular Perturbation Effects
- **作者**：Shi-ang Qi（Vector Institute / Microsoft Research），P. Chapfuwa 等
- **年份/来源**：bioRxiv 2026.05.08.723625 **v2**（8 月修订，新增 K562 分析与 Nadig 多细胞系）
- **核心方法**：借用因果**可迁移理论（transportability）**，论证跨上下文泛化由**共享因果机制**而非分布相似性主导；构建**因果模拟器（CausalDGP）**生成带可调机制分歧的半合成 Perturb-seq，提供已知真值因果结构的基准；将 **Vendi 多样性分数**适配为扰动场景的 **mode collapse 诊断**（标准逐扰动指标看不见的失败模式）；在 4 个深度学习模型 + 6 个简单基线上揭示**跨上下文泛化缺口**。
- **与本项目相关点**：① 这是对本项目 ood_agent / ood_neuro「跨机制外推」假说的**理论+实证托底**——即使真值因果结构完全已知，机制不同的上下文间也无模型能泛化；② 其 CausalDGP（可调机制分歧 A/B/C 跨上下文）**直接类比本项目 Norman 的 CausalDGP / epistasis 代理扫描**，可相互印证；③ **Vendi 多样性诊断**是可移植工具——用于检测本项目「区间饱和 / ECE=0.2125 / 覆盖恒 1.000」式 mode collapse，补强 `verify_collapse`；④ within-context vs cross-context 拆分策略（held-out perturbation×context 配对）可借鉴进本项目双栏拆分设计。

### 3.2 CauFinder: Steering Cell-State and Phenotype Transitions by Causal Disentanglement Learning
- **作者**：Chengming Zhang, Zexi Chen … Luonan Chen（陈洛南, SJTU）、Kazuyuki Aihara（东大）、Jian Liu 等
- **年份/来源**：*Advanced Science* 2026-06-16，DOI 10.1002/advs.76177
- **核心方法**：因果解耦学习 + **do-演算**估计因果表示对状态预测的**因果信息流（causal information flow）**，最大化该流使混杂/虚假相关信号被压低；结合 SHAP/梯度把因果信号映射回基因空间；再与先验基因互作网络结合，用**网络控制**筛主调控因子（master regulators）。在 EGFR-TKI 耐药中发现 **DAAM1** 为既往未识别驱动因子（敲低增敏奥希替尼）。
- **与本项目相关点**：① 把 **do-演算**落到「从观测转录组识别因果调控因子」的可计算流程，与本项目「do-演算 + 组合扰动把相关升级为可证伪预测」主线的**生物侧对接**；② 因果解耦 + 信息流最大化 ↔ 本项目 φ（基因效应嵌入）的「解耦背景混杂」目标（呼应 08-07 CDE-δ 减法抵消背景）；③ master-regulator 网络控制视角可启发孪生因果图的 GRN 边权初始化。

---

## §4 OOD Generalization（分布外泛化）

- **虚拟细胞侧（见 §1.1 VCBench）**：基准级证据——简单基线在跨物种/GRN/时序维度胜出，直指本项目「OOD 强制 + 须超越线性基线」铁律。
- **Perturb-seq 侧（见 §3.1 Mechanisms Matter）**：跨上下文泛化缺口 = 本项目 OOD 子集（ood_agent / ood_neuro）的同类现象；Vendi 多样性诊断可作 mode-collapse 探测器。
- **概念性方法注记（低置信、工程/医学源，仅作不确定度层参考）**：**Conformal Risk Control (CRC)** 把常规共形预测（CP）从「控制覆盖」扩展为「控制任意单调损失的期望值」（如精度、校准、任务特定风险），已在医学影像不确定性框架（Nature Biomedical Engineering 2026）落地。对本项目 P4-1（单标量 conformal `_q` 假设 ID/OOD 同分布 → 区间饱和）的启示：**从覆盖率控制升级到分 subset / localized 的风险控制**——CRC 提供「指定任意单调损失 + 二元搜索阈值」的可复现配方，可直接迁移到我们的 ECE/覆盖分层报告。

---

## §5 Causal Representation Learning（因果表征学习）

### 5.1 Anchor–Stabilizer Theory for Identifiable Representation Learning
- **作者**：（preprints.org 手稿 202606.0188，未标注全作者）
- **年份/来源**：preprints.org 2026.06（预印本，**未同行评审**）
- **核心方法**：系统梳理**可辨识性锚定**理论——把「对称性破缺资源」分类为分布锚、干预锚、因果机制锚、关系/对比锚等；论证**干预/因果机制锚**（pre/post 配对、多环境未知目标干预）能在弱监督下识别模块化机制而非仅潜坐标，残差模糊度应按图恢复/干预预测/反事实等因果任务判定。
- **与本项目相关点**：① 为「**φ 只吃基因效应、ψ 只吃交互/几何量**」的分解提供**可辨识性理论底座**——我们的分解本质是「干预锚 + 机制模块化」假设，本文给出该假设成立的系统性条件；② 直接支撑 P3 候选（CDE-δ 反平行几何、ACC-CRL 干预一致性锚点）的「为何分解可外推」论证；③ 警示：若缺乏干预锚（纯观测），分解退化为分布锚，模糊度不可消解——呼应本项目「纯观测 scRNA 学不到因果」的硬约束。

### 5.2 CauFinder（详见 §3.2）
- 因果解耦 + do-演算信息流，属 CRL 在单细胞状态转换的落地实例。

---

## §6 Calcium Imaging（钙成像）

### 6.1 VADER1: A red-emitting, genetically encoded indicator for two-photon voltage recording in vivo
- **作者**：Shuyuan Yang, A. J. McDonald, X. Lu … F. St-Pierre, N. Ji, V. Emiliani 等
- **年份/来源**：bioRxiv 2026.06.01.726307（预印本）；同期 *Nature Methods* 正式发表
- **核心方法**：首个**全基因编码红光 2P 电压成像 GEVI**；在高通量多参数筛选平台演化得到，2P 激发下稳定记录至皮层第 5 层，SNR 与绿光 JEDI-2P 相当；可与 **GCaMP（绿光钙指示剂）双色复用**，实现「快电压 + 慢钙」同步、全光电路 interrogate。
- **与本项目相关点**：① 提供**全光闭环读端**关键使能——红光电压 + 绿光钙双色，可同时读快（动作电位）慢（钙瞬变）信号，支撑本项目「钙成像湿实验闭环验证机制泛化」；② 红光×蓝光视蛋白光谱分离，避免 PinkyCaMP（已索引）仍受蓝光电生理串扰的局限，是更干净的全光读写同体方案；③ 提示我们湿实验协议（`export_protocol.py`）可升级为「电压+钙双通道」读端。

### 6.2 All-optical analysis of electrical coupling（cOVC）
- **作者**：Nora Elvers, Amelie Bergs … Alexander Gottschalk（Goethe Univ.）
- **年份/来源**：*eLife* Reviewed Preprint 2026-07-15，DOI 10.7554/eLife.111279.1
- **核心方法**：用 QuasAr2 电压指示剂 + **细胞特异光遗传电压钳（cOVC）**，在完整活体 C. elegans 非侵入测量缝隙连接电导，解析特定 innexin 对肌肉同步与运动的贡献。
- **与本项目相关点**：全光电压钳方法学先例，验证「光读写同体测细胞间耦合」可行，对我们设计 ood_neuro（调质改变耦合强度）湿实验闭环有方法借镜。

---

## §7 Optogenetics（光遗传）

### 7.1 Neuropixels Opto
- **作者**：Matteo Carandini（UCL）、Karolina Socha 等；UCL + Allen Institute 联合
- **年份/来源**：*Nature Methods*（2026，具体卷期本期检索未锁定；见 labnews 报道）
- **核心方法**：在**单根超薄硅探针**（窄于发丝）上集成数百记录位点 + 微型发光器，实现深脑**电生理记录与光遗传刺激同步**（同一实验中既测又控），已在小鼠验证。
- **与本项目相关点**：① 提供**湿实验闭环写端**使能——同一探针「观察+干预」同体，正是本项目「预测→光遗传干预→误差回流」闭环所需的硬件；② 揭示「皮层神经元活动比预期更局部化」的初步发现，对孪生因果图的空间尺度假设有反向提示。

### 7.2 All-optical cOVC（见 §6.2）
- 光遗传电压钳属光遗传×电压成像交叉，列此备查。

---

## §8 Neuromodulator（神经调质）

### 8.1 Dalla Porta et al. — Spatially structured heterogeneity shapes large-scale cortical dynamics
- **作者**：Leonardo Dalla Porta, Maria V. Sanchez-Vives 等（IDIBAPS / EBRAINS）
- **年份/来源**：*PNAS* 2026，DOI 10.1073/pnas.2532072123
- **核心方法**：用 **The Virtual Brain (TVB)** 全脑仿真平台，把 **68 个脑区的毒蕈碱型 ACh 受体密度图**叠加到真实结构连接上；模拟从清醒到睡眠多状态，发现**生物锚定的受体异质性**提升脑区协调与信息流，并自发复现「局部睡眠样慢波」现象；证明**同一调质因受体空间密度不同而效应异**。
- **与本项目相关点**：① 直击 ood_neuro「**调制场先验应是空间异质而非全局标量**」——与本项目 MEMORY 中 Colangelo 开放模型（突触级门控而非全局标量）同向；② 多层尺度桥接范式（分子受体密度 → 全脑动力学）正是「多尺度虚拟孪生」的教科书级样板，可直接启发孪生因果图的调质分支建模；③ 数据/模型已开放（EBRAINS），可作 ood_neuro 调制场先验的外部佐证。

### 8.2 状态更新（非新增，仅记录期刊化）
- **Colangelo et al. 调质生物物理开放模型**已于 *PLOS Computational Biology* 2026 正式发表（DOI 10.1371/journal.pcbi.1014460，开放代码/数据 zenodo.14587678），即 08-02 索引的「Colangelo 调质生物物理开放模型」正式版。本期不重复计入精选，但确认其已期刊化、可正式引用。

### 8.3 概念性注记（低置信、综述源）
- Frontiers「predictive processing / precision」综述把**神经调质 = 预测误差的精度加权（precision weighting）**：ACh 选择性放大失匹配响应、DA/5-HT 去同步慢振荡。对本项目不确定度层的启示：**调质可解释为不确定度门控 g(u) 的生物对应**——高 epistemic 不确定时调质上调「放大预测误差」使模型收缩到加性（呼应 08-07 ACC-CRL 的 g(u) 门控）。仅作理论呼应，非正式引用。

---

## §9 不确定度与校准专栏（跨方向）

| 来源 | 方法 | 对本项目不确定度层的映射 |
|---|---|---|
| VCBench (§1.1) | spread-error correlation probe + Contamination Reporting Schema | 认知校准探针 → 并入 `verify_collapse`；训练 manifest SOP |
| Mechanisms Matter (§3.1) | Vendi 多样性分数 → mode collapse 诊断 | 检测区间饱和 / 覆盖恒 1.000 式退化 |
| CRC (§4 注记) | 共形风险控制：任意单调损失期望控制 | P4-1 单标量 → 分 subset / localized 风险控制 |
| Dalla Porta (§8.1) | 调质=精度加权（综述） | g(u) 门控的生物对应 |

> 主线收敛（与 08-07 / 08-11 一致）：**表示层、融合层、不确定度层都必须 OOD 感知**。本期新增的可移植工具链 = Vendi 多样性诊断（检测退化）+ spread-error 认知校准探针（量化退化）+ CRC（分 subset 风险控制）。

---

## §10 基准动态（非论文）

- **Arc 2026 Virtual Cell Challenge**：8/4 开放预注册，8/20 正式开赛；任务 = **zero-shot 多细胞系基因敲除响应预测**，用新生成实验数据评分。这恰是本项目 ood_agent（未见细胞系/主体）的外推测试范式。2025 首届冠军 BM_xTVC 用「深度学习 + 蛋白嵌入 + 公共扰动数据 + 统计特征」混合系统（非端到端 DL 独胜），再次印证本项目基线结论。→ 建议把本项目 Norman/GOAI 的 ood_agent 指标与该挑战的 zero-shot 多细胞系设定对齐，作为对外可比对锚点（内部使用，不外发）。

---

## §11 本周最值得借鉴的方法 + 对多尺度虚拟孪生的启示

**方法一：VCBench 的「spread-error correlation 认知校准探针 + 污染报告 schema」**。它把一个长期被我们定性讨论的问题（「深度模型未稳定超越线性基线」「区间饱和是测量伪影还是机制缺失」）变成**可复现、可审计的操作**——spread-error 探针直接量化「预测展布 vs 误差」的相关性，正好是本项目 ECE=0.2125 / 覆盖恒 1.000 类退化的诊断器；污染 schema 把「必须发布训练清单」从口号变成标准。

**方法二：Mechanisms Matter 的「CausalDGP 可调机制分歧模拟器 + Vendi 多样性诊断」**。它给出**跨上下文（=我们的 ood_agent/ood_neuro）泛化的可证伪评测协议**——允许我们像它那样在 Norman 上构造「机制相同 / 仅 A 变 / 仅 B 变 / 两者皆变」的半合成梯度，并用 Vendi 分数监控预测多样性是否坍缩。

**对多尺度虚拟孪生建模层的三层启示**（承接 08-11 框架）：
1. **表示层（φ）**：VCBench 证明「基础模型未赢 4/5 维度」是领域级事实 → φ 的任何增益都必须过 **CauFinder 式因果解耦 / CDE-δ 减法**的混杂诊断（08-11 元主题延续），否则只是分布内伪迹。
2. **融合层（ψ）**：Mechanisms Matter 的跨机制不可泛化结论 → ψ 的 g(u) 门控在 epistemic 高时必须**收缩到加性主干**（ACC-CRL 方向），并把「机制分歧梯度」作为 ψ 的显式训练信号。
3. **不确定度层**：spread-error 探针 + Vendi 诊断 + CRC 分 subset 风险控制三者组装成本项目 **P4-1 的 localized conformal 落地配方**——单标量 `_q` 退役，改为按 ood_* 子集分别报覆盖 / ECE / 多样性，并对退化自动报警。

---

*去重声明：本期 9 篇精选 + 1 基准动态均为相对 07-31 / 08-01 / 08-02 / 08-07 / 08-11 INDEX 的新条目。已索引条目（OCOO-T、Chatzimichail eaed9309、ACP、GraCE-VAE、PinkyCaMP、ARGEN、PerturbMatch、CDE、SLIM、GeneGeoFlow、Colangelo 等）本期未重复计入。预印本/聚合源条目引用前须核对 DOI 与版本。未外发。*
