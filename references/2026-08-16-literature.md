# 文献监测周报 · 2026-08-16

> 执行：自动化 `automation-1785494084767`（虚拟镜像神经元-文献监测）
> 范围：virtual cell / mirror neuron / perturb-seq·scRNA-seq perturbation / OOD generalization / causal representation learning / calcium imaging / optogenetics / neuromodulator
> 去重基线：相对 07-31 / 08-01 / 08-02 / 08-07 / 08-11 / 08-12 六期 INDEX 去重，本期新增 15 篇精选（含 1 项开源硬件工具）+ 数条概念性注记（均标注来源层级）。
> ⚠️ 未外发 / 未发布，仅写文件与汇总。
> 🔬 严谨性：本期所有入选论文/工具均经二次检索核对 DOI/arXiv 编号、作者与核心方法（封面 15 篇全部命中原始出处）；预印本/聚合源条目引用前仍须核对版本。

---

## §0 TL;DR 优先级表（本期新精选）

| # | 方向 | 论文/来源 | 一句话价值 | 与本项目关联强度 |
|---|---|---|---|---|
| 1 | virtual cell | **Reliable single-cell perturbations explain and improve model performance**（bioRxiv 2026.08.11.744177） | 对扰动做可靠性分级，仅用「可靠」子集训练即匹配/超越全量；直击 P4 数据质量 + baseline gap 元主题 | ★★★ 直接加固「须显式超越线性基线 + 数据质量须审计」SOP |
| 2 | calcium imaging | **CAPT**（arXiv:2607.23258，清华） | 连续自回归群体 Transformer，冻结主干 + 跨数据集/跨物种适配模块迁移 | ★★★ φ 表示层 OOD 感知的可复现配方，佐证 P4-3 |
| 3 | mirror neuron | **Bougou et al., Cell 2026**（DOI 10.1016/j.cell.2026.07.032） | 人类 Utah 阵列颅内记录：SPL 支持意图/观察共享表征、MC 意图主导、SPL 情境门控 | ★★★ 与本项目「部分重叠 + 执行侧私有残差 + 相位条件化」修订互证 |
| 4 | CRL（理论） | **Adversarial Causal Intervention Falsification (ACIF)**（arXiv:2608.06427） | 对抗式「实验家选最致命干预证伪生成器」，worst-intervention IPM，区分观测拟合/干预等价/点识别 | ★★★ 把本项目「湿实验闭环证伪」升级为 principled 主动实验设计协议 |
| 5 | CRL | **DAG-FM**（arXiv:2607.11510 v2，浙大/清华/稳准智能） | 异质 FCM 下「几乎必然可识别」+ 先叶子后父母自回归建图，单卡 24G 吃千维 | ★★★ 为 φ/ψ 因果图发现提供可辨识性底座 + 闭环 GRN 初始化 |
| 6 | OOD / 校准 | **CaPTURe**（arXiv:2608.09166，COPA'26） | 基于粒子的接触感知共形，分层构型空间局部校准，进入/脱离接触均达标覆盖 | ★★★ 把「局部校准」从时间序列搬到「机制状态切换」——直击 P4-1 分 subset 校准 |
| 7 | virtual cell | **CELLens**（arXiv:2608.08430，湖南大学） | 人工引导式因果知识注入 + 反事实分析，让专家审图/改图/重训 | ★★ 因果图人机协同闭环，对标本项目孪生因果图先验注入 |
| 8 | perturb-seq | **SCITO-Perturb-seq**（bioRxiv 2026.08.08.743670） | 首个体外基因组尺度 CRISPRa + 201 表面蛋白 profiling，3.6M CD4 T 细胞，semi-NMF 模块 | ★★ 多模态扰动效应矩阵，对 ood_action 表型轴扩展 |
| 9 | perturb-seq | **scDRP**（bioRxiv 2025.11.21.689783 v3，2026-08-13 修订） | beta-VAE 解耦 + 条件最优传输，ITE 估计，泛化未见剂量/组合 | ★★ 解耦表示 + 反事实，对标 CompositionalTwin 的 φ/ψ 分解 |
| 10 | CRL | **Causal World Models (CWM)**（arXiv:2608.13456，Imperial） | 因果世界模型形式化定义 + 可辨识性层级，连接 CRL/因果发现/SCM/决策 | ★★ 为「多尺度虚拟孪生」提供术语与可辨识性底座 |
| 11 | OOD / 校准 | **RCCP**（arXiv:2608.10553，CIKM'26） | 检索校正共形：取相似历史残差作局部证据 + 标量共形校正，时间序列目标覆盖 | ★★ 局部残差检索 → P4-1 localized conformal 的可迁移配方 |
| 12 | OOD / 校准 | **Confidence Calibration of Deep Learning Systems**（arXiv:2608.12100，thesis） | 噪声感知共形(NACP) + 无监督域适应校准 + 局部 DP 共形 | ★★ 噪声/域偏移下校准，直击 P4-1（标签噪声/域偏移→饱和伪影） |
| 13 | optogenetics | **TINIscope**（USTC，开源头戴式 4 脑区 miniscope） | 0.43–0.48g 头戴式显微镜，4 脑区同步钙成像，可接光遗传模块闭环 | ★★ 湿实验闭环读+写使能（自由活动多脑区） |
| 14 | neuromodulator | **Cholinergic modulation of dopamine release drives effortful behaviour**（Nature 2026, DOI 10.1038/s41586-025-10046-6） | 伏隔核局部 ACh→DA 微环路驱动努力行为；光遗传/GRAB 双通道因果验证 | ★★ ood_neuro 第三轴（调质局部门控 DA 释放）的湿实验范本 |
| 15 | 概念性（理论） | **CWM 可辨识性 / ACIF 主动实验设计**（综述/理论） | 因果表示何时可恢复、对抗式主动干预设计 | ★★ 仅方法论启发，非引用级（见 §5 注记） |

---

## §1 Virtual Cell（虚拟细胞）

### 1.1 Reliable single-cell perturbations explain and improve model performance
- **作者**：Xi Wang, Jack Kuipers, Florian Hugi, Randall J. Platt, Niko Beerenwinkel（ETH Zürich + CZI）
- **年份/来源**：bioRxiv 2026.08.11.744177（预印本，DOI 10.64898/2026.08.11.744177）
- **核心方法**：提出**扰动可靠性分级**框架——把单细胞扰动实验按效应质量分为 reliable / shared / specific 等层级；论证大量「不可靠」扰动（弱效应/低信噪/批次混杂）会**污染模型比较与基准排名**，而非提供信息；报告对 7,170 个扰动的分级中约 65% 为 unreliable、11% shared、24% specific（数字引自预印摘要，引用前核实）；仅用 reliable 子集训练即用约 55% 数据**匹配/超越全量训练**，并在 28 细胞 pilot 上预测可靠性。
- **与本项目相关点**：① **直击 P4「数据质量 vs 信号」元主题**——我们 Norman 基准的 guide 多重度混杂、弱效应扰动同样可能是「不可靠扰动」，会系统性放大 baseline gap 假象；② **加固「机制模型须显式超越简单基线」SOP**——若把 unreliable 扰动剔除后线性基线仍胜，则「深度模型未胜」是数据层事实而非架构失败；③ 可操作化：把「扰动可靠性分级」作为我们 `regress_check` / 训练 manifest 的新前置过滤器（类比 VCBench 的 Contamination Reporting Schema，08-12 索引）；④ 与 08-07「ood_action 无判别力可能是第二次基准伪影」判断同向——shared/specific 区分正是 ood_action 该测的「真正组合效应」vs「单基因可解释部分」。

### 1.2 CELLens: Human-Guided Causal Knowledge Injection for Virtual Cells
- **作者**：Pengcheng Wang, Changjian Chen, Zhuo Tang, You Wu, Long Wang, Feng Yu, Kenli Li（湖南大学 / 岳麓山实验室）
- **年份/来源**：arXiv:2608.08430（2026-08-09，cs.HC / cs.AI）；代码 https://github.com/hnu-vis/CELLens
- **核心方法**：**人工引导式因果知识注入**——针对「自动挖掘的因果图常错、专家手绘图常不完整」的互补缺口，提供基因相似性感知的因果图可视化（混合优化 + Voronoi 边距注入），并配**反事实分析策略**（因果路径可视化 + 反事实聚类可视化）；专家可拖拽拆分/合并概念、增删连边，系统据此重训虚拟细胞；在 116,564 水稻单细胞（8 器官）上做两个案例，获 5 位领域专家正面反馈。
- **与本项目相关点**：① 把「因果图 = 专家确认的 Knowledge 而非自动生成的结论」这一立场操作化，正是本项目孪生因果图（φ/ψ 分解 + 组合先验）预期的**人类在环闭环**——我们 export_protocol 的湿实验回流可借鉴其「反事实干预→观察下游→精炼图」三步；② 反事实聚类（基因用 [语义, 反事实响应] 双特征聚类）↔ 本项目 `do(action)` 反事实预测的可视化审计；③ 局限（Pith 审稿指出）也值得记：精炼循环自指——专家靠同一模型的反事实响应改图，若模型有偏会强化错误；这恰是本项目「反事实 `do(action=other)` 与 base 逐位相等」疑点的方法论预警（见 MEMORY §4 镜像轴已死）。

---

## §2 Mirror Neuron（镜像神经元）

### 2.1 Bougou et al. — Hierarchical and context-dependent encoding of actions in human posterior parietal and motor cortex
- **作者**：Vasiliki Bougou, Jorge Gamez, Emily R. Rosario, Charles Liu, Kelsie Pejsa, Ausaf Bari, Richard A. Andersen（Caltech，Andersen 组）
- **年份/来源**：*Cell* 2026，DOI 10.1016/j.cell.2026.07.032（Received 2025-11-21；Accepted 2026-07-22；Published online 2026-08-11）；PMID 42580336
- **核心方法**：在两例四肢瘫痪受试者 Utah 阵列颅内记录中，比较**运动皮层（MC）**与**后顶叶上区（SPL）**在「意图执行」vs「观察」手动作时的编码；发现 **MC 强编码意图、观察时仅群体级弱响应**；**SPL 在单单元与群体级均支持意图/观察共享表征**；当指令动作与观察动作不一致同时呈现时，**SPL 仅在行为相关时才编码观察动作（情境门控）**，而 MC 保持意图主导。
- **与本项目相关点**：① **直接互证** MEMORY 中 Chatzimichail（Sci Adv 2026）的「执行↔观察不对称性」修订——本项目孪生因果图已改为「**部分重叠 + 执行侧私有残差 + 相位条件化**」，本文给出人脑实证：MC=执行私有（意图主导）、SPL=共享+情境门控，正对应「部分重叠潜空间 + 观察分支门控」；② **情境门控** = 本项目 g(u) 不确定度门控的生物原型——SPL 按行为相关性决定是否把观察信息并入共享表征，类比我们的 ψ 在 epistemic 高时收缩到加性；③ 方法学：用「不一致动作对」范式分离意图 vs 观察，可借鉴为我们湿实验「反事实 `do(action=other)` 预测 vs base」的方向性判据增强。

> 本期镜像神经元**新增原发性研究即此 1 篇**；其余多为该文媒体延续。celestinemoon 所述「镜像神经元追踪运动而非目标」文章对应历史已索引的 Chatzimichail eaed9309，不重复计入。

---

## §3 Perturb-seq / scRNA-seq Perturbation

### 3.1 SCITO-Perturb-seq: A genome-wide CRISPR activation map of surface protein expression in human CD4 T cells
- **作者**：Yutong V. Wang, Junha Park, Min Cheol Kim, … Alexander Marson, Jimmie Ye, Byungjin Hwang（UCSF / UC Berkeley）
- **年份/来源**：bioRxiv 2026.08.08.743670（预印本，DOI 10.64898/2026.08.08.743670）
- **核心方法**：**首个体外基因组尺度 CRISPRa 筛选 + 直接高维表面蛋白 profiling**——组合索引单细胞流式测序（combinatorial-indexed cytometry-seq）耦联 pooled CRISPRa，绘制 **201 个表面蛋白**在 **3.6M 人 CD4 T 细胞**上的因果调控图；16% 激活基因显著改变至少一个表面蛋白；用 **semi-NMF** 分解扰动效应矩阵得 5 个模块（对应已知 CD4 T 细胞状态），模块按「对扰动的共享响应」而非「静息共表达」聚类。
- **与本项目相关点**：① **表型轴扩展**——表面蛋白是转录下游的「非加和」层（翻译后调控），印证 U-Pert（08-12）「扰动表型含表达态之外轴」；② 多模态效应矩阵（基因×蛋白）↔ 我们 GOAI 侧「蛋白均值基线 / Matched Control」评测语境，提示 ood_action 不能只看 RNA 位移；③ semi-NMF 模块 ↔ 我们 epistasis 代理扫描的「rich-epistasis 代理」可作为机制级而非表型级分解。

### 3.2 scDRP: Single-cell disentangled representations for perturbation modeling and treatment effect estimation
- **作者**：Jianle Sun, Petar Stojanov, Kun Zhang（Broad / CMU / 张坤组）
- **年份/来源**：bioRxiv 2025.11.21.689783 **v3**（2026-08-13 修订；DOI 10.1101/2025.11.21.689783）
- **核心方法**：**生成式解耦框架**——用稀疏正则化 β-VAE 分离「扰动相关 / 扰动无关」潜变量（带渐近正确性保证），在混杂条件下假设扰动效应分位保持，于潜空间做**条件最优传输（conditional OT）**推断反事实态并估计**个体化治疗效应（ITE）**；在模拟与真实数据上准确估计 ITE，且**泛化到未见扰动剂量与组合**。
- **与本项目相关点**：① **解耦 + 条件 OT 反事实** ↔ CompositionalTwin 的 φ（基因效应嵌入）/ ψ（交互头）分解的可计算托底——scDRP 证明「解耦潜变量 + OT 反事实」能泛化未见组合，正是我们 ood_action 想要的属性；② ITE + 未见剂量/组合泛化 ↔ 我们「组合双扰动·唯一 should_generalize」子集的直接对标；③ 与 08-11 GeneGeoFlow / 08-07 CDE-δ 同向：解耦是 OOD 泛化的前提。

---

## §4 OOD Generalization（分布外泛化）

- **虚拟细胞侧（见 §1.1 Reliable single-cell perturbations）**：可靠性分级证明「不可靠扰动污染基准排名」——本项目 OOD 强制须先过数据质量审计，否则 baseline gap 可能是伪影。
- **Perturb-seq 侧（见 §3.2 scDRP）**：解耦 + 条件 OT 在未见剂量/组合上泛化，是 ood_action 的可计算范本。
- **校准侧（见 §5/§9 的 CaPTURe / RCCP / NACP）**：局部校准从「分 subset」走向「机制状态切换 / 检索相似残差 / 噪声感知」三方向。

---

## §5 Causal Representation Learning（因果表征学习）

### 5.1 Adversarial Causal Intervention Falsification (ACIF)
- **作者**：Mojtaba Eslami
- **年份/来源**：arXiv:2608.06427（2026-08-05，cs.LG）
- **核心方法**：把因果生成建模形式化为**序贯博弈**——结构因果生成器提议观测/干预分布，**对抗式实验家**选择「最能证伪生成器」的干预；判别器以干预为索引，检验生成器是否复现对应 post-intervention 律；将对抗目标**精确归约为 worst-intervention IPM**；证明：(i) 干预等价类内识别（separating 干预族下点识别）；(ii) 混合策略均衡存在；(iii) 有限样本一致收敛 + 基于间隔的模型选择保证；(iv) 分歧驱动的序贯设计的对数级消除保证；并给线性高斯例：两条观测不可区分的因果方向被单个精心干预分离。
- **与本项目相关点**：① **把本项目「湿实验闭环证伪」升级为 principled 主动实验设计**——我们现状是「预测→光遗传干预→误差回流」启发式；ACIF 给出「选哪个干预最能区分候选机制」的最优准则（worst-intervention IPM），可直接用作我们 E2 湿实验的**干预选择策略**（优先选候选模型分歧最大的 `do(action)`）；② 区分「观测拟合 / 干预等价 / 点识别」三件事，正是我们双栏（记忆 vs 机制泛化）判据的理论对应——**仅观测拟合 ≠ 机制识别**，须用干预等价类收窄；③ 与 08-07 ACC-CRL 干预一致性锚点、08-11 UMNI-CRL 同构。

### 5.2 DAG-FM: A Foundation Model for Causal Discovery under Heterogeneous Causal Mechanisms
- **作者**：Yikang Chen, Zhengkang Guan, Haoyuan Qian, Xingxuan Zhang, Peng Cui, Yi Yang, Fei Wu, Kun Kuang（浙江大学 / 清华大学 / 稳准智能）
- **年份/来源**：arXiv:2607.11510 **v2**（2026-08-02；v1 2026-07-13）
- **核心方法**：**因果发现基础模型**——把 DAG 恢复解耦为「先预测叶节点、再为每个节点预测父节点」的双阶段自回归，用 **Mixture-of-Leaf-Experts (MoLE)** 动态路由到适配不同 FCM 家族（LiNGAM/ANM/HNM/PNL）的专家；证明在异质机制先验空间下**真实 DAG 几乎必然可识别**；单卡 24G BF16 吃下 1,000 维 / 100k 样本，SHD/F1 超 NOTEARS/CAM/GOLEM/Foundation-Causal。
- **与本项目相关点**：① **为 φ/ψ 因果图发现提供可辨识性底座**——我们孪生因果图的 GRN 边权初始化可从 DAG-FM 式「异质机制感知」因果发现取先验；② **先叶子后父母的自回归建图**天然规避非法环路，可借鉴为我们因果图构建的**结构合法性保证**（避免 P4-2 加法反解时的伪环）；③ 与 08-11 Anchor–Stabilizer Theory（干预/机制锚定可辨识性）同向，且给出「异质机制下仍可识别」的更强结论。

### 5.3 Causal World Models (CWM)
- **作者**：Avinash Kori, Fabrizio Russo（Imperial College London）
- **年份/来源**：arXiv:2608.13456（2026-08-13，cs.AI）
- **核心方法**：给出**因果世界模型（CWM）的形式化定义**，将其锚定在「要支持的任务」上，连接因果表征学习、以对象为中心的学习、因果发现、结构因果模型与基于模型的决策；提出四级「因果阶梯」：Rung 1 感知→Rung 2 干预/效用→Rung 3 反事实/生成推理；澄清 CWM 各组件**何时可从数据恢复、到哪类等价**的可辨识性结论。
- **与本项目相关点**：① 为本项目「多尺度虚拟孪生」提供**统一术语与可辨识性层级**——我们 φ(基因效应)/ψ(交互)/g(u)(不确定度门控) 正是 CWM 中「实体属性 / 实体间交互 / 实体-环境交互」的实例化；② 其「预测器≠表征≠因果模型」的区分，呼应我们铁律「OOD 指标达标 ≠ 学到正确机制」（07-31 警示）；③ 仅理论框架，非新方法，作概念底座。

---

## §6 Calcium Imaging（钙成像）

### 6.1 CAPT: A Multi-task Continuous Autoregressive Transformer enabling Cross-dataset and Cross-species Transfer for Calcium Population Dynamics
- **作者**：Xinhong Xu, Yimeng Zhang, Yuanlong Zhang（清华大学）
- **年份/来源**：arXiv:2607.23258（2026-07-25 v1，2026-08-04 v2；代码 https://github.com/TSuXinH/CAPT）
- **核心方法**：**连续自回归群体 Transformer**——对连续钙迹做 continuous patch tokenization（patch 长 L=8 样本），双轴 Transformer（时间因果注意力 + 群体轴注意力）以 MSE 预测下一连续 patch；在大规模小鼠钙成像上预训练，**迁移时冻结主干、仅更新神经元/会话嵌入或任务头**；在 8 个独立数据集（小鼠/斑马鱼幼虫/C. elegans）上做神经群体预测与行为解码，一致超专用/通用基线；C. elegans 上 CAPT 嵌入形成跨数据集共享功能空间并捕获细胞身份结构（NeuroPAL）。
- **与本项目相关点**：① **φ 表示层 OOD 感知的可复现配方**——「冻结预训练主干 + 轻量适配模块」正是我们 P4-3「φ 须 OOD 感知」的工程范本：跨物种/跨范式迁移靠的是适配模块而非重训，类比我们 ψ 的 g(u) 门控在 epistemic 高时收缩；② 连续 tokenization 避开离散化信息损，对我们处理连续扰动强度（Perturbation Curve 式连续效应，见 08-12 概念性注记）有借镜；③ **警示（Pith 审稿）**：固定 L=8 patch 在不同采样率下跨物种时间不对齐，可能让「跨物种迁移」部分依赖巧合 token 尺寸——提醒我们适配模块的「时间轴对齐」须显式处理，否则是伪迁移。

---

## §7 Optogenetics（光遗传）

### 7.1 TINIscope: Tightly Integrated Neuronal Imaging microscope
- **作者**：zhoupc 等（中国科学技术大学，USTC）
- **年份/来源**：开源硬件，GitHub https://github.com/zhoupc/TINIscope（论文发表于《国家科学评论》NSR；0.43–0.48g 头戴式）
- **核心方法**：**超紧凑头戴式荧光显微镜**——单台 0.43g，可 4 台同戴（总重 <2g）；FOV 450×450μm、光学分辨率 ≈2.2μm、40Hz、适配 GRIN lens 深脑成像；支持**最小 1.02–1.2mm 间距的 4 脑区同步钙成像**；可接**光遗传 / 电刺激 / LFP 模块**做多模态实验；换相器解决自由活动动物线缆缠绕；开源光学/电子/机械设计。
- **与本项目相关点**：① **湿实验闭环读+写使能（自由活动多脑区）**——比 08-11 索引的 Adesnik 两光子全息介观显微镜（固定制备、近单细胞）更贴近「自由行为多脑区」场景；我们 E2 湿实验证伪「镜像轴」若要做自由行为多脑区，TINIscope 是_low-cost 开源替代；② 与 08-12 Neuropixels Opto（深脑读写同体）互补：TINIscope 管皮层多脑区钙成像 + 光遗传，Neuropixels Opto 管深脑电生理 + 光遗传；③ 开源 → 可直接 clone 进我们 export_protocol 工具链的成本估算。

> 本期光遗传**无新增期刊化原发性研究**；TINIscope 为开源硬件工具（高价值使能，非论文）。Adesnik 全息介观显微镜已于 08-11 索引，不重复。

---

## §8 Neuromodulator（神经调质）

### 8.1 Cholinergic modulation of dopamine release drives effortful behaviour
- **作者**：G. C. Touponse, M. B. Pomrenze, T. Yassine 等；Neir Eshel 组（Stanford）
- **年份/来源**：*Nature* 2026，DOI 10.1038/s41586-025-10046-6（Online 2026-01-28；卷 651, 1020–1029）
- **核心方法**：用行为任务（FR1→FR46 鼻触换相同奖励）证明**努力放大对相同奖励的 DA 反应**；用 **GRAB-DA + GRAB-ACh** 双通道光纤光度 + **光遗传**（直接刺激 NAc DA 轴突、沉默 VTA 细胞体、抑制 ChAT 中间神经元）因果验证：**伏隔核局部 ACh 中间神经元在奖励前 ~400ms 放电峰，经 α4/α6 烟碱受体局部增强 DA 轴突释放**；阻断该微环路仅损害高努力情境下的努力行为，低努力消费不受影响。
- **与本项目相关点**：① **ood_neuro 第三轴范本**——调质（ACh）**局部门控** DA 释放（而非全局标量），正是 MEMORY 中 Colangelo（突触级门控）/ Dalla Porta（空间异质）的湿实验实证；② 方法学闭环：光遗传直接刺激轴突 + 双通道光纤光度 + 药理学阻断三路交叉验证，可作为我们 E2 湿实验「证伪模型预测」的**金标准协议**（多路独立因果验证，避免单路假阳性）；③ 与 08-11 Khaliq「运动路由」、08-12 Dalla Porta 同列 ood_neuro 调质局部门控证据链。

> 注：本文 online 为 2026-01，非本周新发，但为本周期新纳入监测的高价值 ood_neuro 锚点（前期未索引），按「新发现条目」计入并标注真实日期。

### 8.2 概念性注记（低置信、综述源）
- 「预测编码 / 精度加权」综述把调质 = 预测误差精度加权（ACh 放大失匹配、DA/5-HT 去同步慢振荡），与 08-12 §8.3 同向——调质是 g(u) 门控的生物对应。仅理论呼应，非正式引用。

---

## §9 不确定度与校准专栏（跨方向）

| 来源 | 方法 | 对本项目不确定度层的映射 |
|---|---|---|
| Reliable single-cell perturbations (§1.1) | 扰动可靠性分级 → 剔除 unreliable 再比模型 | P4 数据质量审计前置（`regress_check` 新过滤器） |
| CaPTURe (§5/本栏) | 基于粒子的接触感知共形，分层构型空间局部校准 | P4-1「机制状态切换」下分 subset 校准的可操作范本 |
| RCCP (本栏) | 检索相似历史残差作局部证据 + 标量共形校正 | P4-1 localized conformal 的「局部残差检索」配方 |
| NACP / Confidence Calibration thesis (本栏) | 噪声感知共形（标签噪声下重建无噪 conformity）+ 无监督 DA 校准 + 局部 DP 共形 | P4-1 在标签噪声/域偏移下退化的对症解法 |
| CAPT (§6.1) | 冻结主干 + 适配模块跨分布迁移 | φ 表示层 OOD 感知（P4-3）范本 |
| ACIF (§5.1) | worst-intervention IPM 主动证伪 | E2 湿实验「选最致命干预」策略 |

> 主线收敛（承接 07-31→08-12）：**表示层、融合层、不确定度层都必须 OOD 感知**。本期新增可移植工具链 = ① 扰动可靠性分级（数据质量门控，治 P4 伪影根）；② CaPTURe 的「机制状态切换局部校准」（把分 subset 从静态划分推进到动态状态切换）；③ ACIF 的主动干预证伪（把湿实验闭环从启发式升级为 worst-case 最优）。

---

## §10 基准动态（非论文）

- **Arc 2026 Virtual Cell Challenge**：8/20 正式开赛（zero-shot 多细胞系基因敲除响应预测）进入倒计时（见 08-12 §10）。建议把本项目 Norman/GOAI 的 ood_agent 指标与该挑战 zero-shot 多细胞系设定对齐，作为对外可比对锚点（内部使用，不外发）。本期无新增赛制变更，仅作时间提醒。
- **Reliable single-cell perturbations（§1.1）隐含基准启示**：若把「扰动可靠性」作为新评测维度加入我们 ood_* 子集，可区分「真 OOD 泛化缺口」vs「数据不可靠造成的假缺口」——建议补进每周新证据槽的方法论候选。

---

## §11 本周最值得借鉴的方法 + 对多尺度虚拟孪生的启示

**方法一：Reliable single-cell perturbations 的「扰动可靠性分级」框架**。它把一个长期被我们定性讨论的问题（「baseline gap 可能是数据伪影」「ood_action 无判别力可能是第二次基准伪影」）变成**可操作的前置过滤器**——先按效应质量把扰动分为 reliable/shared/specific，仅用 reliable 子集做模型比较。这直接回答了我们 P4 的核心焦虑：当我们抱怨「线性基线赢」时，先要确认比较是在干净数据上做的；不可靠扰动会同时拖垮深度模型与污染基准排名。

**方法二：ACIF 的「worst-intervention IPM 主动证伪」**。它把本项目「湿实验闭环证伪机制泛化」从启发式升级为**有理论保证的主动实验设计**——对抗式实验家选「最能区分候选机制」的干预，判别器以干预为索引检验生成器是否复现 post-intervention 律，并证明 worst-intervention 目标等价于 IPM、在 separating 干预族下点识别。这正好填补我们 E2 湿实验「该做哪个干预」的策略空白。

**对多尺度虚拟孪生建模层的三层启示**（承接 08-12 框架）：
1. **表示层（φ）**：Reliable perturbations 证明「数据质量决定基准可信度」→ φ 的增益比较必须先过**扰动可靠性分级**审计；CAPT 的「冻结主干 + 适配模块」给出 φ OOD 感知（P4-3）的可复现工程样板。
2. **融合层（ψ）**：ACIF 的「干预等价类收窄」→ ψ 不应只追平加性，而须显式建模「哪些 `do(action)` 能区分候选机制」；CaPTURe 的「机制状态切换局部校准」提示 ψ 的 g(u) 门控应按**当前机制状态**（而非全局）切换加性/交互。
3. **不确定度层**：CaPTURe（接触/非接触状态切换下分别达标覆盖）+ RCCP（检索相似残差）+ NACP（标签噪声/域偏移下重建无噪 conformity）三者组装成本项目 **P4-1 的 localized conformal 落地配方**——单标量 `_q` 退役，改为按 ood_* 子集 + 机制状态分别报覆盖 / ECE / 多样性，并对「不可靠扰动造成的饱和」自动报警。

---

*去重声明：本期 15 篇精选（含 TINIscope 开源硬件）均为相对 07-31 / 08-01 / 08-02 / 08-07 / 08-11 / 08-12 INDEX 的新条目。已索引条目（OCOO-T、Chatzimichail eaed9309、VCBench、U-Pert、Mechanisms Matter、CauFinder、Anchor–Stabilizer、VADER1、cOVC、Neuropixels Opto、Dalla Porta、Colangelo、Adesnik/Abdeladim、GeneGeoFlow、SLIM、CDE、CAPT 之外的历史钙成像条目等）本期未重复计入。ACh→DA 文（Nature 2026-01-28）为本周期新纳入的高价值 ood_neuro 锚点，已标注真实日期。预印本/聚合源条目引用前须核对 DOI 与版本。未外发。*
