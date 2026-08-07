# 文献监测周报 · 2026-08-02（增量，对比 2026-07-31 / 2026-08-01）

> **范围**：virtual cell、mirror neuron、perturb-seq / scRNA-seq perturbation、OOD generalization、causal representation learning、calcium imaging、optogenetics、neuromodulator
> **方法**：Web 检索（截至 2026-08-02），相对 `2026-07-31-literature.md` 与 `2026-08-01-literature.md` 去重，仅收录本周新出现的论文/预印本或显著进展；每条附「与本项目相关点」映射至三条硬约束（记忆 vs 机制泛化 / OOD 子集 / 不确定度）及建模层（多尺度虚拟孪生、do-演算、闭环层）。
> **注意**：部分条目来自预印本/聚合源，正式引用前请核对原始 DOI/venue。本报告仅入库，不外发。
> **已覆盖、本周不再重列**：SCALE / OCOO-T / D-SPIN / Chreode / X-Cell（08-01 §1）、Chin 神经算子共形 / PI-SCM（08-01 §4）、Afshar-Saber hiPSC 全光管线 / 记忆印迹 / Passmore 结果驱动显微 / STN-LID（08-01 §6–7）、SCCVAE / ADAPRE / Mixscale（07-31 §3）、Chatzimichail et al. Science Advances 2026-06-19 镜像神经元群体几何（07-31 §2，本周仅媒体延续）。

---

## 1. Virtual Cell（虚拟细胞）· 新增

### 1.1 PerturbDiff — Functional Diffusion for Single-Cell Perturbation Modeling
- **作者**：Xinyu Yuan, Xixian Liu, Yashi Zhang, Zuobai Zhang, Hongyu Guo, Jian Tang（Mila – Québec AI Institute 等）
- **年份·来源**：arXiv:2602.19685, 2026（报道为 ICML 2026）
- **核心方法**：把单细胞扰动响应建模从「单细胞」抬升到「整个分布」。关键洞察：scRNA-seq 是破坏性的（同一细胞前后不可配对），且未被观测的隐变量（微环境、批次效应）使**同一条件对应一族可能的响应分布**，而非唯一固定分布。做法：① 用**核均值嵌入（kernel mean embedding）**把响应分布映射为 RKHS 中的点，直接在**函数空间定义 DDPM 式扩散**；② 推导出的去噪目标在 RKHS 距离上**数学等价于 MMD**（分布感知损失，消融实验去掉 MMD 仅留 MSE 即崩溃——因零膨胀稀疏性 MSE 会陷入「预测全 0」）；③ **边缘分布预训练（marginal pretraining）**：用 CellxGene 6100 万无扰动细胞预训练无条件流形，赋予零样本/低数据适应能力；④ 在 Tahoe100M（1 亿+细胞/1100 药）、PBMC、Replogle 上 14 项指标 SOTA，DE（差异表达）恢复显著领先。
- **与本项目相关点**：**直击本项目 epistasis/OOD 结论的底层问题**。① 它显式建模「未被观测隐因子 → 响应分布的流形」——正是本项目 Norman 真实基准上「双扰动非加和（epistasis）、加法先验天花板≈0.63」的结构性来源；把 epistemic 不确定性从黑箱残差提升为**分布族的可解释变异性**，可直接对接本项目「epistemic/aleatoric 分解」不确定度目标；② 「分布级建模 + 零样本泛化到未见扰动」对齐本项目 ood_action（未见基因对）轴；③ 边缘预训练提供「用无扰动分布打底、扰动数据做条件偏移」的可复用训练范式，呼应 Chreode（08-01 §1.4）的「共享潜几何预训练」思路，但更聚焦**分布级**而非单步动力学；④ MMD 内生为损失（而非启发式正则）是评测「生物学忠实分布匹配」可借鉴的配方。

### 1.2 Lingshu-Cell — A Generative Cellular World Model toward Virtual Cells
- **作者**：阿里巴巴达摩院团队（张晗等）
- **年份·来源**：arXiv:2603.25240, 2026（v1 2026-03；配套新闻 2026-04/06）
- **核心方法**：生成式**细胞世界模型**，采用**掩码离散扩散（masked discrete diffusion, MDDM）**架构，在单细胞转录组的**离散 token 空间**（~1.8 万基因全转录组，无需预先选基因）直接建模稀疏/非连续数据；条件输入细胞类型 + 扰动（敲除基因 / 细胞因子），预测扰动后全转录组状态分布；在 Virtual Cell Challenge 基因扰动基准（25 队）综合第一，并成功预测 90 种细胞因子对供体 PBMC 的反应。
- **与本项目相关点**：① 「扰动预测 = 条件生成世界模型、输出状态分布而非点估计」= 本项目 `do-` 干预后分布预测的同类范式，且离散扩散天然兼容 scRNA-seq 稀疏性（呼应 PerturbDiff 对 MSE 失效的观察）；② 强调「跨组织/物种捕捉状态分布与细胞亚型比例」对应本项目 OOD（未见主体 ood_agent）轴；③ 可作为本项目孪生生成头的**架构候选**（离散扩散 vs Chreode 的 DiT 连续动力学）——建议对比二者在 Norman 基准上的机制判别力。

---

## 2. Mirror Neuron（镜像神经元）· 新增（少量原发性）

> 本周出现 1 篇新原发性 + 1 篇学位论文级资源，主体仍围绕「执行/观察在群体几何层对齐」（07-31 §2 已覆盖）。Chatzimichail et al. Science Advances 2026-06-19 群体几何文本周仅媒体延续，不计新。

### 2.1 GABAergic/glutamatergic E-I balance shapes motor resonance during action observation
- **作者**：Aynur Ragimova, Madina Imanaeva, Carlos Nieto-Doval, Victoria Moiseeva, Matteo Feurra 等（HSE University / Research Center of Neurology, Moscow）
- **年份·来源**：Front. Hum. Neurosci., 2026-03-23, 20:1746409（DOI 10.3389/fnhum.2026.1746409）
- **核心方法**：用单脉冲 TMS 测人在观察别人手指动作时的运动诱发电位（MEP，即人镜像神经系统 MNS 的生理标记——运动共振 motor resonance）；以 ICF（谷氨酰胺能易化）/ SICI（GABA-A 介导抑制）刻画 M1 皮层兴奋-抑制（E-I）平衡。发现 AO 后 ICF/SP 通路 MEP 升高、SICI 抑制减弱；个体运动共振强度受稳定 E-I 特性（ICF/SICI）约束。
- **与本项目相关点**：① 把「运动共振（MNS 的人类间接标记）」锚定到**皮层 E-I 平衡**——本项目「感知/动作共享潜空间对称性」在环路尺度的可测生理基底；② E-I 平衡是本项目孪生「突触→环路」尺度可直接建模的调节变量，且 E-I 失衡对应 ood_neuro（新调质/病理状态）轴的环路表型；③ 方法学上提示：闭环层校验 twin 预测时，可用 TMS 诱导的 E-I 指标作为**人脑侧的可外推因果验证**锚点。

### 2.2（学位论文）Predictable Context–Based Encoding of Observed Actions in Mirror Neurons of Macaque Premotor Area F5
- **作者**：（Tübingen 大学博士论文，2026-05-26 提交，口试 2026-03-16）
- **年份·来源**：hsbiblio.uni-tuebingen.de, 2026（PhD Thesis）
- **核心方法**：用 video-blocked / rule-blocked 双范式训练两只猕猴根据示指动作选择自身动作；记录 F5 区 859 + 288 个神经元（58.2% / 79.2% 为镜像神经元），用**降维 + 信息分配**把群体活动分解为「行为规则 / 观察动作 / 执行动作」三类驱动并追踪时序动态。结论挑战「行动理解 = 映射到自身运动储备」主流框架，支持**反应选择假说（response selection hypothesis）**：观察期视觉信息驱动有限，主导驱动是观察者即将执行动作的计划、并随执行临近渐增。
- **与本项目相关点**：① 与本项目「执行/观察同源悖论」同源——它给出的答案是「观察编码首先服务于自身的反应选择，而非对称镜像」，为「自我/他者同源」悖论提供另一视角（镜像性可能源于**动作规划空间的共享**而非逐神经元对称）；② 降维 + 信息分解的群体分析方法可直接复用为闭环层校验「twin 是否在正确子空间编码动作」的方案。

---

## 3. Perturb-seq / scRNA-seq Perturbation · 新增

> 本方向「可解释/机制化」主干已由 D-SPIN、Chreode、X-Cell（08-01 §1）覆盖。本周新增聚焦**因果图发现中的混杂鲁棒性**与**分布级建模**（PerturbDiff 见 §1.1）。

### 3.1 Confounder-robust causal discovery in Perturb-seq via proxy and instrumental variables
- **作者**：Kwangmoon Park, Hongzhe Li（University of Pennsylvania）
- **年份·来源**：arXiv:2601.01830, v3 2026-06（初版 2026-01）
- **核心方法**：针对 Perturb-seq 中**未观测混杂（细胞周期、染色质可及性、成本约束导致的未测基因）**破坏因果 DAG 推断的问题，提出用**代理变量（proxy）+ 工具变量（IV）**策略，利用多扰动结构对「任意被省略混杂」做无偏 DAG 估计；在 K562 CRISPRi 上优于忽略未测混杂的基线，回收已知功能模块（胞质/线粒体翻译、有丝分裂检查点）。
- **与本项目相关点**：① 直接服务本项目「机制先验 + 因果图」底座——它把 **do-演算所需的无混杂假设**用 proxy/IV 在技术层加固，正是本项目孪生因果图在真实 Perturb-seq（含 Norman）上落地时必须面对的偏差来源；② 与 ADAPRE（07-31 §3，把 CRISPR 当 IV 处理异质敲低效率）互补：ADAPRE 解决**扰动强度异质 + 可环结构**，Park&Li 解决**未观测混杂**，二者共同构成「Perturb-seq → 可审计因果图」的完整鲁棒化工具箱；③ 建议把 proxy/IV 思路纳入本项目 ood_action 因果图构建 SOP，避免「看起来泛化、实则被混杂驱动」的假机制。

---

## 4. OOD Generalization（分布外泛化）· 新增

> Chin 神经算子共形（08-01 §4.1）、PI-SCM 因果共形（08-01 §4.2）已覆盖。本周新增聚焦**分布漂移下的覆盖保证**与**目标域自适应共形**，均直接服务「不确定度强制」硬约束。

### 4.1 Coverage Guarantees for Pseudo-Calibrated Conformal Prediction under Distribution Shift
- **作者**：Farbod Siahkali, Ashwin Verma, Vijay Gupta
- **年份·来源**：arXiv:2602.14913, v3 2026-07（初版 2026-02）
- **核心方法**：在**有界 label-conditional covariate shift** 模型下，用 domain adaptation 工具推导目标覆盖率的**下界**（用源域分类器损失 + Wasserstein 漂移度量表示），据此设计**伪校准（pseudo-calibration）集合**——通过松弛参数膨胀 conformal 阈值，把目标覆盖率维持在预设水平之上；提出 source-tuned 伪校准，在硬伪标签与随机标签间插值。
- **与本项目相关点**：给出「漂移多大 → 覆盖掉多少 → 阈值该放多宽」的**可量化下界**，比 Chin 的经验覆盖更可审计；本项目 ood_action/ood_agent/ood_neuro 子集本质是 covariate shift，可用 Wasserstein 度量各 OOD 轴的漂移幅度并据下界设定 conformal 阈值，使「不确定度强制」从经验覆盖变成**有界保证**。

### 4.2 Audited Conformal Prediction (ACP) for Classification under Unknown Distribution Shift
- **作者**：Yanfei Zhou, Rizal Fathony, Nam H. Nguyen, Matteo Sesia
- **年份·来源**：arXiv:2606.14909, 2026-06
- **核心方法**：用目标群体的一小份有标数据训练**辅助审计模型**，识别遗留模型可能失效的输入；把审计输出并入 conformal 框架，在保边际覆盖的同时显著提升**条件覆盖**（含显式 group-conditional 覆盖保证）。两种集成策略：边际覆盖+改进条件性能、显式组条件覆盖。
- **与本项目相关点**：① 「少量目标域标注 + 审计模型定位失效样本」= 本项目闭环层「湿实验少量标定 → 回流修正预测」的共形版表述；② **组条件覆盖**直接对应本项目按 `ood_action / ood_agent / ood_neuro` 分组的覆盖审计需求（目前 write_report 的校准表只查边际覆盖），建议扩展为分组条件覆盖；③ 可直接把 ACP 接进每周预测清单的「先挑哪些细胞做光遗传验证」主动实验设计。

### 4.3 RLSCP / WQLCP — Reconstruction-Loss-Scaled Conformal Prediction under Shift
- **作者**：（WQLCP 综述/TheMoonlight，源论文 2026）
- **年份·来源**：2026（weighted/adaptive conformal under distribution shift）
- **核心方法**：RLSCP 用 VAE **重构损失**作为样本不确定性的代理（视为 epistemic uncertainty），发现重构损失与分布漂移严重度强相关；用测试集重构损失的 1−α 分位动态缩放传统 CP 分数，漂移越大预测集越宽以恢复覆盖。WQLCP 进一步做加权自适应。
- **与本项目相关点**：把「重构损失 = epistemic 不确定性」直接变成分布漂移下的**自适应区间宽度**——与本项目孪生若采用 VAE/扩散生成头天然兼容（重构损失现成可得），可作为 Chin 归一化共形（MC Dropout 版）之外的**VAE 原生**不确定度配方。

---

## 5. Causal Representation Learning（因果表征学习）· 新增（本周强相关）

> 07-31 §5（CausalVerse / Causal Differentiating Concepts / Reward-oriented CRL，NeurIPS 2025）已覆盖。本周新增**直接落在遗传扰动 + 隐因果图 + 未见干预组合**上的方法，与本项目核心最贴合。

### 5.1 GraCE-VAE — Graph-aware Causal disEntanglement VAE
- **作者**：Zhang et al.（arXiv:2509.01916, v2 2026-06）
- **核心方法**：图感知变分自编码器做**因果解耦**：① 编码器把**异质网络（已知实体 + 群组节点）**作为结构侧信息注入隐表征，偏置模型朝向符合关系模式的解释；② 解码器在隐变量上实例化一个 **SCM / 学习到的 DAG**；干预通过「干预编码器把干预指示映射为对隐变量的软选择 + 机制修改」实现，只改动被选隐变量的因果机制、其余不动；③ 在**三个遗传扰动数据集**上学习到的隐 DAG 支持准确的**干预结果预测**，尤其对**未见扰动组合（unseen perturbation combinations）**给出有生物学意义的假设；④ 继承既有因果 VAE 的**可辨识性保证**（i.i.d.-within-regime）。
- **与本项目相关点**：**本周与本项目核心最贴合的方法之一（见文末总结）**。① 它把「结构先验（基因/实体网络）+ 干预数据 → 隐因果 DAG + do-干预结果」端到端学出来，正是本项目「机制先验 + 不纯黑箱 + do-演算」在表征尺度的可直接复用模板；② 「未见扰动组合预测」= 本项目 ood_action（held-out 基因对）轴的同义表述，且它显式给出可辨识性保证——回应本项目一直被追问的「OOD 好是否=学到正确机制」；③ 建议把 GraCE-VAE 的「干预编码器 + DAG 解码器」结构用作本项目孪生隐空间的因果层，替代/补充当前 CompositionalTwin 的加性 φ。

### 5.2 CREATOR — Linear Causal Representation Learning by Topological Ordering, Pruning, Disentanglement
- **作者**：（arXiv:2509.22553, 2025-09 预印，2026 活跃）
- **核心方法**：提出线性 CRL 算法 CREATOR，在 ≥d 个环境数据下，通过「拓扑排序 + 特征恢复 → DAG 剪枝 → 特征解耦」三步，在无限数据极限下**辨识隐特征与真实因果 DAG**（至多 sur-等价）；理论基于多环境下的分布可辨识性。
- **与本项目相关点**：提供「多环境（multi-environment）→ 因果 DAG 可辨识」的**白箱线性**基线方法；本项目若有多个细胞情境/批次/调质状态作为「环境」，可直接用 CREATOR 类方法做可辨识性对照，验证孪生因果图是否真被数据支撑而非过参数化记忆。

### 5.3 Mechanism Sparsity Partial Disentanglement（JMLR 2026）
- **作者**：（JMLR 27:24-0771, 2026，CLeaR 2022 扩展）
- **核心方法**：提出**机制稀疏性正则**作为解耦新原理——当隐因子稀疏依赖观测辅助变量/过去隐因子时，同时学隐因子与稀疏因果图即可**非参数可辨识**（至多「一致性」等价，允许部分因子仍纠缠=partial disentanglement）；给出完全解耦的图判据，并用 VAE + 稀疏约束在合成数据验证，含多节点未知目标干预的解耦。
- **与本项目相关点**：① 「干预（含未知目标）→ 解耦」直接对齐本项目闭环层「光遗传干预 → 回流解耦因果」；② 机制稀疏性可作为本项目孪生隐空间的**结构正则**，在不过度假设 DAG 无环的前提下获得可解释因果模块。

---

## 6. Calcium Imaging（钙成像）· 新增（闭环层工具）

> Afshar-Saber hiPSC 全光管线 / 记忆印迹（08-01 §6）已覆盖。本周新增聚焦**传感器与全光串扰消除**，直接支撑闭环层硬件。

### 6.1 PinkyCaMP — mScarlet-based red calcium sensor
- **作者**：Ryan Fink, Shosei Imai, …, Olivia Andrea Masseck 等
- **年份·来源**：Nature Methods, 2026-04-24, 23:998–1010（DOI 10.1038/s41592-026-03065-2）
- **核心方法**：基于亮红荧光蛋白 mScarlet 的 GECI，亮度/光稳定性/SNR 显著优于既有红色指示剂，且**完全兼容蓝光光遗传与双色成像**（在蓝光下无不良光切换）；在体外/体内（纤维光度、宽场、微型显微镜、双光子清醒小鼠）均耐受无毒性。
- **与本项目相关点**：红色 GECI 与蓝光 opsin 的光谱分离是**全光（成像+光遗传同用蓝光）闭环**的关键使能器——直接降低本项目闭环层「钙成像读 + 光遗传写」的串扰约束，使 Afshar-Saber 管线（08-01 §6.1）可升级为真正同步全光。

### 6.2 APPC — Active Pixel Power Control for crosstalk-free all-optical interrogation
- **作者**：Gewei Yan, Guangnan Tian, …, Jianan Y. Qu（HKUST Qu Lab）
- **年份·来源**：Nature Communications, 2026
- **核心方法**：在双光子全光（刺激+成像）中，成像激光会在扫描每个像素时**动态调节功率**，使对准后续要光刺激的 opsin 表达神经元时功率降到激活阈值以下，从而抑制成像激光误激活 ChR 造成的伪迹；在幼体斑马鱼脑验证，保 GECI 信号质量同时显著压制光遗传串扰，可无缝集成现有双光子系统。
- **与本项目相关点**：① 「单飞秒激光同时刺激+成像、近零串扰」是本项目闭环层**因果读写同体**的理想硬件方案——twin 预测→光遗传干预→钙成像验证可在同一视野同一次实验闭环完成；② 其「按目标像素动态降功率」思路可抽象为闭环控制算法模块，接入 Passmore 结果驱动显微（08-01 §7.1）的主动刺激调度。

---

## 7. Optogenetics（光遗传）· 新增（闭环层硬件/控制）

> Passmore 结果驱动显微 / STN-LID（08-01 §7）已覆盖。本周新增聚焦**一体化可植入闭环接口**与**相位靶向闭环刺激**。

### 7.1 Closed-loop-capable neural interface: non-viral delivery + NIR optogenetics + electrophysiology
- **作者**：Chao Yi Chu, Zih Huei Chen, …, San Yuan Chen（成功大学等）
- **年份·来源**：Advanced Science, 2026-02-03, 13(7):e15060（DOI 10.1002/advs.202515060）
- **核心方法**：一体化可植入神经接口，把①**非病毒**递送 ChR2 质粒（PEI-neurotensin 载体固定到电极位点，电穿孔转染）、②**无光纤近红外（NIR）光遗传**（上转换纳米颗粒 UCNP + GelMA，局域 LSPR 增强 NIR→蓝光转换远程激活）、③**电生理记录**集成到单个器件；3D 金反蛋白石（AuIO）微电极提供高表面积促转染与信号采集；海马 DG 体内单步手术验证成功 opsin 表达与实时光诱发活动。
- **与本项目相关点**：① 「非病毒 + 无光纤 + 记录/刺激一体」大幅降低闭环层在体部署门槛，是本项目「AI 预测→光遗传验证→误差回流」在**深脑/自由活动**场景可落地的硬件底座；② NIR 无线激活避免光纤束缚，使 ood_neuro（新调质/病理状态）在体验证可在自然行为下进行；③ 建议把此接口列为闭环层「湿实验候选平台」之一，与 Afshar-Saber 人源 iPSC 体外管线互补（体外机制 vs 在体系统）。

### 7.2 Phase-targeted closed-loop optogenetic stimulation of excitatory/inhibitory neurons modulating cortical theta/gamma
- **作者**：（OscillTrack 算法，2026，Cell Reports 类，DOI 前缀 S1935861X26000112）
- **年份·来源**：2026（closed-loop theta-phase-targeted optogenetic stimulation）
- **核心方法**：用 OscillTrack 实时追踪小鼠运动皮层 LFP 的 theta 相位，在四个目标相位触发蓝光脉冲，分别刺激 RBP4-Cre（锥体兴奋性）与 PV-Cre（抑制性中间神经元）的 ChR2。发现：闭环节点靶向刺激以**相位依赖**方式调节 theta 功率，兴奋/抑制神经元的放大相位**相差 90°**（与计算模型预测一致）；gamma 协议选择性放大 gamma 功率；开放环路 replay 同刺激模式无法复现相位特异效应，证明**实时闭环交互的必要性**。
- **与本项目相关点**：① 「相位依赖 + 兴奋/抑制差 90°」是本项目孪生「突触→环路」尺度**闭环控制协议**的直接范本——twin 若预测某振荡态异常，可类似地以相位靶向光刺激施加 do-干预并验证；② 其「开放环路 replay 失败 → 必须闭环」结论强化了本项目闭环层「预测须由实时反馈校正」的立场，而非离线一次性刺激。

---

## 8. Neuromodulator（神经调质）· 新增（本周强原发性集群）

> 前两周 neuromodulator 仅在 §6–7 交叉覆盖（TSC/癫痫、LID）。本周出现 4 篇直接相关原发性，且含一个**开放生物物理模型**，对 ood_neuro（新调质/病理状态）轴价值最高。

### 8.1 Quantitative anatomy + biophysical modeling of ascending neuromodulatory systems
- **作者**：Colangelo, Muñoz, Antonietti, …, Srikanth Ramaswamy（Newcastle + Blue Brain Project/EPFL + 西班牙团队）
- **年份·来源**：PLOS Computational Biology, 2026, 22(6):e1014460（DOI 10.1371/journal.pcbi.1014460；开放数据/代码 zenodo.14587678）
- **核心方法**：用免疫染色 + 立体学**高精度测绘**大鼠体感皮层各层 ACh/DA/5-HT 调质纤维密度与空间分布（胆碱能纤维密度是 5-HT 的 2.3 倍）；把这些解剖测量整合进**生物物理详细的皮层微环路模型**，首次模拟三系统如何改变数万神经元的节律电活动。关键结论：① ACh 强力抑制慢波（delta，深睡相关），且效应由**突触-到-突触（非体积释放）**信号解释——解决长期争论；② DA/5-HT 也去同步化皮层活动；③ 5-HT 独特诱发更快 theta；④ DA 解剖影响最广（支配所有皮层层的兴奋/抑制神经元）。**全部数据与生物物理模型开源**。
- **与本项目相关点**：**本周对 ood_neuro 轴价值最高（见文末总结）**。① 它给出「调质纤维密度 + 生物物理微环路」的**可复用开放模型**——本项目孪生「新调质状态」可直接把此模型作为调制场先验（把 ACh/DA/5-HT 水平作为状态变量注入离子通道→突触尺度），而非凭空加偏置；② 「ACh 突触级而非体积释放」的结论提示本项目建模调质须用**靶向突触门控**而非全局标量偏置（呼应 STN-LID 的 U 形状态依赖动态，08-01 §7.2）；③ 开放数据/代码可直接拿来校准本项目 ood_neuro 子集的调制动力学。

### 8.2 Acetylcholine demixes heterogeneous dopamine signals for learning and moving
- **作者**：（Nature Neuroscience, 2026, s41593-026-02227-x）
- **核心方法**：在背内侧纹状体（DMS）光学同步测 DA 与 ACh 释放，发现奖励线索诱发**胆碱能暂停（dip）与 DA 的相位关系决定信号含义**：DA 滞后 ACh dip → DA 预测未来行为且 DMS 放电率在下 trial 相关；DA 先于 ACh dip → 与学习无关；DA 与 ACh burst 同时 → 预测对侧定向运动的 vigor。即 **ACh 动态决定 DA 此刻驱动「学习」还是「运动 vigor」**，取决于即时行为情境。
- **与本项目相关点**：① 「同一 DA 信号被 ACh 上下文解复用为学习 vs 运动」是**状态/情境依赖调制**的范例——直接支持本项目「新调质状态改变同一干预的因果效应」建模需求（ood_neuro）；② 光学同步双调质测量是闭环层在调质尺度做因果验证的方法模板。

### 8.3 Cholinergic modulation of dopamine release drives effortful behavior
- **作者**：Touponse, Pomrenze, …, Malenka, Eshel（Stanford）
- **年份·来源**：Nature, 2026-01-28, s41586-025-10046-6
- **核心方法**：努力程度放大相同奖励的 DA 反应，且该放大依赖**局部 ACh 调质**：高努力奖励诱发 NAc 局部中间神经元快速 ACh 释放，ACh 结合 DA 轴突终端烟碱受体，在奖赏送达时增强 DA 释放；阻断 ACh 调质仅在高努力情境选择性削弱 DA 释放与努力行为，低努力消费不受影响。
- **与本项目相关点**：① 「上下文依赖的局部 ACh–DA 轴突终端交互」提供**调质-调质耦合**的可计算机制（可作为孪生环路尺度的耦合项）；② 其「高努力 vs 低努力」双情境对应本项目 OOD 中「同一干预在不同内部状态效果不同」的情形，强化 ood_neuro 须刻画状态依赖动态。

### 8.4 Accumbal acetylcholine signals associative salience during learning
- **作者**：Zhewei Zhang, Kauê Machado Costa, …, Geoffrey Schoenbaum（NIDA/NIH + UAB + 北大/麦戈文）
- **年份·来源**：bioRxiv/PMC11741319, 2026
- **核心方法**：在大鼠三学习任务中同步记录 NAc 的 ACh 与 DA 动态，揭示学习期神经调质响应的演化模式：先出现无差别 ACh dip，随后 DA 按预测值分化，ACh dip 也按值变化；稳定表现需要完整模式；ACh dip 可由改变奖励 contingencies 与值解耦。ACh 动态拟合混合注意-联想学习模型的**联想显著性（associative salience）**项——DA 与 ACh 互补决定学习的内容与速率。
- **与本项目相关点**：① 「ACh=显著性/学习率信号、DA=预测误差」的互补框架，可映射为孪生**学习/可塑尺度**的调制变量（本项目多尺度孪生含突触可塑性）；② 提示 ood_neuro 不仅改稳态响应，还改**学习率/可塑性动态**——twin 若预测新调质下的可塑，应建模该显著性门控。

---

## 本周最值得借鉴的方法（1–2 个）及对建模层（多尺度虚拟孪生）的启示

**方法一：GraCE-VAE 的「图感知因果解耦 + 隐 DAG + 干预编码器」（Zhang et al., 2026）。**
它把**结构先验（基因/实体关系网络）+ 干预数据**端到端学成一个带 SCM 解码器的隐因果图，并能对**未见干预组合**给出有生物学意义的预测，且带可辨识性保证。对建模层的启示：本项目 CompositionalTwin 当前的加性 φ（P1）+ 对称交互头 ψ（P2）本质是**在观测空间做加性/交互近似**，而 GraCE-VAE 提示更机制化的路线——在**隐空间显式建一张可学习 DAG**，用「干预编码器把 `do(p,act)` 映射为对隐节点的软选择 + 机制修改」，只拨动被干预节点的因果机制、其余不变。这直接回应本项目两条痛点：① ood_action（held-out 基因对）可被表述为「对隐 DAG 施加未见节点组合的 do-干预」，泛化有可辨识性托底；② 缓解「OOD 好但是否真学到机制」的质疑（GraCE-VAE 明确保证 + 可解释 DAG）。建议作为 P3 候选架构对照。

**方法二：Colangelo et al. 的「调质纤维解剖 + 生物物理微环路开放模型」（PLOS Comp Biol 2026）。**
它首次把 ACh/DA/5-HT 的皮层纤维密度实测整合进可微生物物理皮层模型，并**全量开源数据与代码**。对建模层的启示：本项目一直把 ood_neuro（新调质/病理状态）当作最难、信号最少的 OOD 轴（真实 Norman 无调质维度），而此文给出**现成的调制场先验**——可直接把调质水平作为状态变量注入孪生「离子通道→突触」尺度，且必须用**突触级靶向门控**（ACh 非体积释放的结论）而非全局标量偏置。这能让 ood_neuro 从「无数据可外推」升级为「有生物物理约束的可计算外推」，并可用其开放模型校准本项目调质动力学。

**配套（评测/闭环侧，与硬约束同源）**：① OOD 不确定度侧，本周 §4.1–4.3 把共形从「经验覆盖」（Chin, 08-01 §4.1）推进到**有界保证 + 组条件覆盖 + VAE 原生自适应**：建议本项目校准表从边际覆盖扩到 **ood_action/ood_agent/ood_neuro 分组条件覆盖**，并可用 Siahkali 的 Wasserstein 下界按漂移幅度设阈值、用 ACP 的「少量目标域标注+审计模型」做每周预测清单的主动实验选择；② 闭环层硬件侧，PinkyCaMP（§6.1）+ APPC（§6.2）组合使「蓝光全光读写同体、近零串扰」成为现实，可直接升级 Afshar-Saber 人源 iPSC 管线为同步闭环验证平台，呼应本周 §7 的相位靶向闭环（§7.2）与一体可植入接口（§7.1）。
