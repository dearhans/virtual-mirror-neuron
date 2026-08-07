# 文献监测周报 · 2026-07-31

> **范围**：virtual cell、mirror neuron、perturb-seq / scRNA-seq perturbation、OOD generalization、causal representation learning、calcium imaging、optogenetics、neuromodulator
> **方法**：Web 检索（截至 2026-07-31），按方向去重与筛选；每条附「与本项目相关点」映射至三条硬约束（记忆 vs 机制泛化 / OOD 子集 / 不确定度）及建模层（多尺度虚拟孪生、do-演算、闭环层）。
> **注意**：部分条目来自新闻聚合/预印本摘要，正式引用前请核对原始 DOI。本报告仅入库，不外发。

---

## 1. Virtual Cell（虚拟细胞）

### 1.1 Virtual cells aim to turn raw data into predictive models of biology
- **作者**：Nature News & Views（综述性报道）
- **年份·来源**：Nature, 2026
- **核心方法**：梳理虚拟细胞现状——从微分方程系统生物学（PhysiCell、Luthey-Schulten 的支原体分裂模拟）到 AI 基础模型（scBaseCount 约 5 亿细胞）。Theis 直言「真正虚拟细胞」仍远未达成，静态状态可捕获、动态变化预测仍弱。
- **与本项目相关点**：印证本项目「机制先验不可丢」立场——纯基础模型在动态扰动响应上存疑，需结合机制。直接对齐 CHARTER 层 3–4：扰动→测量→学因果。

### 1.2 Tahoe-100M / Tahoe-x1 (Tx1) / Rhaister
- **作者**：Tahoe Therapeutics（Biotech，新闻报道）
- **年份·来源**：2025-10 (Tx1) / 2026-06 (Rhaister)，公开数据集与开源权重
- **核心方法**：Tahoe-100M 是公开首个亿级单细胞药物扰动数据集（50 癌细胞系 × 1100+ 小分子，约此前公开总和 50 倍）。Tx1 为 30 亿参数扰动预训练模型，FlashAttention v2 + 全分片并行，展现跨细胞类型零样本泛化。Rhaister 反其道行之——回归基础统计规律，仅用汇总统计 + 少量参考上下文学习，数秒训练、毫秒预测，部分指标接近实验噪声，**零样本版 Rhaister-O 仅靠基线表达即预测**。
- **与本项目相关点**：① 对「OOD（未见化合物/未见菌株）」的零样本泛化有直接可比性；② Rhaister 的「轻量统计 + 强基线」是本项目对照**简单基线（均值/线性）**的绝佳参照——若机制模型未显著超越 Rhaister 式基线，即触发「疑似仅记忆」告警（见 `goai_virtualcell_format.md` §4）。

### 1.3 灵枢细胞 (Lingshu-Cell)
- **作者**：阿里达摩院
- **年份·来源**：2026-06（新闻报道）
- **核心方法**：全转录组尺度生成式「细胞世界模型」，约 1.8 万基因 token 化，以遮挡基因表达做填空式预测，输出细胞状态分布与扰动后动态。
- **与本项目相关点**：生成式「世界模型」思路可借鉴于虚拟孪生状态分布预测；但属黑箱端到端，需以机制先验补强可解释性。

### 1.4 MultiVCDiff — A generative framework for predicting cellular morphological and transcriptomic perturbation responses
- **作者**：王劲卓团队（北京大学未来技术学院）
- **年份·来源**：Cell Reports Methods, 2026-05-21
- **核心方法**：多模态生成式虚拟细胞框架，直接以**扰动本身**（化合物结构 / 基因表示）为条件，在统一扩散生成框架中同步生成细胞形态图像 + 转录组表达谱，无需扰动后实测读数即可 de novo 预测。整合 Cell Painting + L1000，构建 4 个多模态数据集。
- **与本项目相关点**：① 「扰动即条件输入」与本项目 `do-` 干预定义同构；② 多模态（形态 + 转录组）对应本项目多尺度（离子通道→突触→环路）同源思想；③ 真正 de novo 虚拟筛选能力，恰是 OOD（未见扰动）的终极检验。

### 1.5 RegVelo (Regulatory Velocity)
- **作者**：Fabian J. Theis（亥姆霍兹慕尼黑）、Tatjana Sauka-Spengler（牛津）
- **年份·来源**：2026（新闻报道，端到端深度学习 + 微分方程）
- **核心方法**：将神经网络与细胞动态微分方程融合，从观测轨迹同时推断驱动分化的内在调控逻辑，并模拟「敲除转录因子 / 下调回路」后的新轨迹。应用于小鼠胰腺、人造血、斑马鱼神经嵴多谱系分化。
- **与本项目相关点**：① 微分方程（机制）+ 神经网络（拟合）的混合范式，正是本项目「机制先验 + 不迷信端到端」的范本；② 「模拟干预后新轨迹」= do-演算的可计算化；③ 对未见调控干预的外推能力即机制泛化证据。

---

## 2. Mirror Neuron（镜像神经元）

### 2.1 Dynamic population coding of kinematic structure across executed and observed actions in primate premotor cortex
- **作者**：Konstantinos Chatzimichail, Christos Paschalidis, Eleftheria Tzamali, Vassilis Papadourakis, Vassilis Raos 等
- **年份·来源**：Science Advances, 2026-06-19 (12(25), DOI 10.1126/sciadv.aed9309)
- **核心方法**：在 2 只猕猴 PMd/PMv 记录 433 个神经元（285 MirN + 148 non-MirN），执行/观察 4 种抓握。结合群体解码、交叉时间解码、CCA、多元线性回归。发现：① 抓握信息以分布式、时间特异的群体状态承载（非少数固定细胞）；② 执行与观察通过**部分重叠的共享群体几何**对齐，对齐在执行与保持期最强；③ 神经活动系统性耦合多维手部运动学，且该关系**跨神经元群体、跨主体（agent）泛化**。执行→观察存在不对称（执行含额外特异过程）。
- **与本项目相关点**：**核心支撑证据**。① 直接证实本项目层 1 现象「自我/他者同源」的可计算落点是**群体几何的共享潜空间**，而非单神经元严格镜像（单神经元一致性仅机会水平）——这正是本项目层 2「感知与动作在共享潜空间对称性」的神经实证；② 「跨 agent 泛化」给出本项目 OOD（未见主体）的生物学合法性；③ 方法上提示我们评测 self/other 对称性应在**群体几何/低维子空间对齐**层做，而非单神经元指标。

### 2.2 Effects of intention understanding and brief imitative experience on the mirror neuron system
- **作者**：Tomoki Osaki, Takehiro Minamoto
- **年份·来源**：PLOS One, 2025-12-19 (20(12): e0335885)
- **核心方法**：EEG 测量日本手语观察/模仿三阶段（模仿前/中/后）的 mu 抑制（MNS 指标），操纵意图理解水平（计数 vs 猜含义）。结果：两种任务 mu 抑制相当，模仿经验无显著增强（归因于文化因素使 MNS 难学新手语动作）。
- **与本项目相关点**：① mu 抑制作为非侵入 MNS 指标，可用于本项目闭环层的行为/神经对齐校核；② 提示「模仿经验增强 MNS」并非普适，类比本项目——机制泛化需在特定训练/经验条件下才成立，OOD 评估须考虑经验分布。

---

## 3. Perturb-seq / scRNA-seq Perturbation

### 3.1 SCCVAE — Learning genetic perturbation effects with variational causal inference
- **作者**：Emily Liu, Jiaqi Zhang, Caroline Uhler
- **年份·来源**：PLOS Computational Biology, 2026-02 (22(2): e1013194)
- **核心方法**：提出 Single Cell Causal Variational Autoencoder——将**机制因果模型（学得调控网络，扰动建模为沿网络传播的 shift intervention）**嵌入变分自编码器。机制部分提供外推能力，深度部分提供丰富表征。在未见扰动响应外推上优于 SOTA；潜空间可识别功能扰动模块、模拟不同外显率的单基因敲低。
- **与本项目相关点**：**最值得借鉴方法之一（见文末总结）**。① 机制因果 + 变分深度混合，正是本项目「机制先验 + 不纯黑箱」的直接范式模板；② shift intervention 沿网络传播 = 本项目 `do-` 干预在图上的具体化；③ 「外推至未见扰动」即机制泛化，可直接对标本项目 ood_action 子集；④ 潜空间功能模块可解释性对齐本项目双栏可解释要求。

### 3.2 ADAPRE — Causal gene regulatory network inference from Perturb-seq via adaptive instrumental variable modeling
- **作者**：Zhongxuan Sun, Hyunseung Kang, Sündüz Keleş
- **年份·来源**：bioRxiv, 2026-02-18 (DOI 10.64898/2026.02.18.706642)
- **核心方法**：将 CRISPR 干预视为工具变量（IV），在 Poisson-lognormal 观测层显式建模 UMI 计数，并以基因特异性自适应惩罚纠正 CRISPRi **异质敲低强度导致的 degree bias**（强敲低基因被高估为 hub）。可恢复潜在环状结构，优于 DoTEARS/BICYCLE/INSPRE/LLCB。
- **与本项目相关点**：① 直接警示本项目因果图建模中的**混杂与测量偏差**——异质扰动强度会伪造 hub，须显式建模；② IV 框架与 do-演算同源；③ 对 GOAI 虚拟细胞赛题（化合物/菌株异质效应）的偏差校正有借鉴。

### 3.3 Mixscale — Systematic reconstruction of molecular pathway signatures using scalable single-cell perturbation screens
- **作者**：Satija Lab 等
- **年份·来源**：Nature Cell Biology, 2025
- **核心方法**：pooled CRISPRi + scRNA-seq，跨 6 细胞系 × 5 信号情境 >1500 扰动；提出 Mixscale 框架校正扰动效率的个体差异，学习差异表达基因列表与保守分子签名，推断体内/原位信号通路激活。开源 R 包。
- **与本项目相关点**：大规模扰动签名 atlas 思路——本项目可建立「动作/主体/调质」扰动签名库，作为 OOD 评估的参照基线。

---

## 4. OOD Generalization（分布外泛化）

### 4.1 Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks
- **作者**：NeurIPS 2025（arXiv 2505.09716）
- **年份·来源**：NeurIPS 2025
- **核心方法**：构造带明确 OOD 指标的 ARC-like 任务（几何变换、颜色-形状组合），证明 MLP/CNN/Transformer 在组合 OOD 上近 0 准确率；更关键：**即使架构带正确归纳偏置、OOD 准确率近完美，仍可能学到错误的组合特征**——提出验证 OOD 须同时验证「特征正确性」。
- **与本项目相关点**：**方法论级警示，直接服务硬约束**。① 本项目「记忆 vs 机制泛化」双栏若只看 metric 达标，可能落入「OOD 好但特征错」陷阱；② 必须在 ood_* 子集上**额外验证学到的表征/因果结构是否正确**（如对照 SCCVAE 潜空间模块、对照 Science Advances 群体几何），而非仅报 PCC/RMSE；③ 呼应 TRIZ 物理矛盾——高训练依赖度（记忆）与低训练依赖度（外推）须用条件分离显式暴露差额。

### 4.2 Unlocking OOD Generalization in Transformers via Recursive Latent Space Reasoning
- **作者**：Awni Altabaa, Siyu Chen, John Lafferty, Zhuoran Yang（Yale）
- **年份·来源**：arXiv 2510.14095, 2025-10
- **核心方法**：在 Transformer 上探究四种提升 OOD 的机制：① 输入自适应循环（按复杂度动态分配算力）；② 算法监督（结构化中间表征）；③ 离散瓶颈锚定潜表征；④ 显式误差校正。配合机械化可解释性分析揭示 OOD 能力来源。
- **与本项目相关点**：① 离散瓶颈锚定 + 误差校正，可借鉴为虚拟孪生中**不确定度合成**（epistemic + 新颖度）的机制化实现；② 算法监督 = 把「机制先验」显式注入训练，与本项目立场一致。

### 4.3 Towards Better Generalization via Distributional Input Projection Network (DIPNet)
- **作者**：Yifan Hao, Yanxin Lu, Hanning Zhang, Xinwei Shen, Tong Zhang
- **年份·来源**：ICLR 2026 投稿（OpenReview）
- **核心方法**：逐层将输入投影为可学习分布（DIPNet），诱导更平滑的输入损失景观，理论上降低 Lipschitz 常数；在 ViT/LLM/ResNet/MLP 上一致提升标准、对抗、OOD、推理性能。
- **与本项目相关点**：分布化输入表征可降低过拟合、提升 OOD 鲁棒性，可试用于本项目孪生输入（扰动向量 P + 上下文）的平滑化。

---

## 5. Causal Representation Learning（因果表征学习）

### 5.1 CausalVerse — A Comprehensive Benchmark for Causal Representation Learning with Controllable High-Fidelity Simulations
- **作者**：Guangyi Chen, Kun Zhang 等
- **年份·来源**：NeurIPS 2025 Spotlight
- **核心方法**：面向 CRL 的基准，提供**可控高保真仿真**环境，系统化评测因果解耦/干预/反事实能力。
- **与本项目相关点**：① 建议本项目自建「镜像神经元因果表征」基准时借鉴其可控仿真范式（可程序化生成 self/other 对齐度可调的合成数据）；② 直接服务「OOD 强制 + 不确定度强制」评测标准化。

### 5.2 Causal Differentiating Concepts: Interpreting LM Behavior via Causal Representation Learning
- **作者**：Navita Goyal, Hal Daumé III, Alexandre Drouin, Dhanya Sridhar
- **年份·来源**：NeurIPS 2025
- **核心方法**：无监督算法识别「因果差异化概念」——LM 激活中须被改变才能引发不同行为的可解释潜方向，用稀疏约束对比学习目标，恢复真实因果因子。
- **与本项目相关点**：无监督解耦出「行为相关因果因子」的思路，可用于从钙成像/电生理群体活动中解耦出「动作语义」与「运动学」因子（呼应 Science Advances 的运动学编码）。

### 5.3 Reward-oriented Causal Representation Learning
- **作者**：Zirui Yan, Emre Acartürk, Ali Tajer
- **年份·来源**：NeurIPS 2025
- **核心方法**：提出「以奖励为导向的 CRL」——不追求完美恢复潜表征，只学到优化下游任务（奖励）所需的最粗粒度；形式化为在所有干预空间上优化可观测量函数，并设计自适应探索算法序贯识别最优干预子集，给出 regret 上/下界。
- **与本项目相关点**：① 与本项目闭环层「据湿实验误差回流、聚焦高价值刺激参数」高度同构；② 自适应干预选择 = 把 do-演算用于**主动实验设计**，可直接支撑每周预测清单的「下一步该测哪组扰动」决策。

---

## 6. Calcium Imaging（钙成像）

### 6.1 CalM — A Self-Supervised Foundation Model for Population Dynamics in Calcium Imaging Data
- **作者**：Xinhong Xu, Yimeng Zhang, Qichen Qian, Yuanlong Zhang
- **年份·来源**：arXiv 2604.04958, 2026-04（v3 2026-06），代码已发布
- **核心方法**：纯钙信号自监督神经基础模型：高性能 tokenizer 将单神经元轨迹映射到共享离散词表 + 双轴（神经元轴 × 时间轴）自回归 Transformer。在多动物、多 session 数据集上，预测（forecasting）优于专用基线，加任务头后行为解码优于监督模型；线性分析揭示可解释功能结构。
- **与本项目相关点**：**直接服务闭环层**。① 钙成像群体动态基础模型可作本项目闭环验证的**预训练骨干**（forecasting + decoding）；② 双轴依赖建模对应本项目「神经元 × 时间」动态；③ 自监督 + 线性探针的可解释性路线，契合本项目不纯黑箱立场。

### 6.2 Deep Representation Learning on Whole-Brain Population Dynamics Uncovers Geometrically Separable Neural Codes
- **作者**：Abdelbaki, Bandow, Cheng, Grunwald Kadow, Nawrot, Rostami
- **年份·来源**：bioRxiv, 2026-05-12 (DOI 10.64898/2026.05.12.724368)
- **核心方法**：接线无关（wiring-agnostic）深度学习框架，卷积编码器 + 时序 Transformer，直接学全脑果蝇体积钙成像的紧凑表征；分类 16 种（代谢态 × 感觉模态 × 刺激效价）条件，发现状态/模态/效价沿**近正交三轴**可分离，无需解剖标注。
- **与本项目相关点**：① 「近正交轴分离不同因子」= 本项目共享潜空间中「动作/主体/调质」因子解耦的可视化模板；② 无需神经元身份/连接即可学表征，降低了本项目多尺度对齐的工程门槛。

### 6.3 POYO-CAP — Decoding Dynamic Visual Experience from Calcium Imaging via Cell-Pattern-Aware Pretraining
- **作者**：Sangyoon Bae, Mehdi Azabou, Blake Richards, Jiook Cha
- **年份·来源**：arXiv 2510.18516, 2026-05-08
- **核心方法**：针对神经记录异质性（统计规则神经元 vs 高度随机神经元混在同一数据集），先用掩码重建 + 轻量辅助监督在「规则神经元」（偏度/峰度识别）上预训练，再微调随机群体，将异质性转为可扩展学习优势（Allen Brain Observatory 上 12–13% 相对提升，且随模型规模单调提升）。
- **与本项目相关点**：异质性即信号——本项目多尺度孪生聚合「离子通道→环路」异质读数时，可借鉴其「按统计可预测性做课程/数据选择」策略，避免混合分布拖垮表征学习。

---

## 7. Optogenetics（光遗传）

### 7.1 A Closed-Loop-Capable Neural Interface Platform for Deep Brain Modulation
- **作者**：Chao Yi Chu, Zih Huei Chen, San Yuan Chen 等
- **年份·来源**：Advanced Science, 2026-02-03 (13(7): e15060)
- **核心方法**：一体化可植入器件，集成① 非病毒递送 ChR2 质粒（NT-PEI 载体，电穿孔转染）；② 无光纤近红外（NIR）光遗传刺激（上转换纳米颗粒 UCNP + 金反蛋白石微电极，LSPR 增强 NIR→蓝光）；③ 同步电生理记录。单次手术在海马齿状回实现 opsin 表达 + 实时光诱发活动。
- **与本项目相关点**：**直接强化闭环层可行性**。① 非病毒 + 无光纤 + 记录/刺激一体，正是本项目「AI 预测 → 光遗传验证 → 误差回流」所需的低侵入闭环硬件范式；② NIR 远程激活降低本项目湿实验的硬件耦合度。

### 7.2 Closed-Loop Control of Cerebral ATP Transients by an Optogenetic Nanosensor
- **作者**：Qi Lu, Zhengbing Liang, Yaru Sun 等
- **年份·来源**：Advanced Functional Materials, 2026-05（DOI 10.1002/adfm.202523676）
- **核心方法**：经颅 NIR 光遗传纳米探针，实时精准监测并按需控制小鼠 mPFC 星形胶质细胞 ATP 释放；首次给出 ATP 缓解抑郁样行为的有效浓度窗（65–105 μM），超出则触发神经炎症。
- **与本项目相关点**：光遗传 + 神经调质（ATP）闭环的范例，跨接「光遗传」与「神经调质」两轴——示范本项目 OOD（新神经调质状态）如何在闭环中以「浓度窗/状态窗」形式被建模与验证。

---

## 8. Neuromodulator（神经调质）

### 8.1 Acetylcholine demixes heterogeneous dopamine signals for learning and moving
- **作者**：Hee Jae Jang, Royall McMahon Ward, Carla E. M. Golden, Christine M. Constantinople（NYU）
- **年份·来源**：Nature Neuroscience, 2026-03-25 (DOI 10.1038/s41593-026-02227-x)
- **核心方法**：Neuropixels 记录大鼠纹状体数百神经元跨试次放电；发现约 51.6% 神经元在「多巴胺紧随乙酰胆碱之后（~100 ms 窗口）」的大多巴胺信号后发生**持久可塑性更新**（跨试次保持），而多巴胺先于 ACh 时无此效应。提出：ACh 下降使 normally 抑制 DA 强化突触的受体短暂离线，开启 DA 驱动可塑性的时间窗。
- **与本项目相关点**：**直接服务 OOD（新神经调质状态）轴**。① 证明调质效应高度依赖**时序/状态窗口**而非单纯浓度——本项目建模「新调质状态」须以时序门控（timing gate）建模，而非静态加法偏置；② 给出「调质创造可塑性窗」的机制先验，可嵌入孪生突触可塑性尺度；③ 提示本项目湿实验验证调质扰动时须精确控制 ACh/DA 时序。

### 8.2 Cholinergic modulation of dopamine release drives effortful behaviour
- **作者**：Neir Eshel 团队（Stanford）
- **年份·来源**：Nature, 2026-01-28 (DOI 10.1038/s41586-025-10046-6)
- **核心方法**：努力增强对相同奖励的多巴胺反应，且该放大取决于 ACh 对 DA 轴突末梢的局部调节（高努力触发 NAc 局部中间神经元快速释放 ACh，结合 DA 轴突烟碱受体，奖励时放大 DA 释放）；阻断胆碱能调节特异性损害高努力情境 DA 释放与努力行为。
- **与本项目相关点**：ACh↔DA 局部交互的体内实证，支持本项目将「调质协变量 g」作为**情境依赖门控**而非标量偏置；为「调质状态改变动作价值编码」提供环路级证据，呼应层 5 OOD（新调质）。

---

## 本周最值得借鉴的方法（1–2 个）及对建模层（多尺度虚拟孪生）的启示

**方法一：SCCVAE 的「机制因果 + 变分深度」混合范式（Liu, Zhang, Uhler, PLOS Comput Biol 2026）。**
它把扰动建模为沿**学得的调控网络传播的 shift intervention**，机制部分提供对未见扰动的外推、深度部分提供丰富表征——这正是本项目「机制先验 + 不迷信端到端」在扰动响应预测上的可直接复用模板。对建模层的启示：多尺度虚拟孪生的核心算子应显式实现为「`do-` 干预 → 沿因果图（离子通道→突触→环路）传播」的模块，而非把扰动当成黑箱条件向量；预测头应输出**沿尺度传播后的状态分布 + epistemic 不确定度**，使 ood_action 子集的外推可被审计。

**方法二：Science Advances 镜像神经元「共享群体几何」（Chatzimichail et al., 2026）的实证结论。**
它用 433 神经元证明执行/观察通过**部分重叠的低维群体几何**对齐、且运动学耦合关系跨 agent 泛化，而单神经元镜像一致性仅机会水平。对建模层的启示：本项目层 2「感知与动作在共享潜空间对称性」不应在单神经元层校验，而应在**群体几何/低维子空间对齐**层做（如 CCA / 子空间夹角）；同时把「跨主体（agent）对齐」作为 OOD（未见主体）的内建评测维度，并以近正交轴分离「动作/主体/调质」因子（呼应 CalM/Whole-brain 的解耦几何）。

**补充警示（来自 OOD Generalisation is Hard, NeurIPS 2025）**：OOD 指标达标 ≠ 学到正确机制结构。本项目双栏（记忆 vs 机制）评测除报 PCC/RMSE 外，须在 ood_* 子集上**额外验证学到的因果/表征结构正确性**（如对照 SCCVAE 潜空间模块、对照 Science Advances 群体几何），否则可能落入「OOD 好但特征错」的伪机制陷阱——这恰是 TRIZ 物理矛盾（记忆红利 vs 外推）须用条件分离暴露差额的原因。
