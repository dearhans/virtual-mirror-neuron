# 文献监测周报 · 2026-08-01（增量，对比 2026-07-31）

> **范围**：virtual cell、mirror neuron、perturb-seq / scRNA-seq perturbation、OOD generalization、causal representation learning、calcium imaging、optogenetics、neuromodulator
> **方法**：Web 检索（截至 2026-08-01），相对 `2026-07-31-literature.md` 去重，仅收录本周新出现的论文/预印本或显著进展；每条附「与本项目相关点」映射至三条硬约束（记忆 vs 机制泛化 / OOD 子集 / 不确定度）及建模层（多尺度虚拟孪生、do-演算、闭环层）。
> **注意**：部分条目来自预印本/新闻聚合，正式引用前请核对原始 DOI。本报告仅入库，不外发。
> **本周无新增原发性研究的方向**（仅延续上周条目，不再重列）：mirror neuron（Science Advances 2026-06-19 群体几何文持续被媒体解读，但无新 Primary）、neuromodulator（本周未检索到显著新的原发性调质机制论文；其表型在 §6–7 的钙成像/光遗传闭环新进展中交叉出现，如 TSC 癫痫、LID）。

---

## 1. Virtual Cell（虚拟细胞）· 新增

### 1.1 SCALE — Scalable Conditional Atlas-Level Endpoint transport
- **作者**：Shuizhou Chen, Lang Yu, Kedu Jin, Songming Zhang, Hao Wu, Lei Bai, Siqi Sun, Zhangyang Gao 等（上海 AI Lab）
- **年份·来源**：arXiv 2603.17380, 2026-03（v2 2026-03-19）
- **核心方法**：面向虚拟细胞扰动预测的专用大规模基础模型。① BioNeMo 训练/推理框架（12.51× 预训练加速、1.29× 推理加速）；② 把扰动预测**形式化为条件传输（conditional transport）**，用 set-aware flow 架构（LLaMA 细胞编码 + endpoint 监督），在**无序细胞集合**上学习群体级状态转移（而非逐细胞配对重建）；③ 在 Tahoe-100M 上以**生物学忠实指标（PDCorr、DE Overlap）**评测，PDCorr +12.02%、DE Overlap +10.66%（优于 STATE）。
- **与本项目相关点**：① 「扰动预测 = 控制→扰动群体的条件传输」直接对齐本项目 `do-` 干预后的状态分布预测；② set-aware（无序集合、群体级）思想呼应本项目「不该在单神经元层校验，而在群体几何层」（见 07-31 §2.1）；③ 强调「用生物学忠实指标而非重建误差」评测——与本项目「OOD 强制 + 不确定度强制」的评测立场一致；④ Tahoe-100M 仍是可直接复用的零样本 OOD（未见化合物/菌株）基准。

### 1.2 OCOO-T — A Simple and Scalable Virtual Cell Model
- **作者**：（arXiv 2606.12838, 2026-06，作者组未全列）
- **年份·来源**：arXiv 2606.12838, 2026-06
- **核心方法**：主张**简单可扩展**的转录扰动响应预测模型；文献综述中明确引用 Ahlmann-Eltze et al. *Nature Methods* 2025 「Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines」，并综述 STATE / CellFlow / scDFM / AlphaCell / X-Cell / scLong / Virtual Cell Challenge (Cell 2025) / GEARS / Norman 2019 等主线。
- **与本项目相关点**：**直接佐证本项目硬约束**。Ahlmann-Eltze 2025 的实证结论——深度学习扰动模型仍未系统性胜过简单线性基线——正是本项目在 Norman 真实基准上反复验证的现象（线性 RMSE≈0.627 vs compositional 0.636，CI 重叠）。提示本项目任何「机制模型」都必须**显式报告均值/线性简单基线**并证明其显著超越，否则即触发「疑似仅记忆」告警；同时警示 epistasis 信号量小（见项目 P2 结论）时加法先验很难挣脱基线。

### 1.3 D-SPIN — constructs regulatory network models from scRNA-seq
- **作者**：姜家隆（Caltech）等
- **年份·来源**：Cell, 2026-05-12（在线发表）
- **核心方法**：从海量 scRNA-seq（含 Perturb-seq 多扰动）自动构建**可解释全局基因调控网络**的概率图模型。统一处理基因敲除、小分子药物、生长因子等多种扰动，甚至把健康/疾病状态本身视为「生理扰动」建模；网络具清晰机制解释性（可看清哪些互作线路被扰动拨动）→ 从黑箱到白箱。
- **与本项目相关点**：**最值得借鉴方法之一（见文末总结）**。① 把 Perturb-seq 直接建模为**可解释概率图（因果图）**而非黑箱映射，正是本项目「机制先验 + 不迷信端到端」在 GRN 尺度的范本；② 「预测细胞对全新药物/基因扰动的反应」= do-演算的可计算化与机制泛化；③ 白箱网络可直接对接本项目孪生「离子通道→突触→环路」中的基因/回路尺度，作为可审计的因果图底座。

### 1.4 Chreode — A Cell World Model for One-Step Temporal Dynamics and Perturbation Prediction
- **作者**：Mufan Qiu, Genhui Zheng, Yinuo Xu, Ruichen Zhang, Ying Ding, Qi Long, Tianlong Chen
- **年份·来源**：2026（预印本）
- **核心方法**：单步「细胞世界模型」，用**结构化残差转移算子**预测动作条件（action-conditioned）的 cell-state 转移；把分布演化从推理时移到训练时，保留 Waddington 式分解（downhill landscape flow + rotational in-tangent dynamics + stochastic spread）。在 2.4M 细胞小鼠胚胎 atlas 预训练（scVI 编码器 + DiT 动力学骨干）。**作为 GEARS 的可迁移基因状态嵌入，将 Norman Perturb-seq 的 DE20 MSE 从 0.2121 降到 0.1858（相对 −12.4%）**，且未改动 GEARS 训练流程；并产生具竞争力的零样本克隆命运分数。
- **与本项目相关点**：**直击本项目基准**。① 它提升的对象就是本项目真实数据 Norman 2019 Perturb-seq——证明**预训练的发育轨迹动力学是可迁移到 CRISPR 状态偏移的「分化原语」**（二者共享潜几何）；② 对建模层的启示：本项目多尺度孪生应先在「尺度/情境共享潜几何」上预训练转移算子，再把扰动作为 action-conditioned 残差施加，而非从零学每个扰动；③ 单步残差 + Waddington 分解提供可解释的动力学先验。

### 1.5 X-Cell — Scaling Causal Perturbation Prediction via Diffusion Language Models
- **作者**：Chloe Wang, Mehran Karimzadeh, Neal G. Ravindra, Lexi R. Bounds 等
- **年份·来源**：bioRxiv 2026.03.18 (DOI 10.64898/2026.03.18.712807)
- **核心方法**：用扩散语言模型在多样化细胞情境间扩展**因果扰动预测**（被 OCOO-T 综述引用）。
- **与本项目相关点**：因果扰动预测的「扩散 + 语言模型」路线，可作本项目孪生生成头的可选架构参照；其「跨多样细胞情境」扩展对应本项目 OOD（未见主体/情境）轴。

---

## 3. Perturb-seq / scRNA-seq Perturbation · 新增（承接 07-31 §3）

> 本周新增集中于「可解释/机制化」扰动模型：**D-SPIN（§1.3）**、**Chreode（§1.4）**、**X-Cell（§1.5）** 已覆盖。SCCVAE / ADAPRE / Mixscale（07-31 §3.1–3.3）仍为本方向主干，本周无新增 Primary 超越其结论。补充一条评测侧警示：**Ahlmann-Eltze et al., Nature Methods 2025**（被 OCOO-T 引用）实证深度学习扰动模型未胜过线性基线——与本项目 Norman 真实基准结论互证，建议固化为评测 SOP。

---

## 4. OOD Generalization（分布外泛化）· 新增（承接 07-31 §4）

### 4.1 Conformal Prediction for Neural Operators: Distribution-Free UQ in Physics Simulation
- **作者**：Michael Chin（独立研究者）
- **年份·来源**：arXiv 2606.09923, 2026-06
- **核心方法**：首次将 **split conformal prediction** 用于神经算子（FNO 等 PDE 代理模型），给出**有限样本覆盖保证** ℙ(Y∈𝒞(X))≥1−α 的分布无关预测区间；进一步提出**归一化共形**（用 MC Dropout 不确定性生成自适应宽度区间），并在稳态热传导基准上达 89.1% 经验覆盖（α=0.1）；给出** epistemic（68%，可约）/ aleatoric（32%，不可约）不确定度分解框架**。
- **与本项目相关点**：**直接服务「不确定度强制」硬约束的可操作配方**。① 本项目孪生本质是「神经代理模拟器」（离子通道→突触→环路的状态转移），与本文物理仿真代理同构——可直接套用 split/normalized conformal 给扰动响应预测区间；② epistemic/aleatoric 分解正是本项目一直想显式产出的不确定度类型（呼应 07-31 §4.2 离散瓶颈 + 误差校正思路）；③ 分布无关覆盖保证使 OOD（ood_action/ood_agent/ood_neuro）子集的区间可靠性可被审计。

### 4.2 Robust Conformal Prediction under Distribution Shift via Physics-Informed SCM
- **作者**：Rui Xu, Yue Sun, Chao Chen, Parv Venkitasubramaniam, Sihong Xie
- **年份·来源**：ICML（arXiv 2403.15025）
- **核心方法**：标准共形仅在校准/测试边缘分布不同时仍保覆盖，但条件分布 P(Y|X) 漂移时会失效。提出用**物理信息结构因果模型（PI-SCM）**上界覆盖损失并用 Wasserstein 距离最小化，在交通速度、疫情传播多域实测上提升跨域覆盖鲁棒性。
- **与本项目相关点**：把**因果结构**引入共形以抗分布漂移——与本项目「机制先验赋予 OOD 鲁棒性」立场同构；建议本项目在 conformal 校准时**以因果图（do-演算结构）约束非一致性分数**，而非仅用残差，从而在 ood_* 子集上获得更稳的覆盖。

---

## 5. Causal Representation Learning（因果表征学习）· 本周动态

> 本周未检索到显著超越 07-31 §5（CausalVerse / Causal Differentiating Concepts / Reward-oriented CRL，均 NeurIPS 2025）的新 Primary。但 **§4.2 的 PI-SCM + 共形** 把因果结构嵌入分布外不确定度，可视为 CRL 与本项目「机制先验 + 不确定度」的交叉落地；建议把其「因果约束非一致性分数」思路纳入本项目 conformal 校准 SOP。

---

## 6. Calcium Imaging（钙成像）· 新增（承接 07-31 §6）

### 6.1 An open-source pipeline for calcium imaging and all-optical physiology in human stem cell-derived neurons
- **作者**：Wardiya Afshar-Saber, Federico M. Gasparoli, Ziqin Yang, Kellen D. Winden, Cidi Chen, Mustafa Sahin 等
- **年份·来源**：2026（PMID 40777501 / PMC13159138）
- **核心方法**：可扩展开源平台，整合**光遗传刺激 + 钙成像 + 自动采集 + 全流程分析**。用 CRISPR-Cas9 在 AAVS1 安全港敲入 GCaMP6s 的 hiPSC 系；模块化开源采集（micromanager-gui）+ 深度学习分割 + `cali` 单钙动态量化。在 CDKL5 缺陷、SSADH 缺陷、TSC 等神经发育疾病模型上复现活动表型，并用钾通道调节剂展示 TSC 网络亢进的药理功能 rescue。
- **与本项目相关点**：**闭环层直接可用的「全光 + 人源 iPSC 神经元」湿实验底座**。① 光遗传 + 钙成像一体 + 开源，正是本项目「AI 预测 → 光遗传验证 → 误差回流」在**人源细胞**上的可落地硬件/流程；② CRISPR 敲入 GCaMP 同时是**基因扰动（do-）的载体**，与本项目扰动响应预测同源；③ TSC/癫痫表型对应本项目 OOD（新调质/病理状态）的可测表型。

### 6.2 Memory Engrams Show Distinct Learning Ensembles
- **作者**：Pouget, Morier, Autore 等
- **年份·来源**：Nature Neuroscience, 2026
- **核心方法**：海马 dCA1 钙成像以高时间精度标记恐惧联想学习不同时段的神经元；发现不同阶段招募**互不重叠的 temporally-tiered 群体**，光遗传兴奋特定时段群体可在无原刺激下重新激活恐惧表达（因果角色）。
- **与本项目相关点**：① 「记忆由时间分层、非重叠群体 mosaic 编码」进一步支持本项目「编码在群体几何/子空间而非单神经元」立场；② **钙成像 + 光遗传因果操控**确认群体而非单元的因果权重——方法论模板可直接用于闭环层校验本项目 twin 预测的因果正确性（呼应 07-31 §4.1「OOD 好但特征错」的验证需求）。

---

## 7. Optogenetics（光遗传）· 新增（承接 07-31 §7）

### 7.1 Closed-loop optogenetic control of cell biology enables outcome-driven microscopy
- **作者**：Josiah B. Passmore, Alfredo Rates, Jakob Schröder, Lukas C. Kapitein 等（TU Delft）
- **年份·来源**：Nature Communications, 2026, 17:1087（DOI 10.1038/s41467-025-67848-5）
- **核心方法**：提出 **「结果驱动显微（outcome-driven microscopy）」**——把智能显微（实时分析 + 自适应采集）与光遗传结合，以达成**预定义细胞行为结果**为目标闭环控制。用光控细胞迁移与核质运输验证，在单细胞与群体水平实现稳健时空控制。
- **与本项目相关点**：**闭环层范式的直接范本**。① 「以期望结果为目标的闭环光遗传」= 本项目「预测 → 光遗传验证 → 误差回流」的通用化表述；② 把光遗传从「施加刺激」升级为「按预测结果自适应调控」，可直接支撑每周预测清单的**主动实验设计（do-演算用于实验选择）**；③ 单细胞/群体双尺度控制呼应本项目多尺度孪生。

### 7.2 Distinct contributions of two subpopulations of subthalamic neurons to levodopa-induced dyskinesia
- **作者**：Shen 等（复旦大学 王坚团队）
- **年份·来源**：Science Advances, 2026, 12:eaed2912
- **核心方法**：光纤**钙成像**记录 LID（左旋多巴诱发异动症）状态下 STN 两亚群（STN^EP、STN^RtTg）的 U 形钙动态差异；**光遗传**证实激活 STN^EP 改善 LID、抑制加重，激活 STN^RtTg 加重；逆行示踪给出差异上游连接的解剖依据。
- **与本项目相关点**：① 钙成像（光纤光度）+ 光遗传 + 病理状态（LID，属「新调质/病理状态」OOD）三联，交叉覆盖本项目 §6–7–8；② 提示本项目建模「新调质/病理状态」须刻画**亚群特异、状态依赖的动态**（U 形窗口），而非单一 scalars 偏置；③ 为闭环层在疾病模型中验证 twin 预测提供可操作范式。

---

## 本周最值得借鉴的方法（1–2 个）及对建模层（多尺度虚拟孪生）的启示

**方法一：D-SPIN 的「可解释概率图 GRN」（姜家隆等，Cell 2026）。**
它从 scRNA-seq / Perturb-seq 直接构建**白箱基因调控网络**并预测对全新扰动的反应，把扰动当成沿网络的可解释干预而非黑箱映射。对建模层的启示：多尺度虚拟孪生应在「基因/回路」尺度显式落地一张**可审计的因果图**（节点=基因/离子通道/突触群体，边=调控/门控），`do-` 干预沿图传播产生状态偏移——这正是本项目「机制先验 + 不纯黑箱」立场在 GRN 尺度的可直接复用模板，且能让 ood_action 的外推被逐项解释（哪条线路被拨动）。

**方法二：Chreode 的「可迁移单步动力学世界模型」（Qiu et al., 2026）。**
它在小鼠胚胎 atlas 预训练一个结构化残差转移算子，**作为 GEARS 的嵌入直接把 Norman Perturb-seq 的 DE20 MSE 降低 12.4%**——而 Norman 2019 恰是本项目真实基准。其结论「发育轨迹动力学是可迁移到 CRISPR 状态偏移的分化原语、二者共享潜几何」对建模层的启示：本项目孪生应先在**跨尺度/跨情境共享潜几何**上预训练转移算子，再把扰动作为 action-conditioned 残差叠加，而非从零学每个扰动；这能在不牺牲机制可解释性的前提下，直接提升本项目 ood_action / ood_agent 子集的外推。

**配套警示（评测侧，与硬约束同源）**：本周 OCOO-T 综述再次点名 Ahlmann-Eltze *Nature Methods* 2025——深度学习扰动模型仍未系统性胜过简单线性基线，与本项目 Norman 真实基准（线性≈0.627 vs compositional 0.636）互证。因此：① 任何机制模型必须**显式报告均值/线性基线**并证明其显著（CI 不重叠）超越，否则即「疑似仅记忆」；② 不确定度侧可直接采用本周 §4.1 的 **split/normalized conformal（神经算子共形）+ epistemic/aleatoric 分解**给 OOD 子集分布无关覆盖区间，使「不确定度强制」从原则变成可复现 SOP。
