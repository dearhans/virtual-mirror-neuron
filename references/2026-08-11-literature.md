# 虚拟镜像神经元 · 每周文献监测（2026-08-11）

> 增量周报。去重基准：`2026-07-31-literature.md`、`2026-08-01-literature.md`、`2026-08-02-literature.md`、`2026-08-07-literature.md`（已逐条比对标题/作者/DOI，重复项不再收录）。
> 本期新收录 **11 篇**（虚拟细胞 5 / 镜像神经元 2 / OOD 1 / CRL 1 / 光遗传 2 / 钙成像与神经调质经交叉覆盖）。
> 检索轮次：本期在 08-07 的 9 轮检索基础上，对 6 个高价值新候选做了**二次核验**（逐篇确认 arXiv/bioRxiv/Nature DOI、作者、方法细节），以保证写入磁盘的条目可被未来引用核对。
> ⚠️ 含预印本与未同行评审稿件（已逐条标注）。正式引用前须核对原始 DOI 与版本号。

---

## 0. 本期最高优先级（TL;DR）

| 排序 | 论文 | 为什么对本项目是"必读" |
|---|---|---|
| ★★★ | **GeneGeoFlow**（arXiv:2608.06824） | control-anchored residual flow + 基因几何先验 + **Delta-correlation 对齐**，Norman 加性 Pearson Δ 0.8979 / ComboSciPlex 5 个 held-out 药物组合 0.9088——独立验证了 08-07 的 CDE-δ 方向，并给"结构先验 > 容量"再添一枚砝码 |
| ★★★ | **SLIM**（bioRxiv 2026.08.07）+ **in-the-wild benchmark**（arXiv:2604.27646） | 640 参数双线性 + STRING 嵌入 + 闭式岭回归，CPU<10s，单基因 8/12 指标登顶；线性加性模型在 unseen-cell / unseen-perturbation 下最稳——和我们"线性件=远端外推件"的跨域结论完全同构 |
| ★★★ | **两光子全息介观显微镜**（Abdeladim/Adesnik, Nat Neurosci 2026） | 首个近单细胞分辨率（~9×32μm）的**读写同体介观平台**，光刺激 FOV 扩 10× 至 ~3.2×3.2mm，可跨多个远程皮层区近同时 `do-` 干预并同步钙读出——我们的光遗传/钙成像湿实验闭环从设想变硬件可行 |
| ★★☆ | **Confound-diagnostic toolkit**（Qiu&Zhao, bioRxiv 2026.08.04） | Geneformer 嵌入 delta **在 held-out 上无超出基因身份的增益**，表观效应溯源到 library-size / 广谱响应性 / 自指评分——直接警示我们的 φ 与 GOAI 特征工程：基础模型"扰动读数"可能只是伪迹 |
| ★★☆ | **Bilevel LLM Calibration**（arXiv:2608.07419） | 熵最大化 + 双层优化，指出**事后温度缩放是域依赖的、跨域不泛化**——这正是我们 P4-1（conformal 单标量 `_q` 假设 ID/OOD 同分布 → ECE 饱和 0.2125）的根因同构 |
| ★★ | **UMNI-CRL**（Varici et al., NeurIPS 2024） | 未知多节点干预在"充分多样"下**与单节点干预可辨识性等价**，"聚合即力量"——为我们 P4-2（加法先验反解 held-out 基因效应）提供了理论托底 |

---

## 1. Virtual Cell（虚拟细胞 / 扰动响应预测）

### 1.1 GeneGeoFlow: Control-Anchored Residual Flow Matching Conditioned on Gene Geometry
- **作者**：Quanquan Li, Yihe Chi, Liuyang Song, Hongbo Zhang, Jingyu Li, Xidong Xi, Conghua Wei, Yijie Sun, Yu Chen, Xin Liu, Qi Hu, Jing Ke, Guitao Cao
- **年份/来源**：2026-08-07，arXiv `2608.06824`（q-bio，未同行评审）
- **核心方法**：
  - 批判现有图模型"用同一张网络既结构基因表征、又中介基因间交互"，把**稳定的关联（GO、对照衍生共表达）误当成干预特异的响应传播方向**。
  - 提出 **GeneGeoFlow**：从 GO + 对照衍生共表达网络抽取**多尺度谱坐标**作为基因几何；用"扰动条件化的逐基因门控"选择相关结构尺度与网络来源，得到**干预特异的基因几何**；该几何条件化一个 **control-anchored residual flow**（**默认模型不含图消息传递层**）。
  - **condition-wise 最优传输**耦合未配对的对照与扰动群体；**Delta-correlation 目标**对齐预测与观测的"条件级表达偏移方向"。
  - 结果：Norman 加性基准 Pearson Δ **0.8979**；ComboSciPlex 固定切分下 5 个 held-out 药物组合 **0.9088**。
- **与本项目相关点（★★★，最高）**：
  - **方向全中我们的 P3 候选**：① "control-anchored"（以对照为锚）→ 与我们"减 null"的 NTC 配对思想一致；② "Delta-correlation 对齐条件级偏移方向"→ 与 08-07 的 **CDE 差向量 δ** 同构；③ "基因几何作结构先验但不沿图显式传播"→ 直接呼应我们"正确分解 + 正确损失 + 正确 CV"、反对纯黑箱端到端。
  - **对"容量迷信"的又一记**：默认模型**无图消息传递层**却拿 SOTA 级 Pearson Δ——再次证明本领域的增益来自**正确的归纳偏置与损失设计**，而非层数/参数量。
  - **诚实边界（必带）**：它报的是 **Pearson Δ（方向相关性）**，而本期 §1.4 的 in-the-wild 基准明确指出 PCC 类指标**会高估**、且不同指标会重排模型排名。故 0.8979/0.9088 应视为"方向对齐强"，**不能直接换算成我们双栏 RMSE 口径的胜利**——这正是 08-07 我们已立的"指标 SOP 升级" 共识。建议把它纳入我们 canonical 评测的对照方法时，主报 DE/判别性指标，Pearson Δ 仅作辅证。

### 1.2 SLIM: A small linear model with STRING embeddings for single-cell genetic perturbation prediction
- **作者**：Dewei Hu, Marc Pielies Avellí, Lars Juhl Jensen, Simon Rasmussen
- **年份/来源**：2026-08-07，**bioRxiv** `10.64898/2026.08.07.743481`（未同行评审）
- **核心方法**：Ahlmann-Eltze 双线性模型的轻量扩展。扰动用 **STRING 蛋白网络导出的 64 维嵌入**表示，均值转录响应由**闭式岭回归估计器**预测；单细胞群体通过检索训练细胞 + 逐基因重缩放匹配预测均值构造。模型 **640 个可训练参数**，每个基准数据集 **CPU 上 <10 秒**拟合。在 4 个单基因 + 1 个组合扰动数据集上对比 4 个深度学习模型与 2 个简单基线：单基因 12 项"数据集×指标"比较中 **8 项登顶**，且 MMD 显著更低。
- **与本项目相关点（★★★）**：
  - **跨域铁证再临**：我们 Norman 真实基准上 CompositionalTwin P1/P2 与线性 0.627 CI 重叠、以及"线性/加法=最稳远端外推器"的跨数据集（Norman+GOAI）、跨方法（MLP 记忆保持率 34.7%）结论，被独立第三方用 640 参数模型在 5 个数据集上复现。**这是我们的核心 SOP 诊断的第四重互证**（继 Ahlmann-Eltze 2025、Systema 2025、跨域 MISATO 之后）。
  - **直接可用**：SLIM 的 STRING 64 维嵌入 + 闭式岭回归，可原样作为我们 GOAI 赛道的**公开对照预测器**（科学意义 30 分 / 技术 45 分档的"强简单基线"证据）。代码已开源（`github.com/RasmussenLab/SLIM`）。

### 1.3 A confound-diagnostic toolkit for in silico perturbation with single-cell foundation models
- **作者**：Ru Qiu, Mengnan M. Zhao（Mingyao Zhao 组，中山眼科中心 / 多伦多大学）
- **年份/来源**：2026-08-04，**bioRxiv** `10.64898/2026.08.04.732812`（未同行评审）
- **核心方法**：针对"删掉输入序列里的基因 token 即计算机模拟敲除"这一流行策略，指出嵌入 delta 可能**不代表生物学敲除响应**，而是反映基因身份、广谱响应性、tokenization 覆盖不足、library-size 污染或环状状态评分。提出**混杂诊断框架**（held-out 增量测试 + 响应性调整 + 覆盖门控 + library-size 诊断 + 去环状态偏移分析）+ 冻结 Geneformer 扰动引擎的数值对齐复现。在 Frangieh 与 Replogle 数据集、线性与非线性读出上：**原生嵌入 delta 在 held-out 上无超出基因身份的可复现增益**；信号注入标定显示该测试能检出注入残差，而原生增量低于检测地板；匹配对照将表观阳性溯源到 raw-count library-size 结构、广谱响应性与自指评分。
- **相关点（★★☆，对 φ 与 GOAI 特征工程是直接警示）**：
  - **我们的 φ 现在吃的是基因 one-hot**，尚不依赖基础模型嵌入；但若 P3 计划引入 GOAI 蛋白组基础模型表征、或把 Geneformer 类嵌入当扰动特征，这篇告诉我们**必须先过混杂诊断**——否则"扰动读数"测的可能是 library-size/响应性伪迹，而非生物学机制。
  - **对 ood_agent 的旁证**：该文"无超出基因身份的 held-out 增益"恰是我们 P4-2 曾遇到的"held-out 基因效应信号微弱"的**上游同型现象**——提示我们 P4-2 的加法反解路径（从共扰动反推单基因效应）在方法论上是对的，但要警惕"看似学到基因效应、实为身份/批次混肴"的Failure mode。
  - **可移植为一道回归守卫**：把它诊断框架的"held-out increment + library-size 诊断"简化成我们 `scripts/regress_check.py` 的一个子检查，防止 φ 的更新只在测身份。

### 1.4 Benchmarking virtual cell models for in-the-wild perturbation response
- **作者**：Xinjie Mao, Songming Zhang, Qianhong Wen, Xiangyu Wen, Kedu Jin, Hao Wu, Shuizhou Chen, Yuqiang Li, Lei Bai, Qi Liu, Ning Ding, Siqi Sun, Zhangyang Gao
- **年份/来源**：2026-04-30，arXiv `2604.27646`（q-bio，未同行评审；虽早于本监测启动，但前 4 期未收录，本期补入）
- **核心方法**：模块化基准，在三类"实战"场景评测——**unseen 细胞背景 / unseen 扰动 / 跨数据集**。发现：严格切分下几乎所有模型性能**显著下降**；**线性加性模型（BioLORD、scLAMBDA）最稳**；朴素数据集聚合**反而降性能**；**PCC 类全局指标会高估**，扰动聚焦指标（PDCorr、DEG 相关）更能区分模型；HVG 上的好表现≠对特定扰动生物学效应的精确量化。
- **与本项目相关点（★★★）**：
  - 本报告把我们 08-07 立的"指标 SOP 升级"从**单篇证伪升级为系统性证据**：三个独立团队（Empirical Comparison / Response Magnitude / 本文）现在共同指向"全基因 RMSE/PCC 不可信、必须上 DE/判别性/分布级指标"。
  - **"线性加性最稳"第四次互证** → 我们双栏 SOP 的阶段 D 诊断（"ood_action 无判别力"可能是基准/指标伪影）进一步坐实。建议把本文的 Energy Distance / Sinkhorn Divergence 也加进我们 canonical 评测表，与 bootstrap CI 并列。

### 1.5 scDFM: Distributional Flow Matching Model for Robust Single-Cell Perturbation Prediction
- **作者**：Chenglei Yu, Chuanrui Wang, Bangyan Liao, Tailin Wu
- **年份/来源**：2026，**ICLR 2026**，arXiv `2602.07103`（代码 `github.com/AI4Science-WestlakeU/scDFM`）
- **核心方法**：基于**条件流匹配**的生成框架，建模以对照状态为条件的扰动后细胞**完整分布**（非仅均值）。引入 **MMD 目标**对齐扰动/对照群体（超越细胞级对应）；骨干 **PAD-Transformer** 用基因交互图 + 微分注意力捕获上下文特异表达变化。**组合扰动设定下 MSE 比最强基线 CellFlow 低 19.6%**。
- **与本项目相关点（★★）**：
  - **组合（非加和）扰动才是深度模型真正增益阵地**——这与 08-07 Empirical Comparison 第三条、Response Magnitude 的结论**完全一致**，也正好对应我们 P2·epistasis 扫描的发现（richer epistasis 被黑箱 MLP 吃到、结构化先验只在纯加法 regime 占优）。
  - **"分布级建模而非均值"= 我们不确定度强制的天然盟友**：scDFM 直接产出方差/亚群结构，而我们当前不确定度是事后 bootstrap。若未来引入流匹配组件，可参考它"MMD + 微分注意力"来同时改善组合外推与分布校准。
  - 注：与 08-02 的 PerturbDiff（函数空间扩散、MMD 内生）是**不同方法**（流匹配 vs 扩散），本期不重复收录 PerturbDiff。

---

## 2. Mirror Neuron（镜像神经元）

### 2.1 Computational Mirror Neurons Emerge from Motor-Routed Observation and Hebbian Learning
- **作者**：Shamim Khaliq（独立研究者，单一作者稿件）
- **年份/来源**：2026-04-08，**philarchive 手稿**（未同行评审；⚠️ 该作者另有**已被 Research Square 撤回**的预印本——因把 LLM 列为作者违反署名政策——故本条目须以"未同行评审手稿"对待，引用前务必谨慎）
- **核心方法**：在人工传感运动网络中识别计算镜像神经元涌现的**两个必要条件**。(Exp1) 仅双向预测训练**不产生**镜像神经元——两个标准架构学到平行但不可迁移的执行/观察编码。(Exp2) 把观察经运动通路路由（感觉与运动共享丘脑网关）是**必要非充分**——即使强制共享嵌入，反向传播仍找到分离编码。(Exp3) 加入模拟尖峰网络 Hebbian 共激活的对齐信号后，镜像神经元**稳健涌现**：**30–40% 隐单元成为镜像神经元**，跨模态解码达约 3× 机会水平（绝对精度 0.31–0.36），反镜像单元几近消失。**两个条件（运动路由架构 + Hebbian 学习）缺一不可**。
- **与本项目相关点（★★）**：
  - **直接给 08-07 Chatzimichail"执行↔观察不对称性"补了计算机制**：不对称性不是 bug，而是"观察必须经运动通路路由 + Hebbian 对齐"这两步才涌现镜像。落到我们镜像模块：(a) 观察编码器不能直连共享潜空间，必须**经执行/运动通路路由**；(b) 需加 **Hebbian 共激活对齐**作为监督外的结构信号。这与我们 08-07 定的"部分重叠 + 执行侧私有残差 + Hebbian"架构暗示高度吻合。
  - **可证伪预测**：若真实数据也呈现"仅双向预测不够、必须运动路由+Hebbian"，则在 ood_agent（跨主体）上，加了运动路由的架构应显著优于纯共享 encoder。这是一条可写进本周预测清单的双栏假设。
  - ⚠️ 因作者有撤稿史且为单作者未审稿件，**仅作方法论启发，不直接作为证据引用**。

### 2.2 Motor resonance in functional movement disorders: repertoire-congruent mechanisms of action observation
- **作者**：Gloria P. Mingolla, Elena Antelmi, Angela Sandri, Mehran Emadi Andani, Ilaria A. Di Vico, Mirta Fiorio, Michele Tinazzi
- **年份/来源**：2025–2026，TMS + fMRI 临床研究（功能性运动障碍 FMD 患者 vs 健康对照）
- **核心方法**：两项实验。(1) TMS 测 corticospinal 兴奋性：健康对照在观察强/弱力手部动作时呈现力依赖调制；**FMD 功能性无力患者在此调制上消失**，尤其在观察超出其运动库（repertoire）的强动作时。(2) fMRI 观察健康 vs 病理步态：患者**对与其自身运动库一致的病理步态激活更强**，健康对照相反。结论：运动表征完好但增益降低，**运动共振偏好 repertoire-congruent（与自身运动库一致）的动作**。
- **与本项目相关点（★）**：
  - **"运动库（embodiment）调制共振"= 预测编码的具身先验**：观察者的自身运动经验塑造其镜像响应。我们的镜像模块若要做成"预测编码"式，应**以 agent 自身的运动库/具身状态为条件**——这与 08-07 Tsubamoto"集体预测编码"、Nieto-Doval"运动周边抑制"共同指向"镜像不是通用匹配，而是受自身经验门控的对比增强"。
  - **湿实验设计启示**：闭环读出窗口与"动作库一致性"应作为分组变量；预测→干预→读出的误差，可能随"该动作是否在 agent 运动库内"系统性变化——可写成 ood_agent 之外的第五类 OOD（未见动作库）。

---

## 3. Perturb-seq / scRNA-seq Perturbation

> 本期无独立于 08-07 的新原发方法。08-07 收录的"多重扰动细胞联合分析（sprinter + PerturbMatch）"仍是当期主线；本期 §1.4（in-the-wild benchmark）的跨数据集/跨扰动切分口径可视为对该方向的基准级补强。故本节不再单列新条目，避免重复。

---

## 4. OOD Generalization（分布外泛化）

### 4.1 Beyond Post-Hoc Temperature Scaling: Bilevel Optimization for LLM Calibration
- **作者**：Ruochen Jin, Zhanliang Wang, Zongyu Dai, Jiancong Xiao, Bojian Hou
- **年份/来源**：2026-08-07，arXiv `2608.07419`（cs.LG / ML，未同行评审）
- **核心方法**：指出偏好对齐常使 LLM **过度自信且校准差**；传统**事后温度缩放是域依赖的**——在一个域拟合的温度**跨域不泛化**。故改为在训练中修改参数：以**最大化预测分布熵**为校准目标（直接针对过度自信），并用**双层优化**实现（下层在参数化损失下训模型，上层选损失超参以最大化熵）；用一阶近似避免二阶计算，在多项选择与开放式生成上均取得良好校准且**OOD 泛化优势明显**。
- **与本项目相关点（★★☆，直击 P4-1）**：
  - **根因同构**：我们 P4-1 卡点是"conformal 单标量 `_q` 假设 ID/OOD 同分布 → ECE 饱和 0.2125、覆盖恒 1.000"。本文证明"事后全局标量校准跨域崩"是**通用现象**，不是我们实现 bug。**破法一致**：必须做**分 subset / localized conformal**（条件分离），而非一个全局温度/分位数。这把我们 P4-1 从"本地待办"升级为"有独立文献支撑的必做项"。
  - **可移植**：把"熵最大化校准目标"思想借来监控我们 conformal 的覆盖——若某 subset 的预测置信熵异常低（过度自信），即触发 localized 重标定。

---

## 5. Causal Representation Learning（因果表征学习）

### 5.1 Linear Causal Representation Learning from Unknown Multi-node Interventions (UMNI-CRL)
- **作者**：Burak Varici, Emre Acarturk, Karthikeyan Shanmugam, Ali Tajer
- **年份/来源**：NeurIPS 2024，arXiv `2406.05937`（代码 `github.com/acarturk-e/score-based-crl`）
- **核心方法**：突破 CRL 长期依赖的**单节点干预（SNI）假设**——现实中干预常同时影响多个未知节点。提出 **UMNI-CRL**（基于 score 函数 ∇log p(x) 的算法）：跨环境的 score 差编码因果几何；四阶段恢复潜伏变量与 DAG（基 score 差 → 拓扑序 → 祖先可达闭包 → 硬干预解混）。**核心定理"聚合即力量（Aggregation is Power）"**：多个未知多节点（UMN）干预在"干预签名矩阵满秩（充分多样）"时，**可辨识性与严格 SNI 设定等价**；硬干预下完美恢复，软干预下恢复到祖先。合成 SEM 上 MCC>0.90。
- **与本项目相关点（★★）**：
  - **为我们 P4-2 加法先验反解提供理论托底**：我们 P4-2 用 `ê_u = z_au − b̂ − ê_a` 从共扰动反解 held-out 基因单效应，本质是"用多个（可能多节点、未知靶向的）共扰动聚合出单基因等效信号"——UMNI-CRL 证明这在前述"充分多样"条件下**理论可行**。
  - **前提警示（可直接用）**：UMNI-CRL 的"充分多样"=我们的"共扰动条数 62~2971 差 47×、等权平均被低证据基因拉偏"（P4-3 `ê_u` 逆方差收缩）。该文告诉我们：**聚合能否还原单节点信号，取决于干预集合的多样性与满秩性**——这正是我们 P4-3 判据（var_ratio>0.1）的理论注解。若某 held-out 基因的共扰动不够多样，反解就不可靠，应如实报告而非强行插补。

---

## 6. Calcium Imaging（钙成像）

> 本期无独立于前 4 期的新原发 GECI/读出工具（PinkyCaMP 已录 08-02、OCaMP 已录 08-07）。但本期 §7.1 的两光子全息介观显微镜**原生同步钙读出**，钙成像在本周经 §7.1 交叉覆盖。故本节不单列新条目。

---

## 7. Optogenetics（光遗传）

### 7.1 Probing inter-areal computations with a two-photon holographic mesoscope
- **作者**：Lamiae Abdeladim, Uday K. Jagadisan, Hyeyoung Shin, Mora B. Ogando, Hillel Adesnik（UC Berkeley）
- **年份/来源**：2026，**Nature Neuroscience**，DOI `10.1038/s41593-026-02350-9`
- **核心方法**：在商用 2p-RAM 介观显微镜上集成 3D 扫描less 全息光遗传（**3D-SHOT 时间聚焦**）。近单细胞分辨率（侧向 ~9μm / 轴向 ~32μm）。**光刺激 FOV 扩 10× 至 ~3.2×3.2mm**。可单/多皮层区内**全息精确激活时空序列神经元**，同时读出多个下游区的群体钙活动；首次实现**近同时跨多个远程皮层区的特定神经集合光激活**。验证：仅从下游突触后神经元活动调制即可解码光刺激身份；可在小鼠视觉皮层跨区全息重建并传递"类视觉"信息。
- **与本项目相关点（★★★，湿实验闭环直接对口）**：
  - **我们的闭环验证路线从设想变硬件可行**：预测 → 光遗传 `do(神经元群 X = 激活)` → 钙成像读出群体几何变化 → 误差回流。该平台把"写入（全息光遗传）"与"读出（介观钙成像）"在**同一台、近单细胞、跨多远程区**上统一，正是我们多尺度孪生做"因果跨脑区连接图谱"的现成工具。
  - **对 §2 镜像不对称性假说的使能**：Chatzimichail/Tsubamoto 的"执行↔观察不对称"需在**群体尺度、跨脑区**上才能证伪——本平台恰好提供该尺度。
  - **写进 wetlab/**：把"介观读写同体"列为闭环平台升级目标；短期仍以啮齿类全光方案（APPC + PinkyCaMP/OCaMP）推进，中长期评估 2p-RAM+3D-SHOT。

### 7.2 Stimulation modulates gene-linked cell assemblies in the human brain
- **作者**：Haley Moore, Mantre Dehnad, Anne Freelin, Bryan Granger, Suganya Subramanian, Tjitse van der Molen, Ashwinikumar Kulkarni, Stefano Berto, Bradley C. Lega, **Genevieve Konopka**
- **年份/来源**：2026-08-05，**Nature**，DOI `10.1038/s41586-026-10879-9`（数据 GSE288939；代码 `github.com/BioinformaticsMUSC/Moore_etal_MEA`）
- **核心方法**：为直接研究人脑刺激的神经调制机制，开发**离体平台**——微电极阵列（MEA）刺激 + 同步记录 + **单核基因组学**（来自神经外科切除的人颞叶皮层）。发现刺激**增强细胞集合**，并将该效应关联到**细胞类型特异的基因调控网络（GRN）**；在体刺激后的人皮层中鉴定出共同的细胞类型特异基因表达特征。
- **与本项目相关点（★★，ood_neuro / 调质调制场先验）**：
  - **"刺激 → 细胞集合增强 → 细胞类型特异 GRN"** 把"神经活动"与"基因程序"在**人脑**尺度上因果串联——正是我们 ood_neuro（新调质/新神经状态）子集想建模的"调制场 → 基因响应"链路的实验原型。
  - **对建模层指令**：调质/刺激不应建模为"每个 episode 一个常数标量"，而应建模为**细胞类型特异、作用于 GRN 的门控场**（呼应 08-02 Colangelo、08-07 5-HT 时变门控结论）。可作为我们 ood_neuro 先验的"人脑验证"引用。
  - 注意：该研究是**离体 + 人组织**，与我们啮齿类在体闭环是互补而非替代；但提供了"刺激如何改写可靶向遗传特征"的直接证据，对 GOAI/评委科学意义档都有用。

---

## 8. Neuromodulator（神经调质）

> 本周原发性研究主要经 §7.2（刺激 → 细胞类型特异 GRN）与 §7.1（因果跨脑区读写平台）交叉覆盖，无独立新增条目。08-07 收录的 5-HT / ACh / DA 集群（dorsal raphe、Colangelo 等）仍是该方向当期主线。

---

## 9. 不确定度与校准（跨方向，本项目硬约束专栏）

### 9.1 Bilevel LLM Calibration 的"校准必须 OOD 感知"（见 §4.1）
- 核心警示：事后全局标量校准**跨域不泛化**。映射到我们 P4-1：conformal 单标量 `_q` 假设 ID/OOD 同分布 → ECE 饱和 0.2125 是同一类错误。→ **行动项（已立项）**：把 conformal 从单标量升级为分 subset / localized conformal，并在 `code/benchmark_ood.py` 报告里逐 subset 报覆盖@level 与 ECE。

### 9.2 混杂诊断框架的"选择性预测"启示（见 §1.3）
- Qiu&Zhao 证明基础模型扰动 delta 在 held-out 上无超出基因身份的增益。这提示我们的**不确定度不仅要覆盖 aleatoric/epistemic，还要覆盖"表征伪迹"风险**——即 φ 的"扰动读数"本身可能不可信。→ 可把"held-out increment + library-size 诊断"简化成 `regress_check.py` 子检查，防止 φ 更新只在测身份/批次。

### 9.3 与 08-07 不确定度专栏的衔接
- 08-07 收录的 PRESCRIBE（epistemic/aleatoric 分离 + pseudo E-distance）仍是内化不确定度的首选配方；本期新增的 scDFM（分布级 MMD，§1.5）与 Bilevel LLM Calibration（OOD 感知校准，§4.1）分别补齐了**分布级产出**与**OOD 感知重标定**两个拼图。三者合起来正好对应我们硬约束"不确定度强制"的三种来源：内生（PRESCRIBE）、分布级（scDFM）、OOD 感知（Bilevel）。

---

## 10. 本周总结：最值得借鉴的两个方法，及其对建模层（多尺度虚拟孪生）的启示

本周最值得借鉴的是 **GeneGeoFlow 的"control-anchored residual flow + 基因几何先验 + Delta-correlation 对齐"** 与 **"混杂暴露"这条贯穿全周的元主题（以 Qiu&Zhao 的 confound-diagnostic 与 Bilevel LLM Calibration 为两端）**，前者给了我们 P3 方向一份带 SOTA 数字的独立背书，后者则把我们的两个软肋（φ 的扰动表示、conformal 的不确定度）一次性钉在了"必须 OOD 感知"这同一个靶上。GeneGeoFlow 的洞见是：虚拟细胞的增益不来自"沿基因图显式传播信号"这种容量堆叠，而来自**三件朴素的事**——(1) 以对照为锚（control-anchored），即所有响应都相对非靶向对照来定义；(2) 把 GO/共表达网络的几何当作**结构先验去条件化**流，而不是当作消息传递骨架去硬塞机制；(3) 用一个 **Delta-correlation 目标直接对齐"预测的条件级表达偏移方向"与"观测到的偏移方向"**。这三点与我们 08-07 立的 CDE 差向量 δ（φ 应作用在 control/perturbed 配对上、减法抵消共享背景）几乎逐字对应，且现在有了 Norman Pearson Δ 0.8979 / ComboSciPlex 0.9088 的硬数字——只不过必须带着 §1.4 in-the-wild 基准的提醒读它：它报的是**方向相关性**，不能换算成我们双栏 RMSE 口径的胜利，否则又掉回"PCC 类指标高估"的坑。落到我们 P3，这意味着 CompositionalDeltaTwin（φ 改潜空间差分 + ψ 改吃 (‖δ_a‖,‖δ_b‖,cos∠) 几何量）的方向**不是拍脑袋，而是已被两个独立团队（CDE + GeneGeoFlow）从不同数据模态验证过的收敛解**，可以放心推进。

而真正让本周"连点成线"的，是混杂暴露这条主线。一端是 Qiu&Zhao：他们证明 Geneformer 类基础模型的"删 token → 嵌入 delta"在计算机模拟敲除上，**在 held-out 基因上没有任何超出基因身份的可复现增益**，表观效应全可溯源到 library-size、广谱响应性、自指评分——换句话说，最时髦的"基础模型扰动读数"测的很可能只是**伪迹**。另一端是 Bilevel LLM Calibration：他们证明事后温度缩放这种"全局标量校准"是**域依赖的、跨域直接崩**，我们 conformal 单标量 `_q` 假设 ID/OOD 同分布导致 ECE 饱和 0.2125，正是同一个错。把两端叠到我们项目上，结论非常锋利：**我们的 φ（扰动怎么被表示）和我们的不确定度（conformal 怎么给区间）当前都可能是"只看分布内、没看分布外"的产物**——φ 可能只在记基因身份/批次，conformal 可能只在 ID 上校准、OOD 上形同虚设。这正好分别命中我们 P4-3（ê_u 逆方差收缩，判据 var_ratio>0.1）与 P4-1（localized conformal）两个待办，且把它们的优先级从"本地优化"抬升为"有独立文献支撑的必做"。对多尺度虚拟孪生的启示因此是一条贯穿三层的指令：**表示层要 OOD 感知**（φ 的更新必须过混杂诊断，不能只优化 ID 损失——可把 Qiu&Zhao 的 held-out increment + library-size 检查简化成 `regress_check` 子项）；**融合层要 OOD 感知**（ψ 的交互增益门控 g(u) 在 epistemic 高时必须收缩到加性先验，而非无条件叠加）；**不确定度层要 OOD 感知**（conformal 从单标量升级为分 subset/localized，逐 subset 报覆盖与 ECE，呼应 Bilevel 的"校准必须随域变化"）。三者加起来，我们"不确定度强制"这条硬约束才真正从"事后 bootstrap CI"进化成"从表示到校准全链路 OOD 感知"——而这不是加一层网络能解决的，是按 TRIZ 条件分离把每一个组件在 ID 与 OOD 上分别问责。

---

*生成时间：2026-08-11 · 自动化 `automation-1785494084767` · 仅写入本地文件，未外发*
