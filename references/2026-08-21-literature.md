# 文献监测 · 2026-08-21（W34 轮）

> 8 方向周监测，对比 07-31 / 08-01 / 08-02 / 08-07 / 08-11 / 08-12 / 08-16 去重。
> 本轮 web 检索 8 方向中 6 方向返回真实结果；**OOD 校准 / CRL 两方向检索临时失败（fetch error），其候选沿用 08-16 轮 + 本周 P0 内部校准结论**，已在对应方向显式标注。
> 诚实声明：以下条目的 DOI/作者/方法除标注「(08-16 轮收录)」外，均为本轮（2026-08-21）web 检索实回结果；未做二次人工核验的以原始 URL 锚定，不担保最终发表版本。

---

## 1. 虚拟细胞（virtual cell）

- **Virtual Cell Challenge 2026**（virtualcellchallenge.org，8/20 开赛，11 月截止）：今年升维到 **multi-context 泛化 + zero-shot 预测**——给定非靶向对照谱，预测多细胞系对基因敲低的响应，评分用 Arc Institute 新实验扰动数据（模型未在其上训练）。恰为 **ood_agent 外推锚点**；不要求开源权重，鼓励工业界参与（$100k/$50k/$25k）。→ 与本项目「记忆 vs 机制泛化」硬约束直接同构，可作对外评测基准候选。
- **VCBench**（Weidener, Brkić, Jovanović, Ulgac, Meduri；AppliedScientific；bioRxiv，github.com/AppliedScientific/VCBench）：7 维基准（扰动响应/跨物种/GRN/模态整合/时序/多尺度/体外实验），评 5 个基础模型 vs **预注册线性 + 最近邻基线**——基线在 5 个可测维度中 4 个 ≥ 基础模型（第 6 次互证「DL 未胜线性」）。附 **Contamination Reporting Schema + spread-error 认知校准探针**。→ 与本周 P0 内部结论同源：spread-error 探针（预测展布 vs 实际误差相关性）正是检出饱和/坍缩的诊断器，建议并入 `verify_collapse`（与 W33 §4.3 失效模式 C 的口径核算双检）。
- **OCOO-T**（Jiang, An, Zhao, Lai；arXiv Jun 11 2026）：极简 flow-matching vanilla Transformer AIVC，扰动预测转为连续时间去噪，自适应 LayerNorm + in-context token 注入扰动/剂量/细胞型；Tahoe100M/Replogle/PBMC **SOTA**。→ 反向佐证本周 P0 立场：「结构手术不如表示/标定」——OCOO-T 赢在极简流匹配表示，而非复杂辅助模块。

## 2. Perturb-seq / do-演算

- **Park & Li**（UPenn；arXiv:2601.01830v2）：用 proxy + 工具变量（IV）做 **混杂鲁棒因果发现**，处理 Perturb-seq 未测混杂；K562 上优于忽略混杂的基线。→ 直击本项目「混杂暴露」元主题（Qiu&Zhao + Bilevel，08-11/08-16 轮）；φ/GOAI 特征工程须 OOD 感知。
- **ADAPRE**（Sun & Keleş；bioRxiv 10.64898/2026.02.18.706642）：把 CRISPRi 干预当 IV，嵌 **Poisson-lognormal 观测层**解耦测量与表达，基因特异自适应惩罚纠 **强度依赖度偏置**（强敲低基因被误判为 hub）。→ 与本周 P0 同源精神：显式建模 UMI 测量误差 ↔ 我们的 aleatoric σ 分层；异构敲低强度须建模否则拓扑失真。
- **Latent Linear DAGs for interventional count data**（arXiv:2603.25838v1）：潜线性高斯 DAG + **Poisson 测量误差 + 潜混杂**，mean-shift 干预下建立可辨识性（不依赖 faithfulness），给出有限样本 DAG 恢复率。→ 把「测量误差解耦」从 GRN 推到一般干预计数数据；托底 P4 类「必须 OOD 感知」论断。

## 3. 光遗传（optogenetics）

- **STIMscope**（bioRxiv 10.64898/2026.05.27.728160）：厘米级全光（宽场成像 + DMD 图案化刺激），**CRISPI 闭环管线**（GPU 加速校准投影 + 在线 trace 提取 + 帧精度同步），BOM <$5000、全开源。→ 湿实验闭环使能（W33 E7）；低成本全光平台降低准入门槛。
- **FARO**（Hinderling et al.；bioRxiv 10.1101/2025.08.17.670729v2，github.com/pertzlab/FARO）：Feedback Adaptive Real-time Optogenetics，自动分割/跟踪/特征提取 + 自适应硬件控制，按活细胞行为实时更新刺激图案；开源 MIT。→ 闭环光遗传靶向范式，可直接承载 E7 的「刺激→读 Ca→调刺激」闭环。
- **双光子光遗传 + BMI 综述**（PMC12959727）：螺旋扫描/时聚焦/3D-CGH 三策略，脑-AI 闭环（钙成像→AI→光遗传）愿景。→ ood_neuro 第三轴闭环读写的远期使能。

## 4. 钙成像（calcium imaging）

- **PinkyCaMP**（Fink et al.；Nature Methods 2026，s41592-026-03065-2）：mScarlet 红 GECI，亮度 14× jRCaMP1a、2× RCaMP3，**无蓝光光切换**、与蓝光电遗传（stCoChR/stGtACR2）兼容、可双色（GCaMP8s/sDarken）。→ **全光实验读端使能**（W33 E7）：红读 + 蓝写零串扰；代价是动力学慢（上升 ~670ms）。
- **ScaRCaMP**（Zhang et al.；bioRxiv 10.64898/2026.02.28.708321）：光电遗传兼容红 GECI，蓝光光活化可忽略（>200 mW/mm² 稳定），K132Y 突变将响应提至 ΔF/F0=−22%。→ 与 PinkyCaMP 互补的红读选项。
- **SomaFRCaMPi**（ebraincase 新闻，mApple 衍生）：胞体靶向红 GECI，密集区 ROI 检出 3.3×、事件 +50%、串扰 −30%。→ 高密度群体读。

## 5. 神经调质（neuromodulator）

- **Touponse et al.（Eshel, Stanford）**（Nature 2026，s41586-025-10046-6）：**ACh→DA 局部调制驱动努力行为**——高努力奖励触发 NAc 胆碱能中间神经元（ChAT-IN）先放（奖励前 ~400ms），经 α4/α6-nAChR 放大 DA 轴突末梢释放；阻断 ACh 只损高努力 DA、低努力不变。→ ood_neuro 第三轴范本（08-16 轮已收录，本轮检索复核确认）。
- **Constantinople et al.（NYU）**（Nature Neuroscience 2026，s41593-026-02227-x）：**ACh-DA 释放时机（ tens of ms）决定 DA 是促学习还是促运动 vigor**——DA 与 ACh 下降同步→学习；与 ACh 爆发同步→运动 vigor。→ ood_neuro 调制场**时间异质先验**新证据：调质效应是门控时序而非静态增益，直击 P3 增益轴「乘法放大不存在」的细化方向。

## 6. 镜像神经元（mirror neuron）

- **Chatzimichail et al.**（Science Advances 2026，10.1126/sciadv.aed9309）：猕猴前运动皮层 433 神经元执行/观察抓握，**共享部分重叠群体几何**，运动学信息跨 agent 泛化，最强对齐在运动/保持期。→ 与本周 P0 同源：共享群体几何（cf. 我们的 mirror_k 轴）是「执行-观察匹配」的承载层，而非单神经元严格一致。
- **Khaliq**（philarchive，archival 2026-04-08）：计算镜像神经元——**运动路由架构 + Hebbian 对齐**双条件齐备才涌现（30–40% 隐单元成镜像元），跨模态解码 ~3× 机会。→ 08-16 轮标「未审手稿须谨慎」，本轮确认已归档；与 W33 §5 E2（镜像轴反向证伪）互证「路由+Hebbian 缺一不可」。
- **Cell 2026「Hierarchical and context-dependent encoding of actions in human posterior parietal and motor cortex」**（qq.com 报道，BMI 单神经元、人 SPL+MC）：12 条件（左右手×抬起/滑动/旋转×左右），意图 vs 观察分离记录——人脑动作理解是**脑区分工 + 层级 + 任务目标调节**的动态表征，挑战简单镜像账户。→ 人脑单神经元直接证据，补猕猴群体几何的另一面（情境/层级门控）。

## 7. OOD 泛化 / 校准（conformal）

> ⚠️ 本轮 web 检索临时失败（fetch error），以下候选**沿用 08-16 轮（未本轮核验）**，并补本周 P0 内部校准结论。
- 08-16 轮候选：CaPTURe（arXiv:2608.09166，机制状态切换局部校准）、RCCP（arXiv:2608.10553，检索校正共形）、Confidence Calibration thesis（arXiv:2608.12100，NACP 噪声感知共形）。
- **本周 P0 内部贡献（GOAI 真实数据）**：localized conformal（中心=P1 宽=fglex σ）在 val-CV 五子集 cov@0.95 全入带 [0.935,0.955]；独立 test 划分 ood_action cov@0.95=0.9322（ECE 0.0579 ✓）。ECE canonical 4 档在 ood_s3/id 微超 0.08（残差 platykurtic，95% 水平精准）→ 与 VCBench spread-error / CaPTURe 局部校准同源，证成本周「校准须 OOD 感知 + 须申明档位口径（R5）」。

## 8. 因果表示学习（CRL）

> ⚠️ 本轮 web 检索临时失败（fetch error），候选沿用 08-16 轮（未本轮核验）。
- 08-16 轮候选：ACIF（arXiv:2608.06427，worst-intervention IPM 主动证伪）、DAG-FM（arXiv:2607.11510v2，异质 FCM 近乎必然可辨识）、Causal World Models（arXiv:2608.13456，Imperial）。
- 本周关联：P0 #2 fglex 用**连续化合物表征（官能团指示向量）**救 H1（σ 单调于流形距离，ood_action mono +0.99），本质是「把 one-hot 离散编码换成连续因果表示」——与 CRL「连续干预编码」方向一致；但属表征层改动，非新 CRL 方法。

---

## 本周对建模层的启示（汇总）

1. **校准须 OOD 感知 + 须申明档位（R5）**：VCBench spread-error 探针、CaPTURe 局部校准、本周 P0 localized conformal 三路同源 → 把「饱和/坍缩」诊断并入基准流水线（`verify_collapse` 双检）。
2. **连续表征救 H1**：fglex 官能团向量在 GOAI 真实数据上修复 one-hot 导致的 epistemic 退化（mono ood_action +0.99）→ 与 CRL 连续干预编码、OCOO-T 极简流表示互补。
3. **湿实验闭环工具齐备**：PinkyCaMP（光电遗传兼容红读）+ STIMscope/FARO（闭环光遗传）使 W33 E7 闭环光遗传从「提议」进入「可实施」。
4. **ood_neuro 第三轴细化**：Constantinople ACh-DA 时序门控 → 增益轴非静态乘法，而是时机依赖门控，修正 P3 增益轴假设。
5. **对外评测锚点**：Virtual Cell Challenge 2026（8/20 开赛，zero-shot 多细胞系）可作 ood_agent 外推的公开第三方佐证（若组委会允许），避免用外部集冒充官方独立集。
