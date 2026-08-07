# GOAI 虚拟细胞 · 数据到手 · 工作计划

> 日期：2026-08-06 | 数据已解压到 `data/raw/goai/`

## 0. 数据实测总结

### 0.1 与教程预期的差异（列名需映射）
| 教程假设 | 真实列名 |
|---|---|
| `strain` | `Strains` |
| `chemical` | `perturbation_no_concentration`（中文，如"Amphotericin B"）+ `pert_id`（缩写如"#9"） |
| `medium` | `Medium` |
| `temperature` | `Temperature` |
| `time` | `pert_time`（数字）+ `pert_time_unit`（"min"） |
| `plate` | `Yeast_cell_plate` |
| `product_id` | 无此列——对照样本嵌在 `perturbation_no_concentration` 中（Water/DMSO/EDTA） |
| 无 | 新增 `strain_role`（train/val/test）+ `chemical_role`（train/val/test）|

### 0.2 实测统计
- **样本**：train 5,920 / val_strain_only 1,547 / val_chem_only 1,065 / val_both 269 / val_time 157 → **共 8,958**
- **测试提交**：test_chem_only 1,640 / test_strain_only 1,534 / test_both 1,129 / test_time 151 → **共 4,454**
- **菌株 6**：train={BAH, CEK, CGD, DHY210} | val-only={BAI} | test-only={CRD}
- **化合物 46**：42 种真实扰动 + EDTA/Water/DMSO（对照）+ Quality Control（质控）| train/val 34+3 | test-only 11 种（如 MMS, H2O2, Tamoxifen 等）
- **蛋白质 5,245 维**：整体 NA 率 27.4%；817 蛋白 ≥80% 缺失 → 过滤后约 **4,428**（教程预期 4,232）
- **条件**：2 培养基（glucose/galactose）、2 温度（30/37）、6 时间点（15/30/60/90/120/240 min）

---

## 0. ⚠️ 赛制澄清：官方**不给**逐次分数反馈（2026-08-06 修订）

原计划 Day1 写的"提交一次拿官方反馈指标"是**错误假设**，已作废。依据（均来自官方教程 PDF《虚拟细胞-解题思路》）：

| 证据 | 出处 | 推论 |
|---|---|---|
| 初赛提交物 = **方案说明文档 + 技术路线概述**，不强制提交代码 | §1.6 | 初赛是**人工评审**，不是 Kaggle 式自动判分 |
| 官网提交上限 **3 次，以最后一次为准** | §1.5 | 若有实时回分，不会限次且"以最后一次为准"——限次正说明**没有探路价值** |
| 统一排名按 45 技术性能 / 30 科学意义 / 20 方法创新 / 5 开源，三方向同台归一化 | §1.5 | **55% 是评委主观分**，无法由分数反馈优化 |
| 教程通篇强调"内部研究必须分场景分别报告"、"Global R² 区分度极低" | §1.3 / §1.4 | 官方在**教你自建评测**，因为你拿不到线上分 |

**三条硬结论**：
1. **本地评测器 = 唯一裁判**。必须自建严格对齐官方 6 模块权重 × 4 场景的打分脚本（任务 1.3），优先级高于任何建模。
2. **3 次提交不做探路**，全部押后；中途只在本地打分迭代。
3. **文档权重 ≥ 代码权重**（55% 主观分），Day7 的方案文档不是收尾杂活，是主战场。

> 待用户侧确认（唯一权威）：官网/选手群是否提供验证集在线打分。若确有回分，则第 1 次提交可提前用于校准本地评测器与官方口径的偏差；在得到确认前，一律按"无反馈"执行。

---

## 1. 分阶段工作计划（Week 1 = 8/6–8/12，初赛 8.16 截止 = 仅 10 天）

### Day 1（8/6 当天）：数据流水线 + 基线对齐 — PRIORITY

| 任务 | 产出 | 时间 |
|---|---|---|
| **1.1 修正 `goai_virtualcell_adapter.py` 的 COLUMN_MAP**：Strains→strain, perturbation_no_concentration→chemical, Medium→medium, Temperature→temperature, pert_time→time, Yeast_cell_plate→plate, pert_id→compound_id；新增 strain_role/chemical_role 字段 | 适配器就位，`--run` 可读真实数据 | 30min |
| **1.2 数据预处理**：sample_ID 对齐元数据×蛋白矩阵 → 仅训练行算缺失率过滤（≥80% 删）→ mask 矩阵 → log2 转换 | 可用的 X/y/mask 三件套 | 20min |
| **1.3 本地评测器（⚠️ 最高优先级，替代"靠提交拿反馈"）**：按官方 6 模块权重实现 `code/eval_goai.py`（FC-PCC 25% / 上下文均值残差 20% / 药物均值残差 20% / 绝对保真 R²·PCC 20% / 双重未知·时间外推 10% / DEP 检出 5%），四场景分桶（val_chem_only / val_strain_only / val_both / val_time） | **唯一裁判**。见 §0 赛制澄清 | 1.5h |
| **1.4 蛋白均值基线**：每个蛋白用训练集非缺失 log2 均值预测 → 用 1.3 本地评测器打分，对齐赛题报告 Global R²≈0.87（±5% 即校验通过）→ **不提交**（3 次机会全部留到最后） | 评测器自校验 + 保底 prediction.csv 存档 | 15min |
| **1.5 Matched Control 基线**：同 plate(板)/同 instrument(仪器)/同 strain/medium/temp/time 的 Water/DMSO/EDTA 对照样本 log2 均值作预测（零训练，代表"最强不建模基线"） | Δ = model − control 的基准，确认 model 学到东西 | 30min |
| **1.6 条件编码 MLP**：mask-aware MSE，one-hot 编码 5 条件 → 2~3 层 MLP → 4,428 维输出。快速跑通（<1M 参数，几分钟） | 第一个有意义模型，确认超过 Control 基线 | 1h |

**Day 1 决策门**：若 MLP 不超 Control → 问题在特征工程（one-hot 无法外推）→ 跳 Day 2 加统计先验特征。若超出 → 继续按路线。

### Day 2（8/7）：提分特征工程 — 提升 val_both

| 任务 | 产出 |
|---|---|
| **2.1 菌株统计先验特征**：每个菌株在训练集 Δ 均值（4,428 维向量）拼入条件编码 → 增强 id/val_chem_only |
| **2.2 化合物 Δ 先验**：训练集中每种化合物的 (treat−matched_control) 均值 → 拼入特征 |
| **2.3 条件交叉特征**：strain×medium / chem×temp / strain×chem（组合 one-hot/hash）→ 提升 val_strain_only |
| **2.4 时间周期性编码**：pert_time → sin/cos |
| **2.5 mask-aware 多目标 loss**：主 MSE + FC-Pearson loss + 残差 L2 正则 |

### Day 3–4（8/8–8/9）：CompositionalTwin 赛题版 + 外部表征

| 任务 | 产出 |
|---|---|
| **3.1 残差分解 = CompositionalTwin 迁移**：共享头(control 均值) + φ 加法基（strain+chem 独立效应） + ψ 对称交互头（依赖表征，不查表）。输出 Δ = ŷ_treat − control | 与评测 65% Δ 指标对齐的模型 |
| **3.2 化合物 SMILES 抓取**：46 种化合物名 → SMILES（PubChem/RCSB，复用 `fetch_misato_smiles.py` 管线）→ RDKit Morgan 2048 指纹 + 描述符（沙箱已有 RDKit） |
| **3.3 菌株基因组表征**：5 株训练+1 株测试 → Peter et al. 2018 的 1011 株 S. cerevisiae 基因组（DHY210→S288C 代理，CRD/BAI/BAH/CEK/CGD 需定位到 1011 project 文件）→ k-mer 或基因组 embedding |
| **3.4 蛋白侧**：保留蛋白列表 4,428 个 → SGD 蛋白序列 → ESM embedding（本机装 esm 或用预计算 embedding） |

### Day 5–6（8/10–8/11）：全量 OOD 评估 + 消融

| 任务 | 产出 |
|---|---|
| **4.1 三场景 × 四子集分桶报告**：train/id/val_chem_only/val_strain_only/val_both/val_time，6 个评测模块分别报（FC 25% / 上下文残差 20% / 药物残差 20% / 绝对保真 20% / 双重未知 10% / DEP 5%） |
| **4.2 Bootstrap 95% CI + 校准**：复用 `benchmark_ood.py` 报告管线，产双栏（记忆 vs 机制）+ ECE/覆盖率 |
| **4.3 消融实验**：φ 置零 / ψ 置零 / 表征 shuffle / 表征换 hash → 量化每个组件贡献（教程 §6.1 评委核心考察点） |

### Day 7（8/12）：方案文档 + 外部数据虚拟实验

| 任务 | 产出 |
|---|---|
| **5.1 外部数据虚拟实验**：挑 3~5 个 test-only 化合物（如 MMS/H2O2/Tamoxifen），预测其蛋白组响应 → 与文献已知的 DNA damage/oxidative stress/雌激素通路机制对比（隐藏机理评估，评审维度②科学意义 30% + 评特别奖最高杠杆） |
| **5.2 方案文档撰写**：四章（概述/科学问题/技术方案/可行性验证），重点：实证故事（消融量化 + 机制解读）+ 差异化（不用模板堆砌） |
| **5.3 提交**：prediction.csv + 方案文档 |

---

## 2. 模块化实施依赖图

```
Day1 适配器 ─┬→ 本地评测器(6模块×4场景) ★唯一裁判，先于建模
             ├→ 蛋白均值基线 → 评测器自校验(R²≈0.87) → 存档不提交
             ├→ Matched Control → Δ 基准
             └→ 条件 MLP → Day2 特征工程
                              ├→ 统计先验
                              └→ 时间编码
                                        ↓
                                  Day3 CompositionalTwin ← RDKit SMILES(已完成)
                                        ↓
                                  Day4 基因组表征 / 蛋白 ESM(本机)
                                        ↓
                                  Day5 OOD 评估 + 消融
                                        ↓
                                  Day6-7 文档 + 外部验证 + 提交
```

---

## 3. 关键风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| CompoundTwin 加法先验同样退化为均值/线性（类似 norman/MISATO 结果） | 中高 | 这本身就是科学结论——在文档里诚实报告"加法可分解性边界"，转叙事为"OOD 框架 + 机制诊断"，不是精度军备竞赛 |
| 菌株 6 株太少，统计先验不可靠 | 中 | 靠基因组表征（Peter 2018）弥补；对照组打底 |
| 化合物名称→SMILES 匹配失败（音译/异构体） | 中 | 复用 `fetch_misato_smiles.py` 的双源（RCSB+PubChem）容错逻辑；人工复核 top-10 |
| 初赛截止只剩 10 天，资源极紧 | 高 | Day1 必需产出**可提交的 prediction.csv 存档**（蛋白均值基线）→ 保底；Day4 菌株基因组/蛋白 ESM 若过慢可降级为教程说的 k-mer/统计量（不阻塞主线） |
| **无官方分数反馈 → 本地评测器与官方口径存在偏差**（见 §0） | 中高 | ①严格按教程 §1.4 的 6 模块定义实现；②**matched-control 绝对 R²≈0.98 是唯一硬锚点**（已验证 0.979 通过）；蛋白-mean 0.87 是教程"不回加对照"旧版，本评测器回加后为 0.96（方法相关，仅作参考）；③所有对比只看**相对排序 + FC/残差模块**，不迷信绝对分 |
| 3 次提交机会误用于探路 | 中 | §0 已定：中途不提交，全部本地打分；最后一次提交前做 checklist（列顺序/sample_ID 对齐/维度 4,428/无 NaN） |
| val_both（269 样本）+ test_both（1,129）双重未知过于极端、所有模型都拉不开 | 中 | 若所有模型包括 MLP 都无法超越 Control → 诚实报告"当前方法边界"，转向「主动学习/实验推荐」叙事而非预测精度 |

---

## 4. 进度快照（2026-08-06，目标=晋级复赛）

**已完成（M1+M2）：**
- ✅ 实测真实 schema：`split_final` = **官方 OOD 四场景**（chem_only/strain_only/both/time），直接对标本项目 SOP 三轴+时间外推，无需伪造切分。
- ✅ 写 `code/data/goai_loader.py`：方法-B 按 `sample_ID` join 独立蛋白矩阵（5243 维）；`perturbation_no_concentration∈{Water,DMSO}` 为对照；匹配键 (Strains,Medium,Temp,pert_time,data_source) 算 Δ；**raw intensity 已转 log2**；官方 `split_final`→子集语义。
- ✅ 写 `code/goai_eval.py`（6 模块评测器骨架 + 基线 + 锚点校验）。**评测器已校准：matched-control 绝对 R²=0.979 吻合教程 0.98 锚点**。产物 `experiments/goai_baseline_anchors.json`。

**关键战术结论（决定打法）：**
- 绝对保真饱和（基线 0.96–0.98）→ **65% 权重在 FC(Δ PCC)+两残差**才是区分度所在。
- 基线 FC(Δ PCC) 仅 **0.14**（ID）且 OOD 更低 → **CompositionalTwin φ+ψ 的战场是 FC/残差**，必须显著超过 0.14。这正好对应教程"残差分解架构必须含交互项"。

**下一步（M3）：**
1. 补全评测器残差类（上下文均值残差/药物均值残差，各 20%）+ DEP 检出（5%）模块 → 全 6 模块×4 场景矩阵；
2. 写 `code/goai_benchmark.py`：baselines + CompositionalTwin 赛题版（φ 加法基 + ψ 对称交互头）+ 两栏(记忆vs机制) + bootstrap CI + 校准 → 直接喂 Day7 方案文档；
3. 菌株泛化：strain one-hot 仅 ID 可用，OOD 菌株需基因组/k-mer 表征（沙箱不可达→本机预处理）。

**✅ M3 已完成（2026-08-06）—— CompositionalTwin 登顶！**
- 5 模型跑通（2 baseline + Ridge + MLP(256,128) + CompositionalTwin），产物 `experiments/goai_benchmark.json`。
- **加权总分 Top: Twin 0.713 > Ridge 0.621 > MLP 0.612 > ProteinMean 0.439 > MatchedCtrl 0.196。**
- **Twin ID FC 0.549 / ood_action 0.283 / ood_agent 0.245 / ood_s3 0.120** —— 全部 OOD 轴均为最佳或并列最佳。
- M3 化合物效应 PCC 0.904 / M4 菌株效应 PCC 0.855 → 残差分解实证捕获生物信号。
- **MLP 黑箱外推陷阱**：ID FC 0.544 → ood_action 仅 0.189 → 标准过拟合叙事，决赛文档核心论据。
- ood_s3(双未知)所有模型仍弱(~0.12)— 需菌株基因组/SMILES 表征的交互头才能突破（本机预处理阻塞）。
- 关键代码：`code/goai_metrics.py`（6模块评测器）、`code/goai_benchmark.py`（基准跑分）、`code/goai_compositional_twin.py`（残差分解模型）。

## 5. 下一步行动（M4 → Day7 文档）

1. ✅ **M2 完成**：loader + eval + 锚点 0.979。
2. ✅ **M3 完成**：6 模块评测器 + CompositionalTwin 登顶 0.713。
3. **M4·文档冲刺**：从 benchmark.json 生成 LaTeX 表格 + 两栏(记忆vs机制) + bootstrap CI + 校准曲线 → Day7 方案文档。
4. **M4·外部表征**（用户本机）：菌株 k-mer / 化合物 SMILES → 替换 one-hot → 再跑 Twin 期望 ood_s3 突破 0.12。
5. **M4·提交**：生成 prediction.csv（Twin 预测的绝对丰度）→ 最终提交。

进入 M4？
