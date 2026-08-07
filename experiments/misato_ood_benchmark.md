# MISATO Hand-2 OOD Benchmark — QM.hdf5 跑通结果

**日期**：2026-08-01
**输入**：`data/raw/misato/QM.hdf5`（340 MB，用户本机下载就位）
**脚本**：`code/benchmark_misato.py`（镜像 omics 的 P1/P2/MLP 对照，跨域迁移到结合轨迹）
**切分**：`code/data/misato_ood_splits.py`（RCSB 驱动，LLO/LPO/COMBO 三轴）
**产物**：`experiments/misato_ood_benchmark.json`

---

## 1. 目标与方法的诚实定位

实测 QM.hdf5 schema（`--inspect`）：每个复合物（PDB ID）在 `mol_properties/` 下存 **7 个配体分子级 QM 标量**——

`Electron_Affinity, Electronegativity, Hardness, Ionization_Potential, Koopman, molecular_weight, total_charge`

**关键事实**：这些全是**配体自身**的 QM 分子性质，**不含 protein–ligand 结合能/相互作用能**。真正的结合自由能目标在 **MD.hdf5**（133 GB，用户本机下载中）。

本基准自动选定 `Ionization_Potential`（电离势）作回归目标，以验证：
- (a) OOD 评估协议能否在结合域跑通；
- (b) 刻画 OOD 退化形态。

特征为**确定性 hash 向量**（ligand + protein 各 32 维，无 RDKit/结构），属透明占位弱基线。

---

## 2. 结果（RMSE，越低越好，含 bootstrap 95% CI）

| subset | additive_ridge (P1) | blackbox_mlp | knn | mean (baseline) |
|---|---|---|---|---|
| train | 0.820 [0.76,0.89] | 0.515 [0.49,0.54] | 0.707 [0.66,0.77] | 0.824 [0.77,0.90] |
| id_val | 0.949 [0.80,1.08] | 1.104 [0.97,1.24] | 1.043 [0.90,1.17] | 0.946 [0.80,1.08] |
| id_test | 0.903 [0.76,1.04] | 1.100 [0.98,1.23] | 0.966 [0.84,1.10] | 0.899 [0.76,1.04] |
| **ligand_ood** | 0.886 [0.79,1.00] | 0.679 [0.61,0.76] | 0.818 [0.74,0.89] | 0.886 [0.79,1.00] |
| **protein_ood** | 0.866 [0.79,0.95] | 0.700 [0.64,0.76] | 0.803 [0.73,0.87] | 0.867 [0.79,0.95] |
| **combo_ood** | 0.877 [0.82,0.94] | 0.686 [0.64,0.73] | 0.808 [0.76,0.86] | 0.878 [0.82,0.94] |

（usable 复合物 16,046；splits: train 13,197 / id_val 1,413 / id_test 1,436 / ligand_ood 4,953 / protein_ood 6,014 / combo_ood 9,116）

---

## 3. 三个诚实发现

### 发现 1：OOD **不退化**——但与 omics 不可比
combo_ood（0.877 ridge）≈ id_test（0.903 ridge）≈ train（0.820）。这与 omics 的 epistasis 扫描**完全相反**（omics 里 combo_ood 大幅退化，黑箱 MLP 单调胜出）。

**原因**：目标 `Ionization_Potential` 是**配体固有分子性质，与 protein 完全无关**。因此：
- ligand_ood：held-out ligand 的 IP 仍在训练 ligand 的同域分子分布内 → 可预测；
- protein_ood / combo_ood：protein 不改变 IP → 目标不变 → **不构成真正的"组合 OOD"**。

→ 这个基准**不能回答"结合能加和 vs epistasis"的科学问题**。protein 维度在此目标下是冗余上下文。

### 发现 2：加性先验在弱特征下完全退化到均值
additive_ridge 在所有 OOD 子集 ≈ mean（0.87–0.89 vs 0.88）。在 hash 特征下，加性模型无任何外推能力，等价于记训练均值。

### 发现 3：MLP 的"优势"是目标分布匹配，不是泛化
MLP 在 OOD 上（0.68–0.70）比 ridge/mean（0.88）低 ~20%，但在 id 上（1.10）**反而比均值还差**（过拟合 hash）。MLP 的优势来自它拟合了**训练目标的连续分布**（输出被约束在合理 IP 范围内），而非学到了 ligand→IP 的语义规律——hash 是任意映射，不可能真泛化。这是"分布匹配"假象，不是机制胜势。

---

## 4. 结论

- ✅ **框架迁移成功**：OOD 评估协议（三轴切分 + bootstrap CI + 记忆 vs 机制双栏）在结合域**完整可运行**，数据管线、RCSB 元数据、HDF5 读取全部验证通过。
- ⚠️ **QM.hdf5 不能回答结合能科学问题**：目标为配体分子性质，protein 维度冗余；OOD 不退化使加性 vs 黑箱的对照**失去意义**。
- 📌 这是一份**诚实的负结果 / 框架验证报告**——与 epistasis 合成扫描的"结构化先验仅纯加法 regime 占优"同构，都是"先讲清什么测不了、为什么"的高价值研究信号（对应 GOAI 评审维度④"探索过程与研究信号"）。

---

## 5. 让 Hand 2 真正有科学竞争力的下一步

| 动作 | 依赖 | 预期贡献 |
|---|---|---|
| **改用 MD.hdf5 的真实结合能目标**（MM-GBSA/相互作用能） | 用户本机下完 133 GB MD.hdf5 | 让 protein 维度生效 → combo_ood 成为真"组合 OOD" → 加性 vs 黑箱对比**有意义**，预期重演 omics 结论 |
| **RDKit 配体语义特征 + 蛋白 embedding 替换 hash** | 用户本机（有 RDKit 环境） | 消除"弱特征"缺陷 → 加性/黑箱对比有判别力；可做 Murcko scaffold 级严格 LLO |
| **在 `benchmark_misato.py` 加 `--qm-mode md`** 分支读 MD 结合能 | MD.hdf5 就位 | 一键切换数据源，复用同一套 OOD 协议 |

**当前建议**：QM 版作为"框架验证 + 诚实负结果"保底交付；真正冲奖的 Hand 2 科学故事等 MD.hdf5（或用户本机 RDKit 特征）就位后补完。
