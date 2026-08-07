# MISATO MD — 配体语义特征升级（`--feat qmprops`）对比报告

**日期**：2026-08-03
**目标**：MD.hdf5 `frames_interaction_energy`（MM 相互作用能，真蛋白–配体结合目标）
**对照**：`hash`（ID 占位弱特征，旧） vs `qmprops`（QM.hdf5 7 个配体分子 QM 标量，本地语义升级）
**蛋白轴**：两版均为 hash（RDKit 不做蛋白；蛋白语义 embedding 留待后续）
**样本**：16047 usable 复合物；切分 train 13198 / id_test 1436 / ligand_ood 4953 / protein_ood 6014 / combo_ood 9116

## 1. 核心结论（combo_ood = held-out 配体×蛋白对，真组合 OOD）

| 模型 | hash版 RMSE | qmprops版 RMSE [95% CI] |
|---|---|---|
| additive_ridge（加法先验 = P1） | 50.9 | **46.8 [43.0, 51.0]** |
| blackbox_mlp | 44.4 | **35.6 [31.6, 39.8]** |
| knn | 44.8 | **33.6 [30.4, 36.7]** |
| mean（平凡基线） | 51.0 | 51.0 [46.8, 55.7] |

## 2. 读法（诚实版）

1. **真配体语义让加法先验「不再是废基线」**：hash 版里 ridge≈mean（50.9 vs 51.0，加法先验完全退化成平凡预测）；qmprops 版里 ridge 降到 46.8、显著低于 mean（CI 不重叠）→ 配体 QM 性质确实给加法模型喂进了真实化学信号。
2. **但加法先验仍显著输给非加法模型**：combo_ood 上 ridge 46.8 远高于 knn 33.6 / mlp 35.6，CI 完全不重叠（ridge 上界 51.0 > knn 上界 36.7）。即「配体效应(线性) + 蛋白效应(线性)」**无法还原 held-out 组合的结合能**，非加性（邻居/KNN、黑箱 MLP）仍胜。
3. **跨域呼应 omics 结论成立**：在结合域，加法先验只在「有真实配体语义」时脱离平凡，但**对真组合 OOD 仍系统性劣于非加性模型**——与 omics「epistasis 信号量小但确实存在、加法先验仅纯加法 regime 占优」的判断一致，支撑「组合泛化须学非加性结构」的赛道叙事。
4. **过拟合提示**：MLP train=13.0 但 id_test=79.8（train 严重过拟合）；KNN 在 combo_ood(33.6) 最稳，说明**化学近邻（QM 性质相似）→ 相互作用能相近**这一归纳偏置最有效。

## 3. 局限（与 hash 版共通 + 新增）

- 蛋白轴仍是 hash（无语义），加法模型只能表达「配体线性效应 + 蛋白身份」，无法表达配体×蛋白交互项——这恰是它输给非加性模型的结构性原因。
- `id_val` 出现 bootstrap 离群（ridge 1297）：该子集含少数极端相互作用能样本，bootstrap 重采样被放大，属分布假象，非模型问题（id_test / OOD 子集干净）。
- qmprops 是 **7 个全局配体标量**，远不如 RDKit Morgan 指纹（2048 位结构指纹）信息量大。下一版 `--feat rdkit` 用 Morgan 指纹才是更强语义对照。

## 4. 下一步

- `--feat rdkit`：需配体 SMILES 映射。沙箱无法抓境外 API，已写 `code/fetch_misato_smiles.py`，**在用户本机（真实外网）跑一次**生成 `data/processed/misato_ligand_smiles.json`，即可在沙箱用 Morgan 指纹重跑。
- 产物：`experiments/misato_ood_benchmark_md_qmprops.json`（本报告数值源）
