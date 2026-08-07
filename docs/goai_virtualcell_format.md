# GOAI 赛道三·方向一「虚拟细胞」数据格式与适配层说明

> 用途：把 GOAI 世界人工智能开源大赛「前沿探索丨AI for Research」赛道三、方向一
> （虚拟酵母扰动响应预测）的官方数据接入本项目 OOD 外推基准，并产出可提交预测。
> 源依据：赛道手册 PDF（p15-18）。**数据尚未由组委会下发，本文与适配层为草稿骨架。**

---

## 1. 赛题数据格式（依据手册）

组委会提供两个输入文件：

| 文件 | 内容 | 用途 |
|---|---|---|
| `train_val` | 样本 metadata + 生物条件 + 测量上下文 + 蛋白质组标签（**有真值**） | 训练 + 内部验证（可自选 train/val 划分） |
| `test` | 样本 ID + 生物条件 + 测量上下文（蛋白真值**保留、离线评分**） | 最终提交预测 |

每个样本 = 一个**固定蛋白顺序的 log2 蛋白丰度向量**（蛋白质组响应）。

输入字段（手册「建议公开字段」）：

- **生物条件**：菌株 `strain`、化合物/处理 `compound`、培养基 `medium`、温度 `temp`、时间点 `time`、重复 `replicate`
- **测量上下文**：数据来源 `source`、仪器 `instrument`、板号 `plate`、批次 `batch`
- **响应目标**：蛋白丰度（log2 向量）+ 对照（匹配到 DMSO 或 Water）

评测集三类泛化场景（**核心**）：

| 场景 | 划分方式 | 考察能力 |
|---|---|---|
| **S1** | 已见菌株 + **未见化合物** | 对新扰动的外推 |
| **S2** | **未见菌株** + 已见化合物 | 对新遗传背景的外推 |
| **S3** | **未见菌株 + 未见化合物** | 对双重未知组合的泛化 |

评分模块（权重）：绝对保真度 20% / 匹配对照原始 FC 25% / 上下文均值残差 20% /
药物均值残差 20% / 双重未知·时间外推 10% / 高效应蛋白 DEP 检出 5%。
核心指标 Δ = `treat − control`，计算 `PCC(Δ_pred, Δ_true)`（占 25%，所有 OOD 划分的核心）。

---

## 2. 字段映射：比赛 → 本项目统一接口

本项目 `benchmark_ood.py` 接口：`X=[P(Dp), A(2), G(1)]`，`y` 响应（可向量），
`meta=[action_idx, agent_idx, neuro_idx]`，`split` 子集标签。

| 比赛概念 | 本项目字段 | 说明 |
|---|---|---|
| 化合物 `compound` | **P（扰动向量）one-hot** | 被预测外推的「扰动身份」 |
| 培养基/温度/时间/source/仪器/板号/批次 | **P（上下文段）one-hot** | 吸收测量变异（赛题设计：上下文用于去批次，泛化由菌株/化合物可见性定义） |
| 动作 A[2] | 退化 `[1,0]` | 本赛题无 self/other 语义 |
| 神经调质 g | 退化 `1.0` | 该数据集无调质协变量 |
| **菌株 `strain`** | **meta[agent_idx]（仅 meta，不进 X）** | 混杂因子 / OOD 锚点（符合 CHARTER「agent 不进特征」） |
| 响应 y | **Δ = log2(treat) − log2(matched_control)** 蛋白质组向量 | 默认 `target: delta`，匹配 25% 权重 FC；可选 `absolute` 对应 20% 绝对保真度 |

对照匹配规则（赛题要求）：treatment 减去**同 (strain, medium, temp, time, 测量上下文)**
的 control（DMSO/Water）。无匹配对照的样本丢弃（无法算 Δ）。

---

## 3. OOD 三轴映射（直接对齐本项目硬约束）

| 赛题场景 | 项目子集 | 机制语义 | 应外推？ |
|---|---|---|---|
| S1 已见菌株+未见化合物 | `ood_action` | 对新扰动外推（FC 核心指标） | ✅ 应外推 |
| S2 未见菌株+已见化合物 | `ood_agent` | 对新遗传背景外推 | ✅ 应外推 |
| S3 未见菌株+未见化合物 | `ood_s3`（自定义子集） | 双重未知组合——机制泛化终极检验 | ✅ 应外推 |
| 已见菌株+已见化合物 | `train` / `id` | in-distribution（held-out 细胞） | — |

实现：在 `train_val` 内按 `heldout_compound_frac` / `heldout_strain_frac` 抽 held-out
化合物集与菌株集，依「双未知 > 仅化合物未知 > 仅菌株未知 > 皆已见」优先级互斥切分。
这与赛题最终测试集 S1/S2/S3 的定义同构，可内部复现评测逻辑。

---

## 4. 本项目硬约束如何被满足

1. **记忆 vs 机制泛化**：`train`/`id`（记忆）与 `ood_*`（机制泛化）分开评测；
   `flags` 在「应外推」子集上若未显著优于简单基线（均值/线性/KNN），触发
   「疑似仅记忆」告警 → 启动 SOP 阶段 D 误差回流 + TRIZ 修正。
2. **OOD 子集必含**：S1/S2/S3 三轴全齐（外加 id/train），缺失子集会在日志/报告明标。
3. **不确定度校准**：RMSE 带 bootstrap 95% CI；覆盖度/ECE 衡量区间可靠性；
   虚拟孪生不确定度 = 集成 epistemic + 新颖度(OOD) 合成，缓解黑箱同方差过度自信。

---

## 5. 提交流程

1. 数据到手后放入 `data/raw/goai_virtualcell/`（train_val + test）。
2. 核对官方字段字典，更新 `code/data/goai_virtualcell_adapter.py` 的 `COLUMN_MAP`
   与 `CONTROL_TOKENS`（核心逻辑无需改动）。
3. 跑真实 OOD 基准：
   `python code/benchmark_ood.py --config configs/benchmark_ood_goai.yaml`
4. 生成提交：另写 `predict_goai_submission.py`（加载 train_val → 拟合 twin →
   预测 test → 用 `write_submission` 写出 `prediction.csv`，sample_id + 蛋白宽表）。
   注意：若以 `target='delta'` 训练，提交前须把 Δ 加回 matched_control 得到绝对 log2 丰度。

---

## 6. 数据到手 Checklist

- [ ] 官方字段字典到手 → 填 `COLUMN_MAP` / `CONTROL_TOKENS`
- [ ] 确认蛋白向量是宽表列（`protein_*` 前缀）还是分离矩阵（扩展点 TODO #1）
- [ ] 确认对照标识（DMSO/Water 是否区分两类）→ 必要时扩展 `CONTROL_CLASS`
- [ ] 确认 `train_val`/`test` 格式（csv/tsv/parquet）→ parquet 需 pandas+pyarrow
- [ ] 跑冒烟测试 `_smoke_goai.py` 确认链路
- [ ] 决定 `target: delta`（默认）还是 `absolute`
- [x] （开放知识榜）接化合物可泛化特征 —— ✅ 已实现：`code/data/goai_compound_features.py` + `GoaiConfig.use_compound_descriptors` 开关（默认 False；冲开放知识榜时设 True）。把化合物 one-hot 段拼接 PubChem 2D 描述符，使 unseen compound(S1/S3) 具备可外推结构信号（TRIZ 参数变化 #35，详见 `docs/triz_contradiction_analysis.md` §4.3）。MoA 类别为后续 TODO（当前占位 0）。
      —— 这是让 unseen compound(S1/S3) 能真正外推、破解「记忆 vs 机制泛化」矛盾的破局点

---

## 7. 文件索引

- `code/data/goai_virtualcell_adapter.py` — 适配层（草稿，核心逻辑完整，字段占位）
- `configs/benchmark_ood_goai.yaml` — 基准配置（含 S3 子集与标签覆盖）
- `code/benchmark_ood.py` — 已接线：新增 `goai` 数据源分发 + 报告动态子集定义/响应描述
- `_smoke_goai.py` — 合成数据冒烟测试（验证解析→Δ→切分→字典全链路）
