# GOAI 虚拟细胞赛道 · 初赛提交资料清单（INDEX）

本目录包含初赛全部提交资料。按官方 §1.6 要求，初赛提交物为 **方案说明文档 + 技术路线概述**，代码与预测文件作为初步实验 / 可行性验证附件。

---

## 一、核心提交物（评委必读）

| 文件 | 内容 | 对应评分维度 |
|---|---|---|
| `goai_提交_方案说明文档.md` | 完整方案：问题理解 / 方法 / 实验 / 科学意义 / 创新 / 取舍归因 | 技术45 + 科学30 + 创新20 |
| `goai_提交_阶段性实验结果报告.md` | 阶段性实验结果 / 可行性验证（独立成篇：锚点校准 + 5模型主结果 + 记忆vs机制两栏 + 关键发现 + 复现） | 技术45 |
| `goai_提交_技术路线概述.md` | 精炼版技术路线（架构 / 实证 / 里程碑） | 概览 |

## 二、可行性验证附件

| 文件 | 内容 |
|---|---|
| `data/submissions/prediction.csv` | test 处理组 CompositionalTwin 预测（4,252 行 × 5,243 列），log₂ 绝对丰度 |
| `experiments/goai_benchmark.json` | 5 模型 × 6 模块 × 5 场景完整评分卡（本报告 §2/§3 数据源） |
| `experiments/goai_bootstrap_ci.json` | bootstrap 95% 置信区间（不确定度，N=200/seed=42，**已生成**，已回填方案文档 §5 与阶段性报告 §3.1/§5） |
| `experiments/goai_workplan.md` | 工作计划与里程碑快照 |

## 三、代码（开源可复现）

```
code/
  data/goai_loader.py          # 真实 loader：sample_ID join 蛋白矩阵 + log2 + 官方 split_final
  goai_metrics.py              # 6 模块评测器（NaN-safe，权重 25/20/20/20/10/5）
  goai_benchmark.py            # 基线 + Ridge + MLP + CompositionalTwin 打分
  goai_compositional_twin.py   # CompositionalTwin：μ₀+μ_s+μ_c+ψ
  goai_submit.py               # test → prediction.csv 生成
  goai_bootstrap.py            # bootstrap CI
```

## 四、复现命令

```bash
PY="C:/Users/cc/.workbuddy/binaries/python/versions/3.13.12/python.exe"
export PYTHONPATH="C:/Users/cc/WorkBuddy/2026-07-31-18-03-27/.pylibs"
# 1. 评测器 + 基线（校准锚点）
$PY code/goai_eval.py
# 2. 全量 benchmark
$PY code/goai_benchmark.py
# 3. test 预测
$PY code/goai_submit.py
# 4. 不确定度 CI
$PY code/goai_bootstrap.py
```

## 五、一句话结论

CompositionalTwin 在官方 OOD 切分上加权分 **0.427**，与 Ridge(0.424)/MLP(0.422) **几乎持平**（闭合榜无压倒性 SOTA，诚实结论）；但其 **FC（25% 核心指标）= 0.446 [0.440, 0.451] 全场最高**，且 bootstrap 95% CI 与 Ridge [0.412, 0.421] **不重叠 → 领先统计显著**；MLP 在 OOD 上系统性崩塌（id 0.544→ood_action 0.189）仍正面支撑机制结构优于记忆拟合的核心论点；残差模块 M3/M4 Twin 未占优已诚实归因并给出复赛路线。
