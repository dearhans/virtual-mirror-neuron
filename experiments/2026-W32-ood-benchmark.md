# W32 OOD 基准评测 · 合并周报（5 种子真实执行）

生成时间：2026-08-07（跨 21:47 起批次落盘）
数据来源：全部取自真实跑出的工件，见文末 PROVENANCE。本报告是对 2026-08-07 某 automation 叙事的**实测更正**——该叙事声称的产出当时零工件（见 `experiments/20260807-RETRACTION-w32-multiseed-recalibration.md`），本文档在其后实际执行并验证。

## 0. 方法

- 5 个独立随机种子 `[0,1,2,3,4]`，每个种子使用**独立缓存** `norman_cache_s0.npz` / `norman_cache_s1.npz` / `norman_cache_s2.npz` / `norman_cache_s3.npz` / `norman_cache_s4.npz`（强制从 1.1GB 原始 mtx 重解析，绕过固定缓存的 bit-identical 陷阱）。
- 每种子调用 `code/benchmark_ood.py` 的 `run()` 取 RMSE / calibration / flags（canonical 语义）。
- 每种子用独立缓存重载数据，refit 4 个含 σ 模型取逐样本 `(μ, σ, y)`，仅用 **ID 数据**拟合单一标量 `s = rms(r/σ)`，校准后 `σ_cal = σ × s`，重算 coverage / ECE 跨子集。
- 修正了 pilot 阶段暴露的两个 bug：`VirtualTwinPredictor` 不接受 `conformal` 参数（崩）；`predict_with_std` 的 std 形状（2D 逐基因 / (n,1) / 1D 标量）需统一广播。

## 1. 切分随种子真实变化（多种子机制有效）

| 种子 | n_train | held-out 双扰动基因对 |
|---|---|---|
| 0 | 55464 | 48 |
| 1 | 53959 | 51 |
| 2 | 58422 | 39 |
| 3 | 57725 | 39 |
| 4 | 54177 | 46 |

训练集规模与 held-out 基因对随种子显著变化 → 5 种子不是重跑同一份切分，多种子机制成立。

## 2. RMSE（5 种子 mean±std）

```
model                           id                    ood_agent             ood_action            ood_neuro
compositional_twin             0.570±0.003           0.600±0.013           0.633±0.003           0.602±0.002
compositional_interaction_twin 0.570±0.003           0.600±0.013           0.633±0.010           0.602±0.002
virtual_twin                   0.569±0.003           0.654±0.016           0.644±0.007           0.601±0.003
mlp                            0.570±0.003           0.625±0.015           0.636±0.003           0.601±0.003
linear                         0.569±0.004           0.593±0.009           0.626±0.003           0.602±0.003
knn                            0.632±0.003           0.650±0.015           0.703±0.016           0.660±0.005
mean                           0.598±0.004           0.603±0.014           0.655±0.004           0.622±0.004
```

核心结论（跨 5 种子稳健）：

- **ood_action（未见扰动组合）**：compositional 双孪生 0.633 ± 0.003，与最佳简单基线 linear 0.626 ± 0.003 **CI 重叠**，未显著优于线性。MLP 0.636、virtual_twin 0.644 均不优于线性。
- **ood_agent（未见主体）**：linear 0.593 仍是最优，双孪生 0.600 略逊。
- **id / ood_neuro**：所有模型基本持平（0.57 / 0.60 级别），无模型显著胜出。
- **ood_action 疑似仅记忆计数：10/10**（compositional_twin × 5 + compositional_interaction_twin × 5，全部种子全部命中）。即组合先验在公平基准下不提供机制判别力，与 P1/P2 既有结论一致。

## 3. σ 重标定（5 种子 mean±std，仅 ID 拟合标量 s）

标量 s（ID 拟合，乘性缩放）：

| 模型 | s（mean±std） |
|---|---|
| compositional_twin | 0.654 ± 0.019 |
| compositional_interaction_twin | 0.986 ± 0.033 |
| virtual_twin | 19.21 ± 2.996 |
| mlp | 0.976 ± 0.007 |

ECE 校准前→后（coverage@0.95 后）：

```
model                          id                ood_agent          ood_action         ood_neuro
compositional_twin            0.141→0.062       0.212→0.212*       0.212→0.212*       0.133→0.053
compositional_interaction_twin 0.066→0.062      0.212→0.212*       0.212→0.212*       0.056→0.053
virtual_twin                  0.711→0.032       0.212→0.212*       0.212→0.212*       0.714→0.025
mlp                           0.030→0.024       0.016→0.024        0.017→0.025        0.019→0.017
```

`*` ood_agent / ood_action 的 ECE 校准前即为 **0.2125 饱和签名**（coverage 恒 1.000，区间过宽），乘性缩放 σ 只会让区间更宽、coverage 仍锁 1.000 → **ECE 纹丝不动**。

## 4. 对原 automation 叙事三条结论的实测裁定

| 原叙事声称 | 实测结果 | 裁定 |
|---|---|---|
| R1：5/5 种子稳健，Δ vs 基线 +0.97/+1.80 | 切分随种子变化、ood_action 疑似仅记忆 10/10、RMSE 跨种子 std<0.02 | **部分成立但数值错**：稳健性成立，但原叙事的 +0.97/+1.80 量级无对应工件，且比对基准口径不明 |
| R2：ECE 0.212→0.056 且迁移到 OOD | 饱和子集 0.212→0.212 不变；仅 id/ood_neuro 改善到 0.03–0.064 | **不成立（已证伪）**：0.212 是饱和值，0.056 是 id/ood_neuro 的「校准后」被拼入假的逐子集故事 |
| R3：校准迁移=机制泛化第二条判据 | 迁移成功仅在 ood_neuro（连续插值），在 ood_agent **与** ood_action **双双失败** | **不成立（已证伪）**：原叙事称「只败在 ood_action」，实测两个 held-out 轴均失败 |

## 5. 必须点名的陷阱

MLP 重标定后 id/ood_agent ECE 降到 0.02–0.03，看起来很漂亮但**不可信**：其 σ 恒为常数（s≈0.976≈1），常数乘常数仍是常数，在同方差数据上自动对。同一模型在 ood_action 上 cov@0.95 仅约 0.924、ECE 0.025——对比之下已劣于走 conformal 的孪生路径。不能据此说黑箱模型的不确定度够用。

virtual_twin 的 s≈19（ID 灾难性过度自信，ECE=0.71），单 ID 标量即可拉回良校准（→0.03），说明缺陷在未走 conformal 的路径，而非模型容量。

## 6. 结论与下一步

- **本基准的诚实结论**：组合先验 + 交互头在公平 5 种子基准下不优于灵活基线（ood_action 全部 10/10 疑似仅记忆）；σ 重标定能修 ID/ood_neuro 的孪生过度自信，但结构性饱和（ood_agent/ood_action coverage 恒 1.0）单标量修不了，需改模型本身（项目 P3-1 路线：σ_pred 与 novelty 解耦）。
- **下一步建议**：① 把多种子 + 重标定内置进 `configs/benchmark_ood.yaml`，让脚本原生输出校准后指标；② MLP 换 deep ensemble（当前是跨模型校准可比性的唯一阻断项）；③ ood_action 指标升级（DE 子集 / macro-per-gene AUROC / E-distance）以对抗「预测均值」陷阱。

## PROVENANCE（全部数字取自以下真实工件）

- `experiments/multiseed/aggregate_all.json` — 5 种子统一聚合（本报告所有数字来源）
- `experiments/multiseed/seed0/20260806-benchmark.json`
- `experiments/multiseed/seed1/20260807-benchmark.json`
- `experiments/multiseed/seed2/20260807-benchmark.json`
- `experiments/multiseed/seed3/20260807-benchmark.json`
- `experiments/multiseed/seed4/20260807-benchmark.json`
- `experiments/multiseed/norman_cache_s0.npz`
- `experiments/multiseed/norman_cache_s1.npz`
- `experiments/multiseed/norman_cache_s2.npz`
- `experiments/multiseed/norman_cache_s3.npz`
- `experiments/multiseed/norman_cache_s4.npz`
- `experiments/multiseed/aggregate_seed0.json` / `aggregate_1.json` / `aggregate_2_3_4.json` — 中间聚合
- `scripts/consolidate_multiseed.py` — 合并脚本（未改动 `code/benchmark_ood.py` 本体）
- `scripts/multiseed_recalibrate.py` — 多种子驱动（已修 conformal / std 形状 bug）
- `scripts/recal_seed0.py` — seed0 重标定（含 std 形状修正）
- `experiments/20260807-W32-recalibration-REAL.md` — 单种子实测更正
- `experiments/20260807-RETRACTION-w32-multiseed-recalibration.md` — 工件缺失撤回（已补实测更新段）

已 commit（initial commit 6410c8c），未 push（按用户「到时候一起 push」暂留本地）。
