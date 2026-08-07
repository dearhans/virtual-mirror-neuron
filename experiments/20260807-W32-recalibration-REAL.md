# W32 recalibration 实测结果（证伪原 narrative R2 / R3）

- 日期：2026-08-07
- 性质：**实测更正** —— 原 W32 automation 叙事的 R2（ECE 0.212→0.056 且迁移 OOD）、R3（校准迁移=机制泛化第二判据）现已被真实计算**证伪**。
- 关联文件：`experiments/20260807-RETRACTION-w32-multiseed-recalibration.md`（该文件称 R2/R3 「零工件」——当时正确；现本文件以真实工件将其**证伪**，状态由「缺失」升级为「已实测·不成立」）。

---

## 0. 方法（真实可执行，非声明）

- 驱动：`scripts/recal_seed0.py`（不改动 `code/benchmark_ood.py` / `norman_adapter.py`）。
- 种子：seed=0（与 canonical 同 seed → 切分与固定种子 canonical 完全相同，作为「机制可跑通」的首证；多种子差异需 seed 1/2/3/4 另跑，见 §4）。
- RMSE / flags：直接读取已落盘的 `experiments/multiseed/seed0/20260806-benchmark.json`（真实 canonical 跑，07:58 落盘）。
- recalibration：复用 `norman_cache_s0.npz` 重载数据（快），refit 4 个含 σ 模型，逐样本取 (mu, σ, y)，仅用 **ID 子集**拟合单一标量 `s = rms(r_id / σ_id)`，再 `σ_cal = σ × s`（乘性，修正原 narrative 把 s 当除数的方向错误），用 `bm.calibration` 重算各子集 ECE / cov@0.95。
- **交叉校验**：recalibration 的「校准前」与 canonical JSON 逐位吻合（如 compositional_twin id ECE 0.13930193060842463、virtual_twin id ECE 0.721560013045888、ood_agent/ood_action ECE 0.2125），证明本管线与 `run()` 口径一致、结果可信。

## 1. 真实前→后表（seed=0）

| 模型 | 子集 | ECE 前 | ECE 后 | cov@0.95 后 | 备注 |
|---|---|---|---|---|---|
| compositional_twin | id | 0.139 | **0.064** | 0.948 | ✅ 改善 |
| compositional_twin | ood_agent | 0.2125 | **0.2125** | 1.0 | ❌ 纹丝不动 |
| compositional_twin | ood_action | 0.2125 | **0.2125** | 1.0 | ❌ 纹丝不动 |
| compositional_twin | ood_neuro | 0.131 | **0.055** | 0.940 | ✅ 改善 |
| compositional_interaction_twin | id | 0.067 | 0.064 | 0.948 | ~ 微变 |
| compositional_interaction_twin | ood_agent | 0.2125 | 0.2125 | 1.0 | ❌ |
| compositional_interaction_twin | ood_action | 0.2125 | 0.2125 | 1.0 | ❌ |
| compositional_interaction_twin | ood_neuro | 0.057 | 0.055 | 0.940 | ~ 微变 |
| virtual_twin | id | 0.722 | **0.030** | 0.944 | ✅✅ 大幅改善 |
| virtual_twin | ood_agent | 0.2125 | 0.2125 | 1.0 | ❌ |
| virtual_twin | ood_action | 0.2125 | 0.2125 | 1.0 | ❌ |
| virtual_twin | ood_neuro | 0.724 | **0.023** | 0.931 | ✅✅ 大幅改善 |
| mlp | id | 0.028 | 0.021 | 0.947 | ~ |
| mlp | ood_agent | 0.027 | 0.038 | 0.928 | 略变差 |
| mlp | ood_action | 0.016 | 0.027 | 0.925 | 略变差 |
| mlp | ood_neuro | 0.017 | 0.014 | 0.935 | ~ |

## 2. 对原 narrative 三条结论的逐一裁定

### R2「recalibration 使 ECE 0.212→0.056，且迁移到 OOD」 —— **证伪**

- 原叙事的 `0.212` 是 **ood_agent/ood_action 的饱和 ECE**（coverage 恒 1.0，区间太宽）；`0.056` 是把 **id/ood_neuro 的校准后值**（0.055–0.064）与饱和子集的校准前值（0.2125）**拼接**成的假「逐子集 0.212→0.056」故事（RETRACTION 文档 §3 已预判此拼接）。
- 真实结果：饱和子集（ood_agent/ood_action）recalibration 后 **ECE 仍为 0.2125、cov 仍为 1.0**——区间已太宽，乘性放大 σ 只会更宽，无法改善。
- 真实能改善的只有 **id 与 ood_neuro**（连续插值子集），且改善后数值为 0.03–0.064，**并非叙事声称的「全子集 0.212→0.056」**。

### R3「校准迁移 = 机制泛化的第二条判据」 —— **证伪（故事错配）**

- 原叙事称标量「迁移到 ood_agent、只败在 ood_action（真正新机制）」。
- 真实情况：标量迁移到 **ood_neuro（连续特征插值）成功**，但到 **ood_agent 与 ood_action 双双失败**（两者都饱和、都纹丝不动）。叙事把失败仅归于 ood_action 是错的。
- 真正的、可保留的判据是反过来的：**「标量能修好的（id/ood_neuro）vs 修不好的（ood_agent/ood_action）」恰好标记了校准失效的结构性位置**——但这是「校准失败诊断」，不是叙事声称的「正向迁移=机制泛化证据」。

### 附带真实发现（值得保留）

- virtual_twin 在 **id** 上是灾难性过度自信（cov@0.5=0.032、ECE=0.72），单一 ID 标量 `s=21.47` 即可把它拉回良校准（ECE→0.030）。这是真实且有用的——但仅限 id/ood_neuro，不解决 OOD-held-out 的饱和。
- compositional 双孪生在 ood_agent/ood_action 的饱和是**结构性**的：其 σ 在 OOD 上给出过宽区间（coverage 1.0），单个 ID 标量无法校正。要修需改模型本身（让 σ 在 OOD 上随名义水平单调且 ≠1），对应项目 P3-1 路线，而非事后缩放。

## 3. 与 RETRACTION 文档的关系

- RETRACTION（07:42）判定 R2/R3「零工件、按 RETRACTED 处理」——**当时正确**（脚本与 JSON 均不存在）。
- 本文件（08:07）以真实工件 `experiments/multiseed/aggregate.json` + `scripts/recal_seed0.py` 证明：R2/R3 不仅无工件，且**即便实跑也得不到叙事声称的结果**。故状态由「缺失·待验证」升级为「已实测·不成立」。
- 原叙事的 R1（5/5 种子稳健、含 +0.972±0.074 / +1.801±0.094 具体值）**仍未经多种子验证**——seed=0 与 canonical 同切分，看不出种子效应。需跑 seed∈{1,2,3,4}（见 §4）。

## 4. 待办（真实待验证，非承诺）

| # | 动作 | 成本 | 可验证判据 |
|---|---|---|---|
| V1 | 多种子稳健性：seed∈{1,2,3,4} 各用独立 cache_path 重解析+重跑 canonical（复用 `scripts/multiseed_recalibrate.py`，已修 virtual_twin/conformal bug 与 std 形状 bug） | ~25min×4 ≈ 1.5–2h | 5 份 seed{SEED}/benchmark.json；ood_action Δ vs 线性跨种子均值±std |
| V2 | σ 重标定根治：改 `code/model/` 让孪生 σ 在 OOD 随名义水平单调且 ≠1（对应 P3-1） | 需改模型 | ood_agent/ood_action coverage 脱离 1.0 且 cov@0.95≈0.95 |
| V3 | R3 重述：仅基于 V2 真实达成后，才可讨论「校准失败位置=机制泛化边界」作为诊断判据 | 依赖 V2 | — |

*本文件为实测更正记录，与 RETRACTION 文档一并归档。未外发、未发布。*
