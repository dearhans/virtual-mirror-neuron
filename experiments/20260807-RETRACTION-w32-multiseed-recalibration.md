# 撤回声明 · W32 multiseed / recalibration 结论作废

**日期**：2026-08-07
**性质**：无工件声明（claims without artifacts）——报告文本被生成，对应计算从未执行
**状态**：`RETRACTED`，禁止进入任何决策、周报或 GOAI 提交叙事

---

## 1. 被撤回的三条结论

以下结论出现在某次 automation 叙事中，**现全部作废**：

| # | 被撤回的结论 | 声称的支撑工件 | 磁盘实况 |
|---|---|---|---|
| R1 | 多种子扫描显示 **5/5 种子稳健** | `scripts/multiseed_ood_sweep.py` + multiseed 聚合 JSON | **不存在**（`scripts/` 目录本身不存在） |
| R2 | recalibration 使 **ECE 0.212 → 0.056**，且该改善**迁移到 OOD** | `scripts/recalibrate_sigma.py` + recalibration JSON | **不存在** |
| R3 | **「校准迁移」可作为机制泛化的第二条判据** | 上述两项的联合证据 | **无任何支撑** |

另：声称的汇总报告 `experiments/2026-W32-ood-benchmark.md` 亦不存在。

## 1.5 实测更新（2026-08-07 08:07）— R2/R3 已由「缺失」升级为「已实测·不成立」

经真实重跑（`scripts/recal_seed0.py` + `experiments/multiseed/aggregate.json`，seed=0），R2/R3 **不仅无工件，且实跑亦得不到叙事声称的结果**：

- R2「ECE 0.212→0.056 且迁移 OOD」：**证伪**。原 `0.212` 是 ood_agent/ood_action 的饱和 ECE（coverage 恒 1.0），recalibration 后纹丝不动（仍为 0.2125）；`0.056` 是把 id/ood_neuro 的校准后值（0.055–0.064）与饱和子集校准前值拼接成的假故事。真实能改善的仅 id/ood_neuro（→0.03–0.064）。
- R3「校准迁移=机制泛化第二判据」：**证伪（故事错配）**。真实是标量迁移到 ood_neuro（连续插值）成功、到 ood_agent 与 ood_action **双双失败**，非叙事所称「只败在 ood_action」。

详细真实前→后表与裁定见 `experiments/20260807-W32-recalibration-REAL.md`。R1（5/5 种子稳健）**仍未经多种子验证**（seed=0 与 canonical 同切分）。

## 2. 取证方法与结论强度

三重独立检查，全部零命中：

```
find . -iname "*multiseed*" -o -iname "*recalibrat*" -o -iname "*2026-W32-ood*"   → 空
grep -rn -il "multiseed|recalibrat|2026-W32-ood" .                                → 空
ls scripts/                                                                        → 目录不存在
```

**关键判读**：连**字符串提及**都在全库为零，说明这不是「产物写到别处后被清理」，
而是**该段计算从未发生**。两份运行日志 `experiments/w32_benchmark_run.log`（1047 B）与
`experiments/w32_benchmark_run_0806.log`（1047 B）内容除输出文件名外逐字节相同，
均只记录了单次 canonical 跑（seed=0），无任何多种子或重标定的执行痕迹。

## 3. 与之对照：本周**确实成立**的部分

不因一处造假而否定全部，边界必须划清：

| 项 | 状态 | 证据 |
|---|---|---|
| canonical 基准跑（7 预测器 × 4 子集，seed=0，bootstrap=100） | ✅ 真实 | `w32_benchmark_run_0806.log` exit 0；`20260806-benchmark.json` sha256[:16]=`087b7cfff42c61fc` |
| ood_action 判定「疑似仅记忆」 | ✅ 真实 | log 中判定行 + JSON `flags` 段 |
| 三跑 bit-identical（20260802/20260805/20260806） | ✅ 真实 | `results` 段 sha256[:16]=`caf29680042baf6f` 三跑一致 |
| 附录 A 校准诊断（ECE=0.2125 饱和签名、virtual_twin 过度自信） | ✅ 真实且可独立复算 | 28/28 条目从原始 `coverage` 重算与存储值差 < 1e-6；6 条退化条目与理论饱和值 `mean(\|1−level\|)=0.2125` 差 **0.00000** |

即：**W32 报告的第 1–8 节（含附录 A/B）成立；multiseed 与 recalibration 相关的一切不成立。**

> 注意 R2 的诱骗性：`0.212 → 0.056` 这个数对在真实数据里**有原型**——
> 0.2125 是饱和 ECE，0.057 是 P2 交互项孪生在 ood_neuro 的真实 ECE。
> 即幻觉是把两个**真实但无因果关系**的数字拼成了一个「重标定成功」的故事。
> 这正是本项目黑历史（eval 口径 bug、ECE 饱和伪影）的同型失败：**看起来最漂亮的数字最该被质证。**

## 4. 结构性修复（已落地，非承诺）

新增 `code/verify_artifacts.py`：声明↔工件强绑定校验器。

- 扫描报告 `.md` 与指标 `.json` 中出现的所有本地路径声明，逐一校验存在性并计算 sha256；
- 任一 MISSING 即 `exit 1`，并打印「不得进入决策或提交叙事」告警；
- 支持 `--require` 显式断言、`--manifest` 写出 PROVENANCE 清单。

**双向验收（已执行）**：

```
# 真报告 → PASS
python code/verify_artifacts.py --report experiments/20260806-benchmark.md \
    --json experiments/20260806-benchmark.json --manifest experiments/20260806-PROVENANCE.json
→ 合计 2 条声明 | PASS 2 | MISSING 0 | EXIT=0

# 本次三条幻觉声明 → FAIL（精确抓出）
python code/verify_artifacts.py --require scripts/multiseed_ood_sweep.py \
    --require scripts/recalibrate_sigma.py --require experiments/2026-W32-ood-benchmark.md \
    --require experiments/figures/fig1_rmse_by_subset.png
→ 合计 4 条声明 | PASS 1 | MISSING 3 | EXIT=1
```

**首次实战即暴露自身 bug（如实记录）**：把校验器指向本撤回文件时，
裸文件名（正文写 `20260806-benchmark.json` 而文件实际在 `experiments/` 下）
被按项目根解析 → 产生 **3 条假阳性 MISSING**。若不修，闸门会误杀真实报告，
反而制造「校验器说有问题所以别信报告」的噪声。
已修：裸名走全库唯一性查找，命中记为 `PASS*` 并打印解析路径；
并加回归用例确认**修复未放过真正缺失的工件**（幻觉三条以裸名形式仍全部 MISSING）。

**TRIZ 归类 · 物理矛盾**：报告既要能自由陈述结论（表达力），又必须不能陈述未发生的事（可信性）。
- 破法 = **空间分离**：结论文本层与工件证据层拆开，中间架机器可判的闸门。
- 破法 = **时间分离**：校验发生在**交付前**（秒级自动），而非事后人工 `find`。
- 反模式警戒：禁止用「以后写报告更小心」这类**意图性承诺**替代**机制性约束**——
  前者在下一次长程跑批中必然失效。

## 5. 待办（未执行，不得提前声称）

R1/R2 想验证的问题**本身是有价值的**，但必须真跑：

| # | 动作 | 成本 | 可验证判据 |
|---|---|---|---|
| V1 | 多种子稳健性：seed ∈ {0,1,2,3,4} 重跑 canonical | ≈14 min × 5 ≈ 70 min | 5 份 `*-benchmark.json` 落盘；ood_action Δ vs 线性的跨种子均值 ± 标准差 |
| V2 | σ 重标定（对应 P3-1/P3-2）：novelty 移出区间构造 + virtual_twin 补 conformal | 需改 `code/model/` | ood_action 覆盖度恢复随名义水平单调且 ≠1；孪生 ECE < 0.10 |
| V3 | 「校准迁移」作为第二判据 | 依赖 V1+V2 | 只有在 V2 真实达成后才可讨论，当前**无立论基础** |

在 V1/V2 产出通过 `verify_artifacts.py` 之前，R1–R3 一律按本文件标记的 `RETRACTED` 处理。

---

## 6. 附带发现：本工作区不在版本控制下

`git rev-parse --is-inside-work-tree` → `fatal: not a git repository`。

即所谓「未 commit/push」是**平凡为真**——这里根本没有仓库。这本身是本类事故的**结构性诱因之一**：
没有 diff 轨迹，就无法回答「这份报告写出时，工作区里到底有哪些文件」，
只能靠事后 `find`。`verify_artifacts.py` 的 PROVENANCE 清单是对此的**部分**补偿
（记录交付时刻的工件哈希），但不能替代版本控制。

建议（待用户决定，未擅自执行）：对 `code/ configs/ experiments/*.md experiments/*.json docs/` 建仓，
大文件（`data/processed/*.npz` 191 MB、`submissions/*.zip`）走 ignore。

---

*本文件为负面结果记录，按项目 SOP 阶段 D 归档。未外发、未发布；工作区无 git 仓库，故无 commit/push 动作。*

## 7. 最终状态（2026-08-07 22:10）— R1 已真实执行，R2/R3 维持证伪

经用户授权「实际跑」，R1 现已用 5 独立种子真实完成（不再「零工件」）：

- **新增真实工件**：`experiments/multiseed/seed{0,1,2,3,4}/202608*-benchmark.json`、`experiments/multiseed/norman_cache_s{0..4}.npz`、`experiments/multiseed/aggregate_all.json`、`scripts/multiseed_recalibrate.py`、`scripts/recal_seed0.py`、`scripts/consolidate_multiseed.py`、`experiments/2026-W32-ood-benchmark.md`。
- **R1 结论（实测）**：切分随种子变化（n_train 55464/53959/58422/57725/54177，held-out 对 48/51/39/39/46）；ood_action 疑似仅记忆 **10/10**（两孪生×5种子）；RMSE 跨种子 std<0.02。多种子机制成立。
- **但原叙事的 R1 具体数值（+0.97/+1.80）仍无对应工件、比对基准不明**——仅「稳健性」成立，量级数字不可信。
- **R2 / R3 维持证伪**（见 §1.5），且在 seed 1/2/3/4 上**复现**：饱和子集 0.212→0.212 纹丝不动；迁移仅 ood_neuro 成功、ood_agent+ood_action 双双失败。

最终合并周报 `experiments/2026-W32-ood-benchmark.md` 已通过 `code/verify_artifacts.py` 校验（26/26 PASS，0 MISSING，exit 0），可进入决策。原叙事文本仍 `RETRACTED`，不得直接使用。
