# code/ · 建模与基准代码

本目录承载多尺度虚拟孪生的实现。`benchmark_ood.py` 为**完整可运行骨架**，已接入项目内模块：

```
code/
├── requirements.txt          # 依赖（managed 运行时，保持隔离）
├── benchmark_ood.py          # ✅ OOD 外推基准入口：合成数据→基线对比→校准→双栏报告+反事实演示
├── configs -> ../configs/benchmark_ood.yaml
├── model/
│   ├── twin.py               # ✅ VirtualTwin（多尺度：离子通道/突触/环路 + 因果头 + 集成/新颖度不确定度）
│   └── twin.py:MLPTwin       # ✅ 朴素 MLP 黑箱基线（对照「带生物学先验 > 端到端黑箱」）
├── perturbation/
│   └── graph.py              # ✅ CausalGraph + do() 干预（反事实演示用）
└── data/
    └── loaders.py            # ✅ 合成数据生成 + 真实 perturb-seq/钙成像加载接口（pandas/anndata 懒加载）
```

## 模块职责
- **data/loaders.py** — 统一数据接口。内置 `generate_mirror_neuron_dataset()`（扰动→响应，含
  动作/主体/调质上下文）；`load_perturb_seq()` / `load_calcium_imaging()` 为真实数据加载桩，
  懒加载 pandas / anndata，缺失时给出清晰提示，不阻塞骨架运行。
- **model/twin.py** — `VirtualTwin`：细胞尺度非线性机制 φ(p)（集成 MLP）+ 动作对称性权重共享
  （self/other 共享 φ → 镜像性）+ 神经调质乘法增益 g（因果头显式）。不确定度 = 集成 epistemic
  + 新颖度(OOD, ReLU) 合成，使 OOD 不再过度自信。
- **perturbation/graph.py** — 因果图：声明可干预节点（perturbation/action/neuromodulator），
  `do()` 改写特征，供反事实演示验证因果头。
- **benchmark_ood.py** — 调用上述模块；预测器对比 均值/线性/KNN（项目强制简单基线）+ MLP(黑箱)
  + VirtualTwin(多尺度)；切分 ID 与三类 OOD；输出校准后 RMSE/MAE/R² + 覆盖度 + ECE + bootstrap CI
  + 双栏报告 + do-干预反事实表。

## 设计原则（与 CHARTER 对齐）
- **机制先验不可丢**：模型须含可解释因果结构，不接受纯黑箱端到端（基准内 MLP 黑箱仅作对照）。
- **双栏报告**：记忆（in-distribution）vs 机制泛化（OOD）必须同时输出。
- **不确定度校准**：所有指标先校准再比较（参考 Shift Bioscience metric calibration）。
- **可复现**：随机种子、配置、数据版本全部显式。

## 运行

```bash
# 默认读取 configs/benchmark_ood.yaml，使用内置合成数据，开箱即跑
python code/benchmark_ood.py

# 快速自检（小数据）
python code/benchmark_ood.py --quick

# 指定配置 / 根目录
python code/benchmark_ood.py --config configs/benchmark_ood.yaml --root .
```

运行后会在 `experiments/` 生成 `YYYYMMDD-benchmark.md`（双栏报告）与 `.json`（全部指标）。
当模型在「机制应外推」的 OOD 子集上未显著优于简单基线时，报告会触发 SOP 阶段 D 误差回流 + TRIZ 修正提示。

## 已知局限（诚实记录）
- 合成数据的「机制」为真乘法增益 + 对称非线性；真实数据接入后须重新验证外推结论。
- `load_perturb_seq` / `load_calcium_imaging` 目前为接口桩，真实 schema 映射待补。
- 虚拟孪生用 sklearn MLP 集成；接入真实多尺度物理模型（离子通道 ODE / 可塑性规则）是下一阶段。
