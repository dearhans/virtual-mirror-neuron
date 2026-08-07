# 虚拟镜像神经元 · Virtual Mirror Neuron

> 计算研究项目：以「扰动响应预测」为轴，构建多尺度虚拟孪生，用 `do-演算` + 组合扰动把「相关」升级为「可证伪的预测」，并以光遗传 / 钙成像湿实验闭环验证**机制泛化**。

## 一句话定位
把单个神经元当作一个**预测编码单元**；镜像性不过是「感知」与「动作」在共享潜空间里的等价 / 对称性。我们要造的，是一台能对**未见条件（OOD）**给出*带不确定度*的扰动响应预测的机器——能外推，才说明学到了机制，而不是记住了分布。

## 七层研究主线（Nobel 思维）
1. **现象层** — 死磕悖论：同一块神经基质既编码「我在抓花生」也编码「我看你抓花生」。这不是装饰性发现，而是「自我 / 他者」在计算上同源的直接证据。研究的起点永远是问对问题。
2. **抽象层** — 翻译成可计算对象：单神经元 = 预测编码单元；镜像性 = 感知 / 动作在共享潜空间的对称性。这一步决定你后面建的是模型还是黑箱。
3. **建模层** — 多尺度虚拟孪生：离子通道 → 突触可塑性 → 环路；「给一个刺激 / 扰动」= 图上的 `do-` 干预。数据来自电生理、钙成像、单细胞核组学（与 perturb-seq 同思想）。
4. **因果层** — 用 `do-演算` + 组合扰动把相关升级为可证伪预测。虚拟细胞领域（GEARS、scPerturb、Virtual Cell Challenge 2025）已证明：带生物学先验的模型优于纯端到端黑箱，别迷信黑箱。
5. **泛化层** — 真正的考验是 OOD 外推：未见过的动作、没见过的主体、新的神经调质状态。模型只记住训练分布就没价值；能外推才说明学到机制。
6. **闭环层** — 绝不停在 benchmark：预测 → 光遗传 / 钙成像湿实验验证 → 误差回流 → 再训练。
7. **意义层** — 反哺社会认知理论、孤独症机制假说、脑机接口，以及让机器具备「看样学样」的内禀共情能力。

## 硬约束（不可妥协）
- 所有预测必须**区分「记忆」与「机制泛化」**。
- 评测必须含**未见条件（OOD）子集**。
- 报告必须**给出不确定度**（校准后指标，参考 Shift Bioscience 的 metric calibration 思路）。

## 目录结构
```
virtual-mirror-neuron/
├── docs/            # 方法论、SOP、文献笔记
│   ├── methodology.md   # 七层框架 + TRIZ 矛盾映射 + mirror-neuron 自学习映射
│   └── SOP.md           # 闭环标准作业流程
├── code/            # 建模 / 扰动 / 基准代码（多尺度虚拟孪生）
├── data/            # 电生理、钙成像、snRNA-seq（占位，内容不入库）
├── experiments/     # OOD 基准运行记录（占位）
├── wetlab/          # 每周导出的可执行湿实验方案
├── automations/     # 3 个自动化（文献监测 / 周报 / OOD 基准）规格
├── references/      # 文献与专利归档（占位）
├── configs/         # 模型 / 评测 / 刺激参数配置（占位）
├── CHARTER.md       # 项目章程（角色 / 硬约束 / 闭环规则 / 决策原则）
└── LICENSE
```

## 自动化
见 [automations/README.md](automations/README.md)：文献监测 / 周报 / OOD 基准评测（项目强制）。

## 快速开始
```bash
# 1. 克隆
git clone https://github.com/<owner>/virtual-mirror-neuron.git
cd virtual-mirror-neuron
# 2. 安装（managed 运行时，保持隔离）
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
# 3. 跑一次最小 OOD 基准（依赖 code/ 落地后可用）
python code/benchmark_ood.py --config configs/baseline.yaml
```

> 注：仓库当前为**骨架阶段**，`code/` 与 `configs/` 待按 SOP 落地。闭环与自动化已就绪，可先于代码运行以规范流程。
