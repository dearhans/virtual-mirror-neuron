# 虚拟镜像神经元项目 — 跨机器迁移指南 (MIGRATION)

> 用途：把本机已完成的全部工作打包带到另一台电脑继续。
> 生成日期：2026-07-31。来源机器用户：`dearhans`，WorkBuddy 托管运行时。

---

## 一句话结论（回答「要不要重新建项目」）

**不需要从零重建项目内容。** 项目的四块交付物（仓库文档 / 专家包 / SOP skill / OOD 基准代码）都在本包里，直接解压即可用。

但 WorkBuddy 把三类**本地状态**绑定在每台机器上，不会随文件自动迁移，必须在新电脑上「恢复」一次：
1. **专家包** → 放在新电脑的用户级插件目录（或直接导入）。
2. **自动化** → 存在本地 SQLite，不随文件走，需按本指南在新电脑重新创建（参数已附，可一键复制）。
3. **助手身份文件**（SOUL/IDENTITY/USER）→ 若新电脑登录同一 WorkBuddy 账号可能已云端同步；否则把 `identity/` 三文件放到 `~/.workbuddy/`。

> SOP skill（`.codebuddy/skills/`）已随 `project/` 带走，新电脑打开项目工作区即自动可用，无需额外操作。

---

## 包内容清单

```
virtual-mirror-neuron-migration/
├── MIGRATION.md            # 本文件
├── project/                # = 完整项目目录（含 .codebuddy/skills/ 下的 SOP skill）
│   ├── README.md  CHARTER.md  LICENSE  .gitignore
│   ├── docs/        (methodology.md, SOP.md)
│   ├── automations/ (README.md)
│   ├── code/        (benchmark_ood.py, model/twin.py, data/loaders.py, perturbation/graph.py, README.md, requirements.txt)
│   ├── configs/     (benchmark_ood.yaml)
│   ├── experiments/ (20260731-benchmark.md, .json)
│   ├── data/ wetlab/ references/  (占位 .gitkeep)
│   └── .codebuddy/skills/virtual-mirror-neuron-sop/  (SKILL.md + references/)
├── expert-package/         # 专属专家包「虚拟镜像神经元研究专家」
│   ├── .codebuddy-plugin/plugin.json
│   ├── agents/virtual-mirror-neuron.md
│   ├── avatars/expert.png
│   └── README.md
└── identity/               # 助手稳定身份（Mira 🪞）
    ├── SOUL.md  IDENTITY.md  USER.md
```

---

## 在新电脑上的恢复步骤

### 1) 项目文件（必做，无依赖）
把 `project/` 整个目录解压到新电脑的任意 WorkBuddy 工作区（或任意目录）。
- 代码立刻可跑：`cd project && python code/benchmark_ood.py`（需 numpy/sklearn/pyyaml，详见 `code/requirements.txt`）。
- SOP skill 已就位：WorkBuddy 在该目录打开对话即能调用 `virtual-mirror-neuron-sop`。

### 2) 专家包（推荐，让用户级专家可用）
把 `expert-package/` 内容放到新电脑：
```
~/.workbuddy/plugins/marketplaces/my-experts/plugins/virtual-mirror-neuron/
```
（若该路径不存在先建好）。放入后专家即出现在专家中心。
> 也可在 WorkBuddy 用 expert-manager 的「导入」流程重新注册（若客户端提供）。

### 3) 自动化（必做 — 因本地 db 不随文件走）
WorkBuddy 自动化存在本地数据库，**无法导出为文件**。在新电脑的工作区对话里，让我（或你）用 `automation_update` 重新创建以下三个（完整参数见下方「自动化完整定义」）。创建时把 `cwds` 改成新电脑的项目路径（即第 1 步解压后的 `project/` 绝对路径）。

### 4) 助手身份（可选 — 取决于账号是否同步）
若新电脑登录同一 WorkBuddy 账号后身份未同步，把 `identity/` 三文件覆盖到新电脑 `~/.workbuddy/`。
名字是 **Mira 🪞**（取自 mirror，呼应「观察-执行共振」）。若想改名，改 `IDENTITY.md` 与 `USER.md` 即可。

### 5) GitHub 远端（待办 — 最后一起建）
本机始终没建成 GitHub 远端（连接器只读 + 本机出网被拦）。待新电脑网络/授权放通后，二选一：
- **网页端上传**：在 github.com 建 `virtual-mirror-neuron`（Public），把 `project/` 内容拖进去提交。
- **能出网的机器**：用 git 推送。
专家包、SOP skill、自动化配置均不依赖远端仓库，本地即可运转。

---

## 自动化完整定义（在新电脑重建时用）

> 三个均为 `status: ACTIVE`、`scheduleType: recurring`。创建时 `cwds` 改为新电脑的 `project/` 绝对路径，`expertId` 填 `virtual-mirror-neuron`（若专家包已恢复），否则去掉 `expertId`。

### A. 文献监测（每周一）
- name: `虚拟镜像神经元-文献监测`
- rrule: `RRULE:FREQ=WEEKLY;BYDAY=MO`
- prompt:
```
你负责「虚拟镜像神经元」项目的每周文献监测。请检索并梳理以下方向的最新进展：virtual cell（虚拟细胞）、mirror neuron（镜像神经元）、perturb-seq / scRNA-seq perturbation、OOD generalization（分布外泛化）、causal representation learning（因果表征学习）、calcium imaging（钙成像）、optogenetics（光遗传）、neuromodulator（神经调质）。对每篇值得关注的新论文/预印本，提取：标题、作者、年份/来源、核心方法、与本项目（记忆 vs 机制泛化、OOD、do-演算）的相关点。将结果写入项目 references/ 目录（文件名 references/<YYYY-MM-DD>-literature.md），并在 references/INDEX.md 追加一行索引（若不存在则创建）。去重。最后用一段话总结本周最值得借鉴的 1-2 个方法，及其对建模层（多尺度虚拟孪生）的启示。不要外发或发布，仅写文件与汇总。
```

### B. 周报与湿实验方案（每周五）
- name: `虚拟镜像神经元-周报与湿实验方案`
- rrule: `RRULE:FREQ=WEEKLY;BYDAY=FR`
- prompt:
```
你负责「虚拟镜像神经元」项目的每周周报与湿实验方案导出（闭环层）。步骤：1) 汇总本周模型预测（来自 experiments/ 与当前模型状态，若尚无代码则用占位说明并列出待落地项）。2) 挑选 Top-K 最高信息增益 / 最高不确定度的预测。3) 生成可执行湿实验方案，写入 wetlab/<YYYY-Www>.md，包含：靶点（光遗传 / 钙成像）、刺激参数（频率、强度、时程、组合扰动设计）、预期响应 + 不确定度 + 证伪条件。4) 汇总本周进度、未解矛盾（用 TRIZ 做矛盾分类：本项目核心是物理矛盾「记忆 vs 外推」）、下一步。周报本身写入项目根目录 WEEKLY-<YYYY-Www>.md。所有预测必须区分「记忆（in-distribution）」与「机制泛化（OOD）」双栏。不要外发或发布。
```

### C. OOD 基准评测（每周日，项目强制项）
- name: `虚拟镜像神经元-OOD基准评测`
- rrule: `RRULE:FREQ=WEEKLY;BYDAY=SU`
- prompt:
```
你负责「虚拟镜像神经元」项目的每周 OOD 外推基准评测（项目硬约束，强制）。步骤：1) 加载当前模型与 configs/ 下配置（若代码尚未落地，先生成最小可运行的基线脚本骨架并标注 TODO，再评测）。2) 切分 OOD 子集：未见动作（unseen action）、未见主体（unseen agent）、新神经调质状态（novel neuromodulator）。3) 跑 OOD 外推基准，对比简单基线：线性 / 均值 / 最近邻。4) 输出校准后指标：in-distribution vs OOD 双栏、置信区间、校准曲线、ECE（expected calibration error），写入 experiments/<YYYYMMDD>-benchmark.md 与指标 JSON。5) 判据：若 OOD 指标未显著优于简单基线，标记为「疑似仅记忆」，触发 SOP 阶段 D 误差回流与 TRIZ 修正。参考 Shift Bioscience 的 metric calibration 思路——指标必须先校准再比较。报告必须给出不确定度。不要外发或发布。
```

---

## 当前进展与已知局限（接手时务必知道）

- **已验证**：`code/benchmark_ood.py` 实跑通过。虚拟孪生（多尺度+集成/新颖度不确定度）在 ID 与「应外推」OOD 子集上显著优于简单基线（均值/线性/KNN）与黑箱 MLP；`ood_action`(imitation) 作为真正新机制被如实报为大误差、不误报「仅记忆」。
- **校准**：此前同方差不确定度在 OOD 上过度自信（覆盖度 0.17），改用「集成+新颖度(ReLU)」后 ID/ood 覆盖度 0.94–0.98、ECE 0.12–0.15；黑箱 MLP 在 ood_action 仍 ECE=0.70（灾难性过度自信）。
- **局限**：合成数据机制为真乘法增益+对称非线性，接真实数据须重验；`load_perturb_seq`/`load_calcium_imaging` 仍是接口桩（schema 映射待补）；孪生用 sklearn MLP 集成，未接真实物理多尺度模型（离子通道 ODE/可塑性）。
- **PAT 安全**：此前对话中出现过的 GitHub PAT 已视为泄露，请确认已在 GitHub 撤销。

---

## 快速验证（新电脑解压后）

```bash
cd <解压后的 project 目录>
python code/benchmark_ood.py          # 应生成 experiments/<日期>-benchmark.md + .json
python -c "import numpy,sklearn,yaml;print('deps OK')"
```

若依赖缺失，用托管 Python 建 venv：`python -m venv venv && venv/bin/pip install numpy scikit-learn pyyaml`。
