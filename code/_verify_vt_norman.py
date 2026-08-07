"""_verify_vt_norman.py — 验证 benchmark_ood 的 VirtualTwinPredictor 在
Norman 组合性重排路径下是否真的会因 g 误切产生 NaN/inf。

关键对照：真实 pipeline 用 `VirtualTwinPredictor.fit`，其中
    self.Dp = X.shape[1] - 3   # 组合性布局 [pa,pb,A(2),g(1)] -> Dp = 2*comp_Dp
而此前手动复现用了 `VirtualTwin(Dp=comp_Dp)`，那是错误用法，才触发 NaN。

本脚本复刻真实 fit 路径，仅用少量 train 细胞（g 切分与样本量无关，数量足够暴露 bug），
检查 virtual_twin 在 ood_action 上的 RMSE 是否有限。
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from benchmark_ood import (
    norman_to_compositional_X, VirtualTwinPredictor, load_config, run,
)
from data.norman_adapter import load_norman, NormanConfig

ROOT = os.path.dirname(HERE)

# ---- 加载 Norman 真数据（用与 compositional config 一致的参数）----
ncfg = NormanConfig(
    raw_dir=os.path.join(ROOT, "data", "raw", "norman2019"),
    panel_k=256, seed=0, heldout_gene_frac=0.20, heldout_gemgroup=8,
    train_frac=0.70, double_train_frac=0.50,
    cache_path=os.path.join(ROOT, "data", "processed", "norman_cache.npz"),
)
data = load_norman(ncfg)
X, y, split = data["X"], data["y"], data["split"]
print(f"raw X shape={X.shape}  comp_Dp={X.shape[1]-3}")

# ---- 组合性重排（与 run() 一致）----
Xc = norman_to_compositional_X(X)
print(f"compositional Xc shape={Xc.shape}  -> VirtualTwinPredictor 将推导 Dp = {Xc.shape[1]-3} (=2*comp_Dp)")

cfg = load_config(os.path.join(ROOT, "configs", "benchmark_ood_norman_compositional.yaml"))
train_mask = split == "train"
# 子采样 train 以加速（g 切分与样本量无关）
rng = np.random.default_rng(0)
tr_idx = np.where(train_mask)[0]
tr_sub = rng.choice(tr_idx, size=min(3000, len(tr_idx)), replace=False)
Xtr, ytr = Xc[tr_sub], y[tr_sub]

vt = VirtualTwinPredictor(cfg)
try:
    vt.fit(Xtr, ytr)
    print(f"FIT OK  -> VirtualTwin.Dp={vt.twin.Dp}  A_dim={vt.twin.A_dim}")
    # 预测 ood_action
    oa = split == "ood_action"
    oa_idx = np.where(oa)[0][:600]
    pred = vt.predict(Xc[oa_idx])
    rmse = float(np.sqrt(np.mean((y[oa_idx] - pred) ** 2)))
    finite = bool(np.all(np.isfinite(pred))) and np.isfinite(rmse)
    print(f"PRED ood_action n={len(oa_idx)}  RMSE={rmse:.4f}  finite={finite}")
    if not finite:
        print("!!! 真实 pipeline 仍产生非有限值 -> 确有 bug")
    else:
        print(">>> 真实 pipeline 正常：VirtualTwinPredictor 正确推导 Dp=2*comp_Dp，g 列无误切。此前 NaN 系手动以 Dp=comp_Dp 构造所致（测试误用）。")
except Exception as e:
    print(f"FIT/PRED RAISED: {type(e).__name__}: {e}")
    print("!!! 真实 pipeline 抛错 -> 确有 bug")
