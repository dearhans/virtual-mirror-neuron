#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_ood.py — 虚拟镜像神经元 · OOD 外推基准评测（多尺度孪生版）
================================================================

项目硬约束（见 CHARTER.md / SOP.md）：
  1. 所有预测必须区分「记忆（in-distribution）」与「机制泛化（OOD）」。
  2. 评测必须含未见条件（OOD）子集：未见动作 / 未见主体 / 新神经调质状态。
  3. 报告必须给出不确定度；指标必须先校准再比较（参考 Shift Bioscience 的 metric calibration）。

本脚本调用项目内模块：
  - data.loaders.generate_mirror_neuron_dataset  : 合成「扰动→响应」数据（含动作/主体/调质上下文）
  - model.twin.VirtualTwin                        : 多尺度虚拟孪生（离子通道/突触/环路 + 因果头 + 集成/新颖度不确定度）
  - model.twin.MLPTwin                           : 朴素 MLP 黑箱基线（对照「带生物学先验 > 端到端黑箱」）
  - perturbation.graph                           : 因果图 + do-干预（反事实演示）

预测器对比：均值 / 线性 / KNN（简单基线，项目强制） + MLP(黑箱) + VirtualTwin(多尺度结构化)。
OOD 子集：未见主体 / 未见动作(imitation) / 新神经调质。指标：RMSE(含 bootstrap 95% CI) / MAE / R²
          + 区间覆盖度 + ECE（校准后）。在「机制本应外推」的 OOD 子集上若未显著优于最佳简单基线 → 标记「疑似仅记忆」。

运行：
  python benchmark_ood.py --config ../configs/benchmark_ood.yaml
  python benchmark_ood.py --quick            # 小数据快速自检
  python benchmark_ood.py --root <项目根目录> # 覆盖输出根
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

# 确保 code/ 在路径中（无论从何处调用）
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from data.loaders import (  # noqa: E402
    generate_mirror_neuron_dataset,
    generate_compositional_dataset,
    load_perturb_seq,
    load_calcium_imaging,
    to_benchmark_format,
)
from data.norman_adapter import load_norman, NormanConfig  # noqa: E402
from data.goai_virtualcell_adapter import load_goai, GoaiConfig  # noqa: E402
from model.twin import VirtualTwin, MLPTwin  # noqa: E402
from model.compositional import CompositionalTwin  # noqa: E402
from model.compositional import CompositionalInteractionTwin  # noqa: E402
from perturbation.graph import build_mirror_neuron_graph, col_map  # noqa: E402


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "data": {
        "path": None,  # 若指向存在文件则加载真实数据（loader 待实现）
        "synthetic": {"n_train": 2400, "n_test": 900, "seed": 7, "noise_std": 0.15},
    },
    "model": {
        "checkpoint": None,        # 真实孪生检查点；None 时使用 VirtualTwin 结构化模型
        "hidden": [32, 32],        # 虚拟孪生·细胞尺度 MLP
        "n_ensemble": 5,           # 集成规模（epistemic 不确定度）
        "novelty_k": 0.8,          # 新颖度（OOD）不确定度权重
        "random_state": 0,
    },
    "baselines": {"knn_k": 5},
    "eval": {
        "bootstrap_samples": 200,
        "ci_alpha": 0.95,
        "calibration_levels": [0.5, 0.8, 0.9, 0.95],
        # 模型相对最佳简单基线的 OOD RMSE 提升若 ≤ 此阈值（且 bootstrap CI 重叠），
        # 则标记「疑似仅记忆」
        "memorization_flag_delta": 0.0,
    },
    "output": {"dir": "experiments", "prefix": "benchmark"},
}


def load_config(config_path: Optional[str]) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # 深拷贝默认值
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        _deep_update(cfg, user_cfg)
    return cfg


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_or_generate(cfg: dict, root: str) -> Dict[str, np.ndarray]:
    dpath = cfg["data"].get("path")
    source = cfg["data"].get("source")
    if source == "norman":
        nc = cfg["data"].get("norman", {})
        raw_dir = nc.get("raw_dir") or os.path.join(root, "data", "raw", "norman2019")
        cache = nc.get("cache_path") or os.path.join(root, "data", "processed", "norman_cache.npz")
        ncfg = NormanConfig(
            raw_dir=raw_dir,
            panel_k=int(nc.get("panel_k", 256)),
            seed=int(nc.get("seed", 0)),
            heldout_gene_frac=float(nc.get("heldout_gene_frac", 0.20)),
            heldout_gemgroup=int(nc.get("heldout_gemgroup", 8)),
            train_frac=float(nc.get("train_frac", 0.70)),
            double_train_frac=float(nc.get("double_train_frac", 0.50)),
            cache_path=cache,
        )
        return load_norman(ncfg)
    if source == "goai":
        gc = cfg["data"].get("goai", {})
        raw_dir = gc.get("raw_dir") or os.path.join(root, "data", "raw", "goai_virtualcell")
        cache = gc.get("cache_path") or os.path.join(root, "data", "processed", "goai_cache.npz")
        gcfg = GoaiConfig(
            raw_dir=raw_dir,
            train_val_file=gc.get("train_val_file", "train_val.csv"),
            test_file=gc.get("test_file", "test.csv"),
            target=gc.get("target", "delta"),
            heldout_compound_frac=float(gc.get("heldout_compound_frac", 0.20)),
            heldout_strain_frac=float(gc.get("heldout_strain_frac", 0.20)),
            train_frac=float(gc.get("train_frac", 0.70)),
            seed=int(gc.get("seed", 0)),
            cache_path=cache,
        )
        return load_goai(gcfg)
    if dpath:
        full = dpath if os.path.isabs(dpath) else os.path.join(root, dpath)
        if os.path.isfile(full):
            ext = os.path.splitext(full)[1].lower()
            if ext in (".csv", ".tsv", ".npz"):
                # 文件名含 calcium 视为钙成像，否则默认 perturb-seq（二者宽表同 schema）
                if "calcium" in os.path.basename(full).lower():
                    ds = load_calcium_imaging(full)
                else:
                    ds = load_perturb_seq(full)
                return to_benchmark_format(ds)
            if ext == ".h5ad":
                ds = load_perturb_seq(full)
                return to_benchmark_format(ds)
            raise NotImplementedError(f"未支持的真实数据格式: {ext}（{full}）")
    s = cfg["data"]["synthetic"]
    if s.get("compositional"):
        return generate_compositional_dataset(
            s["n_train"], s["n_test"], s["seed"], s["noise_std"],
            Dp=8,
            mag_train=float(s.get("mag_train", 0.6)),
            mag_ood=float(s.get("mag_ood", 1.4)),
            epistasis=float(s.get("epistasis", 0.0)),
        )
    return generate_mirror_neuron_dataset(s["n_train"], s["n_test"], s["seed"], s["noise_std"])


# --------------------------------------------------------------------------- #
# 预测器（统一接口）
# --------------------------------------------------------------------------- #
class Predictor:
    name = "base"

    def fit(self, X: np.ndarray, y: np.ndarray, split=None, z_comp=None) -> "Predictor":
        self._y_train_std = float(np.std(y)) if len(y) else 1.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_with_std(self, X: np.ndarray, subset=None) -> Tuple[np.ndarray, np.ndarray]:
        mu = self.predict(X)
        return mu, np.full_like(mu, self._y_train_std)


class MeanBaseline(Predictor):
    name = "mean"

    def fit(self, X, y, split=None, z_comp=None):
        super().fit(X, y, split=split, z_comp=z_comp)
        # 向量响应时存每基因均值（shape (K,)），标量时存标量
        self._mu = np.mean(y, axis=0) if len(y) else np.array(0.0)
        return self

    def predict(self, X):
        mu = np.asarray(self._mu)
        return np.broadcast_to(mu, (len(X),) + mu.shape)


class LinearBaseline(Predictor):
    name = "linear"

    def fit(self, X, y, split=None, z_comp=None):
        super().fit(X, y, split=split, z_comp=z_comp)
        self._m = Ridge(alpha=1.0).fit(X, y)
        resid = y - self._m.predict(X)
        self._std = float(np.std(resid)) if len(resid) else self._y_train_std
        return self

    def predict(self, X):
        return self._m.predict(X)

    def predict_with_std(self, X, subset=None):
        return self.predict(X), np.full(len(X), self._std)


class KNNBaseline(Predictor):
    name = "knn"

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X, y, split=None, z_comp=None):
        super().fit(X, y, split=split, z_comp=z_comp)
        self._y_train = y
        self._m = KNeighborsRegressor(n_neighbors=self.k).fit(X, y)
        return self

    def predict(self, X):
        return self._m.predict(X)

    def predict_with_std(self, X, subset=None):
        neigh = self._m.kneighbors(X, return_distance=False)
        mu = self._m.predict(X)
        std = np.array([np.std(self._y_train[idxs]) for idxs in neigh])
        std = np.where(np.isfinite(std) & (std > 0), std, self._y_train_std)
        return mu, std


class VirtualTwinPredictor(Predictor):
    """包装 model.twin.VirtualTwin（多尺度结构化孪生）。"""

    name = "virtual_twin"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.Dp = None

    def fit(self, X, y, split=None, z_comp=None):
        super().fit(X, y, split=split, z_comp=z_comp)
        self.Dp = X.shape[1] - 3  # act(2) + g(1)
        A_dim = X.shape[1] - self.Dp - 1
        m = self.cfg["model"]
        self.twin = VirtualTwin(
            Dp=self.Dp,
            A_dim=A_dim,
            hidden=tuple(m["hidden"]),
            n_ensemble=m["n_ensemble"],
            novelty_k=m["novelty_k"],
            random_state=m["random_state"],
        ).fit(X, y)
        return self

    def predict(self, X):
        mean, _, _ = self.twin.predict(X)
        return mean

    def predict_with_std(self, X, subset=None):
        mean, std, _ = self.twin.predict(X)
        return mean, std


class MLPTwinPredictor(Predictor):
    """包装 model.twin.MLPTwin（朴素 MLP 黑箱基线）。"""

    name = "mlp"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, X, y, split=None, z_comp=None):
        super().fit(X, y, split=split, z_comp=z_comp)
        m = self.cfg["model"]
        self.twin = MLPTwin(hidden=(64, 64), random_state=m["random_state"]).fit(X, y)
        return self

    def predict(self, X):
        return self.twin(X)

    def predict_with_std(self, X, subset=None):
        return self.twin.predict(X)


class CompositionalTwinPredictor(Predictor):
    """包装 model.compositional.CompositionalTwin（组合性先验 + 课程学习 + 可选共形校准）。

    - 组合性先验：共享组件编码器 phi + 加法组合，使「双扰动 = 两条单扰动效应之和」。
    - 课程学习：stage1 单扰动 → stage2 双扰动（warm_start）。
    - 共形校准（conformal=True）：训练拆出校准子集，按 subset 计算绝对残差分位数，
      预测时按测试 subset 用该分位数定区间半宽 → 各子集覆盖率≈名义 0.9。
    """

    name = "compositional_twin"

    def __init__(self, cfg: dict, conformal: bool = False):
        self.cfg = cfg
        self.conformal = conformal

    def _component_dims(self, X):
        A_dim = 2  # 组合数据动作 one-hot 固定 2 维
        comp_Dp = (X.shape[1] - 1 - A_dim) // 2
        return comp_Dp, A_dim

    def fit(self, X, y, split=None, z_comp=None):
        super().fit(X, y, split=split, z_comp=z_comp)
        m = self.cfg["model"]
        comp_Dp, A_dim = self._component_dims(X)
        if self.conformal and split is not None:
            self.twin = self._fit_conformal(X, y, split, z_comp, comp_Dp, A_dim, m)
        else:
            self.twin = CompositionalTwin(
                Dp=comp_Dp, A_dim=A_dim, hidden=tuple(m["hidden"]),
                n_ensemble=m["n_ensemble"], novelty_k=m["novelty_k"],
                random_state=m["random_state"], curriculum=bool(m.get("curriculum", True)),
            ).fit(X, y, z_comp=z_comp)
        return self

    def _fit_conformal(self, X, y, split, z_comp, comp_Dp, A_dim, m):
        rng = np.random.default_rng(int(m.get("random_state", 0)))
        idx = np.arange(len(X))
        rng.shuffle(idx)
        ncal = max(1, int(len(idx) * float(m.get("conformal_calib_frac", 0.2))))
        cidx = idx[:ncal]
        fidx = idx[ncal:]
        twin = CompositionalTwin(
            Dp=comp_Dp, A_dim=A_dim, hidden=tuple(m["hidden"]),
            n_ensemble=m["n_ensemble"], novelty_k=m["novelty_k"],
            random_state=m["random_state"], curriculum=bool(m.get("curriculum", True)),
        )
        twin.fit(X[fidx], y[fidx],
                 z_comp=(z_comp[fidx] if z_comp is not None else None))
        mu_cal, std_cal, _ = twin.predict(X[cidx])
        # 归一化共形分数：|残差| / 模型异方差 std（含新颖度）→ 区间随点自适应缩放，
        # 使大误差的 OOD（ood_action 大幅度双扰动）自动获得更宽区间。
        scores = np.abs(y[cidx] - mu_cal) / np.clip(std_cal, 1e-9, None)
        alpha = float(m.get("conformal_alpha", 0.9))
        self._q = float(np.quantile(scores, alpha)) if len(scores) else 1.645
        self._alpha = alpha
        return twin

    def predict(self, X):
        mean, _, _ = self.twin.predict(X)
        return mean

    def predict_with_std(self, X, subset=None):
        mean, std, _ = self.twin.predict(X)
        if self.conformal and hasattr(self, "_q"):
            # 归一化共形：区间半宽 = q × 模型异方差 std（按点自适应）
            return mean, self._q * std
        return mean, std


class CompositionalInteractionPredictor(Predictor):
    """包装 model.compositional.CompositionalInteractionTwin（P2：加法基 + 可学习基因对交互项）。

    - 加法组合基：与 CompositionalTwin 同（共享 φ，单扰动监督）。
    - 基因对交互头 ψ：在训练双扰动上拟合并泛化到 held-out 基因对(ood_action)，拟合真实 epistasis。
    - 同输入（Norman→组合性布局重排）公平对比 CompositionalTwin(P1) 与基线。
    """

    name = "compositional_interaction_twin"

    def __init__(self, cfg: dict, conformal: bool = False):
        self.cfg = cfg
        self.conformal = conformal

    def _component_dims(self, X):
        A_dim = 2
        comp_Dp = (X.shape[1] - 1 - A_dim) // 2
        return comp_Dp, A_dim

    def fit(self, X, y, split=None, z_comp=None):
        super().fit(X, y, split=split, z_comp=z_comp)
        m = self.cfg["model"]
        comp_Dp, A_dim = self._component_dims(X)
        if self.conformal and split is not None:
            self.twin = self._fit_conformal(X, y, split, z_comp, comp_Dp, A_dim, m)
        else:
            self.twin = CompositionalInteractionTwin(
                Dp=comp_Dp, A_dim=A_dim, hidden=tuple(m["hidden"]),
                n_ensemble=m["n_ensemble"], novelty_k=m["novelty_k"],
                random_state=m["random_state"], curriculum=bool(m.get("curriculum", True)),
                interaction_hidden=tuple(m.get("interaction_hidden", (64,))),
                interaction_reg=float(m.get("interaction_reg", 1e-1)),
            ).fit(X, y, z_comp=z_comp)
        return self

    def _fit_conformal(self, X, y, split, z_comp, comp_Dp, A_dim, m):
        rng = np.random.default_rng(int(m.get("random_state", 0)))
        idx = np.arange(len(X))
        rng.shuffle(idx)
        ncal = max(1, int(len(idx) * float(m.get("conformal_calib_frac", 0.2))))
        cidx = idx[:ncal]
        fidx = idx[ncal:]
        twin = CompositionalInteractionTwin(
            Dp=comp_Dp, A_dim=A_dim, hidden=tuple(m["hidden"]),
            n_ensemble=m["n_ensemble"], novelty_k=m["novelty_k"],
            random_state=m["random_state"], curriculum=bool(m.get("curriculum", True)),
            interaction_hidden=tuple(m.get("interaction_hidden", (64,))),
            interaction_reg=float(m.get("interaction_reg", 1e-1)),
        )
        twin.fit(X[fidx], y[fidx],
                 z_comp=(z_comp[fidx] if z_comp is not None else None))
        mu_cal, std_cal, _ = twin.predict(X[cidx])
        scores = np.abs(y[cidx] - mu_cal) / np.clip(std_cal, 1e-9, None)
        alpha = float(m.get("conformal_alpha", 0.9))
        self._q = float(np.quantile(scores, alpha)) if len(scores) else 1.645
        self._alpha = alpha
        return twin

    def predict(self, X):
        mean, _, _ = self.twin.predict(X)
        return mean

    def predict_with_std(self, X, subset=None):
        mean, std, _ = self.twin.predict(X)
        if self.conformal and hasattr(self, "_q"):
            return mean, self._q * std
        return mean, std


# --------------------------------------------------------------------------- #
# 评测与校准
# --------------------------------------------------------------------------- #
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def bootstrap_ci(y_true, y_pred, n_boot, alpha, seed=0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 2:
        v = rmse(y_true, y_pred)
        return v, v, v
    vals = np.empty(n_boot)
    idx = np.arange(n)
    for b in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        vals[b] = rmse(y_true[s], y_pred[s])
    lo = np.percentile(vals, (1 - alpha) / 2 * 100)
    hi = np.percentile(vals, (1 + alpha) / 2 * 100)
    return float(np.mean(vals)), float(lo), float(hi)


def calibration(y_true, mu, std, levels) -> Dict[str, object]:
    ztab = {0.5: 0.674, 0.8: 1.282, 0.9: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = {lv: ztab.get(lv, 1.645) for lv in levels}
    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)
    # 向量响应 mu(n,K) 时，若 std 为每样本标量(n,)，需按列广播到基因维
    if mu.ndim == 2 and std.ndim == 1:
        std = std.reshape(-1, 1)
    # 用于分箱的「每样本不确定度」标量
    std1d = (std[:, 0] if std.ndim == 2 else std.ravel())

    cov = {}
    for lv in levels:
        lo = mu - z[lv] * std
        hi = mu + z[lv] * std
        cov[lv] = float(np.mean((y_true >= lo) & (y_true <= hi)))

    per_level_dev = {lv: abs(cov[lv] - lv) for lv in levels}
    ece = float(np.mean(list(per_level_dev.values())))

    # 不确定度分箱校准曲线
    n_bins = 5
    edges = np.quantile(std1d, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = std1d.min() - 1e-9, std1d.max() + 1e-9
    bin_info = []
    for b in range(n_bins):
        m = (std1d >= edges[b]) & (std1d < edges[b + 1])
        if m.sum() == 0:
            continue
        lo = mu[m] - z[0.9] * std[m]
        hi = mu[m] + z[0.9] * std[m]
        bin_info.append({
            "std_bin": [float(edges[b]), float(edges[b + 1])],
            "n": int(m.sum()),
            "coverage_0.9": float(np.mean((y_true[m] >= lo) & (y_true[m] <= hi))),
            "nominal_0.9": 0.9,
        })

    return {"coverage": cov, "ece": ece, "calibration_curve": bin_info, "levels": levels}


@dataclass
class SubsetResult:
    subset: str
    n: int
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    calibration: Dict[str, object] = field(default_factory=dict)


def evaluate_predictor(pred: Predictor, Xtr, ytr, Xte, yte, cfg, split_tr=None, z_comp=None):
    pred.fit(Xtr, ytr, split=split_tr, z_comp=z_comp)
    mu, std = pred.predict_with_std(Xte, subset=cfg.get("_current_subset"))
    ev = cfg["eval"]
    rmse_mean, lo, hi = bootstrap_ci(yte, mu, ev["bootstrap_samples"], ev["ci_alpha"])
    cal = calibration(yte, mu, std, ev["calibration_levels"])
    metrics = {
        "rmse": rmse_mean, "rmse_lo": lo, "rmse_hi": hi,
        "mae": mae(yte, mu), "r2": r2(yte, mu),
    }
    return metrics, cal


def counterfactual_demo(Xtr, ytr, cfg, compositional: bool = False) -> Dict:
    """用因果图做 do-干预反事实演示：验证孪生对调质增益的因果响应。

    向量响应时取基因均值作为标量读数（仅作因果头行为的健全性演示）。
    """
    m = cfg["model"]
    if compositional:
        # 组合数据布局：[p_a, p_b, act(2), g]；组件 Dp=8
        comp_Dp = (Xtr.shape[1] - 1 - 2) // 2
        Dp = comp_Dp
        A_dim = 2
        g_idx = 2 * comp_Dp + A_dim
        act_idx = slice(2 * comp_Dp, 2 * comp_Dp + A_dim)
        twin = CompositionalTwin(
            Dp=comp_Dp, A_dim=A_dim, hidden=tuple(m["hidden"]), n_ensemble=m["n_ensemble"],
            novelty_k=m["novelty_k"], random_state=m["random_state"],
            curriculum=bool(m.get("curriculum", True)),
        ).fit(Xtr, ytr)
    else:
        Dp = Xtr.shape[1] - 3
        A_dim = Xtr.shape[1] - Dp - 1
        g_idx = Dp + A_dim
        act_idx = slice(Dp, Dp + A_dim)
        twin = VirtualTwin(
            Dp=Dp, A_dim=A_dim, hidden=tuple(m["hidden"]), n_ensemble=m["n_ensemble"],
            novelty_k=m["novelty_k"], random_state=m["random_state"],
        ).fit(Xtr, ytr)
    graph = build_mirror_neuron_graph(Dp)
    cm = col_map(Dp)
    rng = np.random.default_rng(123)
    idx = rng.choice(len(Xtr), size=3, replace=False)
    rows = []
    for i in idx:
        x0 = Xtr[i:i + 1]
        base_g = float(x0[0, g_idx])
        pred_base = float(twin.predict(x0)[0].ravel().mean())
        x_g = graph.do("neuromodulator", 1.6, x0, cm)
        pred_g = float(twin.predict(x_g)[0].ravel().mean())
        x_other = x0.copy()
        # 动作轴置为 other（one-hot [0,1]）
        x_other[0, act_idx] = [0.0, 1.0]
        pred_other = float(twin.predict(x_other)[0].ravel().mean())
        rows.append({
            "sample": int(i), "base_gain": round(base_g, 3),
            "pred_base": round(pred_base, 3),
            "do_gain_1.6_pred": round(pred_g, 3),
            "scale_ratio": round(pred_g / pred_base, 3) if pred_base != 0 else None,
            "do_action_other_pred": round(pred_other, 3),
        })
    return {"rows": rows, "graph_summary": graph.summary()}


# --------------------------------------------------------------------------- #
# Norman → 组合性布局重排（仅当 source=norman 且启用 compositional 时调用）
# --------------------------------------------------------------------------- #
def norman_to_compositional_X(X: np.ndarray) -> np.ndarray:
    """把 Norman 标准布局 [P(Dp), A(2), g(1)] 重排为组合性孪生布局
    [p_a(Dp), p_b(Dp), A(2), g(1)]：
      - 单扰动：p_a = 该基因 one-hot，p_b = 0
      - 双扰动：p_a / p_b = 两个受扰基因 one-hot（拆成两分量，加法组合先验才生效）
    纯向量化（基于稀疏非零索引），不随细胞数放大成 Python 循环。
    """
    n, d = X.shape
    Dp = d - 3                      # P 维度
    P = X[:, :Dp]
    A = X[:, Dp:Dp + 2]
    g = X[:, Dp + 2:Dp + 3]
    pa = np.zeros((n, Dp), dtype=X.dtype)
    pb = np.zeros((n, Dp), dtype=X.dtype)
    row, col = np.nonzero(P)        # 行优先 → row 已升序，同 row 连续
    # 每行第一个非零给 p_a，第二个（若有）给 p_b；用掩码直接写，避免默认 0 与「gene@0」歧义
    is_first = np.empty(len(row), dtype=bool)
    is_first[0] = True
    is_first[1:] = row[1:] != row[:-1]
    pa[row[is_first], col[is_first]] = 1.0
    pb[row[~is_first], col[~is_first]] = 1.0
    return np.hstack([pa, pb, A, g]).astype(np.float64)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
SUBSETS = ["id", "ood_agent", "ood_action", "ood_neuro"]
# 「机制本应外推」的 OOD 子集（用于「疑似仅记忆」判定）：
#   ood_agent : 主体是混杂因子，机制不含主体 → 应外推
#   ood_neuro : 增益为连续特征 → 应插值/外推
#   ood_action: imitation 是真正新机制 → 不强制要求外推（如实报告）
SHOULD_GENERALIZE = {"ood_agent": True, "ood_neuro": True, "ood_action": False}
SUBSET_LABELS = {
    "id": "ID",
    "ood_agent": "OOD-未见主体",
    "ood_action": "OOD-未见动作",
    "ood_neuro": "OOD-新调质",
}


def _resolve_subsets(cfg: dict, split: np.ndarray):
    ev = cfg["eval"]
    subsets = ev.get("subsets") or list(SUBSETS)
    sg = ev.get("should_generalize") or dict(SHOULD_GENERALIZE)
    labels = ev.get("subset_labels") or dict(SUBSET_LABELS)
    present = [s for s in subsets if (split == s).any()]
    return present, sg, labels


def run(cfg: dict, root: str) -> dict:
    data = load_or_generate(cfg, root)
    X, y, split = data["X"], data["y"], data["split"]
    z_comp = data.get("z_comp")
    source = cfg["data"].get("source")

    present, SHOULD_GENERALIZE_RES, SUBSET_LABELS_RES = _resolve_subsets(cfg, split)

    comp = bool(cfg["model"].get("compositional"))
    # Norman 真实数据走「组合性布局」重排：把多热 P 拆成 p_a/p_b 两分量，
    # 否则 CompositionalTwin 会误把单个 P 向量劈成两半（维度/语义全错）。
    # 重排后 X 同时喂给本 run 所有预测器（同输入公平对比）。
    if comp and source == "norman":
        X = norman_to_compositional_X(X)

    train_mask = split == "train"          # 仅训练集用于拟合（避免 ID 测试泄漏）
    Xtr, ytr = X[train_mask], y[train_mask]
    ztr = z_comp[train_mask] if z_comp is not None else None
    str_split = split[train_mask]
    test_masks = {s: (split == s) for s in present}

    if comp:
        predictors: List[Predictor] = [
            CompositionalTwinPredictor(cfg, conformal=bool(cfg["model"].get("conformal", False))),
            # 同组合性输入上跑 VirtualTwin（不带加法先验）→ 隔离「组合先验」本身的贡献
            VirtualTwinPredictor(cfg),
            MLPTwinPredictor(cfg),
            LinearBaseline(),
            KNNBaseline(k=cfg["baselines"]["knn_k"]),
            MeanBaseline(),
        ]
        # P2：在 P1 组合先验基础上加可学习基因对交互头，攻克 ood_action epistasis
        if cfg["model"].get("compositional_interaction"):
            predictors.append(
                CompositionalInteractionPredictor(cfg, conformal=bool(cfg["model"].get("conformal", False)))
            )
    else:
        predictors: List[Predictor] = [
            VirtualTwinPredictor(cfg),
            MLPTwinPredictor(cfg),
            LinearBaseline(),
            KNNBaseline(k=cfg["baselines"]["knn_k"]),
            MeanBaseline(),
        ]

    results: Dict[str, Dict[str, Dict[str, float]]] = {s: {} for s in present}
    cals: Dict[str, Dict[str, object]] = {s: {} for s in present}

    for pred in predictors:
        for s in present:
            cfg["_current_subset"] = s
            m, c = evaluate_predictor(
                pred, Xtr, ytr, X[test_masks[s]], y[test_masks[s]], cfg,
                split_tr=str_split, z_comp=ztr,
            )
            results[s][pred.name] = m
            cals[s][pred.name] = c
    cfg.pop("_current_subset", None)

    flags = []
    delta_thr = cfg["eval"]["memorization_flag_delta"]
    twin_name = "compositional_twin" if comp else "virtual_twin"
    simple_baselines = ["mean", "linear", "knn"]
    # 受评孪生候选：P1 组合先验 + (若启用) P2 交互头
    twin_candidates = [twin_name]
    if comp and cfg["model"].get("compositional_interaction") and "compositional_interaction_twin" in \
            results[present[0]]:
        twin_candidates.append("compositional_interaction_twin")
    for s in present:
        if not SHOULD_GENERALIZE_RES.get(s):
            continue
        for tn in twin_candidates:
            if tn not in results[s]:
                continue
            twin_rmse = results[s][tn]["rmse"]
            best_base = min(results[s][b]["rmse"] for b in simple_baselines)
            twin_hi = results[s][tn]["rmse_hi"]
            best_base_lo = min(results[s][b]["rmse_lo"] for b in simple_baselines)
            delta = best_base - twin_rmse
            significant = delta > delta_thr and twin_hi < best_base_lo
            verdict = ("机制泛化成立（显著优于简单基线）" if significant
                       else "疑似仅记忆（模型未在应外推子集上显著优于简单基线）")
            flags.append({
                "subset": s,
                "twin": tn,
                "delta_vs_best_baseline": round(delta, 4),
                "verdict": verdict,
            })

    cf = counterfactual_demo(Xtr, ytr, cfg, compositional=comp)

    return {
        "config": cfg,
        "subsets": present,
        "subset_labels": SUBSET_LABELS_RES,
        "subset_definitions": data.get("subset_definitions", {}),
        "response_desc": data.get("response_desc", ""),
        "compositional": comp,
        "n_train": int(train_mask.sum()),
        "n_id_test": int((split == "id").sum()),
        "n_total": int(len(split)),
        "results": results,
        "calibration": cals,
        "flags": flags,
        "counterfactual": cf,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def _fmt_ci(m: Dict[str, float]) -> str:
    return f"{m['rmse']:.3f} [{m['rmse_lo']:.3f}, {m['rmse_hi']:.3f}]"


def _cov_get(coverage: Dict[object, float], level: float) -> float:
    """覆盖度查表：兼容内存态（float 键）与 JSON 重载态（str 键）。"""
    if level in coverage:
        return coverage[level]
    return coverage.get(str(level), float("nan"))


def write_report(report: dict, root: str) -> Tuple[str, str]:
    out_dir = os.path.join(root, report["config"]["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    prefix = report["config"]["output"]["prefix"]
    md_path = os.path.join(out_dir, f"{date}-{prefix}.md")
    json_path = os.path.join(out_dir, f"{date}-{prefix}.json")

    subsets = report["subsets"]
    model_order = list(report["results"][subsets[0]].keys())
    labels = {
        "virtual_twin": "虚拟孪生(多尺度)",
        "compositional_twin": "组合性孪生(组合先验)",
        "compositional_interaction_twin": "组合性孪生(P2·交互项)",
        "mlp": "MLP(黑箱)",
        "linear": "线性基线", "knn": "KNN基线", "mean": "均值基线",
    }
    subset_labels = report["subset_labels"]

    lines = []
    src = report["config"].get("data", {}).get("source")
    if src == "norman":
        src_note = "（真实数据：Norman 2019 perturb-seq）"
    elif src == "goai":
        src_note = "（真实数据：GOAI 赛道三·虚拟细胞 yeast 扰动蛋白质组）"
    else:
        src_note = ""
    lines.append(f"# OOD 外推基准评测 · {date}{src_note}\n")
    lines.append(f"- 生成时间(UTC): {report['generated_at']}")
    lines.append(f"- 总细胞数: {report['n_total']}")
    lines.append(f"- 训练样本（拟合用）: {report['n_train']}")
    lines.append(f"- ID 测试样本（in-distribution 评测）: {report['n_id_test']}")
    lines.append(f"- 模型：VirtualTwin（多尺度+集成/新颖度不确定度）vs 简单基线（均值/线性/KNN）+ 黑箱 MLP")
    if src == "norman":
        default_resp = "向量响应 y = 高变异基因面板（K=256 维），RMSE 跨基因聚合"
    elif src == "goai":
        default_resp = "向量响应 y = 蛋白质/基因表达 Δ（对照匹配后 log2），RMSE 跨蛋白聚合"
    else:
        default_resp = "标量响应 y = g·(p·w + mirror_k(action)·nl(p))，单输出；RMSE 跨测试样本聚合"
    resp_desc = report.get("response_desc") or default_resp
    lines.append(f"- 响应 y：{resp_desc}\n")

    lines.append("\n## 1. RMSE（越低越好，含 95% CI）— 记忆 vs 机制泛化\n")
    header = "| 预测器 | " + " | ".join(subset_labels.get(s, s) for s in subsets) + " |"
    sep = "|" + "---|" * (len(subsets) + 1)
    lines.append(header)
    lines.append(sep)
    for mdl in model_order:
        row = f"| {labels[mdl]} |"
        for s in subsets:
            row += f" {_fmt_ci(report['results'][s][mdl])} |"
        lines.append(row)
    lines.append("\n> 各 OOD 子集机制语义见本报告末尾「子集定义」。\n")

    # —— 记忆 vs 机制泛化 双栏对照（项目硬约束：所有预测必须分双栏）——
    sg_conf = report["config"]["eval"].get("should_generalize") or dict(SHOULD_GENERALIZE)
    mech_subsets = [s for s in subsets if sg_conf.get(s)]
    lines.append("\n### 记忆 vs 机制泛化（双栏对照）\n")
    lines.append("> 左栏：in-distribution（记忆，ID 子集）RMSE；右栏：应外推 OOD 子集（机制泛化）RMSE 均值。")
    lines.append("> 二者差额 = 机制证据：若孪生在右栏显著优于简单基线，则说明学到机制而非仅记忆。\n")
    lines.append("| 预测器 | 记忆·ID RMSE [95%CI] | 机制泛化·OOD RMSE [95%CI] | Δ(OOD−ID) |")
    lines.append("|---|---|---|---|")
    for mdl in model_order:
        idm = report["results"]["id"][mdl]
        if mech_subsets:
            mr = float(np.mean([report["results"][s][mdl]["rmse"] for s in mech_subsets]))
            ml = float(np.mean([report["results"][s][mdl]["rmse_lo"] for s in mech_subsets]))
            mh = float(np.mean([report["results"][s][mdl]["rmse_hi"] for s in mech_subsets]))
            mtxt = f"{mr:.3f} [{ml:.3f}, {mh:.3f}]"
            dtxt = f"{mr - idm['rmse']:+.3f}"
        else:
            mtxt, dtxt = "—（无应外推子集）", "—"
        lines.append(f"| {labels[mdl]} | {_fmt_ci(idm)} | {mtxt} | {dtxt} |")

    lines.append("\n## 2. 校准指标（不确定度可靠性）\n")
    lines.append("| 预测器 | 子集 | ECE | 覆盖@0.9 | 覆盖@0.95 |")
    lines.append("|---|---|---|---|---|")
    for s in subsets:
        for mdl in model_order:
            cal = report["calibration"][s][mdl]
            lines.append(
                f"| {labels[mdl]} | {subset_labels.get(s, s)} | {cal['ece']:.3f} | "
                f"{_cov_get(cal['coverage'], 0.9):.3f} | "
                f"{_cov_get(cal['coverage'], 0.95):.3f} |"
            )

    lines.append("\n## 3. 「记忆 vs 机制泛化」判定\n")
    for f in report["flags"]:
        lines.append(f"- **{subset_labels.get(f['subset'], f['subset'])}**: {f['verdict']} （Δ vs 最佳基线 = {f['delta_vs_best_baseline']}）")
    mem = [f for f in report["flags"] if "疑似" in f["verdict"]]
    if mem:
        lines.append(f"\n⚠️ 触发 SOP 阶段 D 误差回流 + TRIZ 修正：{', '.join(subset_labels.get(f['subset'], f['subset']) for f in mem)}")
    else:
        lines.append("\n✅ 当前在「应外推」子集上均显著优于简单基线，未触发仅记忆告警。")

    lines.append("\n## 4. 反事实 do-干预演示（因果头）\n")
    lines.append("> 固定其他条件，对增益施加 do(gain=1.6) / 对动作轴施加 do(双扰动)，观察孪生响应（向量响应取基因均值作读数）。\n")
    lines.append("| 样本 | 基准增益 | 基准预测 | do(gain=1.6)预测 | 缩放比 | do(双扰动)预测 |")
    lines.append("|---|---|---|---|---|---|")
    for r in report["counterfactual"]["rows"]:
        lines.append(
            f"| {r['sample']} | {r['base_gain']} | {r['pred_base']} | "
            f"{r['do_gain_1.6_pred']} | {r['scale_ratio']} | {r['do_action_other_pred']} |"
        )
    lines.append(f"\n因果图：\n```\n{report['counterfactual']['graph_summary']}\n```")

    lines.append("\n## 5. 不确定度说明\n")
    nb = report["config"]["eval"].get("bootstrap_samples", 200)
    lines.append(f"- RMSE 经 {nb} 次 bootstrap 给出 95% 置信区间；覆盖度/ECE 衡量预测区间是否可靠。")
    lines.append("- 虚拟孪生不确定度 = 集成 epistemic + 新颖度(OOD) 合成；在未见条件下新颖度会撑宽区间，缓解黑箱同方差不确定度的过度自信。")
    lines.append("- 所有 OOD 结论均带 CI；若 CI 重叠则视为「不显著」，避免过拟合式误判。")

    lines.append("\n## 6. 子集定义（OOD 语义，诚实标注）\n")
    sdefs = report.get("subset_definitions") or {}
    if sdefs:
        # 数据源自带子集定义（如 GOAI S1/S2/S3）
        for s in subsets:
            lines.append(f"- {sdefs.get(s, f'**{s}**：（未提供定义）')}")
    else:
        if src == "norman":
            # Norman 专属子集语义（与 configs/benchmark_ood_norman.yaml 的 subset_labels 对齐）
            lines.append("- **ID(已见基因)**：已见单扰动基因、已见批次的 held-out 细胞（in-distribution，与训练同分布）。")
            lines.append("- **OOD-未见基因(ood_agent)**：20% 单扰动基因从未在训练出现——真正未见扰动身份 → **标注为「不强制外推」**（如实报告，不要求外推）。")
            lines.append("- **OOD-组合双扰动(ood_action)**：双扰动中 held-out *基因对*（两基因均见过单扰动、但该具体组合从未训练）——**真正的机制组合泛化检验**（须组合两条已见单扰动效应）；标注为「应外推」。训练集已含部分双扰动（可学 epistasis）。")
            lines.append("- **OOD-未见批次(ood_neuro)**：gemgroup=8 批次从未在训练出现——混杂因子 OOD → **标注为「不强制外推」**。")
            lines.append("\n> 数据来源：真实单细胞 CRISPR 扰动数据 Norman, Replogle et al. 2019 (Science), GEO GSE133344 — 公开 perturb-seq。")
        else:
            # 默认（合成镜像神经元）硬编码说明 —— 与 generate_mirror_neuron_dataset 的真实机制一致
            lines.append("- **ID**：动作=self/other、调质=baseline/dopamine、主体=0/1/2 的 held-out 细胞（in-distribution，与训练同分布）。")
            lines.append("- **OOD-未见主体(ood_agent)**：主体=3 从未在训练出现；动作/调质仍为已见。主体在机制中是零均值混杂、不进入特征 → **标注为「应外推」**（孪生应把它当未知混杂处理，不依赖主体身份）。")
            lines.append("- **OOD-未见动作(ood_action)**：动作=imitation 从未在训练出现（训练仅 self/other）。imitation 的镜像增益 mirror_k=0，是**真正的新机制缺口** → 标注为「不强制外推」（如实报告，不要求外推）。")
            lines.append("- **OOD-新调质(ood_neuro)**：调质=serotonin(gain=1.6) 从未在训练出现（训练仅 baseline=1.0/dopamine=1.3）。孪生因果头 z=y/g 把乘法调质变成可学先验 → **标注为「应外推」**（应插值/外推到新增益）。")
            lines.append("\n> 数据来源：合成「扰动→响应」数据（generate_mirror_neuron_dataset），机制 y = g·(p·w + mirror_k(action)·nl(p))，nl(p)=p0·p1+sin(p2)。")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    return md_path, json_path


# --------------------------------------------------------------------------- #
# 可视化（matplotlib, Agg 后端，无 GUI 依赖）
# --------------------------------------------------------------------------- #
def write_figures(report: dict, root: str) -> List[str]:
    """根据 report 绘制 OOD 评测图，返回生成的 PNG 路径列表。

    Fig1: 各子集 RMSE 分组条形图（virtual_twin vs 基线 vs mlp，带 95% CI 误差棒），
          按 should_generalize 着色背景（应外推=暖色 / 不强制=冷色）。
    Fig2: 校准图——经验覆盖率 vs 名义置信度（对角线），virtual_twin 各子集，标 ECE。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 中文字体（Windows 优先微软雅黑/黑体，避免 CJK 缺字形警告）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    out_dir = os.path.join(root, report["config"]["output"]["dir"], "figures")
    os.makedirs(out_dir, exist_ok=True)
    subsets = report["subsets"]
    labels = report["subset_labels"]
    sg = report["config"]["eval"].get("should_generalize") or {}

    results = report["results"]
    models = list(results[subsets[0]].keys())
    simple = [m for m in ("mean", "linear", "knn") if m in models]
    desired_twins = ["compositional_twin", "virtual_twin"]
    order = [m for m in desired_twins if m in models] + simple + \
            [m for m in models if m not in desired_twins and m not in simple]

    paths = []

    # ---- Fig1: RMSE 分组条形 ----
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(subsets))
    n_models = len(order)
    width = 0.8 / n_models
    colors = {"virtual_twin": "#d1495b", "mean": "#6c757d", "linear": "#6c757d",
              "knn": "#6c757d", "mlp": "#2a9d8f"}
    for mi, mdl in enumerate(order):
        rmses, los, his = [], [], []
        for s in subsets:
            r = results[s][mdl]
            rmses.append(r["rmse"])
            los.append(r["rmse"] - r.get("rmse_lo", r["rmse"]))
            his.append(r.get("rmse_hi", r["rmse"]) - r["rmse"])
        ax.bar(x + (mi - n_models / 2 + 0.5) * width, rmses, width,
               yerr=[los, his], capsize=2, label=mdl,
               color=colors.get(mdl, "#457b9d"),
               edgecolor="white", error_kw={"elinewidth": 1})
    for i, s in enumerate(subsets):
        c = "#ffe8cc" if sg.get(s, False) else "#e7f5ff"
        ax.axvspan(i - 0.5, i + 0.5, color=c, alpha=0.5, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(s, s) for s in subsets], rotation=15, ha="right")
    ax.set_ylabel("RMSE（越低越好）")
    ax.set_title("OOD 外推基准 · 各子集 RMSE（误差棒=95% bootstrap CI）\n"
                 "暖色背景=应外推(OOD) · 冷色背景=不强制外推(ID/混杂)")
    ax.legend(ncol=len(order), fontsize=8, loc="upper left")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "fig1_rmse_by_subset.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    paths.append(p1)

    # ---- Fig2: 校准图 ----
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.6, label="理想校准")
    levels_default = [0.5, 0.8, 0.9, 0.95]
    for s in subsets:
        cal = report["calibration"][s].get("virtual_twin", {})
        cov = cal.get("coverage", {})
        if cov:
            levels = sorted(float(k) for k in cov.keys())
        else:
            levels = levels_default
        ys = []
        for lv in levels:
            v = cov.get(lv)
            if v is None:
                v = cov.get(str(lv))
            ys.append(v if v is not None else float("nan"))
        ece = cal.get("ece", float("nan"))
        ax.plot(levels, ys, "o-", label=f"{labels.get(s, s)} (ECE={ece:.3f})")
    ax.set_xlabel("名义置信度（nominal level）")
    ax.set_ylabel("经验覆盖率（empirical coverage）")
    ax.set_title("不确定性校准 · 各子集 virtual_twin\n经验覆盖率 vs 名义置信度（ECE 越小越好）")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "fig2_calibration.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    paths.append(p2)

    print(f"[benchmark_ood] 图表已写出: {paths}")
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    default_root = os.path.dirname(HERE)  # code/ 的父目录 = 项目根
    ap = argparse.ArgumentParser(description="虚拟镜像神经元 OOD 外推基准评测（多尺度孪生版）")
    ap.add_argument("--config", default=os.path.join(default_root, "configs", "benchmark_ood.yaml"))
    ap.add_argument("--root", default=default_root)
    ap.add_argument("--data", default=None, help="真实数据文件路径（CSV/TSV/NPZ/H5AD），覆盖 config 的 data.path")
    ap.add_argument("--quick", action="store_true", help="小数据快速自检")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.data:
        cfg["data"]["path"] = args.data
    if args.quick:
        cfg["data"]["synthetic"]["n_train"] = 400
        cfg["data"]["synthetic"]["n_test"] = 120
        cfg["eval"]["bootstrap_samples"] = 50

    if not os.path.isdir(args.root):
        os.makedirs(args.root, exist_ok=True)

    print(f"[benchmark_ood] 加载配置: {args.config}")
    print(f"[benchmark_ood] 项目根: {args.root}")
    report = run(cfg, args.root)
    md, js = write_report(report, args.root)
    try:
        figs = write_figures(report, args.root)
    except Exception as e:
        print(f"[benchmark_ood] 图表生成跳过（{type(e).__name__}: {e}）")

    print("\n=== 摘要（RMSE，越低越好）===")
    twin_name = "compositional_twin" if report.get("compositional") else "virtual_twin"
    for s in report["subsets"]:
        twin = report["results"][s][twin_name]
        best_base = min(report["results"][s][b]["rmse"] for b in ["mean", "linear", "knn"])
        print(f"  {s:12s}  {twin_name}={twin['rmse']:.3f}  best_baseline={best_base:.3f}")
    print("\n=== 判定 ===")
    for f in report["flags"]:
        print(f"  {f['subset']:12s}  {f['verdict']}")
    print(f"\n报告已写入:\n  {md}\n  {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
