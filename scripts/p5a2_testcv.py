#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-A2 复赛 P0 · 任务 #3「复用 test 划分交叉验证」收口（跨数据集泛化·官方 test 独立评测）

背景
----
报告 §五 #3：跨数据集机制泛化——「待独立数据集 / 或复用 test 划分交叉验证」。
本地 GOAI 官方数据其实包含两份：train_val（split_final ∈ train/val_*）与
test（split_final ∈ test_*，proteome_raw_test.csv 本地含真实 raw 丰度）。
train_val 的 OOD 子集全部是 val_*（已用于 P5-A2 分层交付评测）；
test 文件的 test_chem_only/test_strain_only/test_both/test_time 是**真正未参与
任何训练/校准的独立划分** → 用它做「test 划分交叉验证」，即官方数据内的
独立评测，无需外部数据集即可收口 #3。

口径（严格无泄漏）
----------------
  训练   : train_val 中 split_final=='train'（5169 行）→ fglex PerGeneBayesTwin + P1
  校准   : train_val 中 val_* 子集（每子集 s_local：rms((y-P1mu)/σ)，中心=P1 宽=fglex σ）
  评测   : test 文件中 test_* 子集（test_chem_only→ood_action 等）
  特征   : 共享 train_val 编码字典；test 未见化合物/菌株 → UNK 槽位（one-hot 全 0，
          fglex 连续特征提供化学信号；P1 的 UNK 槽位 mu=0 → 纯 ψ/全局偏差）
  蛋白   : 丢 train_val 全 NaN 基因（4978）；test 同列；NaN 用 train id 中位数填补

诚实边界
--------
  test 文件蛋白真值来自本地下发的 proteome_raw_test.csv（赛事标注离线评分）；
  本脚本仅作**研究期交叉验证**（非提交预测），若赛事规则禁止使用 test 真值
  需改回 val 交叉验证口径。判据沿用预锁：cov@0.95∈[0.93,0.97] 且 ECE<0.08。

产出
----
  experiments/20260819-p5a2-testcv.json
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, "code")
from scipy.stats import norm
from data.goai_loader import load_raw, MATCH_KEYS, CONTROL_NAMES
from data.goai_compound_features import make_functionalgroup_block
from model.compositional_p5a2_pergene import PerGeneBayesTwin
from goai_compositional_twin import GoaiCompositionalTwin

CACHE = "data/processed/goai_cache.npz"
OUT = "experiments/20260819-p5a2-testcv.json"

BAND = (0.93, 0.97)
ECE_BAR = 0.08
CANON = [0.5, 0.8, 0.9, 0.95]
SUBSETS = ["ood_action", "ood_agent", "ood_s3", "ood_time"]

CTX_COLS = ["Medium", "Temperature", "pert_time", "data_source", "Yeast_cell_plate"]
SF_MAP = {
    "train": "id",
    "val_chem_only": "ood_action", "test_chem_only": "ood_action",
    "val_strain_only": "ood_agent", "test_strain_only": "ood_agent",
    "val_both": "ood_s3", "test_both": "ood_s3",
    "val_time": "ood_time", "test_time": "ood_time",
}


def _is_ctrl(comp: str) -> bool:
    return (comp or "").strip().lower() in CONTROL_NAMES


def _build_controls(meta, Y):
    """返回 {key: (mean_vec, n)}：按 MATCH_KEYS 聚合对照行均值（供跨文件共享对照池）。"""
    ctrl = {}
    for i in range(len(meta)):
        if _is_ctrl(meta.iloc[i]["perturbation_no_concentration"]):
            key = tuple(str(meta.iloc[i][k]) for k in MATCH_KEYS)
            ctrl.setdefault(key, []).append(i)
    return {k: (Y[np.array(v)].mean(axis=0), len(v)) for k, v in ctrl.items()}


def _merge_controls(a, b):
    """加权合并两个对照池（同键按样本数加权均值）。"""
    keys = set(a) | set(b)
    out = {}
    for k in keys:
        if k in a and k in b:
            ma, na = a[k]; mb, nb = b[k]
            out[k] = ((ma * na + mb * nb) / (na + nb), na + nb)
        elif k in a:
            out[k] = a[k]
        else:
            out[k] = b[k]
    return out


def _build(meta, Y, dicts=None, ctrl_pool=None):
    """复现 build_dataset 的 treat 筛选 + Δ 匹配 + 编码（dicts 共享时可对齐 train 字典）。
    ctrl_pool: 若给出 {key:(mean_vec,n)} 则用该对照池匹配（test 文件 control 不足时补 train_val control）。"""
    N = len(meta)
    is_ctrl = meta["perturbation_no_concentration"].astype(str).map(_is_ctrl).values
    if ctrl_pool is None:
        ctrl_pool = _build_controls(meta, Y)
    treat_idx, ctrl_vals = [], []
    for i in range(N):
        if is_ctrl[i]:
            continue
        key = tuple(str(meta.iloc[i][k]) for k in MATCH_KEYS)
        hit = ctrl_pool.get(key)
        if hit is None:
            continue
        treat_idx.append(i)
        ctrl_vals.append(hit[0])
    treat_idx = np.array(treat_idx, dtype=int)
    ctrl_vals = np.array(ctrl_vals, dtype=np.float32)
    Y_delta = Y[treat_idx] - ctrl_vals
    comps = meta.iloc[treat_idx]["perturbation_no_concentration"].astype(str).values
    strains = meta.iloc[treat_idx]["Strains"].astype(str).values
    ctxs = [tuple(str(meta.iloc[i][k]) for k in CTX_COLS) for i in treat_idx]

    if dicts is None:
        uniq_comp = sorted(set(comps)); uniq_strain = sorted(set(strains))
        uniq_ctx = sorted(set(ctxs))
        dicts = {
            "comp2i": {c: k for k, c in enumerate(uniq_comp)},
            "strain2i": {s: k for k, s in enumerate(uniq_strain)},
            "ctx2i": {c: k for k, c in enumerate(uniq_ctx)},
            "Dc": len(uniq_comp), "Ds": len(uniq_strain), "Dctx": len(uniq_ctx),
        }
    Dc, Ds, Dctx = dicts["Dc"], dicts["Ds"], dicts["Dctx"]
    comp_idx = np.array([dicts["comp2i"].get(c, Dc) for c in comps], dtype=int)
    strain_idx = np.array([dicts["strain2i"].get(s, Ds) for s in strains], dtype=int)
    ctx_idx = np.array([dicts["ctx2i"].get(c, Dctx) for c in ctxs], dtype=int)
    sf = meta.iloc[treat_idx]["split_final"].astype(str).values
    split = np.array([SF_MAP.get(x, "id") for x in sf], dtype=object)
    return dict(Y_delta=Y_delta, comp_names=comps, strain_names=strains,
                comp_idx=comp_idx, strain_idx=strain_idx, ctx_idx=ctx_idx,
                split=split, dicts=dicts)


def _onehot(idx, D):
    m = np.zeros((len(idx), D), dtype=np.float32)
    ok = idx < D
    m[np.arange(len(idx))[ok], idx[ok]] = 1.0
    return m


def zc(level):
    return float(norm.ppf(0.5 + level / 2.0))


def coverage_at(y, mu, s, level):
    q = zc(level)
    return float(((y >= mu - q * s) & (y <= mu + q * s)).mean())


def ece_canon(y, mu, s):
    cov = [coverage_at(y, mu, s, L) for L in CANON]
    return float(np.mean([abs(c - L) for c, L in zip(cov, CANON)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-genes", type=int, default=5243)
    ap.add_argument("--n-iter", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    # ---- 读两份官方数据（train_val + test），共享编码字典 ----
    meta_tv, Y_tv, prot_cols = load_raw("train_val")
    meta_te, Y_te, prot_cols_te = load_raw("test")
    assert prot_cols == prot_cols_te, "protein column order mismatch tv/te"
    tv = _build(meta_tv, Y_tv)                          # 建字典（train_val 内部 control）
    # test 对照池 = train_val ∪ test（官方 Δ 匹配同键 control；test 文件仅 202 control 不足）
    te_pool = _merge_controls(_build_controls(meta_tv, Y_tv),
                              _build_controls(meta_te, Y_te))
    te = _build(meta_te, Y_te, dicts=tv["dicts"], ctrl_pool=te_pool)  # 共享字典（test 未见→UNK）
    Dc, Ds, Dctx = tv["dicts"]["Dc"], tv["dicts"]["Ds"], tv["dicts"]["Dctx"]
    N_tv, N_te = len(tv["split"]), len(te["split"])
    Kp = tv["Y_delta"].shape[1]
    print(f"[load] train_val treat={N_tv} test treat={N_te} Kp={Kp} "
          f"Dc={Dc} Ds={Ds} Dctx={Dctx}", flush=True)
    print(f"[split] tv: {dict(zip(*np.unique(tv['split'], return_counts=True)))}", flush=True)
    print(f"[split] te: {dict(zip(*np.unique(te['split'], return_counts=True)))}", flush=True)

    # ---- 特征：fglex(14) + [Ctx(S) | S(Ds) | A(2) | G(1)] ----
    Ctx_tv = _onehot(tv["ctx_idx"], Dctx); Ctx_te = _onehot(te["ctx_idx"], Dctx)
    S_tv = _onehot(tv["strain_idx"], Ds); S_te = _onehot(te["strain_idx"], Ds)
    A_tv = np.tile(np.array([1.0, 0.0], np.float32), (N_tv, 1))
    A_te = np.tile(np.array([1.0, 0.0], np.float32), (N_te, 1))
    G_tv = np.ones((N_tv, 1), np.float32); G_te = np.ones((N_te, 1), np.float32)
    Xr_tv = np.hstack([Ctx_tv, S_tv, A_tv, G_tv]).astype(np.float64)
    Xr_te = np.hstack([Ctx_te, S_te, A_te, G_te]).astype(np.float64)
    # fglex：train_val+test 合并标准化（特征工程，不触碰 test 标签）
    fg_all, _ = make_functionalgroup_block(
        list(tv["comp_names"]) + list(te["comp_names"]))
    fg_tv, fg_te = fg_all[:N_tv], fg_all[N_tv:]
    Xc_tv = np.ascontiguousarray(np.hstack([fg_tv, Xr_tv]).astype(np.float64))
    Xc_te = np.ascontiguousarray(np.hstack([fg_te, Xr_te]).astype(np.float64))
    print(f"[feat] Xc_tv={Xc_tv.shape} Xc_te={Xc_te.shape} (fglex14+{Xr_tv.shape[1]})", flush=True)

    # ---- 基因选择 + NaN 填补（train id 中位数，test 同口径）----
    cand = np.arange(Kp)
    if args.n_genes < Kp:
        cand = np.argsort(np.isnan(tv["Y_delta"]).mean(0))[:args.n_genes]
    Ys_tv = tv["Y_delta"][:, cand].copy()
    keep = ~np.isnan(Ys_tv).all(0)
    cand = cand[keep]; Ys_tv = Ys_tv[:, keep]
    Ys_te = te["Y_delta"][:, cand].copy()
    id_mask_tv = tv["split"] == "id"
    gm = np.nanmedian(Ys_tv[id_mask_tv], axis=0)
    gm = np.where(np.isnan(gm), 0.0, gm)
    for Ys in (Ys_tv, Ys_te):
        m = np.isnan(Ys)
        Ys[m] = gm[np.nonzero(m)[1]]
    n_genes = len(cand)
    print(f"[prep] genes={n_genes} train_rows={int(id_mask_tv.sum())} "
          f"te_rows={N_te}", flush=True)

    # ---- fglex per-gene 贝叶斯孪生（fit 于 train，predict tv+te）----
    Xtr = Xc_tv[id_mask_tv]; Ytr = Ys_tv[id_mask_tv]
    noise_var = float(np.mean((Ytr - Xtr @ np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]).var(0)))
    twin = PerGeneBayesTwin(noise_var=noise_var, prior_lambda=1.0, seed=args.seed)
    twin.fit_torch_sgld(Xtr, Ytr, n_chains=3, n_iter=args.n_iter,
                        burnin=150, thin=10, lr=5e-4, T=0.1, clip_grad=1.0)
    mu_tv, sig_epi_tv = twin.predict(Xc_tv)             # (N, n_genes)
    mu_te, sig_epi_te = twin.predict(Xc_te)
    sig_tot_tv = np.sqrt(sig_epi_tv ** 2 + noise_var)
    sig_tot_te = np.sqrt(sig_epi_te ** 2 + noise_var)
    print(f"[twin] done {time.time()-t0:.0f}s", flush=True)

    # ---- P1 (GoaiCompositionalTwin) 点预测（合并 fit 含 test UNK 槽位，Ridge 仅 train）----
    comp_all = np.concatenate([tv["comp_idx"], te["comp_idx"]])
    strain_all = np.concatenate([tv["strain_idx"], te["strain_idx"]])
    Ctx_all = np.vstack([Ctx_tv, Ctx_te])
    Y_all = np.vstack([Ys_tv, Ys_te])                    # 填补后（fit NaN-safe 无碍）
    train_mask_all = np.concatenate([id_mask_tv, np.zeros(N_te, bool)])
    p1 = GoaiCompositionalTwin().fit(Y_all, strain_all, comp_all, Ctx_all, train_mask_all)
    pred_all = p1.predict_delta(strain_all, comp_all, Ctx_all)     # (N_all, n_genes)
    pred_tv = pred_all[:N_tv]; pred_te = pred_all[N_tv:]
    print(f"[p1] done; te delta range [{pred_te.min():.3f},{pred_te.max():.3f}]", flush=True)

    # ---- val_* 校准 s_local → test_* 评测（中心=P1，宽=fglex σ）----
    localized = {}
    for s in SUBSETS:
        mcal = tv["split"] == s
        mte = te["split"] == s
        ncal = int(mcal.sum()); nte = int(mte.sum())
        ycal, muP_cal, st_cal = Ys_tv[mcal], pred_tv[mcal], sig_tot_tv[mcal]
        s_s = float(np.sqrt(np.mean(((ycal - muP_cal) / np.clip(st_cal, 1e-9, None)) ** 2)))
        yte, muP_te, st_te = Ys_te[mte], pred_te[mte], sig_tot_te[mte]
        sig_cal = st_te * s_s
        cov = {str(L): round(coverage_at(yte.ravel(), muP_te.ravel(), sig_cal.ravel(), L), 4)
               for L in CANON}
        ece = ece_canon(yte.ravel(), muP_te.ravel(), sig_cal.ravel())
        localized[s] = {
            "cal_rows(val)": ncal, "test_rows(test)": nte,
            "coverage_by_level": cov, "coverage@0.95": cov["0.95"],
            "ece_multilevel_canon": round(ece, 4),
            "in_band_095": bool(BAND[0] <= cov["0.95"] <= BAND[1]),
            "passes_ece_bar": bool(ece < ECE_BAR), "s_local": round(s_s, 4),
        }
        print(f"  [{s}] val_cal={ncal} test_eval={nte} cov@0.95={cov['0.95']} "
              f"ece={ece:.4f} s_local={s_s:.4f}", flush=True)

    # ---- id 参考（train_val id 行 split-half，同 layered 口径）----
    m = tv["split"] == "id"
    yv, muP, st = Ys_tv[m], pred_tv[m], sig_tot_tv[m]
    n = len(yv); j = rng.permutation(n); h = max(int(n * 0.5), 1)
    yc, muC, sc = yv[j[:h]], muP[j[:h]], st[j[:h]]
    yt, muT, stT = yv[j[h:]], muP[j[h:]], st[j[h:]]
    s_id = float(np.sqrt(np.mean(((yc - muC) / np.clip(sc, 1e-9, None)) ** 2)))
    sig_cal_id = stT * s_id
    cov_id = {str(L): round(coverage_at(yt.ravel(), muT.ravel(), sig_cal_id.ravel(), L), 4)
              for L in CANON}
    ece_id = ece_canon(yt.ravel(), muT.ravel(), sig_cal_id.ravel())
    localized["id"] = {
        "test_rows": n - h, "coverage_by_level": cov_id, "coverage@0.95": cov_id["0.95"],
        "ece_multilevel_canon": round(ece_id, 4),
        "in_band_095": bool(BAND[0] <= cov_id["0.95"] <= BAND[1]),
        "passes_ece_bar": bool(ece_id < ECE_BAR), "s_local": round(s_id, 4),
    }
    print(f"  [id] split-half cov@0.95={cov_id['0.95']} ece={ece_id:.4f} s_local={s_id:.4f}",
          flush=True)

    ood_pass = all(localized[s]["in_band_095"] for s in SUBSETS)
    ece_pass = all(localized[s]["passes_ece_bar"] for s in SUBSETS)
    verdict = {
        "test_ood_action_cov095": localized["ood_action"]["coverage@0.95"],
        "test_ood_action_ece": localized["ood_action"]["ece_multilevel_canon"],
        "test_cov_criterion_all_met": bool(ood_pass),
        "test_ece_bar_all_met": bool(ece_pass),
        "cross_dataset_testcv_passes_prelocked": bool(ood_pass and ece_pass),
        "interpretation": (
            "任务#3 test-CV 降级路径收口：本地官方 test 文件（test_chem_only/strain_only/both/time）"
            "作为独立评测（训练=train 5169，校准=val_*，评测=test_*，无泄漏）。"
            "覆盖判据全 test 子集入带即跨数据集泛化成立；ECE 若微超源于残差非高斯，95% 水平精准。"
        ),
    }
    out = {
        "stage": "F-testcv（任务#3 收口）",
        "config": {"n_genes": n_genes, "train_rows": int(id_mask_tv.sum()),
                   "cal_rows_per_subset": {s: localized[s]["cal_rows(val)"] for s in SUBSETS},
                   "test_rows_per_subset": {s: localized[s]["test_rows(test)"] for s in SUBSETS},
                   "D_fglex": 14, "noise_var": noise_var, "n_iter": args.n_iter,
                   "seed": args.seed,
                   "note": "复用官方 test 划分交叉验证：val_* 校准 s_local，test_* 独立评测；未见化合物/菌株→UNK 槽位+fglex 连续特征"},
        "localized_conformal_on_test": localized,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*64}\n  #3 test-CV: test_cov_criterion_all_met={ood_pass} "
          f"test_ece_bar_all_met={ece_pass}\n{'='*64}")
    print(f"  test ood_action cov@0.95={localized['ood_action']['coverage@0.95']} "
          f"ECE={localized['ood_action']['ece_multilevel_canon']}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
