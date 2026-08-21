"""P5-A2 · per-gene Bayesian LINEAR twin (real GOAI data, scalable structure).

This is the *proper* per-gene structure the compositional twin was always meant to be
(one Bayesian linear regressor per protein), vectorized so a single SGLD chain covers
all K proteins at once via the weight matrix W (D x K) + bias b (K,).

Why not the shared MLP from compositional_p5a2_bayes.py? That spike used a shared MLP
with 4 outputs on SYNTHETIC data. Real GOAI has Kp=5243 protein outputs -> a shared
MLP with 5243 outputs is intractable for SGLD/NUTS. Per-gene linear (the original
compositional twin design) scales: 5243 independent Bayesian linear regressions, each
D-dim, vectorized as one (D x K) SGLD run.

Two prongs (same as P5-A2 spike):
  (a) torch SGLD (vanilla Welling-Teh, multi-chain/burn-in/thin) -> scalable.
  (b) pymc NUTS (subset of genes, ORACLE) -> validates the posterior is *proper*
      (not just Langevin noise), the PC-3 salvage crux.

SELF-CONTAINED: torch lazy-imported; does not touch .pylibs numpy. Import-safe in base runtime.
"""
import numpy as np


class PerGeneBayesTwin:
    def __init__(self, noise_var=1.0, prior_lambda=1.0, seed=0):
        self.noise_var = float(noise_var)
        self.prior_lambda = float(prior_lambda)
        self.seed = int(seed)
        self._fitted = False
        self.method_ = None

    # ------------------------------------------------------------------ #
    def fit_torch_sgld(self, X, Y, n_chains=3, n_iter=600, burnin=150, thin=10,
                       lr=5e-4, T=0.1, clip_grad=1.0, wdecay=1e-3, device="cpu"):
        torch = __import__("torch")
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        Yt = torch.tensor(Y, dtype=torch.float32, device=device)
        D = X.shape[1]
        K = Y.shape[1]

        def make_params():
            W = torch.empty(D, K, device=device)
            torch.nn.init.xavier_normal_(W, gain=1.0)
            W.requires_grad_(True)
            b = torch.zeros(K, device=device, requires_grad=True)
            return W, b

        kept = []
        for c in range(n_chains):
            torch.manual_seed(self.seed + c * 1000)
            np.random.seed(self.seed + c * 1000)
            W, b = make_params()
            chain = []
            for t in range(n_iter):
                Wd = W.detach().clone().requires_grad_(True)
                bd = b.detach().clone().requires_grad_(True)
                mu = Xt @ Wd + bd
                nll = 0.5 * ((mu - Yt) ** 2).sum() / self.noise_var
                logprior = -0.5 * self.prior_lambda * (Wd.pow(2).sum() + bd.pow(2).sum())
                loss = nll - logprior
                gW, gb = torch.autograd.grad(loss, [Wd, bd], create_graph=False)
                gnorm = float(torch.sqrt(gW.pow(2).sum() + gb.pow(2).sum()).item())
                scale = 1.0
                if gnorm > clip_grad and gnorm > 0:
                    scale = clip_grad / gnorm
                gW = gW * scale
                gb = gb * scale
                nW = torch.randn_like(Wd) * float(np.sqrt(2.0 * lr * T))
                nb = torch.randn_like(bd) * float(np.sqrt(2.0 * lr * T))
                Wn = Wd - 0.5 * lr * (gW + self.prior_lambda * Wd) + nW
                bn = bd - 0.5 * lr * (gb + self.prior_lambda * bd) + nb
                if not (torch.isfinite(Wn).all() and torch.isfinite(bn).all()):
                    Wn, bn = Wd.detach().clone(), bd.detach().clone()
                W, b = Wn, bn
                if t >= burnin and (t - burnin) % thin == 0:
                    chain.append((W.detach().clone(), b.detach().clone()))
            kept.append(chain)
        self._chains = kept
        self._fitted = True
        self.method_ = "torch_sgld"
        return self

    def _sgld_predict(self, X):
        torch = __import__("torch")
        Xt = torch.tensor(X, dtype=torch.float32)
        samples = [s for ch in self._chains for s in ch]
        # 增量 Welford 累积 (N,K) 均值/方差，避免堆叠全部后验样本爆内存（全量 K=5243 时关键）
        mean = None
        m2 = None
        count = 0
        for W, b in samples:
            with torch.no_grad():
                mu = (Xt @ W + b).numpy().astype(np.float64)   # (N, K)
            count += 1
            if mean is None:
                mean = mu.copy()
                m2 = np.zeros_like(mu)
            else:
                delta = mu - mean
                mean += delta / count
                m2 += delta * (mu - mean)
        var = m2 / max(count - 1, 1)
        return mean, np.sqrt(np.maximum(var, 0.0))

    # ------------------------------------------------------------------ #
    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("fit_* not called")
        if self.method_ == "torch_sgld":
            return self._sgld_predict(X)
        raise RuntimeError(self.method_)
