"""P5-A2 — Proper-Bayesian re-implementation of the compositional twin (PC-3 epistemic salvage).

Authorized tool-chain upgrade: torch 2.13 (CPU) + pymc 6.3, isolated in `.venv_p5a2`.
This is the *salvage path* for PC-3 under-scaling after P5-A (numpy SGLD, negative),
P5-B (NCL, double-negative), P5-C (GMM gate, negative) all failed.

Two prongs share one hypothesis: a *proper* posterior over weights yields sigma(x) that is
MONOTONE in distance-to-training-manifold tau(x) (d sigma / d tau > 0) — which numpy-only SGLD
(3 chains, no burn-in/thinning/preconditioning) could not produce.

  (a) torch SGLD (vanilla Welling-Teh form) -> scalable path. Upgrades P5-A's mechanism with:
      burn-in, thinning, multi-chain, grad-clip, weight-decay (RMSprop preconditioning was tried
      but 1/vi exploded on tiny gradients -> deferred; vanilla is stable and already recovers H1-H4).
  (b) pymc NUTS (stratified subset) -> gold-standard posterior ORACLE. Validates that a *proper*
      posterior (not just Langevin noise) recovers under-scaling — the crux of the PC-3 fix.

SELF-CONTAINED: does NOT import compositional.py / .pylibs (the venv's numpy 2.4.6 would clash
with .pylibs' older numpy). torch/pymc are lazy-imported inside methods so this module is
import-safe in the base managed runtime.

Architectural simplification (documented): the compositional twin is per-gene linear regressors;
for the spike we use ONE shared MLP (hidden layers) mapping [pa, pb, act, g-context] -> 4 outputs
as a Bayesian stand-in. Per-gene BNN wiring is the next step after the method is validated.
"""
import numpy as np


class ProperBayesTwin:
    def __init__(self, hidden=(64, 32), noise_var=1.0, prior_lambda=1.0, seed=0):
        self.hidden = tuple(hidden)
        self.noise_var = float(noise_var)
        self.prior_lambda = float(prior_lambda)
        self.seed = int(seed)
        self._fitted = False
        self.method_ = None

    # ------------------------------------------------------------------ #
    # (a) torch preconditioned SGLD
    # ------------------------------------------------------------------ #
    def fit_torch_sgld(self, X, Y, n_chains=4, n_iter=2000, burnin=500, thin=5,
                       lr=5e-4, T=0.1, clip_grad=1.0, wdecay=1e-3, device="cpu"):
        torch = __import__("torch")
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        Yt = torch.tensor(Y, dtype=torch.float32, device=device)
        Din, Dout = X.shape[1], Y.shape[1]
        dims = [Din] + list(self.hidden) + [Dout]
        self._mlp_dims = dims

        def make_params(rng):
            params = []
            for i in range(len(dims) - 1):
                W = torch.empty(dims[i], dims[i + 1], device=device)
                torch.nn.init.xavier_normal_(W, gain=1.0)
                W.requires_grad_(True)
                b = torch.zeros(dims[i + 1], device=device, requires_grad=True)
                params += [W, b]
            return params

        def forward(params, x):
            h = x
            for i in range(0, len(params) - 2, 2):
                h = torch.relu(h @ params[i] + params[i + 1])
            return h @ params[-2] + params[-1]

        kept_per_chain = []
        for c in range(n_chains):
            torch.manual_seed(self.seed + c * 1000)
            np.random.seed(self.seed + c * 1000)
            params = make_params(None)
            chain_samples = []
            for t in range(n_iter):
                params_det = [p.detach().clone().requires_grad_(True) for p in params]
                out = forward(params_det, Xt)
                nll = 0.5 * ((out - Yt) ** 2).sum() / self.noise_var
                logprior = -0.5 * self.prior_lambda * sum((p ** 2).sum() for p in params_det)
                loss = nll - logprior
                grads = torch.autograd.grad(loss, params_det, create_graph=False)
                gnorm = float(np.sqrt(sum(g.detach().pow(2).sum().item() for g in grads)))
                scale = 1.0
                if gnorm > clip_grad and gnorm > 0:
                    scale = clip_grad / gnorm
                new_params = []
                for p, g in zip(params_det, grads):
                    g = g.detach() * scale
                    noise = torch.randn_like(p) * float(np.sqrt(2.0 * lr * T))
                    p_new = p - 0.5 * lr * (g + self.prior_lambda * p) + noise
                    if not torch.isfinite(p_new).all():
                        p_new = p.detach().clone()       # NaN/Inf guard: skip degenerate update
                    new_params.append(p_new)
                params = new_params
                if t >= burnin and (t - burnin) % thin == 0:
                    chain_samples.append([p.detach().clone() for p in params])
            kept_per_chain.append(chain_samples)
        self._sgld_chains = kept_per_chain
        self._fitted = True
        self.method_ = "torch_sgld"
        return self

    def _sgld_predict(self, X):
        torch = __import__("torch")
        Xt = torch.tensor(X, dtype=torch.float32)
        ensemble = [s for chain in self._sgld_chains for s in chain]
        fs = []
        for sample in ensemble:
            with torch.no_grad():
                h = Xt
                for i in range(0, len(sample) - 2, 2):
                    h = torch.relu(h @ sample[i] + sample[i + 1])
                out = h @ sample[-2] + sample[-1]
            fs.append(out.numpy())
        fs = np.stack(fs, axis=0)  # (S, N, Dout)
        return fs.mean(0), fs.std(0)

    # ------------------------------------------------------------------ #
    # (b) pymc NUTS oracle (stratified subset)
    # ------------------------------------------------------------------ #
    def fit_pymc_nuts(self, X, Y, draws=300, tune=300, chains=2):
        pm = __import__("pymc")
        import pytensor.tensor as pt
        Din, Dout = X.shape[1], Y.shape[1]
        dims = [Din] + list(self.hidden) + [Dout]
        self._mlp_dims = dims
        with pm.Model() as model:
            Xd = pm.Data("X_data", X.astype("float32"))  # strict=False by default -> pm.set_data works
            weights, biases = [], []
            for i in range(len(dims) - 1):
                Wi = pm.Normal(f"W{i}", 0.0, 1.0 / np.sqrt(self.prior_lambda),
                               shape=(dims[i], dims[i + 1]))
                bi = pm.Normal(f"b{i}", 0.0, 1.0, shape=(dims[i + 1],))
                weights.append(Wi)
                biases.append(bi)
            h = Xd
            for i in range(len(dims) - 2):
                z = h @ weights[i] + biases[i]
                h = pt.switch(z > 0, z, 0.0)  # ReLU (avoids pytensor.tensor.nnet import)
            mu = h @ weights[-1] + biases[-1]
            pm.Normal("y_obs", mu=mu, sigma=float(np.sqrt(self.noise_var)),
                      observed=Y.astype("float32"))
            self._trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                                    target_accept=0.95, max_treedepth=15,
                                    progressbar=False, random_seed=12345)
            self._model = model
        self._fitted = True
        self.method_ = "pymc_nuts"
        return self

    def _nuts_predict(self, X):
        # Manual posterior-predictive: draw weights from the NUTS trace and
        # forward the MLP in numpy. Avoids sample_posterior_predictive's
        # compiled-shape incompatibility when X rows differ from training.
        post = self._trace.posterior
        dims = self._mlp_dims
        L = len(dims) - 1
        W, b = [], []
        for i in range(L):
            w = post[f"W{i}"].values   # (chain, draw, Di, Di1)
            bb = post[f"b{i}"].values  # (chain, draw, Di1)
            W.append(w.reshape(-1, *w.shape[2:]))   # (S, Di, Di1)
            b.append(bb.reshape(-1, *bb.shape[2:]))  # (S, Di1)
        S = W[0].shape[0]
        Xn = X.astype("float64")
        mus = np.empty((S, Xn.shape[0], dims[-1]))
        for s in range(S):
            h = Xn
            for i in range(L - 1):
                h = np.maximum(0.0, h @ W[i][s] + b[i][s])
            mus[s] = h @ W[L - 1][s] + b[L - 1][s]
        return mus.mean(0), mus.std(0)

    # ------------------------------------------------------------------ #
    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("fit_* not called")
        if self.method_ == "torch_sgld":
            return self._sgld_predict(X)
        if self.method_ == "pymc_nuts":
            return self._nuts_predict(X)
        raise RuntimeError(self.method_)
