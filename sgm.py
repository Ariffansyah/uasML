"""Diffusion-model-assisted supervised learning of generative models (Liu et al., JMLMC 5(1), 2024).

data X --(training-free score, Eqs. 11-12)--> reverse ODE (Eq. 13) --> labeled pairs (y, x)
       --> MLP x = F(y) trained with plain MSE (Algorithm 1). Sampling afterwards is one forward pass.

Schedule follows the authors' code: alpha_t = 1 - 0.99t, beta_t^2 = 0.01 + 0.99t
(the paper's alpha_t = 1 - t, beta_t^2 = t, nudged so b(t) and 1/beta_t^2 stay finite at t = 1 and t = 0).
"""
import numpy as np
import torch
import torch.nn as nn

S = 0.99
def alpha(t): return 1 - S * t
def beta2(t): return 1 - S + S * t
def b(t): return -S / alpha(t)                    # d log(alpha_t) / dt, Eq. (4)
def sigma2(t): return S - 2 * b(t) * beta2(t)     # d beta_t^2 / dt - 2 b(t) beta_t^2, Eq. (4)


def score(z, t, x0):
    """Monte Carlo score estimate, Eqs. (11)-(12), using reference data batch x0.
    softmax == Q(z|x_n) / sum_m Q(z|x_m), but without exp() underflowing to 0/0 at small t."""
    a, v = alpha(t), beta2(t)
    ax0 = a * x0
    # -|z - a x_n|^2 / 2v, minus the per-row constant -|z|^2 / 2v that softmax ignores: one matmul, no cdist
    w = torch.softmax((z @ ax0.T - 0.5 * ax0.square().sum(1)) / v, dim=1)
    return -(z - w @ ax0) / v                     # = sum_n w_n * -(z - a x_n) / v, since sum_n w_n = 1


@torch.no_grad()
def ode(z, x0, steps, path=None):
    """Reverse-time ODE (13) from t=1 (Gaussian) to t=0 (data), explicit Euler. Appends every state to `path` if given."""
    dt = 1 / steps
    for i in range(steps):
        t = 1 - i * dt
        z = z - dt * (b(t) * z - 0.5 * sigma2(t) * score(z, t, x0))
        if path is not None: path.append(z)
    return z


@torch.no_grad()
def make_labels(x, m, steps=500, n_ref=1000, chunk=5000, seed=0, progress=None):
    """Algorithm 1, lines 3-7: draw m Gaussian y's and push them through the ODE.
    Each chunk of trajectories uses its own random reference batch of n_ref data points (as the authors' code)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(m, x.shape[1], generator=g)
    out = []
    for k, z in enumerate(y.split(chunk)):
        x0 = x if len(x) <= n_ref else x[torch.randperm(len(x), generator=g)[:n_ref]]
        out.append(ode(z, x0, steps))
        if progress: progress(k + 1, -(-m // chunk))
    return y, torch.cat(out)


def mlp(d, hidden, layers):
    mods, w = [], d
    for _ in range(layers):
        mods += [nn.Linear(w, hidden), nn.Tanh()]
        w = hidden
    return nn.Sequential(*mods, nn.Linear(w, d))


def fit(y, x, hidden, layers, epochs, lr, batch=None, progress=None):
    """Algorithm 1, line 8: supervised MSE regression x = F(y). batch=None -> full batch."""
    net = mlp(y.shape[1], hidden, layers)
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        for idx in torch.randperm(len(y)).split(batch or len(y)):
            loss = (net(y[idx]) - x[idx]).square().sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if progress: progress(ep + 1, epochs, loss.item())
    return net


def synthesize(arr, n_out, m=20_000, steps=200, hidden=512, epochs=100, progress=None):
    """Tabular use (app.py, balance_experiment.py): learn the rows of `arr` (n x d) and return
    (n_out synthetic rows, z, log_cols). z maps rows into the space the generator works in, for distance/KL checks.
    Positive columns with a long right tail (skew > 1) are modeled as log(x): on the raw scale the generator spills
    mass below the minimum, clipping piles it up there, and the rare large values are lost."""
    arr = np.asarray(arr, np.float64)
    skew = ((arr - arr.mean(0)) ** 3).mean(0) / np.maximum(arr.std(0), 1e-12) ** 3
    logc = (arr.min(0) > 0) & (skew > 1)

    def fwd(a):
        a = np.array(a, np.float64)
        a[:, logc] = np.log(a[:, logc])
        return a

    mu, sd = fwd(arr).mean(0), fwd(arr).std(0)
    sd[sd == 0] = 1
    z = lambda a: torch.from_numpy(((fwd(a) - mu) / sd).astype(np.float32))
    say = progress or (lambda frac, text: None)
    y, lab = make_labels(z(arr), m, steps, progress=lambda k, n: say(k / n, f"Diffusion labeling: chunk {k}/{n}"))
    net = fit(y, lab, hidden, 2, epochs, 2e-3, 1024,
              progress=lambda e, n, l: say(e / n, f"Training generator: epoch {e}/{n}, MSE {l:.4f}"))
    with torch.no_grad():
        out = net(torch.randn(n_out, arr.shape[1])).double().numpy() * sd + mu
    out[:, logc] = np.exp(np.minimum(out[:, logc], 700))   # float64 exp overflows past ~709
    out = np.clip(out, arr.min(0), arr.max(0))
    ints = (arr == np.round(arr)).all(0)
    out[:, ints] = np.round(out[:, ints])
    return out, z, logc


if __name__ == "__main__":
    torch.manual_seed(0)
    # 1) score check: data ~ N(0, I) => Z_t ~ N(0, (alpha^2 + beta^2) I) => exact score = -z / (alpha^2 + beta^2)
    x = torch.randn(50_000, 3)
    z = torch.randn(200, 3)
    for t in (0.2, 0.5, 0.9):
        exact = -z / (alpha(t) ** 2 + beta2(t))
        err = (score(z, t, x) - exact).norm() / exact.norm()
        assert err < 0.05, (t, err)
    # 2) labels check: ODE must land on a 2-mode mixture with the right mean/std per coordinate
    x = torch.cat([torch.randn(2000, 2) * 0.3 + 2, torch.randn(2000, 2) * 0.3 - 2])
    _, lab = make_labels(x, 4000, steps=200, n_ref=4000)
    assert (lab.mean(0) - x.mean(0)).abs().max() < 0.15, lab.mean(0)
    assert (lab.std(0) - x.std(0)).abs().max() < 0.15, lab.std(0)
    print("sgm self-check ok")
