"""Reproduce Sec. 4: python train.py moons spirals power ...   (names: see data.TOY, data.UCI)
Writes runs/<name>/{labels.pt, model.pt, metrics.json, results.png}. Labels are cached; --relabel to redo."""
import argparse, json, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import data, sgm

# Sec. 4.1 (toys) and the authors' UCI script. UCI training is minibatch Adam instead of their full-batch
# GPU loop: 100 epochs x 196 steps ~= their 20k iterations, but CPU-friendly.
TOY = dict(m=1000, steps=100, n_ref=1000, chunk=1000, hidden=100, layers=4, epochs=5000, lr=5e-3, batch=None)
UCI = dict(m=200_000, steps=500, n_ref=1000, chunk=5000, layers=2, epochs=100, lr=2e-3, batch=1024)
HIDDEN = dict(power=256, gas=512, hepmass=1024, miniboone=1400)


def config(name):
    if name in data.TOY:
        return dict(TOY)
    return dict(UCI, hidden=HIDDEN[name], chunk=2000 if name == "miniboone" else 5000)


def marginal_kl(truth, other, n=20_000):
    """Authors' metric: per-dimension KL(truth || other) between 1D KDEs (bw 0.2) on a 1000-point grid,
    except both densities are renormalized on the grid so the sum is a proper (non-negative) KL."""
    kl = []
    for j in range(truth.shape[1]):
        g = np.linspace(truth[:, j].min(), truth[:, j].max(), 1000)
        p, q = gaussian_kde(truth[:n, j], 0.2)(g) + 1e-16, gaussian_kde(other[:n, j], 0.2)(g) + 1e-16
        p, q = p / p.sum(), q / q.sum()
        kl.append(float(np.sum(p * np.log(p / q))))
    return kl


def cross_entropy(samples, test):
    """Fig. 2 metric: -E_test[log q], q = KDE fitted on samples."""
    return float(-np.log(gaussian_kde(samples.T)(test.T) + 1e-12).mean())


def plot_toy(sets, xent, path):
    lo, hi = sets["Ground truth"].min(0) - 0.5, sets["Ground truth"].max(0) + 0.5
    gx, gy = np.mgrid[lo[0]:hi[0]:200j, lo[1]:hi[1]:200j]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, (title, s) in zip(ax, sets.items()):
        a.imshow(gaussian_kde(s.T, 0.08)(np.c_[gx.ravel(), gy.ravel()].T).reshape(gx.shape).T,
                 origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]), cmap="viridis")
        a.set(title=title, xlabel=f"Cross entropy = {xent[title]:.2f}", xticks=[], yticks=[])
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_uci(sets, kl, path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    dims = np.arange(1, sets["Ground truth"].shape[1] + 1)
    j = np.random.default_rng(0).integers(len(dims))
    color = {"Ground truth": "C0", "Labeled data": "C1", "Generated samples": "C2"}
    g = np.linspace(*np.percentile(sets["Ground truth"][:, j], [0.5, 99.5]), 400)
    for title, s in sets.items():
        ax[0, 0].plot(g, gaussian_kde(s[:20_000, j], 0.2)(g), label=title, c=color[title])
        ax[1, 0].plot(dims, s.mean(0), "o-", label=title, ms=4, c=color[title])
        ax[1, 1].plot(dims, s.std(0), "o-", label=title, ms=4, c=color[title])
    for title, k in kl.items():
        ax[0, 1].plot(dims, k, "o-", label=title, ms=4, c=color[title])
    ax[0, 0].set(title=f"(a) 1D marginal PDF, dim {j + 1}")
    ax[0, 1].set(title="(b) marginal KL divergence", xlabel="dimension")
    # paper's axis ranges, widened when a series falls outside them (a fixed range hid HEPMASS's std of ~0.8)
    m = max(0.1, max(np.abs(s.mean(0)).max() for s in sets.values()) + 0.01)
    lo, hi = (f(f(s.std(0)) for s in sets.values()) for f in (min, max))
    ax[1, 0].set(title="(c) marginal mean", xlabel="dimension", ylim=(-m, m))
    ax[1, 1].set(title="(d) marginal std", xlabel="dimension", ylim=(min(0.9, lo - 0.02), max(1.1, hi + 0.02)))
    for a in ax.flat: a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def run(name, epochs=None, relabel=False, eval_only=False):
    c = config(name)
    c["epochs"] = epochs or c["epochs"]
    out = f"runs/{name}/"
    os.makedirs(out, exist_ok=True)
    train, test = data.load(name)
    print(f"[{name}] train {train.shape}, test {test.shape}, config {c}")

    if relabel or not os.path.exists(out + "labels.pt"):
        t0 = time.time()
        y, xl = sgm.make_labels(torch.from_numpy(train), c["m"], c["steps"], c["n_ref"], c["chunk"],
                                progress=lambda k, n: print(f"\r[{name}] ODE labeling chunk {k}/{n}", end="", flush=True))
        torch.save(dict(y=y, x=xl, sec=time.time() - t0), out + "labels.pt")
        print()
    lab = torch.load(out + "labels.pt")

    if eval_only:  # redo samples, metrics and figure from the saved generator
        ck = torch.load(out + "model.pt")
        net = sgm.mlp(ck["d"], ck["hidden"], ck["layers"])
        net.load_state_dict(ck["state"])
        train_sec = json.load(open(out + "metrics.json"))["train_sec"]
    else:
        t0 = time.time()
        every = max(1, c["epochs"] // 20)
        net = sgm.fit(lab["y"], lab["x"], c["hidden"], c["layers"], c["epochs"], c["lr"], c["batch"],
                      progress=lambda e, n, l: e % every == 0 and print(f"[{name}] epoch {e}/{n} mse {l:.4f}", flush=True))
        train_sec = time.time() - t0
        torch.save(dict(d=train.shape[1], hidden=c["hidden"], layers=c["layers"], state=net.state_dict()), out + "model.pt")

    t0 = time.perf_counter()
    with torch.no_grad():
        gen = net(torch.randn(100_000, train.shape[1])).numpy()
    sample_sec = time.perf_counter() - t0

    labels = lab["x"].numpy()
    m = dict(config=c, label_sec=lab["sec"], train_sec=train_sec, sample_100k_sec=sample_sec)
    if name in data.TOY:
        sets = {"Ground truth": train, "Labeled training data": labels, "Generated samples": gen[:len(train)]}
        m["cross_entropy"] = {k: cross_entropy(s, test) for k, s in sets.items()}
        plot_toy(sets, m["cross_entropy"], out + "results.png")
    else:
        sets = {"Ground truth": train, "Labeled data": labels, "Generated samples": gen}
        kl = {"Labeled data": marginal_kl(train, labels), "Generated samples": marginal_kl(train, gen)}
        m |= dict(kl=kl, kl_gen_vs_test=marginal_kl(test, gen),
                  mean={k: s.mean(0).tolist() for k, s in sets.items()},
                  std={k: s.std(0).tolist() for k, s in sets.items()})
        plot_uci(sets, kl, out + "results.png")
    json.dump(m, open(out + "metrics.json", "w"), indent=1)
    print(f"[{name}] labeling {m['label_sec']:.1f}s | training {train_sec:.1f}s | 100k samples {sample_sec:.3f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+", choices=data.TOY + data.UCI)
    p.add_argument("--epochs", type=int)
    p.add_argument("--relabel", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="re-evaluate saved models without training")
    a = p.parse_args()
    for n in a.names:
        run(n, a.epochs, a.relabel, a.eval_only)
