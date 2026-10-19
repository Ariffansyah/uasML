"""Datasets from the paper: 2D toys (Sec. 4.1) and POWER/GAS/HEPMASS/MINIBOONE (Sec. 4.2).
UCI preprocessing is ported from the authors' loaders (MAF preprocessing, Papamakarios et al. 2017),
including their 200k cap on POWER/GAS/HEPMASS training rows. Every loader returns float32 (train, test)."""
from collections import Counter
import pickle
import numpy as np
import pandas as pd

ROOT = "data/"
TOY = ["gaussians", "moons", "spirals", "checkerboard"]
UCI = ["power", "gas", "hepmass", "miniboone"]


def toy(name, n=1000, seed=0):
    r = np.random.default_rng(seed)
    if name == "gaussians":   # ring of 6 modes, as drawn in Fig. 2
        th = 2 * np.pi * r.integers(0, 6, n) / 6
        x = 2.5 * np.c_[np.cos(th), np.sin(th)] + 0.25 * r.standard_normal((n, 2))
    elif name == "moons":
        from sklearn.datasets import make_moons
        x = make_moons(n, noise=0.1, random_state=seed)[0] * 2 + [-1, -0.2]
    elif name == "spirals":   # FFJORD "2spirals"
        t = np.sqrt(r.random((n // 2, 1))) * 3 * np.pi
        d = np.c_[-np.cos(t) * t, np.sin(t) * t] + 0.5 * r.random((n // 2, 2))
        x = np.r_[d, -d] / 3 + 0.1 * r.standard_normal((n, 2))
    elif name == "checkerboard":  # FFJORD; discontinuous density, the hard case for a smooth F(y)
        x1 = r.random(n) * 4 - 2
        x2 = r.random(n) - r.integers(0, 2, n) * 2 + np.floor(x1) % 2
        x = np.c_[x1, x2] * 2
    else:
        raise ValueError(name)
    return x.astype(np.float32)


def _split(a):  # MAF convention: last 10% test, then last 10% of the rest validation
    nt = int(0.1 * len(a)); a, test = a[:-nt], a[-nt:]
    nv = int(0.1 * len(a)); return a[:-nv], a[-nv:], test


def power():
    rng = np.random.RandomState(42)
    a = np.load(ROOT + "power/data.npy")
    rng.shuffle(a)
    a = np.delete(a, [1, 3], axis=1)
    n = len(a)
    voltage, gap, sm = 0.01 * rng.rand(n, 1), 0.001 * rng.rand(n, 1), rng.rand(n, 3)  # same draw order as MAF
    a = a + np.hstack((gap, voltage, sm, np.zeros((n, 1))))
    tr, va, te = _split(a)
    full = np.vstack((tr, va)); mu, s = full.mean(0), full.std(0)
    return (tr[:200_000] - mu) / s, (te - mu) / s


def _maf_pickle(path):
    """MAF's gas file is a Python 2 / pandas 0.x DataFrame pickle that modern pandas can't read.
    Unpickle the pandas objects as plain state holders and rebuild the frame from its single float block."""
    class Stub:
        def __setstate__(self, s): self.state = s

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_new_Index": return lambda cls, d: d["data"]
            if mod.startswith("pandas"): return Stub
            return super().find_class(mod.replace("numpy.core", "numpy._core"), name)

    st = U(open(path, "rb"), encoding="latin1").load().state["_data"].state[3]["0.14.1"]
    (blk,) = st["blocks"]
    return pd.DataFrame(blk["values"].T, columns=st["axes"][0])


def gas():
    df = _maf_pickle(ROOT + "gas/ethylene_CO.pickle").drop(columns=["Meth", "Eth", "Time"])
    while (B := (df.corr() > 0.98).values.sum(1)).max() > 1:  # drop near-duplicate sensors
        df = df.drop(columns=df.columns[np.argmax(B > 1)])
    tr, _, te = _split(((df - df.mean()) / df.std()).values)
    return tr[:200_000], te


def hepmass():
    # Rows are: label, 26 features, mass. MAF's loader loses `mass` through a header quirk; dropped explicitly here.
    tr, te = (pd.read_csv(ROOT + f"hepmass/1000_{s}.csv", header=None, skiprows=1).values for s in ("train", "test"))
    tr, te = tr[tr[:, 0] == 1, 1:-1], te[te[:, 0] == 1, 1:-1]               # signal class only
    mu, s = tr.mean(0), tr.std(0, ddof=1)
    tr, te = (tr - mu) / s, (te - mu) / s
    # MAF's "too many repeated values" filter (it checks the count of the *smallest* value; kept as-is)
    keep = [j for j in range(tr.shape[1]) if Counter(tr[:, j])[tr[:, j].min()] <= 5]
    tr, te = tr[:, keep], te[:, keep]
    return tr[: -int(0.1 * len(tr))][:200_000], te


def miniboone():
    tr, va, te = _split(np.load(ROOT + "miniboone/data.npy"))
    full = np.vstack((tr, va)); mu, s = full.mean(0), full.std(0)
    return (tr - mu) / s, (te - mu) / s


def load(name):
    if name in TOY:
        return toy(name, seed=0), toy(name, n=2000, seed=1)
    tr, te = globals()[name]()
    return tr.astype(np.float32), te.astype(np.float32)
