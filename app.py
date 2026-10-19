"""streamlit run app.py   (from the project folder)
Playground: how the method works, in 2D. Synthetic data: CSV in, synthetic CSV + fidelity/privacy/usefulness report out.
Class balancing: grow the rare classes with per-class generators, and test whether it helps. Paper reproduction: train.py results."""
import json, os, time
import streamlit as st

# Draw the header before the heavy imports: the first load spends ~5 s importing torch/pandas/matplotlib,
# and Streamlit shows only a grey skeleton until the script emits something.
st.set_page_config(page_title="Diffusion-labeled generator", layout="wide")
st.title("Generative models trained with diffusion labels")
st.caption("Liu et al., *Diffusion-model-assisted supervised learning of generative models for density estimation*, JMLMC 5(1), 2024")
with st.spinner("Loading PyTorch (first visit only)…"):
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    import data, sgm
play, synth, bal, repro = st.tabs(["Playground", "Synthetic data", "Class balancing", "Paper reproduction"])
HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"


@st.cache_data(show_spinner="Downloading the heart failure dataset…")
def load_csv(url):
    return pd.read_csv(url)


def scatter(ax, pts, title, color, lim, bg=None):
    if bg is not None:
        ax.scatter(*bg.T, s=2, c="lightgray")
    ax.scatter(*pts.T, s=2, c=color, alpha=0.5)
    ax.set(title=title, xticks=[], yticks=[], xlim=lim[:, 0], ylim=lim[:, 1], aspect="equal")


def sub(a, n=2000):
    return a[np.random.default_rng(1).permutation(len(a))[:n]]


def nearest(a, b):
    """Distance from each row of a to its closest row of b."""
    return torch.cat([torch.cdist(c, b).min(1).values for c in a.split(256)]).numpy()


def utility(tr, hold, out, j):
    """Train-on-synthetic-test-on-real: a random forest fitted on the real training rows vs on the synthetic rows,
    both predicting column j of the real rows the generator never saw. Returns (metric, real score, synthetic score)."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
    Xh, yh = np.delete(hold, j, 1), hold[:, j]
    classes = np.unique(tr[:, j])
    metric = "R²" if len(classes) > 10 else "AUC" if len(classes) == 2 and len(np.unique(yh)) == 2 else "accuracy"
    scores = []
    for a in (sub(tr, 20_000), sub(out, 20_000)):
        X, y = np.delete(a, j, 1), a[:, j]
        if metric == "R²":
            scores.append(r2_score(yh, RandomForestRegressor(200, n_jobs=-1, random_state=0).fit(X, y).predict(Xh)))
            continue
        y = classes[np.abs(y[:, None] - classes).argmin(1)]  # snap synthetic values onto the real classes
        f = RandomForestClassifier(200, n_jobs=-1, random_state=0).fit(X, y)
        # one-class fit -> constant probabilities -> AUC 0.5, which is the honest score for that case
        scores.append(roc_auc_score(yh, f.predict_proba(Xh)[:, -1]) if metric == "AUC" else accuracy_score(yh, f.predict(Xh)))
    return metric, *scores


def class_scores(Xtr, ytr, Xte, yte, c):
    """Random forest fitted on (Xtr, ytr), scored on the real test rows: recall/precision/F1 of class c, balanced accuracy."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
    pred = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(Xtr, ytr).predict(Xte)
    p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=[c], zero_division=0)
    return {"Recall": r[0], "Precision": p[0], "F1": f[0], "Macro F1": f1_score(yte, pred, average="macro", zero_division=0),
            "Balanced accuracy": balanced_accuracy_score(yte, pred)}


with play:
    st.markdown("Reverse ODE (Eq. 13) carries Gaussian noise onto the data, with the score estimated straight from "
                "the data points (no score network). Those (noise, data) pairs train a plain MLP with MSE, "
                "which then samples in a single forward pass.")
    c = st.columns(4)
    name = c[0].selectbox("Dataset", data.TOY)
    n = c[1].slider("Data points", 200, 3000, 1000, 100)
    steps = c[2].slider("ODE steps", 20, 300, 100, 10)
    epochs = c[3].slider("Generator epochs", 500, 5000, 2000, 500)
    if st.button("Run", type="primary"):
        x = torch.from_numpy(data.toy(name, n))
        y = torch.randn(n, 2)
        path = [y]
        lab = sgm.ode(y, x, steps, path)
        bar = st.progress(0.0)
        net = sgm.fit(y, lab, 100, 4, epochs, 5e-3, progress=lambda e, N, l: e % 50 == 0 and
                      bar.progress(e / N, f"Training generator: epoch {e}/{N}, MSE {l:.4f}"))
        bar.empty()
        yb = torch.randn(10_000, 2)
        t0 = time.perf_counter(); sgm.ode(yb, x, steps); ode_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = net(yb).numpy()
        gen_sec = time.perf_counter() - t0
        st.session_state.play = dict(x=x.numpy(), path=torch.stack(path).numpy(), gen=gen, ode_sec=ode_sec, gen_sec=gen_sec)

    if p := st.session_state.get("play"):
        K = len(p["path"]) - 1
        k = st.slider("ODE step (t runs 1 → 0)", 0, K, K)
        lim = np.c_[np.minimum(p["x"].min(0), -3), np.maximum(p["x"].max(0), 3)].T * 1.05
        fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
        scatter(ax[0], p["path"][k], f"ODE state at t = {1 - k / K:.2f}", "C0", lim, bg=p["x"])
        idx = np.random.default_rng(0).choice(p["path"].shape[1], 40, replace=False)
        ax[1].plot(p["path"][:, idx, 0], p["path"][:, idx, 1], lw=0.7)
        ax[1].scatter(*p["path"][-1, idx].T, s=10, c="k", zorder=3)
        ax[1].set(title="40 trajectories, noise → data", xticks=[], yticks=[], xlim=lim[:, 0], ylim=lim[:, 1], aspect="equal")
        scatter(ax[2], p["x"], "Ground truth", "C2", lim)
        scatter(ax[3], p["gen"], "Trained generator F(y), 10k samples", "C3", lim)
        fig.tight_layout()
        st.pyplot(fig)
        m = st.columns(3)
        m[0].metric("Sampling 10k with the ODE", f"{p['ode_sec']:.2f} s")
        m[1].metric("Sampling 10k with the generator", f"{p['gen_sec'] * 1000:.1f} ms")
        m[2].metric("Speed-up", f"{p['ode_sec'] / p['gen_sec']:.0f}×")


with synth:
    st.markdown("Upload a table. The app standardizes the numeric columns, labels them with the training-free diffusion ODE, "
                "trains a generator and returns synthetic rows with a fidelity and privacy report. "
                "Non-numeric and constant columns, and rows with missing values, are dropped.")
    up = st.file_uploader("CSV file", type="csv")
    df = None
    if up:
        df = pd.read_csv(up, sep=None, engine="python")  # sniffs "," vs ";" (Excel in id/eu locales saves ";")
    elif st.checkbox("No file at hand? Use the California housing dataset (20,640 rows, 9 columns)"):
        from sklearn.datasets import fetch_california_housing
        df = fetch_california_housing(as_frame=True).frame

    if df is not None:
        num = df.select_dtypes("number").dropna()
        num = num.loc[:, num.nunique() > 1]
        if skipped := [col for col in df.columns if col not in num.columns]:
            st.caption(f"Ignored non-numeric or constant columns: {', '.join(map(str, skipped))}")
        if num.shape[1] < 2 or len(num) < 50:
            st.error("Need at least 2 numeric columns and 50 complete rows.")
        else:
            st.dataframe(num.head(), width="stretch")
            with st.expander("Settings (defaults follow the paper, scaled for a laptop CPU)"):
                c = st.columns(5)
                M = c[0].number_input("Labeled pairs M", 1000, 200_000, 20_000, 1000)
                steps = c[1].number_input("ODE steps", 50, 500, 200, 50)
                hidden = c[2].number_input("Hidden units", 64, 2048, 512, 64)
                epochs = c[3].number_input("Epochs", 10, 500, 100, 10)
                n_out = c[4].number_input("Synthetic rows", 100, 1_000_000, min(len(num), 1_000_000), 100)

            if st.button("Generate synthetic data", type="primary"):
                arr = num.to_numpy(np.float64)
                perm = np.random.default_rng(0).permutation(len(arr))
                hold, tr = arr[perm[: len(arr) // 5]], arr[perm[len(arr) // 5:]]  # 20% never seen, for the privacy check
                bar = st.progress(0.0)
                out, z, logc = sgm.synthesize(tr, n_out, M, steps, hidden, epochs, progress=bar.progress)
                bar.empty()
                from train import marginal_kl  # pulls in scipy (~1.5 s); only this report needs it
                st.session_state.pop("util", None)
                st.session_state.syn = dict(
                    cols=list(num.columns), real=arr, tr=tr, hold=hold, out=out,
                    logged=[c for c, l in zip(num.columns, logc) if l],
                    kl=marginal_kl(z(arr).numpy(), z(out).numpy()),
                    cr=np.nan_to_num(np.corrcoef(arr.T)), cs=np.nan_to_num(np.corrcoef(out.T)),
                    dcr_syn=np.median(nearest(z(sub(out)), z(tr))),    # synthetic row -> closest training row
                    dcr_new=np.median(nearest(z(sub(hold)), z(tr))))   # unseen real row -> closest training row

            if (s := st.session_state.get("syn")) and s["cols"] == list(num.columns):
                cols, real, out, kl, cr, cs, dcr_syn, dcr_new = (
                    s[k] for k in ("cols", "real", "out", "kl", "cr", "cs", "dcr_syn", "dcr_new"))

                m = st.columns(3)
                m[0].metric("Mean marginal KL (lower is better)", f"{np.mean(kl):.3f}")
                m[1].metric("Mean |Δ correlation|", f"{np.abs(cr - cs).mean():.3f}")
                m[2].metric("Median distance to closest training row", f"{dcr_syn:.2f}",
                            f"{dcr_syn / dcr_new - 1:+.0%} vs unseen real rows ({dcr_new:.2f})")
                if dcr_syn < 0.8 * dcr_new:
                    st.warning("Synthetic rows sit noticeably closer to training rows than unseen real rows do: "
                               "the generator partly reproduces training records. Don't treat this output as anonymized.")
                if s["logged"]:
                    st.caption(f"Modeled on a log scale (long right tail): {', '.join(map(str, s['logged']))}")

                st.dataframe(pd.DataFrame({"real mean": real.mean(0), "synthetic mean": out.mean(0), "real std": real.std(0),
                                           "synthetic std": out.std(0), "marginal KL": kl}, index=cols).style.format("{:.3g}"),
                             width="stretch")

                fig, ax = plt.subplots(1, 2, figsize=(11, 5))
                for a, cm, title in zip(ax, (cr, cs), ("Real correlations", "Synthetic correlations")):
                    a.imshow(cm, vmin=-1, vmax=1, cmap="RdBu_r")
                    a.set_title(title)
                    a.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
                    a.set_yticks(range(len(cols)), cols, fontsize=7)
                fig.tight_layout()
                st.pyplot(fig)

                left, right = st.columns(2)
                with left:
                    col = st.selectbox("Column distribution", cols)
                    j = cols.index(col)
                    fig, a = plt.subplots(figsize=(6, 4))
                    bins = np.histogram_bin_edges(real[:, j], 60)
                    a.hist(real[:, j], bins, density=True, alpha=0.5, label="real")
                    a.hist(out[:, j], bins, density=True, alpha=0.5, label="synthetic")
                    a.legend(); a.set_xlabel(col)
                    st.pyplot(fig)
                with right:
                    c = st.columns(2)
                    bj, bi = np.unravel_index(np.abs(cr - np.eye(len(cols))).argmax(), cr.shape)  # most correlated pair
                    i = cols.index(c[0].selectbox("x", cols, int(bi)))
                    j = cols.index(c[1].selectbox("y", cols, int(bj)))
                    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
                    for a, d, title, color in ((ax[0], real, "Real", "C0"), (ax[1], out, "Synthetic", "C1")):
                        d = sub(d)
                        a.scatter(d[:, i], d[:, j], s=2, alpha=0.4, c=color)
                        a.set(title=title, xlabel=cols[i], ylabel=cols[j])
                    fig.tight_layout()
                    st.pyplot(fig)

                st.subheader("Usefulness: train on synthetic, test on real")
                st.caption("A random forest is trained once on the real training rows and once on the synthetic rows. "
                           "Both predict the chosen column for the 20% of real rows the generator never saw. "
                           "Close scores mean the synthetic data can stand in for the real data.")
                target = st.selectbox("Column to predict", cols, len(cols) - 1)
                if st.button("Run usefulness test"):
                    with st.spinner("Training random forests…"):
                        st.session_state.util = (target, utility(s["tr"], s["hold"], out, cols.index(target)))
                if (u := st.session_state.get("util")) and u[0] == target:
                    metric, real_score, syn_score = u[1]
                    m = st.columns(2)
                    m[0].metric(f"{metric}, trained on real rows", f"{real_score:.3f}")
                    m[1].metric(f"{metric}, trained on synthetic rows", f"{syn_score:.3f}", f"{syn_score - real_score:+.3f}")

                st.download_button("Download synthetic CSV", pd.DataFrame(out, columns=cols).to_csv(index=False),
                                   "synthetic.csv", "text/csv", type="primary")


with bal:
    st.markdown("When one class is rare (say 80% of patients survive and 20% die), a classifier learns to favour the "
                "common class and misses the rare one. This tab trains a diffusion generator on each under-represented "
                "class and adds synthetic rows until every class is as large as the biggest one. It then checks whether "
                "that actually helps, on real rows the generator never saw.")
    up_b = st.file_uploader("CSV file", type="csv", key="bal_csv")
    bdf = None
    if up_b:
        bdf = pd.read_csv(up_b, sep=None, engine="python")
    elif st.checkbox("No file at hand? Use the heart failure dataset (299 patients, 32% died)"):
        try:
            bdf = load_csv(HEART_URL)
        except OSError:
            st.error("Couldn't download the example (no internet?). Upload heart_failure.csv instead.")

    if bdf is not None:
        targets = [c for c in bdf.columns if 2 <= bdf[c].nunique() <= 10]
        if not targets:
            st.error("No column with 2–10 distinct values to use as the class.")
        else:
            target = st.selectbox("Class column", targets, len(targets) - 1)
            feats = bdf.drop(columns=target).select_dtypes("number")
            ok = feats.notna().all(axis=1) & bdf[target].notna()
            feats = feats[ok].loc[:, feats[ok].nunique() > 1]
            yall = bdf.loc[ok, target].to_numpy()
            counts = pd.Series(yall).value_counts()
            st.dataframe(pd.DataFrame({"rows": counts, "share": counts / counts.sum()}).rename_axis(target)
                         .style.format({"share": "{:.0%}"}), width="content")
            if feats.shape[1] < 2 or counts.min() < 10:
                st.error("Need at least 2 numeric feature columns and at least 10 rows in every class.")
            elif counts.min() == counts.max():
                st.info("The classes are already balanced.")
            elif st.button("Balance classes", type="primary"):
                from sklearn.model_selection import StratifiedShuffleSplit
                X = feats.to_numpy(np.float64)
                bar = st.progress(0.0)

                def grow(Xc, n, label):
                    """n synthetic rows like Xc (one class), from a diffusion generator trained on Xc only."""
                    torch.manual_seed(0)
                    return sgm.synthesize(Xc, int(n), m=10_000, steps=200, hidden=256, epochs=60,
                                          progress=lambda f, t: bar.progress(f, f"{label}: {t}"))[0]

                def balanced(Xs, ys, label):
                    n = pd.Series(ys).value_counts()
                    new = [(grow(Xs[ys == c], n.max() - n[c], f"{label}, class {c}"), c) for c in n.index if n[c] < n.max()]
                    return (np.vstack([Xs] + [r for r, _ in new]),
                            np.concatenate([ys] + [np.full(len(r), c, dtype=ys.dtype) for r, c in new]))

                # Does it help? 5 random splits: 30% real rows held out, the generator only sees the other 70%
                rare, before, after = counts.idxmin(), [], []
                for i, (tr, te) in enumerate(StratifiedShuffleSplit(5, test_size=0.3, random_state=0).split(X, yall)):
                    Xb, yb = balanced(X[tr], yall[tr], f"Test {i + 1}/5")
                    before.append(class_scores(X[tr], yall[tr], X[te], yall[te], rare))
                    after.append(class_scores(Xb, yb, X[te], yall[te], rare))
                # The download: generators trained on all rows of each class
                Xf, yf = balanced(X, yall, "Final data")
                bar.empty()
                out = pd.concat([feats.assign(**{str(target): yall, "is_synthetic": 0}),
                                 pd.DataFrame(Xf[len(X):], columns=feats.columns).assign(
                                     **{str(target): yf[len(X):], "is_synthetic": 1})], ignore_index=True)
                st.session_state.bal = dict(target=target, rare=rare, binary=len(counts) == 2, out=out,
                                            before=pd.DataFrame(before), after=pd.DataFrame(after))

            if (b := st.session_state.get("bal")) and b["target"] == target:
                st.dataframe(b["out"].groupby(str(target))["is_synthetic"].agg(
                    real=lambda s: int((s == 0).sum()), synthetic="sum", total="size"), width="content")
                key = "F1" if b["binary"] else "Macro F1"
                st.markdown(f"**Does it help?** A random forest predicting `{target}`, trained without and with the "
                            f"synthetic rows, averaged over 5 random splits and always scored on real rows the generator "
                            f"never saw. Recall, precision and F1 are for the rarest class, `{b['rare']}`.")
                mb, ma = b["before"].mean(), b["after"].mean()
                for col, k in zip(st.columns(5), ma.index):
                    col.metric(k, f"{ma[k]:.3f}", f"{ma[k] - mb[k]:+.3f} vs no balancing")
                gain, wins = ma[key] - mb[key], int((b["after"][key] > b["before"][key]).sum())
                dp = ma["Precision"] - mb["Precision"]
                if gain > 0.01:
                    st.success(f"Balancing helped: {key} {gain:+.3f} on average, better in {wins} of 5 splits. The model "
                               f"catches more `{b['rare']}` cases (recall {ma['Recall'] - mb['Recall']:+.3f}), "
                               + (f"at the cost of some false alarms (precision {dp:+.3f})." if dp < 0
                                  else f"and its precision also rose ({dp:+.3f})."))
                elif gain < -0.01:
                    st.warning(f"Balancing made things worse here: {key} {gain:+.3f} on average "
                               f"(better in only {wins} of 5 splits).")
                else:
                    st.info(f"No clear difference: {key} {gain:+.3f} on average, better in {wins} of 5 splits. "
                            "The imbalance may be too mild to matter; class weights are a simpler alternative.")
                g = b["out"].groupby(str(target))["is_synthetic"]
                thin = [f"`{c}` ({int(n_real)} real → {int(n_syn)} synthetic)"
                        for (c, n_real), n_syn in zip((g.size() - g.sum()).items(), g.sum()) if n_syn > 5 * n_real]
                if thin:
                    st.caption(f"⚠️ Grown more than 5×: {', '.join(thin)}. These synthetic rows are variations of very "
                               "few real rows, so they add little genuinely new information.")
                st.caption("Use the synthetic rows for training only, and always evaluate on real rows "
                           "(filter with the `is_synthetic` column).")
                st.download_button("Download balanced CSV", b["out"].to_csv(index=False), "balanced.csv", "text/csv",
                                   type="primary")

with repro:
    runs =[r for r in data.TOY + data.UCI if os.path.exists(f"runs/{r}/metrics.json")]
    if not runs:
        st.info("No trained runs yet: `python train.py gaussians moons spirals checkerboard power gas hepmass miniboone`")
    else:
        rows = []
        for r in runs:
            m = json.load(open(f"runs/{r}/metrics.json"))
            rows.append({"dataset": r, "labeling (s)": m["label_sec"], "training F (s)": m["train_sec"],
                         "100k samples (s)": m["sample_100k_sec"],
                         "mean KL, labels": np.mean(m["kl"]["Labeled data"]) if "kl" in m else None,
                         "mean KL, generated": np.mean(m["kl"]["Generated samples"]) if "kl" in m else None,
                         "cross entropy, truth / labels / generated":
                             " / ".join(f"{v:.2f}" for v in m["cross_entropy"].values()) if "cross_entropy" in m else None})
        st.markdown("Same stages as the paper's Table 1 (theirs: RTX A5000 GPU; here: this machine).")
        st.dataframe(pd.DataFrame(rows).set_index("dataset"), width="stretch")
        st.image(f"runs/{st.selectbox('Results figure', runs)}/results.png")
