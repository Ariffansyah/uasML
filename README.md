# Diffusion-labeled supervised generative models

Implementation of Liu et al., *Diffusion-model-assisted supervised learning of generative models for density
estimation*, JMLMC 5(1):25–38, 2024, plus a Streamlit application.

| File | What |
|---|---|
| `sgm.py` | The method: training-free score (Eqs. 11–12), reverse ODE (Eq. 13), labeling + MSE training (Algorithm 1). `python sgm.py` runs a self-check. |
| `data.py` | 2D toy sets and POWER / GAS / HEPMASS / MINIBOONE with the authors' (MAF) preprocessing. |
| `train.py` | Reproduces Sec. 4: writes `runs/<name>/{labels.pt, model.pt, metrics.json, results.png}`. |
| `ablation_refs.py` | Experiment behind METODE.md §5.5: how the score's reference batch affects the generator on HEPMASS. |
| `app.py` | Streamlit app: 2D playground, CSV → synthetic data with fidelity/privacy/usefulness checks, class balancing, and a table of your reproduction runs. |
| `balance_experiment.py` | Class balancing experiment (METODE.md §7.3): diffusion generator vs SMOTE, class weights, random oversampling. |

## Setup

```sh
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy scipy pandas matplotlib scikit-learn streamlit imbalanced-learn
# UCI data (857 MB tarball from Papamakarios et al.; only the 4 files used are extracted)
curl -L -o data/data.tar.gz https://zenodo.org/records/1161203/files/data.tar.gz
tar -xzf data/data.tar.gz -C data --strip-components=1 data/power/data.npy data/gas/ethylene_CO.pickle \
    data/hepmass/1000_train.csv data/hepmass/1000_test.csv data/miniboone/data.npy
```

## Run

```sh
.venv/bin/python train.py gaussians moons spirals checkerboard     # Fig. 2, ~1 min total
.venv/bin/python train.py power gas hepmass miniboone              # Figs. 3-6 + Table 1, ~1 h on a laptop CPU
.venv/bin/streamlit run app.py
```

`--epochs N` overrides the training length; labels are cached, so retraining the generator skips the ODE step (`--relabel` to redo it). `--eval-only` redraws metrics and figures from saved models.

## Where this differs from the paper text

- **Schedule.** α_t = 1 − 0.99t, β²_t = 0.01 + 0.99t, as in the authors' code (the paper writes α_t = 1 − t, β²_t = t, which makes b(t) blow up at t = 1 and 1/β²_t at t = 0).
- **Score weights.** Eq. (12) is computed as a softmax over log-densities. Same math, but the authors' raw `exp` underflows to 0/0 near t = 0.
- **Score batch.** 1000 random data points per chunk of trajectories, as in the authors' code (the text says N = 5000).
- **UCI generator.** Two tanh hidden layers and lr 2e-3, as in the authors' code (the text says one layer, lr 0.01). Trained with minibatch Adam (batch 1024, 100 epochs ≈ 20k steps, cosine LR) instead of their full-batch GPU loop, so it runs on a CPU.
- **Toy sets.** "Diffeqml" (torchdyn's logo dataset) is replaced by FFJORD's checkerboard; the Gaussian mixture is a ring of 6 modes, as drawn in Fig. 2.
- **Metrics.** Marginal KL is measured against the training data (the paper's protocol), with both KDEs renormalized on the evaluation grid (the authors' raw sum can go slightly negative). `metrics.json` also has `kl_gen_vs_test` against held-out test data.
- **HEPMASS is 21-D here, not 11-D.** The authors' published loader (MAF preprocessing) yields the standard 21 dimensions; the paper's "11-dimensional" and Fig. 5 can't be reproduced from their code. POWER (6), GAS (8) and MINIBOONE (43) match.
- **GAS file.** MAF's `ethylene_CO.pickle` is a Python 2 / pandas 0.x pickle; `data._maf_pickle` rebuilds it without old pandas.
