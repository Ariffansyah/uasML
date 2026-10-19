"""Class balancing experiment (METODE.md section 7.4): does oversampling the minority class with the
diffusion generator help a classifier, compared with the usual methods?
   python balance_experiment.py        (~5 min on a laptop CPU; writes runs/balance_results.csv)

Data: heart_failure.csv, target DEATH_EVENT (deaths are the minority class).
10 stratified 70/30 splits. The 30% test part is always real and untouched; balancing only ever sees the training part.
Scenarios: the natural ratio (32% deaths) and 80/20 (deaths in the training part subsampled to 20%).
Methods: no balancing, class weights, random oversampling, SMOTE, and the diffusion generator trained on the
training part's minority rows only. Metrics are for the minority class (deaths) on the real test part."""
import os
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sgm

METHODS = ["none", "class weights", "random oversampling", "SMOTE", "diffusion generator"]
MODELS = {
    "logistic regression": lambda cw: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=cw)),
    "random forest": lambda cw: RandomForestClassifier(300, n_jobs=-1, random_state=0, class_weight=cw),
}


def balance(method, X, y, seed):
    if method in ("none", "class weights"):
        return X, y
    if method == "random oversampling":
        return RandomOverSampler(random_state=seed).fit_resample(X, y)
    if method == "SMOTE":
        return SMOTE(random_state=seed).fit_resample(X, y)
    minority = X[y == 1]                         # diffusion generator: learn the minority rows, add the missing ones
    n_new = int((y == 0).sum() - len(minority))
    torch.manual_seed(seed)
    new = sgm.synthesize(minority, n_new, m=10_000, steps=200, hidden=256, epochs=60)[0]
    return np.vstack([X, new]), np.r_[y, np.ones(n_new, int)]


df = pd.read_csv("heart_failure.csv")
X, y = df.drop(columns="DEATH_EVENT").to_numpy(float), df.DEATH_EVENT.to_numpy()
rows = []
for scenario in ("natural (32% deaths)", "80/20"):
    for seed, (tr, te) in enumerate(StratifiedShuffleSplit(10, test_size=0.3, random_state=0).split(X, y)):
        Xtr, ytr = X[tr], y[tr]
        if scenario == "80/20":
            pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
            keep = np.r_[neg, np.random.default_rng(seed).choice(pos, len(neg) // 4, replace=False)]
            Xtr, ytr = Xtr[keep], ytr[keep]
        for method in METHODS:
            Xb, yb = balance(method, Xtr, ytr, seed)
            for model, make in MODELS.items():
                p = make("balanced" if method == "class weights" else None).fit(Xb, yb).predict_proba(X[te])[:, 1]
                pred = (p >= 0.5).astype(int)
                rows.append(dict(scenario=scenario, split=seed, model=model, method=method,
                                 recall=recall_score(y[te], pred), precision=precision_score(y[te], pred, zero_division=0),
                                 f1=f1_score(y[te], pred), balanced_acc=balanced_accuracy_score(y[te], pred),
                                 auc=roc_auc_score(y[te], p)))
        print(f"{scenario}: split {seed + 1}/10 done", flush=True)

res = pd.DataFrame(rows)
os.makedirs("runs", exist_ok=True)
res.to_csv("runs/balance_results.csv", index=False)
pd.set_option("display.width", 200)
for (scenario, model), g in res.groupby(["scenario", "model"], sort=False):
    mean, std = g.groupby("method", sort=False)[["recall", "precision", "f1", "balanced_acc", "auc"]].agg(["mean", "std"]).pipe(
        lambda t: (t.xs("mean", axis=1, level=1), t.xs("std", axis=1, level=1)))
    print(f"\n== {scenario} | {model}  (mean over 10 splits; ± = std of F1)")
    print(mean.round(3).assign(f1_std=std["f1"].round(3)).to_string())
