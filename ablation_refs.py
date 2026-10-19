"""Ablation for METODE.md section 5.5: why does the generator's spread shrink on HEPMASS, and what fixes it?
   python ablation_refs.py [hepmass|miniboone|...]      (~15 min on a laptop CPU)

Same pipeline as train.py at 1/4 scale (50k labels, 60 epochs), with four ways of choosing the score's reference batch:
  A  a new random batch of 1000 points per chunk of 5000 trajectories (authors' code, train.py)
  D  same labels as A, generator also told which chunk (one-hot), random chunk at sampling time
  B  one shared batch of 1000 points for every chunk
  C  one shared batch of 5000 points for every chunk
Reports the generator's mean std (target 1), marginal KL vs train and vs test, and a copying check:
median distance generated -> nearest training row, divided by the same for real test rows (~1 = no copying)."""
import sys, time
import numpy as np
import torch
import torch.nn as nn
import data, sgm, train

torch.manual_seed(0)
name = sys.argv[1] if len(sys.argv) > 1 else "hepmass"
tr, te = data.load(name)
x = torch.from_numpy(tr)
d = x.shape[1]
M, CH, K = 50_000, 5000, 10
y = torch.randn(M, d)
pick = lambda n: x[torch.randperm(len(x))[:n]]


def label(refs):
    t0 = time.time()
    out = torch.cat([sgm.ode(z, refs[k], 500) for k, z in enumerate(y.split(CH))])
    print(f"  labeled in {time.time() - t0:.0f}s", flush=True)
    return out


def fit(inp, out, epochs=60):  # sgm.fit, but input and output sizes may differ (setup D)
    net = nn.Sequential(nn.Linear(inp.shape[1], 1024), nn.Tanh(), nn.Linear(1024, 1024), nn.Tanh(), nn.Linear(1024, d))
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        for idx in torch.randperm(M).split(1024):
            loss = (net(inp[idx]) - out[idx]).square().sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net


def nn_dist(a, b):
    return torch.cat([torch.cdist(c, b).min(1).values for c in a.split(256)])


dcr_test = nn_dist(torch.from_numpy(te[:2000]), x).median().item()


def report(tag, lab, gen):
    gen = gen.numpy()
    print(f"{tag:30s} label std {lab.std(0).mean():.3f} | gen std {gen.std(0).mean():.3f} (min {gen.std(0).min():.3f}) | "
          f"KL train {np.mean(train.marginal_kl(tr, gen)):.4f} test {np.mean(train.marginal_kl(te, gen)):.4f} | "
          f"copying ratio {nn_dist(torch.from_numpy(gen[:2000]), x).median().item() / dcr_test:.2f}", flush=True)


batches = [pick(1000) for _ in range(K)]
shared1k, shared5k = pick(1000), pick(5000)
z = torch.randn(M, d)

print("A: new random 1000-point batch per chunk (authors' code)", flush=True)
lab_a = label(batches)
net = fit(y, lab_a)
with torch.no_grad(): report("A per-chunk batch", lab_a, net(z))

print("D: A's labels, generator conditioned on the chunk", flush=True)
net = fit(torch.cat([y, torch.eye(K).repeat_interleave(CH, 0)], 1), lab_a)
with torch.no_grad(): report("D chunk-conditioned F(y, c)", lab_a, net(torch.cat([z, torch.eye(K)[torch.randint(K, (M,))]], 1)))

print("B: one shared 1000-point batch", flush=True)
lab_b = label([shared1k] * K)
net = fit(y, lab_b)
with torch.no_grad(): report("B shared batch, 1000", lab_b, net(z))

print("C: one shared 5000-point batch", flush=True)
lab_c = label([shared5k] * K)
net = fit(y, lab_c)
with torch.no_grad(): report("C shared batch, 5000", lab_c, net(z))
