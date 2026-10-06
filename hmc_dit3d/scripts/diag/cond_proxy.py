"""Cheap condition-space proxy for prior quality (no DiT): draw 664 conditions from a prior, compare with 664 held-out-of-sampling
real train conditions using L1 on the concatenated measures. Reports 1-NN accuracy (50 = indistinguishable) and coverage.
Row 'trainbank' (another disjoint set of real train conditions) is the reference a perfect prior would match.
usage: cond_proxy.py CAT vae.pt [vae.pt ...]"""
import sys, torch
sys.path.insert(0, 'scripts/diag')
from gmm_prior import fit_gmm, sample_gmm
from hmc_dit3d.data.hmc_condition_bank import load_hmc_condition_bank
from hmc_dit3d.hmc.condition_vae import load_condition_vae_checkpoint
cat = sys.argv[1]; dev = 'cuda'; N = 664; R = 4
bank = load_hmc_condition_bank(f'results/cov_diag/banks/{cat}_train.pt')
D = bank.descriptors.to(dev); S = [s.to(dev) for s in bank.sequences]; n = len(bank)
feat = lambda seqs: torch.cat([s.float() for s in seqs], -1)
T = feat(S)
def metrics(Q, refidx):
    Rf = T[refidx]; dqr = torch.cdist(Q, Rf, p=1); drr = torch.cdist(Rf, Rf, p=1); dqq = torch.cdist(Q, Q, p=1)
    inf = torch.full((len(Q),), float('inf'), device=dev)
    M = torch.cat([torch.cat([drr + torch.diag(inf), dqr.t()], 1), torch.cat([dqr, dqq + torch.diag(inf)], 1)], 0)
    lab = torch.cat([torch.ones(len(Rf)), torch.zeros(len(Q))]).to(dev); pred = lab[M.argmin(1)]
    return 100 * (pred == lab).float().mean().item(), 100 * dqr.argmin(1).unique().numel() / len(Rf), dqr.min(1).values.mean().item()
def report(name, sampler):
    acc = [];
    for r in range(R):
        g = torch.Generator(device=dev).manual_seed(1000 + r); perm = torch.randperm(n, generator=torch.Generator().manual_seed(r)).to(dev)
        refidx, other = perm[:N], perm[N:2 * N]
        acc.append(metrics(sampler(g, other), refidx))
    a = torch.tensor(acc).mean(0).tolist(); print(f'  {name:28s} cond 1-NNA={a[0]:5.1f}  cond COV={a[1]:5.1f}  NN-dist={a[2]:.3f}', flush=True)
print(f'== {cat}: reference'); report('trainbank (real conds)', lambda g, other: T[other])
for path in sys.argv[2:]:
    vae, meta = load_condition_vae_checkpoint(path, device=dev)
    with torch.no_grad():
        mu, logvar = vae.encode(D, S)
        print(f'== {path.split("/")[-1]}  beta={meta.get("beta")} z={meta.get("latent_dim")}')
        dec = lambda z: feat(vae.decode_conditions(z, sequence_threshold=0.1)['sequences'])
        report('trainrecon (decode mu)', lambda g, other: dec(mu[other]))
        report('trainpost (mu + sigma*eps)', lambda g, other: dec(mu[other] + (0.5 * logvar[other]).exp() * torch.randn(N, mu.shape[1], device=dev, generator=g)))
        for temp in (1.0, 1.2): report(f'normal T={temp}', lambda g, other, temp=temp: dec(torch.randn(N, mu.shape[1], device=dev, generator=g) * temp))
        for k in (32, 128, 512):
            gm = fit_gmm(mu.double(), k, generator=torch.Generator(device=dev).manual_seed(7))
            for temp in (1.0,): report(f'gmm K={k} T={temp}', lambda g, other, gm=gm, temp=temp: dec(sample_gmm(gm, N, temperature=temp, generator=g).float()))
