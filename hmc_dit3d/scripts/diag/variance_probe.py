"""Test: is the shrinkage caused by the sampler's fixed-small reverse variance (beta_tilde)?
DDPM drops the term coef1^2 * Var(x0 | x_t) from the reverse variance. That term is negligible when data has
unit scale, but here raw data std is ~0.16, so the shape is decided in the last ~100 steps where coef1 ~ 1/t is large.
We compare reverse variances: small (current code), large (beta_t), analytic (beta_tilde + coef1^2 * Var(x0|x_t),
with Var(x0|x_t) estimated per axis from eps-hat statistics on TRAIN shapes, as in Analytic-DPM)."""
import sys, numpy as np, torch
from pathlib import Path
import hmc_dit3d.train.sample as S
from hmc_dit3d.train.config import load_experiment_config
from hmc_dit3d.data.hmc_condition_bank import load_hmc_condition_bank
from hmc_dit3d.hmc.extractor import HMCFeatureExtractor
torch.manual_seed(0); dev = torch.device('cuda'); B = 48; CAT = sys.argv[1] if len(sys.argv) > 1 else 'car'
cfg = load_experiment_config(Path(f'configs/train_{CAT}_h100_formal_bottleneck.yaml'))
model, diff, ck = S.load_sampling_checkpoint(cfg, Path(f'../checkpoints/dit/{CAT}.pt'), dev); model.eval()
bank = load_hmc_condition_bank(f'results/cov_diag/banks/{CAT}_train.pt'); T = diff.num_timesteps
rng = np.random.default_rng(0); ex = HMCFeatureExtractor(cfg.hmc)
def load(ids):
    pts = np.stack([(lambda p: p[rng.choice(len(p), 2048, replace=False)])(np.load(bank.source_paths[i])) for i in ids]).astype(np.float32)
    return torch.from_numpy(pts).to(dev).transpose(1, 2).contiguous()
# --- 1) estimate g[t, axis] = E[eps_hat^2] on train shapes (bank condition), every timestep
gpath = Path(f'results/vae_retrain/{CAT}/{CAT}_eps2_stats.pt')
gpath.parent.mkdir(parents=True, exist_ok=True)
if gpath.exists(): g = torch.load(gpath)
else:
    g = torch.zeros(T, 3); order = torch.randperm(len(bank), generator=torch.Generator().manual_seed(123)); nb = 24
    with torch.no_grad():
        for t in range(T):
            ids = order[(t * nb) % (len(bank) - nb):][:nb]; x0 = load(ids.tolist())
            tt = torch.full((nb,), t, device=dev, dtype=torch.int64); xt = diff.q_sample(x_start=x0, timesteps=tt, noise=torch.randn_like(x0))
            eh = model(xt, tt, torch.zeros(nb, dtype=torch.int64, device=dev), bank.descriptors[ids].to(dev), [s[ids].to(dev) for s in bank.sequences])
            g[t] = eh.pow(2).mean(dim=(0, 2)).cpu()
    k = torch.ones(1, 1, 9) / 9; g = torch.nn.functional.conv1d(torch.nn.functional.pad(g.t()[:, None], (4, 4), mode='replicate'), k)[:, 0].t().contiguous()  # smooth over t
    torch.save(g, gpath)
ab = diff.alphas_cumprod.double(); c1 = diff.posterior_mean_coef1.double(); small = diff.posterior_variance.double(); large = diff.betas.double()
var_x0 = ((1 - ab) / ab)[:, None] * (1 - g.double()).clamp(0, 1)            # (T,3) expected Var(x0 | x_t) per axis
analytic = torch.minimum(small[:, None] + c1[:, None] ** 2 * var_x0, large[:, None])
print('t      E[eps_hat^2] per axis        sqrt Var(x0|x_t) per axis      var small / analytic(x,y,z) / large', flush=True)
for t in (1, 5, 20, 50, 100, 200, 400, 700):
    print(f'{t:4d}  {[round(v, 3) for v in g[t].tolist()]}   {[round(v, 4) for v in var_x0[t].sqrt().tolist()]}   {small[t]:.2e} / {[f"{v:.2e}" for v in analytic[t].tolist()]} / {large[t]:.2e}', flush=True)
# --- 2) run chains with each variance choice on held-out-of-estimation pairing: same 48 shapes as before
idx = torch.randperm(len(bank), generator=torch.Generator().manual_seed(1))[:B]; x0 = load(idx.tolist())
cond = (bank.descriptors[idx].to(dev), [s[idx].to(dev) for s in bank.sequences]); labels = torch.zeros(B, dtype=torch.int64, device=dev)
den = lambda x, t: model(x, t, labels, cond[0], cond[1])
V = {'small': small[:, None].expand(T, 3), 'analytic': analytic, 'large': large[:, None].expand(T, 3)}
def cd(a, b):
    d = torch.cdist(a.transpose(1, 2), b.transpose(1, 2)).square(); return d.min(2).values.mean(1) + d.min(1).values.mean(1)
@torch.no_grad()
def run(kind, t_start=None):
    torch.manual_seed(7)
    if t_start is None: x = torch.randn_like(x0); t_start = T - 1
    else: x = diff.q_sample(x_start=x0, timesteps=torch.full((B,), t_start, device=dev), noise=torch.randn_like(x0))
    for step in reversed(range(t_start + 1)):
        tt = torch.full((B,), step, device=dev, dtype=torch.int64)
        mean, _, _ = diff.p_mean_variance(den, x, tt)
        x = mean + (V[kind][step].float().to(dev).view(1, 3, 1).sqrt() * torch.randn_like(x) if step > 0 else 0)
    return x
for label, ts in (('chain from t_start=200', 200), ('full chain from pure noise', None)):
    for kind in ('small', 'analytic', 'large'):
        x = run(kind, ts)
        print(f'{label:27s} variance={kind:8s} size ratio per axis {[round(v, 3) for v in (x.std(2) / x0.std(2)).median(0).values.tolist()]}  CD to source x1e3 median {cd(x, x0).median().item() * 1e3:.3f}', flush=True)
