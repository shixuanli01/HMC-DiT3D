"""Where does the size shrinkage come from? Start the reverse chain from q(x_t | real x0) at several t_start
and measure the size of the result against the real shape. A correct model returns ratio 1 for every t_start."""
import sys, numpy as np, torch
from pathlib import Path
import hmc_dit3d.train.sample as S
from hmc_dit3d.train.config import load_experiment_config
from hmc_dit3d.data.hmc_condition_bank import load_hmc_condition_bank
from hmc_dit3d.hmc.extractor import HMCFeatureExtractor
torch.manual_seed(0); dev = torch.device('cuda'); B = 48; CAT = sys.argv[1] if len(sys.argv) > 1 else 'car'
cfg = load_experiment_config(Path(f'configs/train_{CAT}_h100_formal_bottleneck.yaml'))
model, diff, ck = S.load_sampling_checkpoint(cfg, Path(f'../checkpoints/dit/{CAT}.pt'), dev); model.eval()
bank = load_hmc_condition_bank(f'results/cov_diag/banks/{CAT}_train.pt')
idx = torch.randperm(len(bank), generator=torch.Generator().manual_seed(1))[:B]
rng = np.random.default_rng(0); ex = HMCFeatureExtractor(cfg.hmc)
pts = np.stack([(lambda p: p[rng.choice(len(p), 2048, replace=False)])(np.load(bank.source_paths[i])) for i in idx.tolist()]).astype(np.float32)
x0 = torch.from_numpy(pts).to(dev).transpose(1, 2).contiguous()          # (B,3,N)
res = [ex.extract(p) for p in pts]                                         # training-time condition: from the same 2048 points
cond_s = (torch.tensor(np.stack([r.descriptor for r in res]), dtype=torch.float32, device=dev),
          [torch.tensor(np.stack([r.sequences[k] for r in res]), dtype=torch.float32, device=dev) for k in range(len(cfg.hmc.scales))])
cond_f = (bank.descriptors[idx].to(dev), [s[idx].to(dev) for s in bank.sequences])  # bank condition: all 15k points
labels = torch.zeros(B, dtype=torch.int64, device=dev)
def ratio(x):  # per-axis median over shapes of std(gen)/std(real)
    return (x.std(2) / x0.std(2)).median(0).values.tolist()
@torch.no_grad()
def run(t_start, cond, from_noise=False):
    den = lambda x, t: model(x, t, labels, cond[0], cond[1])
    if from_noise: x = torch.randn_like(x0); t_start = diff.num_timesteps - 1
    else: x = diff.q_sample(x_start=x0, timesteps=torch.full((B,), t_start, device=dev), noise=torch.randn_like(x0))
    for step in reversed(range(t_start + 1)):
        x = diff.p_sample(den, x, torch.full((B,), step, device=dev, dtype=torch.int64))
    return x
@torch.no_grad()
def one_step(t, cond):  # x0-prediction from a single denoiser call
    tt = torch.full((B,), t, device=dev, dtype=torch.int64); xt = diff.q_sample(x_start=x0, timesteps=tt, noise=torch.randn_like(x0))
    return diff.predict_xstart_from_eps(xt, tt, model(xt, tt, labels, cond[0], cond[1]))
print('alpha_bar at t=0,10,50,100,200,400,700,999:', [round(float(diff.alphas_cumprod[t]), 5) for t in (0, 10, 50, 100, 200, 400, 700, 999)], '| data std', round(float(x0.std()), 4), flush=True)
for name, cond in (('sampled-2048 cond (as in training)', cond_s), ('bank-15k cond (as in trainbank)', cond_f)):
    print('==', name, flush=True)
    print('  single-call x0-prediction size ratio:', {t: [round(v, 3) for v in ratio(one_step(t, cond))] for t in (0, 5, 20, 50, 100, 200, 400)}, flush=True)
    for t in (0, 5, 20, 50, 100, 200, 400, 700):
        print(f'  chain from t_start={t:4d}: size ratio per axis', [round(v, 3) for v in ratio(run(t, cond))], flush=True)
    print('  chain from pure noise  : size ratio per axis', [round(v, 3) for v in ratio(run(0, cond, from_noise=True))], flush=True)
