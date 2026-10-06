"""Size correction calibrated on TRAIN only.
usage: rescale_eval.py CAT SEED CALIB_TAG TARGET_TAG [TARGET_TAG...]
CALIB_TAG must be a trainbank-mode payload for SEED: sample i was generated from train condition idx[i], so comparing it with
the real train shape idx[i] gives the per-axis size ratio. Targets are divided by that ratio about each shape's centroid."""
import sys, numpy as np, torch
from hmc_dit3d.data.hmc_condition_bank import load_hmc_condition_bank
cat, seed, calib, targets = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
O = f'results/vae_retrain/{cat}/'
bank = load_hmc_condition_bank(f'results/cov_diag/banks/{cat}_train.pt')
g = torch.load(O + f'{cat}_{calib}_seed{seed}.pt', map_location='cpu', weights_only=False); assert g['diag_mode'] == 'trainbank'
s = g['samples'].float(); idx = torch.randperm(len(bank), generator=torch.Generator().manual_seed(seed + 1))[:len(s)]
rng = np.random.default_rng(seed)
r = torch.from_numpy(np.stack([(lambda p: p[rng.choice(len(p), s.shape[1], replace=False)])(np.load(bank.source_paths[i])) for i in idx.tolist()])).float()
torch.save({'samples': r, 'diag_mode': 'realtrain', 'diag_seed': seed}, O + f'{cat}_realtrain_seed{seed}.pt')
ratio = s.std(1) / r.std(1); f = ratio.median(0).values
print(f'{cat} seed {seed}: size ratio gen/source per axis median {[round(v, 4) for v in f.tolist()]} IQR {[round(v, 3) for v in (ratio.quantile(.75, 0) - ratio.quantile(.25, 0)).tolist()]} | train per-shape std mean {[round(v, 4) for v in r.std(1).mean(0).tolist()]}')
torch.save({'factors': f, 'calib': calib, 'seed': seed}, O + f'{cat}_rescale_factors_from_{calib}_seed{seed}.pt')
for tag in targets:
    p = torch.load(O + f'{cat}_{tag}_seed{seed}.pt', map_location='cpu', weights_only=False); x = p['samples'].float()
    p['samples'] = (x - x.mean(1, keepdim=True)) / f + x.mean(1, keepdim=True); p['diag_rescale'] = f.tolist(); p['diag_rescale_calib'] = calib
    torch.save(p, O + f'{cat}_{tag}_rescaled_seed{seed}.pt'); print('saved', f'{cat}_{tag}_rescaled_seed{seed}.pt')
