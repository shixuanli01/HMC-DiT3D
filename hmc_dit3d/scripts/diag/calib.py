"""Calibrate per-axis size factors from a trainbank-mode payload against its source train shapes. usage: calib.py CAT TAG SEED SUFFIX"""
import sys, numpy as np, torch
from hmc_dit3d.data.hmc_condition_bank import load_hmc_condition_bank
cat, tag, seed, suffix = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
O = f'results/vae_retrain/{cat}/'; g = torch.load(O + f'{cat}_{tag}_seed{seed}.pt', map_location='cpu', weights_only=False); assert g['diag_mode'] == 'trainbank'
s = g['samples'].float(); bank = load_hmc_condition_bank(f'results/cov_diag/banks/{cat}_train.pt')
idx = torch.randperm(len(bank), generator=torch.Generator().manual_seed(seed + 1))[:len(s)]; rng = np.random.default_rng(seed)
r = torch.from_numpy(np.stack([(lambda p: p[rng.choice(len(p), s.shape[1], replace=False)])(np.load(bank.source_paths[i])) for i in idx.tolist()])).float()
f = (s.std(1) / r.std(1)).median(0).values; torch.save({'factors': f, 'calib': tag, 'seed': seed}, O + f'{cat}_rescale_factors{suffix}.pt'); print(cat, tag, 'factors', [round(v, 4) for v in f.tolist()])
