"""Apply the fixed, train-calibrated per-axis size correction of a category to payloads. usage: apply_rescale.py CAT in.pt [in.pt ...]  (writes *_rescaled_seedN.pt)"""
import os, sys, re, torch
cat = sys.argv[1]; f = torch.load(f'results/vae_retrain/{cat}/{cat}_rescale_factors' + os.environ.get('RESCALE_SUFFIX', '') + '.pt')['factors']
for path in sys.argv[2:]:
    p = torch.load(path, map_location='cpu', weights_only=False); x = p['samples'].float()
    p['samples'] = (x - x.mean(1, keepdim=True)) / f + x.mean(1, keepdim=True); p['diag_rescale'] = f.tolist()
    out = re.sub(r'_seed(\d+)\.pt$', r'_rescaled_seed\1.pt', path); assert out != path; torch.save(p, out)
