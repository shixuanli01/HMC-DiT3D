"""Score sample payloads under the TopoDiT test.py protocol (val split, refs = npy[10000:12048],
independent 8-vs-8 groups, metrics averaged over groups), adding MMD-CD / MMD-EMD.
Reference/sample order is shuffled; results are averaged over --perms permutations.
usage: val_eval_mmd.py --perms 3 --out out.json payload.pt [...]"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/fsx/weicyang/shix/emd_build'); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from hmc_dit3d.metrics.topodit_batch8 import pairwise_cd_emd, knn_accuracy, coverage

ROOT = '/fsx/weicyang/shix/HMC-DiT3D/ShapeNetCore.v2.PC15k'
SYN = {'chair': '03001627', 'airplane': '02691156', 'car': '02958343'}

def val_refs(cat):
    fs = sorted(glob.glob(f'{ROOT}/{SYN[cat]}/val/*.npy'))
    return torch.stack([torch.from_numpy(np.load(f)[10000:12048]).float() for f in fs])

def group(S, R, dev):
    rs_cd, rs_emd = pairwise_cd_emd(R, S, device=dev)
    rr_cd, rr_emd = pairwise_cd_emd(R, R, device=dev)
    ss_cd, ss_emd = pairwise_cd_emd(S, S, device=dev)
    return {'1-NNA-CD': 100 * knn_accuracy(rr_cd, rs_cd, ss_cd), '1-NNA-EMD': 100 * knn_accuracy(rr_emd, rs_emd, ss_emd),
            'COV-CD': 100 * coverage(rs_cd), 'COV-EMD': 100 * coverage(rs_emd),
            # MMD: mean over references of distance to nearest sample (raw, unscaled)
            'MMD-CD': float(rs_cd.min(1).values.mean()), 'MMD-EMD': float(rs_emd.min(1).values.mean())}

ap = argparse.ArgumentParser(); ap.add_argument('--perms', type=int, default=3); ap.add_argument('--out', required=True)
ap.add_argument('payloads', nargs='+'); a = ap.parse_args()
dev = torch.device('cuda'); cache = {}
out = json.load(open(a.out)) if Path(a.out).exists() else {}
for path in a.payloads:
    if path in out: continue
    cat = [c for c in SYN if f'/{c}/' in path or f'{c}_' in path][0]
    if cat not in cache: cache[cat] = val_refs(cat)
    V = cache[cat]; S = torch.load(path, map_location='cpu', weights_only=False)['samples'].float()
    n = len(V) // 8 * 8; rows = []
    for p in range(a.perms):
        R = V[torch.randperm(len(V), generator=torch.Generator().manual_seed(p))[:n]]
        Sp = S[torch.randperm(len(S), generator=torch.Generator().manual_seed(100 + p))[:n]]
        g = [group(Sp[i:i + 8], R[i:i + 8], dev) for i in range(0, n, 8)]
        rows.append({k: float(np.mean([x[k] for x in g])) for k in g[0]})
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    sd = {k: float(np.std([r[k] for r in rows])) for k in rows[0]}
    out[path] = {'category': cat, 'n_val_used': n, 'perms': a.perms, 'mean': m, 'std': sd, 'per_perm': rows}
    json.dump(out, open(a.out, 'w'), indent=1)
    print(path, ' '.join(f'{k}={m[k]:.4g}' for k in m), flush=True)
