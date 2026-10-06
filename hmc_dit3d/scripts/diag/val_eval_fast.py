"""Same protocol and same numbers as scripts/diag/val_eval_mmd.py, but faster: all pairs of several 8-vs-8 groups go through
the CUDA EMD extension in one batch, and reference-vs-reference matrices (independent of the samples) are cached per category."""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.environ.get('EMD_BUILD', '/workspace/emd_build'))
import emd_cuda
from hmc_dit3d.metrics.topodit_batch8 import knn_accuracy, coverage
ROOT = os.environ.get('SHAPENET_ROOT', '/workspace/HMC-DiT3D/ShapeNetCore.v2.PC15k')
SYN = {'chair': '03001627', 'airplane': '02691156', 'car': '02958343'}
dev = torch.device('cuda'); G = int(os.environ.get('EVAL_G', 2))  # groups per EMD batch (G * 64 pairs, ~1 GB each)
def val_refs(cat):
    fs = sorted(glob.glob(f'{ROOT}/{SYN[cat]}/val/*.npy'))
    return torch.stack([torch.from_numpy(np.load(f)[10000:12048]).float() for f in fs])
def pair_mats(A, B):
    """A, B: (g, 8, N, 3) on device -> CD and EMD matrices (g, 8, 8), entry [i, j] = d(A_i, B_j)."""
    g, k = A.shape[0], A.shape[1]
    x = A[:, :, None].expand(g, k, k, -1, 3).reshape(g * k * k, -1, 3).contiguous(); y = B[:, None].expand(g, k, k, -1, 3).reshape(g * k * k, -1, 3).contiguous()
    cds = []
    for i in range(0, len(x), 64):
        d = torch.cdist(x[i:i + 64], y[i:i + 64]).square(); cds.append(d.min(2).values.mean(1) + d.min(1).values.mean(1))
    m = emd_cuda.approxmatch_forward(x, y); e = emd_cuda.matchcost_forward(x, y, m) / x.shape[1]
    return torch.cat(cds).view(g, k, k).cpu(), e.view(g, k, k).cpu()
def all_groups(A, B):
    A = A.view(-1, 8, *A.shape[1:]); B = B.view(-1, 8, *B.shape[1:]); cd, em = [], []
    for i in range(0, len(A), G):
        for attempt in range(60):  # other jobs share the GPU: wait and retry on out-of-memory
            try:
                c, e = pair_mats(A[i:i + G].to(dev), B[i:i + G].to(dev)); break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); time.sleep(10)
        else: raise RuntimeError('GPU out of memory after retries')
        cd.append(c); em.append(e)
    return torch.cat(cd), torch.cat(em)
ap = argparse.ArgumentParser(); ap.add_argument('--perms', type=int, default=3); ap.add_argument('--out', required=True); ap.add_argument('payloads', nargs='+'); a = ap.parse_args()
cache = {}; out = json.load(open(a.out)) if Path(a.out).exists() else {}
for path in a.payloads:
    if path in out: continue
    cat = [c for c in SYN if f'/{c}/' in path or f'{c}_' in path][0]
    if cat not in cache:
        V = val_refs(cat); n = len(V) // 8 * 8; rrp = Path(f'results/vae_retrain/{cat}/rr_cache_perms{a.perms}.pt')
        if rrp.exists(): rr = torch.load(rrp)
        else:
            rr = [all_groups(R, R) for R in (V[torch.randperm(len(V), generator=torch.Generator().manual_seed(p))[:n]] for p in range(a.perms))]
            torch.save(rr, str(rrp) + f'.tmp{os.getpid()}'); os.replace(str(rrp) + f'.tmp{os.getpid()}', rrp)
        cache[cat] = (V, n, rr)
    V, n, rr = cache[cat]; S = torch.load(path, map_location='cpu', weights_only=False)['samples'].float(); rows = []
    for p in range(a.perms):
        R = V[torch.randperm(len(V), generator=torch.Generator().manual_seed(p))[:n]]
        Sp = S[torch.randperm(len(S), generator=torch.Generator().manual_seed(100 + p))[:n]]
        rs_cd, rs_em = all_groups(R, Sp); ss_cd, ss_em = all_groups(Sp, Sp); rr_cd, rr_em = rr[p]
        g = [{'1-NNA-CD': 100 * knn_accuracy(rr_cd[i], rs_cd[i], ss_cd[i]), '1-NNA-EMD': 100 * knn_accuracy(rr_em[i], rs_em[i], ss_em[i]),
              'COV-CD': 100 * coverage(rs_cd[i]), 'COV-EMD': 100 * coverage(rs_em[i]),
              'MMD-CD': float(rs_cd[i].min(1).values.mean()), 'MMD-EMD': float(rs_em[i].min(1).values.mean())} for i in range(len(rs_cd))]
        rows.append({k: float(np.mean([x[k] for x in g])) for k in g[0]})
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}; sd = {k: float(np.std([r[k] for r in rows])) for k in rows[0]}
    out[path] = {'category': cat, 'n_val_used': n, 'perms': a.perms, 'mean': m, 'std': sd, 'per_perm': rows}
    json.dump(out, open(a.out, 'w'), indent=1)
    print(path, ' '.join(f'{k}={m[k]:.4g}' for k in m), flush=True)
