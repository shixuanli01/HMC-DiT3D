"""Turn (possibly smooth) HMC measures into conditions that look like the DiT's training-time ones.

The DiT (hmc_point_source=sampled) is trained on conditions extracted from the 2,048 points it denoises.
Given a finest-scale measure we draw n points multinomially, re-derive the coarser scales by pooling
(Hilbert order is nested, 8 children per parent) and recompute the descriptor exactly as the extractor does.
"""
import math
import torch


def descriptor_from_sequences(seqs, scales, q_orders, empty_box_epsilon=1e-12):
    x = torch.tensor([math.log(1.0 / 2 ** s) for s in scales], dtype=torch.float64, device=seqs[0].device)
    xc = x - x.mean()
    slope = lambda y: (y * xc).sum(-1) / (xc * xc).sum()  # y: (B, S)
    out = []
    for q in q_orders:
        if abs(q - 1.0) < 1e-8:
            y = torch.stack([torch.where(s > 0, s * torch.log(s.clamp_min(1e-300)), torch.zeros_like(s)).sum(-1) for s in seqs], -1)
            out.append(slope(y))
        else:
            z = torch.stack([torch.where(s > 0, s.pow(q), torch.zeros_like(s)).sum(-1) for s in seqs], -1)
            out.append(slope(torch.log(z.clamp_min(empty_box_epsilon))) / (q - 1.0))
    return torch.stack(out, -1)


def resample_conditions(cond, n, scales, q_orders, generator=None, delta=1e-8):
    fine_p = cond['sequences'][-1].double()
    B, L = fine_p.shape
    idx = torch.multinomial(fine_p, n, replacement=True, generator=generator)
    counts = torch.zeros(B, L, dtype=torch.float64, device=fine_p.device).scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float64))
    fine = counts / (n + delta)
    seqs = [fine.view(B, 8 ** s, -1).sum(-1) for s in scales]
    desc = descriptor_from_sequences(seqs, scales, q_orders)
    dt = cond['descriptors'].dtype
    return {'descriptors': desc.to(dt), 'sequences': [s.to(dt) for s in seqs]}
