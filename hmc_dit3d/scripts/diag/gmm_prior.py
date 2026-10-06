"""Full-covariance Gaussian mixture fitted by EM in torch (no sklearn in the venv)."""
import math, torch

def fit_gmm(x, k, iters=200, reg=1e-4, generator=None):
    n, d = x.shape
    means = x[torch.randperm(n, generator=generator, device=x.device)[:k]].clone()
    cov = torch.cov(x.T) + reg * torch.eye(d, device=x.device, dtype=x.dtype)
    covs = cov.expand(k, d, d).clone(); w = torch.full((k,), 1.0 / k, device=x.device, dtype=x.dtype)
    prev = -math.inf
    for _ in range(iters):
        L = torch.linalg.cholesky(covs)
        diff = x[None] - means[:, None]                                   # k,n,d
        sol = torch.linalg.solve_triangular(L, diff.transpose(1, 2), upper=False)  # k,d,n
        logp = -0.5 * sol.square().sum(1) - torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)[:, None] - 0.5 * d * math.log(2 * math.pi)
        logp = logp + torch.log(w)[:, None]                               # k,n
        ll = torch.logsumexp(logp, 0); r = torch.exp(logp - ll)           # k,n
        nk = r.sum(1) + 1e-10; w = nk / n
        means = (r @ x) / nk[:, None]
        diff = x[None] - means[:, None]
        covs = torch.einsum('kn,knd,kne->kde', r, diff, diff) / nk[:, None, None] + reg * torch.eye(d, device=x.device, dtype=x.dtype)
        cur = ll.mean().item()
        if abs(cur - prev) < 1e-6 * abs(cur): break
        prev = cur
    return {'w': w, 'means': means, 'chol': torch.linalg.cholesky(covs)}

def sample_gmm(g, n, temperature=1.0, generator=None):
    comp = torch.multinomial(g['w'], n, replacement=True, generator=generator)
    eps = torch.randn(n, g['means'].shape[1], 1, device=g['means'].device, dtype=g['means'].dtype, generator=generator)
    return g['means'][comp] + temperature * (g['chol'][comp] @ eps).squeeze(-1)
