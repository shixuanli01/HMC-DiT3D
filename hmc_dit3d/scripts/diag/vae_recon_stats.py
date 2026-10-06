"""Condition-space diagnostics for an HMC condition VAE against its train bank (no DiT involved)."""
import sys, torch
from hmc_dit3d.data.hmc_condition_bank import load_hmc_condition_bank
from hmc_dit3d.hmc.condition_vae import load_condition_vae_checkpoint

bank_path, vae_path = sys.argv[1], sys.argv[2]
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
dev = 'cuda'
bank = load_hmc_condition_bank(bank_path)
vae, meta = load_condition_vae_checkpoint(vae_path, device=dev)
D = bank.descriptors.to(dev); S = [s.to(dev) for s in bank.sequences]
N = len(bank); Ls = [s.shape[1] for s in S]
print(f'bank N={N} split={bank.split} desc_dim={D.shape[1]} seq_lens={Ls}')
print('vae meta:', {k: meta.get(k) for k in ('latent_dim', 'hidden_dims', 'beta', 'epochs', 'seed', 'bank_size', 'bank_split')},
      'params(M)=%.1f' % (sum(p.numel() for p in vae.parameters()) / 1e6))

def ent(p): return -(p.clamp_min(1e-12).log() * p).sum(-1)
with torch.no_grad():
    mu, logvar = vae.encode(D, S)
    kl_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0)
    print(f'|mu| mean={mu.norm(dim=1).mean():.2f}  post std mean={logvar.mul(.5).exp().mean():.3f}  KL total={kl_dim.sum():.1f} nats  '
          f'active dims(KL>0.01)={(kl_dim > 0.01).sum().item()}/{mu.shape[1]}  agg-post std per dim: mean={mu.std(0).mean():.3f}')
    dn, logits = vae.decode(mu)
    dt = (D - vae.descriptor_mean) / vae.descriptor_std
    print(f'descriptor recon MSE (z-scored) = {torch.mean((dn - dt) ** 2):.4f}  (1.0 = predicting the mean)')
    rec = vae.decode_conditions(mu, sequence_threshold=thr)
    rec0 = vae.decode_conditions(mu, sequence_threshold=0.0)
    g = torch.Generator(device=dev).manual_seed(0)
    pri = vae.sample(N, device=dev, temperature=1.2, sequence_threshold=thr, generator=g)
    print(f'--- per scale (threshold={thr}) ---')
    for i, L in enumerate(Ls):
        t, r, r0, p = S[i], rec['sequences'][i], rec0['sequences'][i], pri['sequences'][i]
        ce = -(t * torch.log_softmax(logits[i], -1)).sum(-1)
        klrec = (ce - ent(t)).mean()
        tv = 0.5 * (t - r).abs().sum(-1).mean()
        tm, rm = t > 0, r > 0
        inter = (tm & rm).sum(-1).float()
        prec = (inter / rm.sum(-1).clamp_min(1)).mean(); recl = (inter / tm.sum(-1).clamp_min(1)).mean()
        # mass the decoder puts on voxels that are empty in the target
        leak = (r0 * (~tm)).sum(-1).mean()
        # baseline: how far apart are two different train shapes / a shape and the dataset mean
        perm = torch.randperm(N, device=dev)
        tv_pair = 0.5 * (t - t[perm]).abs().sum(-1).mean()
        tv_mean = 0.5 * (t - t.mean(0, keepdim=True)).abs().sum(-1).mean()
        print(f'scale[{i}] L={L}: occupancy data={tm.sum(-1).float().mean():.0f} recon={rm.sum(-1).float().mean():.0f} prior={(p>0).sum(-1).float().mean():.0f} | '
              f'perplexity data={ent(t).exp().mean():.0f} recon={ent(r).exp().mean():.0f} prior={ent(p).exp().mean():.0f}')
        print(f'          KL(target||recon)={klrec:.3f} nats | TV(target,recon)={tv:.3f}  [TV to random other shape={tv_pair:.3f}, to dataset mean={tv_mean:.3f}] | '
              f'support precision={prec:.3f} recall={recl:.3f} | off-support mass before threshold={leak:.3f}')
    # nearest-neighbour structure in condition space (coarsest two scales, L1)
    def feat(seqs): return torch.cat([seqs[0], seqs[1]], -1)
    T = feat(S)
    def nn_stats(Q, exclude_self=False):
        d = torch.cdist(Q, T, p=1)
        if exclude_self: d.fill_diagonal_(float('inf'))
        m = d.min(1)
        return m.values.mean().item(), m.indices.unique().numel() / N
    for name, Q, ex in (('train (leave-one-out)', T, True), ('recon(mu)', feat(rec['sequences']), False), ('prior T=1.2', feat(pri['sequences']), False)):
        dist, cov = nn_stats(Q, ex)
        print(f'cond-space NN [{name}]: mean L1 to nearest train cond={dist:.3f}  frac of train conds that are someone\'s NN={cov:.3f}')
    idx = torch.cdist(feat(rec['sequences']), T, p=1).argmin(1)
    print(f'recon retrieves its own source as NN: {(idx == torch.arange(N, device=dev)).float().mean():.3f}')
