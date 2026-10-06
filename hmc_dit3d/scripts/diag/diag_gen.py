"""Diagnostic generation: same test references as the VAE protocol, swap the condition source.
modes: vae (baseline), trainbank (random train conditions), gmm (GMM on train posterior means),
trainrecon (random train conditions passed through VAE encode->decode; measures decoder loss only),
testref (paired test conditions; oracle, diagnostic only),
knninterp (interpolate between a random train posterior mean and one of its k nearest train neighbours),
trainpost (train posterior mean + posterior noise; diagnostic upper bound).
Options: --variance large (reverse variance beta_t), --resample N (redraw N points from the finest measure),
--fast (fp16 autocast, validated against fp32), --guidance G (classifier-free guidance on the HMC condition).
See PROGRESS_2026-10-06.md."""
import argparse,sys,torch
sys.path.insert(0,str(__import__('pathlib').Path(__file__).parent))
from pathlib import Path
import hmc_dit3d.train.sample as S
from hmc_dit3d.train.config import load_experiment_config
from hmc_dit3d.data.hmc_condition_bank import load_hmc_condition_bank
from hmc_dit3d.data.shapenet_pc15k import ShapeNetPC15KDataset
p=argparse.ArgumentParser()
p.add_argument('--config');p.add_argument('--checkpoint');p.add_argument('--vae');p.add_argument('--mode')
p.add_argument('--bank');p.add_argument('--out');p.add_argument('--gen-bs',type=int,default=166)
p.add_argument('--guidance',type=float,default=1.0);p.add_argument('--seed',type=int,default=None);p.add_argument('--gmm-k',type=int,default=32);p.add_argument('--gmm-temp',type=float,default=1.0);p.add_argument('--resample',type=int,default=0);p.add_argument('--variance',default='small');p.add_argument('--fast',action='store_true');p.add_argument('--knn-k',type=int,default=5);p.add_argument('--interp-max',type=float,default=0.5);p.add_argument('--interp-min',type=float,default=0.0);p.add_argument('--post-temp',type=float,default=1.0);p.add_argument('--temperature',type=float,default=1.2);p.add_argument('--threshold',type=float,default=0.1)
a=p.parse_args()
if a.fast:  # fp16 autocast + non-deterministic fast kernels (~3x faster; validate against an fp32 run before trusting)
    _sgs=S.set_global_seed;_lsc=S.load_sampling_checkpoint
    def _seed_fast(seed,*x,**k):
        _sgs(seed,deterministic=False);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    def _load_fast(*x,**k):
        model,diff,ck=_lsc(*x,**k);fwd=model.forward
        def fast_forward(*xx,**kk):
            with torch.autocast('cuda',dtype=torch.float16): return fwd(*xx,**kk).float()
        model.forward=fast_forward;return model,diff,ck
    S.set_global_seed=_seed_fast;S.load_sampling_checkpoint=_load_fast
if a.variance=='large':  # DDPM 'fixedlarge': reverse variance beta_t instead of beta_tilde_t
    from hmc_dit3d.train.diffusion import GaussianDiffusion as GD
    _pmv=GD.p_mean_variance
    def _pmv_large(self,denoise_fn,x_t,timesteps,clip_denoised=False):
        m,v,x0=_pmv(self,denoise_fn,x_t,timesteps,clip_denoised)
        return m,self._extract(self.betas.to(x_t.device),timesteps,x_t.shape),x0
    GD.p_mean_variance=_pmv_large
cfg=load_experiment_config(Path(a.config))
if a.seed is not None:
    import dataclasses
    cfg=dataclasses.replace(cfg,train=dataclasses.replace(cfg.train,seed=a.seed))
N=664
orig_load=S.load_condition_vae_checkpoint
class Wrap:
    def __init__(s,vae): s.vae=vae
    def __getattr__(s,k): return getattr(s.vae,k)
    def sample(s,n,device,**kw):
        if a.mode=='vae': return s.vae.sample(n,device=device,**kw)
        bank=load_hmc_condition_bank(a.bank)
        if a.mode=='gmm':  # aggregate-posterior prior: GMM fitted on train posterior means (fixes N(0,I) prior hole)
            from gmm_prior import fit_gmm,sample_gmm
            with torch.no_grad():
                mu=torch.cat([s.vae.encode(bank.descriptors[i:i+512].to(device),[x[i:i+512].to(device) for x in bank.sequences])[0] for i in range(0,len(bank),512)])
            g=torch.Generator(device=device).manual_seed(cfg.train.seed+7)
            gmm=fit_gmm(mu.double(),a.gmm_k,generator=g)
            z=sample_gmm(gmm,n,temperature=a.gmm_temp,generator=g).float()
            print(f'gmm k={a.gmm_k} |mu|={mu.norm(dim=1).mean():.2f} |z|={z.norm(dim=1).mean():.2f}',flush=True)
            return s.vae.decode_conditions(z,sequence_threshold=a.threshold)
        if a.mode=='knninterp':  # latent interpolation between a random train posterior mean and one of its k nearest train neighbours
            g=torch.Generator().manual_seed(cfg.train.seed+1)
            idx=torch.randperm(len(bank),generator=g)[:n]
            with torch.no_grad():
                mu=torch.cat([s.vae.encode(bank.descriptors[i:i+512].to(device),[x[i:i+512].to(device) for x in bank.sequences])[0] for i in range(0,len(bank),512)])
                nn_idx=torch.cdist(mu[idx.to(device)],mu).topk(a.knn_k+1,largest=False).indices[:,1:].cpu()
                j=nn_idx[torch.arange(n),torch.randint(0,a.knn_k,(n,),generator=g)]
                lam=(a.interp_min+torch.rand(n,1,generator=g)*(a.interp_max-a.interp_min)).to(device)
                z=mu[idx.to(device)]+lam*(mu[j.to(device)]-mu[idx.to(device)])
            print(f'knninterp k={a.knn_k} lam_max={a.interp_max} |z|={z.norm(dim=1).mean():.2f}',flush=True)
            return s.vae.decode_conditions(z,sequence_threshold=a.threshold)
        if a.mode=='trainpost':  # aggregate-posterior sample: z = mu_i + sigma_i*eps for random train i (diagnostic upper bound for any latent prior)
            idx=torch.randperm(len(bank),generator=torch.Generator().manual_seed(cfg.train.seed+1))[:n]
            with torch.no_grad():
                mu,lv=s.vae.encode(bank.descriptors[idx].to(device),[x[idx].to(device) for x in bank.sequences])
                if a.post_temp!=1.0:  # push latents away from (or towards) the train latent mean
                    m=torch.cat([s.vae.encode(bank.descriptors[i:i+512].to(device),[x[i:i+512].to(device) for x in bank.sequences])[0] for i in range(0,len(bank),512)]).mean(0,keepdim=True)
                    mu=m+a.post_temp*(mu-m)
                z=mu+(0.5*lv).exp()*torch.randn(mu.shape,device=device,generator=torch.Generator(device=device).manual_seed(cfg.train.seed+11))
            return s.vae.decode_conditions(z,sequence_threshold=a.threshold)
        if a.mode in ('trainbank','trainrecon'):
            idx=torch.randperm(len(bank),generator=torch.Generator().manual_seed(cfg.train.seed+1))[:n]
            if a.mode=='trainrecon':  # same train conditions as trainbank, but reconstructed through the VAE (posterior mean)
                with torch.no_grad():
                    mu=s.vae.encode(bank.descriptors[idx].to(device),[x[idx].to(device) for x in bank.sequences])[0]
                return s.vae.decode_conditions(mu,sequence_threshold=a.threshold)
        else:  # testref: reproduce reference permutation
            ds=ShapeNetPC15KDataset(root_dir=cfg.data.root_dir,categories=cfg.data.categories,split='test',sample_size=cfg.data.sample_size,random_subsample=False)
            ref=torch.randperm(len(ds),generator=torch.Generator().manual_seed(cfg.train.seed))[:n]
            pos={Path(pth).stem:i for i,pth in enumerate(bank.source_paths)}
            idx=torch.tensor([pos[Path(ds[int(j)]['path']).stem] for j in ref])
        return {'descriptors':bank.descriptors[idx].to(device),'sequences':[x[idx].to(device) for x in bank.sequences]}
class Wrap2(Wrap):
    def sample(s,n,device,**kw):
        c=Wrap.sample(s,n,device,**kw)
        if a.resample>0:
            from resample import resample_conditions
            g=torch.Generator(device=c['descriptors'].device).manual_seed(cfg.train.seed+13)
            c=resample_conditions(c,a.resample,list(cfg.hmc.scales),list(cfg.hmc.q_orders),generator=g)
            print('resample',a.resample,'occ',[round((x>0).sum(-1).float().mean().item()) for x in c['sequences']],'desc',c['descriptors'].mean(0).tolist(),flush=True)
        return c
S.load_condition_vae_checkpoint=lambda *x,**k:(lambda r:(Wrap2(r[0]),r[1]))(orig_load(*x,**k))
payload=S.generate_vae_conditioned_samples(cfg,Path(a.checkpoint),Path(a.vae),split='test',limit=N,
    vae_temperature=a.temperature,vae_sequence_threshold=a.threshold,generation_batch_size=a.gen_bs,hmc_guidance_scale=a.guidance)
payload['diag_mode']=a.mode;payload['diag_resample']=a.resample;payload['diag_variance']=a.variance;payload['diag_fast']=a.fast;payload['diag_seed']=cfg.train.seed
S.save_sample_payload(payload,Path(a.out))
print('saved',a.out)
