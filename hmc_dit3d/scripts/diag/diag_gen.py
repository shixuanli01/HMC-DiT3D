"""Diagnostic generation: same test references as the VAE protocol, swap the condition source.
modes: vae (baseline), trainbank (random train conditions), gmm (GMM on train posterior means),
trainrecon (random train conditions passed through VAE encode->decode; measures decoder loss only),
testref (paired test conditions; oracle, diagnostic only)."""
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
p.add_argument('--guidance',type=float,default=1.0);p.add_argument('--seed',type=int,default=None);p.add_argument('--gmm-k',type=int,default=32);p.add_argument('--gmm-temp',type=float,default=1.0);p.add_argument('--temperature',type=float,default=1.2);p.add_argument('--threshold',type=float,default=0.1)
a=p.parse_args()
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
S.load_condition_vae_checkpoint=lambda *x,**k:(lambda r:(Wrap(r[0]),r[1]))(orig_load(*x,**k))
payload=S.generate_vae_conditioned_samples(cfg,Path(a.checkpoint),Path(a.vae),split='test',limit=N,
    vae_temperature=a.temperature,vae_sequence_threshold=a.threshold,generation_batch_size=a.gen_bs,hmc_guidance_scale=a.guidance)
payload['diag_mode']=a.mode;payload['diag_seed']=cfg.train.seed
S.save_sample_payload(payload,Path(a.out))
print('saved',a.out)
