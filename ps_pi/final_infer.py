"""Load the finalized CUB model and predict from frozen RN50 features."""
import argparse
import torch
from torch.nn import functional as F
from .common import ROOT, atomic_save
from .compact_outputs import load_compact_checkpoint
from .final_geometry import remapped_bank
from .anchored_cbl import AnchoredCBL
from .signed_shared_noise import SignedSharedNoise
from .protected_concept_forward import sample_concepts


def load_final(seed=42,mode='adapter_matched',device='cuda'):
    """Default deterministic adapter was chosen from V22 validation before tests."""
    checkpoint=ROOT/f'outputs/cub/final_v1/seed{seed}/{mode}_seed{seed-42}_cw5_lr3e-06/model.pt'
    saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
    cbl,pack=load_compact_checkpoint(ROOT/f'outputs/cub/compact_baselines/seed{seed}/model.pt')
    original=torch.load(ROOT/f'outputs/cub/baseline/seed{seed}/corrected_best_validation.pt',weights_only=False,map_location='cpu')
    bank=remapped_bank(original,pack);config=saved['protocol']['config']
    anchored=AnchoredCBL(cbl,pack['native_protocol']).to(device).eval()
    anchored.load_state_dict(saved['state']['cbl'])
    noise=SignedSharedNoise(bank,len(pack['bias']),mode,config['hidden'],config['max_mean'],config['initial_scale'],config['min_scale'],config['max_scale']).to(device).eval()
    noise.load_state_dict(saved['state']['noise'])
    for key in ['train_mean','train_std','weight','bias']:pack[key]=pack[key].to(device)
    return anchored,noise,pack


@torch.no_grad()
def predict(features,cbl,noise,pack,samples=256,mc_seeds=(9071,9072)):
    device=pack['weight'].device;results=[]
    for seed in mc_seeds:
        generator=torch.Generator(device=device).manual_seed(seed);probabilities=[]
        for batch in features.split(128):
            z,_,_=sample_concepts(cbl,noise,batch.to(device),pack,samples,generator)
            probabilities.append(F.linear(z,pack['weight'],pack['bias']).softmax(-1).mean(0).cpu())
        results.append(torch.cat(probabilities))
    return torch.stack(results).mean(0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,choices=[42,43,44,45,46],default=42)
    p.add_argument('--mode',choices=['adapter_matched','shared_noise','shared_no_sampling','diag_noise','mean_only','clean','fixed_isotropic'],default='adapter_matched')
    args=p.parse_args();torch.set_num_threads(8)
    item=torch.load(args.features,weights_only=True,map_location='cpu');features=item['features'] if isinstance(item,dict) else item
    if features.ndim!=2 or features.shape[1]!=2048 or not torch.isfinite(features).all():raise ValueError('Expected finite [N,2048] frozen RN50 features')
    probabilities=predict(features,*load_final(args.seed,args.mode))
    atomic_save({'probabilities':probabilities,'predictions':probabilities.argmax(-1),'mode':args.mode,'baseline_seed':args.seed},args.output)


if __name__=='__main__':main()
