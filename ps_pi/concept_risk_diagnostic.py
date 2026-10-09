"""Training-only task-risk gradients at the protected concept-noise initialization."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from .common import ROOT,atomic_json,digest,seed_all
from .compact_outputs import load_compact_checkpoint
from .protected_concepts import ProtectedBottleneckNoise
from .protected_concept_forward import remapped_bank,sample_concepts
from .mixup_likelihood import marginal_soft_task_nll


def main():
    torch.set_num_threads(8);seed_all(0)
    original_path=ROOT/'outputs/cub/baseline/seed42/corrected_best_validation.pt'
    compact_path=ROOT/'outputs/cub/compact_baselines/seed42/model.pt'
    cbl,pack=load_compact_checkpoint(compact_path)
    original=torch.load(original_path,weights_only=False,map_location='cpu')
    bank=remapped_bank(original,pack)
    noise=ProtectedBottleneckNoise(bank,len(pack['bias']),'shared_noise',max_mean=.3)
    source=ROOT/'artifacts/cub/train_target.pt'
    train=torch.load(source,weights_only=True,map_location='cpu')
    ids=torch.randperm(len(train['labels']),generator=torch.Generator().manual_seed(7718))[:512]
    x=train['features'][ids]
    views=[x]
    for view in range(2):
        item=torch.load(ROOT/f'artifacts/cub/augmentation_pairs_v13/view{view}.pt',weights_only=False,map_location='cpu')
        if not torch.equal(item['indices'],train['indices']) or item['protocol']['split']!='train':
            raise RuntimeError('Training view order/split differs')
        views.append(item['features'][ids])
    choices=torch.randint(0,3,(len(ids),),generator=torch.Generator().manual_seed(7719))
    x=torch.stack(views)[choices,torch.arange(len(ids))]
    mixing=np.random.default_rng(7720);partner=torch.tensor(mixing.permutation(len(ids)))
    mix=float(mixing.beta(.2,.2));x=mix*x+(1-mix)*x[partner]
    soft=F.one_hot(train['labels'][ids],len(pack['bias'])).float();soft=mix*soft+(1-mix)*soft[partner]
    z,_,_=sample_concepts(cbl,noise,x,pack,32,torch.Generator().manual_seed(7721))
    logits=F.linear(z,pack['weight'],pack['bias'])
    expected=-(logits.log_softmax(-1)*soft.unsqueeze(0)).sum(-1).mean()
    marginal=marginal_soft_task_nll(logits,soft)
    ge=torch.autograd.grad(expected,noise.network[-1].bias,retain_graph=True)[0]
    gm=torch.autograd.grad(marginal,noise.network[-1].bias)[0]
    k=len(pack['concepts']);active=noise.active_mask.bool();groups=noise.factor.ne(0).any(0)
    chunks={}
    for name,index in [('mean',torch.arange(k)[active]),('diagonal_scale',torch.arange(k,2*k)[active]),
                       ('shared_scale',torch.arange(2*k,2*k+len(pack['bias']))[groups])]:
        e=ge[index];m=gm[index]
        chunks[name]={'coordinates':len(index),'expected_gradient_norm':float(e.norm()),
                      'marginal_gradient_norm':float(m.norm()),
                      'expected_positive_fraction':float((e>0).float().mean()),
                      'marginal_positive_fraction':float((m>0).float().mean()),
                      'cosine':float(F.cosine_similarity(e,m,dim=0)),
                      'expected_gradient_sum':float(e.sum()),'marginal_gradient_sum':float(m.sum())}
    report={'split':'train','validation_used':False,'test_used':False,'rows':len(ids),'samples':32,
            'subset_seed':7718,'view_seed':7719,'mixup_seed':7720,'mc_seed':7721,'mixup_weight':mix,
            'baseline_sha256':digest(original_path),'compact_checkpoint_sha256':digest(compact_path),
            'train_data_sha256':digest(source),'source_sha256':digest(__file__),
            'expected_task_ce':float(expected.detach()),'marginal_task_nll':float(marginal.detach()),'gradient_chunks':chunks,
            'interpretation':'Positive scale-logit gradient is an initial shrinking tendency under descent. This compares task terms only on one fixed training subset, not complete optimizer updates or validation gains.',
            'next_change':'If expected task risk shrinks random directions, test marginal class likelihood under the same protected architecture and all controls; do not infer performance from diagnostic.'}
    atomic_json(report,ROOT.parent/'Essay/PS-Shared/report/concept_risk_diagnostic.json')
    print(json.dumps({k:report[k] for k in ['expected_task_ce','marginal_task_nll','gradient_chunks']},indent=2),flush=True)


if __name__=='__main__':main()
