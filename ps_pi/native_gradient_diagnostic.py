"""Measure objective gradient balance on a fixed training-only mixed subset."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from .common import ROOT, atomic_json, digest, official_module
from .native_training import training_attribute_masses, balanced_native_loss


def main():
    torch.set_num_threads(8)
    directory=ROOT/'artifacts/cub';baseline=ROOT/'outputs/cub/baseline/seed42/corrected_best_validation.pt'
    pack=torch.load(baseline,weights_only=False)
    data=torch.load(directory/'train_target.pt',weights_only=True)
    pseudo=torch.load(directory/'train_concept_labels.pt',weights_only=True)
    annotations=torch.load(directory/'native_attributes_v1/train_val.pt',weights_only=True)['train']
    protocol=json.loads((directory/'native_attributes_v1/protocol.json').read_text())
    groups,thresholds,positive,negative=training_attribute_masses(annotations,protocol,data['indices'],len(pack['concepts']))
    context=torch.load(directory/'baseline_concepts'/digest(baseline)/'train.pt',weights_only=True)
    CBL=official_module('model.cbl').ConceptBottleneckLayer
    model=CBL(2048,len(pack['concepts']),pack['config']['cbl_layer_num'],pack['config']['cbl_bias'])
    model.load_state_dict(pack['cbl_state']);model.eval()
    ids=torch.randperm(len(data['features']),generator=torch.Generator().manual_seed(7715))[:512]
    rng=np.random.default_rng(7716);other=torch.as_tensor(rng.permutation(len(ids)))
    mixing=float(rng.beta(.2,.2));x=data['features'][ids]
    x=mixing*x+(1-mixing)*x[other]
    target=pseudo[ids];target=mixing*target+(1-mixing)*target[other]
    labels=F.one_hot(data['labels'][ids],num_classes=len(pack['bias'])).float()
    labels=mixing*labels+(1-mixing)*labels[other]
    p=positive[ids];n=negative[ids]
    p=mixing*p+(1-mixing)*p[other];n=mixing*n+(1-mixing)*n[other]
    ctx=context[ids];ctx=mixing*ctx+(1-mixing)*ctx[other]
    raw=model(x);z=(raw-pack['train_mean'])/pack['train_std'];logits=F.linear(z,pack['weight'],pack['bias'])
    losses={'task':-(labels*logits.log_softmax(-1)).sum(1).mean(),
            'pseudo_concept':F.binary_cross_entropy_with_logits(raw,target),
            'native_attribute':balanced_native_loss(raw.unsqueeze(0),groups,thresholds,p,n),
            'preservation':(z-ctx).square().mean()}
    gradients={}
    for name,loss in losses.items():
        grad=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True)
        gradients[name]=torch.cat([g.reshape(-1) for g in grad]).double()
    norms={name:float(g.norm()) for name,g in gradients.items()}
    if not all(np.isfinite(v) and v>0 for v in norms.values()):
        raise RuntimeError('Expected nonzero finite objective gradient norms')
    cosines={f'{a}/{b}':float(torch.dot(gradients[a],gradients[b])/(norms[a]*norms[b]))
             for i,a in enumerate(gradients) for b in list(gradients)[i+1:]}
    suggested=min(1.,.25*norms['task']/norms['native_attribute'])
    result={'dataset':'cub','split':'train','training_subset':len(ids),'subset_seed':7715,
            'mixup_rng_seed':7716,'mixup_alpha':.2,'mixing':mixing,'original_indices':data['indices'][ids].tolist(),
            'losses':{k:float(v.detach()) for k,v in losses.items()},'gradient_norms':norms,'gradient_cosines':cosines,
            'candidate_native_weight':suggested,'candidate_rule':'min(1, 0.25 * task gradient norm / native gradient norm), measured only at the fixed baseline on the fixed mixed training subset',
            'source_sha256':digest(__file__),'baseline_sha256':digest(baseline),
            'train_features_sha256':digest(directory/'train_target.pt'),
            'native_protocol_sha256':digest(directory/'native_attributes_v1/protocol.json'),
            'native_train_val_sha256':digest(directory/'native_attributes_v1/train_val.pt'),
            'validation_used':False,'test_used':False,'future_cohort_registered':False,
            'limits':'Local training-gradient diagnostic, not evidence of better validation/test performance or causal noise gain. It does not retrospectively change the registered V15 objective or its selection gate. Any calibrated weight requires a new fully matched prospective training cohort.'}
    atomic_json(result,ROOT.parent/'Essay/PS-Shared/report/native_gradient_diagnostic.json')
    print('Training gradient norms:',norms,'cosines:',cosines,'prospective native weight:',suggested,flush=True)


if __name__=='__main__':main()
