"""Reproduce expected versus predictive concept BCE on the isolated validation set."""
import csv
import math
import torch
from torch.nn import functional as F
from .common import ROOT,atomic_json,digest
from .adapt_predictive import marginal_concept_nll

def main():
    torch.set_num_threads(8)
    baseline=ROOT/'outputs/cifar10/baseline/seed42/corrected_best_validation.pt'
    pack=torch.load(baseline,weights_only=False)
    z_path=ROOT/'artifacts/cifar10/baseline_concepts'/digest(baseline)/'val.pt'
    label_path=ROOT/'artifacts/cifar10/val_concept_labels.pt'
    z=torch.load(z_path,weights_only=True);target=torch.load(label_path,weights_only=True)
    raw=z*pack['train_std']+pack['train_mean']
    clean=F.binary_cross_entropy_with_logits(raw,target).item();rows=[]
    for scale in [.03,.1,.2,.3]:
        generator=torch.Generator().manual_seed(901);predictive=0.;expected=0.
        for start in range(0,len(z),256):
            value=z[start:start+256]
            eps=torch.randn(64,*value.shape,generator=generator);eps=torch.cat([eps,-eps],0)
            draws=(value.unsqueeze(0)+scale*eps)*pack['train_std']+pack['train_mean']
            pseudo=target[start:start+len(value)]
            predictive+=marginal_concept_nll(draws,pseudo).item()*len(value)
            expected+=F.binary_cross_entropy_with_logits(draws,pseudo.unsqueeze(0).expand_as(draws)).item()*len(value)
        row={'scale':scale,'clean_bce':clean,'predictive_bce':predictive/len(z),'expected_bce':expected/len(z)}
        if row['predictive_bce']>row['expected_bce']+1e-6:raise RuntimeError('Jensen inequality diagnostic failed')
        rows.append(row)
    report=ROOT.parent/'Essay/PS-Shared/report'
    atomic_json({'dataset':'cifar10','split':'val','n':len(z),'mc_samples':128,'antithetic':True,'seed':901,'test_used':False,
                 'baseline_sha256':digest(baseline),'concepts_sha256':digest(z_path),'pseudo_labels_sha256':digest(label_path),
                 'source_sha256':digest(__file__),'rows':rows},report/'concept_likelihood_diagnostic.json')
    with (report/'concept_likelihood_diagnostic.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(rows,flush=True)
if __name__=='__main__':main()
