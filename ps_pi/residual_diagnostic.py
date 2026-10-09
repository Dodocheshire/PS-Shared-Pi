"""Paired training-only sensitivity diagnostic for semantic versus residual feature noise."""
import argparse
import torch
from torch.nn import functional as F
from .common import ROOT,atomic_json,digest,official_module
from .joint_noise import FeatureNoise
from .adapt_predictive import marginal_concept_nll

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='cub',choices=['cifar10','cifar100','cub'])
    p.add_argument('--rows',type=int,default=512);p.add_argument('--samples',type=int,default=16);args=p.parse_args()
    torch.set_num_threads(8);directory=ROOT/'artifacts'/args.dataset
    baseline_path=ROOT/f'outputs/{args.dataset}/baseline/seed42/corrected_best_validation.pt'
    pack=torch.load(baseline_path,weights_only=False);data=torch.load(directory/'train_target.pt',weights_only=True)
    targets=torch.load(directory/'train_concept_labels.pt',weights_only=True)
    indices=torch.randperm(len(targets),generator=torch.Generator().manual_seed(661))[:args.rows]
    x=data['features'][indices];y=data['labels'][indices];target=targets[indices]
    CBL=official_module('model.cbl').ConceptBottleneckLayer
    cbl=CBL(x.shape[1],len(pack['concepts']),pack['config']['cbl_layer_num'],pack['config']['cbl_bias'])
    cbl.load_state_dict(pack['cbl_state']);raw=cbl(x);context=(raw-pack['train_mean'])/pack['train_std']
    records=[]
    for name,subdir,mode in [('clean','joint_feature_v5','clean'),('semantic_noise','joint_feature_v5','fixed_structured'),
                             ('residual_noise','joint_feature_residual_v6','fixed_structured')]:
        bank_path=directory/subdir/'directions.pt';bank=torch.load(bank_path,weights_only=False)
        noise=FeatureNoise(bank,len(pack['bias']),mode)
        draws=noise.draw(x,context,args.samples,generator=torch.Generator().manual_seed(662))
        prediction=cbl(draws.reshape(-1,x.shape[1])).reshape(len(draws),len(x),-1)
        z=(prediction-pack['train_mean'])/pack['train_std']
        probabilities=F.linear(z,pack['weight'],pack['bias']).softmax(-1).mean(0)
        me,ve=noise.energy(context)
        stack=torch.cat([bank['directions'],bank['group_directions']]);gram=stack@stack.T
        effective_rank=float(stack.square().sum()**2/gram.square().sum())
        record={'variant':name,'accuracy':float(probabilities.argmax(1).eq(y).float().mean()),
                'nll':float(-probabilities[torch.arange(len(y)),y].clamp_min(1e-30).log().mean()),
                'expected_concept_bce':float(F.binary_cross_entropy_with_logits(prediction,target.unsqueeze(0).expand_as(prediction))),
                'predictive_concept_bce':marginal_concept_nll(prediction,target).item(),
                'standardized_concept_drift_rms':float((z-context.unsqueeze(0)).square().mean().sqrt()),
                'expected_white_feature_variance_energy':float(ve),'mean_energy':float(me),
                'actual_white_feature_rms':float(((draws-x.unsqueeze(0))/torch.where(bank['feature_std']>1e-6,bank['feature_std'],torch.ones_like(bank['feature_std']))).square().mean().sqrt()),
                'direction_bank_effective_rank':effective_rank,
                'noise_covariance_effective_rank':0. if mode=='clean' else effective_rank,'bank_sha256':digest(bank_path)}
        records.append(record);print(record,flush=True)
    atomic_json({'dataset':args.dataset,'split':'train','rows':len(indices),'samples':args.samples,
                 'rows_seed':661,'sampling_seed':662,'training_indices':data['indices'][indices].tolist(),
                 'source_sha256':digest(__file__),'baseline_sha256':digest(baseline_path),'records':records,
                 'val_used':False,'test_used':False,
                 'limits':'Frozen encoder on a sampled training subset, common coefficient draws and matched expected standardized feature energy. Diagnostic only, not trained validation evidence.'},directory/'joint_feature_residual_v6/train_sensitivity.json')

if __name__=='__main__':main()
