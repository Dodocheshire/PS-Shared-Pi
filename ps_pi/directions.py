"""Training-only semantic feature directions for structured input noise in joint CBL learning."""
import argparse
import torch
from .common import ROOT,atomic_save,atomic_json,digest
from .noise import sharing_factor

def build_directions(features,targets,class_sets,class_count):
    if features.ndim!=2 or targets.ndim!=2 or len(features)!=len(targets):raise ValueError('Feature/target dimensions differ')
    if not torch.isfinite(features).all() or not ((targets==0)|(targets==1)).all():raise ValueError('Expected finite features and binary training pseudo labels')
    mean=features.mean(0,keepdim=True);std=features.std(0,keepdim=True)
    denominator=torch.where(std>1e-6,std,torch.ones_like(std))
    white=(features-mean)/denominator
    positive=targets.sum(0);negative=len(targets)-positive
    sums=targets.T@white
    valid=(positive>0)&(negative>0)
    directions=sums/positive.clamp_min(1).unsqueeze(1)-(white.sum(0,keepdim=True)-sums)/negative.clamp_min(1).unsqueeze(1)
    directions[~valid]=0
    directions=torch.nn.functional.normalize(directions,dim=1)
    factor=sharing_factor(class_sets,targets.shape[1],class_count)
    grouped=factor.T@directions;norm=grouped.norm(dim=1)
    usable=norm>1e-8
    scaled_factor=factor/torch.where(usable,norm,torch.ones_like(norm)).unsqueeze(0)
    scaled_factor[:,~usable]=0
    grouped=scaled_factor.T@directions
    return {'feature_mean':mean,'feature_std':std,'directions':directions,'factor':scaled_factor,'group_directions':grouped,
            'positive_count':positive,'valid_concepts':valid,'valid_groups':usable}

def expected_feature_energy(mean,diag,group,bank):
    # Whitened feature energy; independent coefficient samples have no cross terms.
    deterministic=(mean@bank['directions']).square().sum(-1)
    variance=diag.square()@bank['directions'].square().sum(-1)+group.square()@bank['group_directions'].square().sum(-1)
    dimension=bank['directions'].shape[1]
    return deterministic.mean()/dimension,variance.mean()/dimension

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='cub',choices=['cifar10','cifar100','cub']);args=parser.parse_args()
    torch.set_num_threads(8);directory=ROOT/'artifacts'/args.dataset
    data=torch.load(directory/'train_target.pt',weights_only=True)
    targets=torch.load(directory/'train_concept_labels.pt',weights_only=True)
    concept_bank=torch.load(directory/'concept_bank.pt',weights_only=False)
    baseline=torch.load(ROOT/f'outputs/{args.dataset}/baseline/seed42/corrected_best_validation.pt',weights_only=False)
    bank=build_directions(data['features'],targets,concept_bank['class_sets'],len(baseline['bias']))
    protocol={'dataset':args.dataset,'selection_split':'train','test_opened':False,'val_used':False,
              'source_sha256':digest(__file__),'training_features_sha256':digest(directory/'train_target.pt'),
              'training_pseudo_sha256':digest(directory/'train_concept_labels.pt'),'concept_bank_sha256':digest(directory/'concept_bank.pt'),
              'description':'positive-minus-negative training feature directions in standardized coordinates; sharing support projected to normalized class group directions'}
    bank['protocol']=protocol;path=directory/'joint_feature_v5/directions.pt';atomic_save(bank,path)
    status={'phase':'prepared','dataset':args.dataset,'features':bank['directions'].shape[1],
            'concepts':len(bank['directions']),'informative_concepts':int(bank['valid_concepts'].sum()),
            'shared_class_directions':int(bank['valid_groups'].sum()),'test_used':False,'val_used':False,
            'protocol':protocol,'artifact_sha256':digest(path)}
    atomic_json(status,directory/'joint_feature_v5/status.json');print(status,flush=True)
if __name__=='__main__':main()
