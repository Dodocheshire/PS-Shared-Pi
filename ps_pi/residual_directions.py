"""Train-only within-class covariance directions orthogonal to concept mean contrasts."""
import argparse
import torch
from torch.nn import functional as F
from .common import ROOT,atomic_json,atomic_save,digest
from .noise import sharing_factor

def build_residual_directions(features,labels,targets,semantic_bank,class_sets,class_count,rank=64,seed=707):
    if len(features)!=len(labels) or len(targets)!=len(features):raise ValueError('Training rows differ')
    if rank<1:raise ValueError('Rank must be positive')
    if not torch.isfinite(features).all() or not ((targets==0)|(targets==1)).all():raise ValueError('Invalid training data')
    if labels.min()<0 or labels.max()>=class_count:raise ValueError('Class labels out of range')
    mean=semantic_bank['feature_mean'];std=semantic_bank['feature_std']
    white=(features.double()-mean.double())/torch.where(std>1e-6,std,torch.ones_like(std)).double()
    centroids=white.new_zeros(class_count,white.shape[1]);centroids.index_add_(0,labels,white)
    counts=torch.bincount(labels,minlength=class_count)
    centroids/=counts.clamp_min(1).unsqueeze(1)
    residual=white-centroids[labels]
    semantic=semantic_bank['directions'].double()
    _,singular,vh=torch.linalg.svd(semantic,full_matrices=False)
    semantic_rank=int((singular>singular.max()*1e-6).sum()) if singular.max()>0 else 0
    q=vh[:semantic_rank]
    if len(q):residual=residual-(residual@q.T)@q
    covariance=residual.T@residual/max(len(residual)-int((counts>0).sum()),1)
    values,vectors=torch.linalg.eigh(covariance)
    available=int((values>values.max().clamp_min(1e-12)*1e-8).sum())
    rank=min(rank,available,white.shape[1]-semantic_rank)
    if rank<1:raise ValueError('No residual subspace remains outside the concept mean span')
    values=values[-rank:].flip(0);basis=vectors[:,-rank:].flip(1)
    # Fix global eigenvector sign without using concept/validation labels.
    anchors=basis.abs().argmax(0);sign=basis[anchors,torch.arange(rank)].sign()
    basis=basis*sign.unsqueeze(0)
    coordinates=residual@basis
    positive=targets.double().sum(0)
    first=targets.double().T@coordinates/positive.clamp_min(1).unsqueeze(1)
    outer=(coordinates.unsqueeze(2)*coordinates.unsqueeze(1)).flatten(1)
    second=(targets.double().T@outer/positive.clamp_min(1).unsqueeze(1)).reshape(len(positive),rank,rank)
    local_cov=second-first.unsqueeze(2)*first.unsqueeze(1)
    local_cov=.5*(local_cov+local_cov.transpose(1,2))
    fallback=(positive<2)|(local_cov.diagonal(dim1=1,dim2=2).sum(1)<=1e-10)
    local_cov[fallback]=torch.diag(values)
    eig,eigvectors=torch.linalg.eigh(local_cov)
    random=torch.randn(len(positive),rank,dtype=torch.float64,generator=torch.Generator().manual_seed(seed))
    coefficients=(eigvectors@(eig.clamp_min(0).sqrt()*(eigvectors.transpose(1,2)@random.unsqueeze(2)).squeeze(2)).unsqueeze(2)).squeeze(2)
    directions=F.normalize(coefficients@basis.T,dim=1).float()
    factor=sharing_factor(class_sets,len(targets[0]),class_count)
    grouped=factor.T@directions;norm=grouped.norm(dim=1);usable=norm>1e-8
    scaled_factor=factor/torch.where(usable,norm,torch.ones_like(norm)).unsqueeze(0)
    scaled_factor[:,~usable]=0;grouped=scaled_factor.T@directions
    return {'feature_mean':mean.clone(),'feature_std':std.clone(),'directions':directions,
            'factor':scaled_factor,'group_directions':grouped,'positive_count':positive.float(),
            'valid_concepts':directions.norm(dim=1)>1e-8,'valid_groups':usable,
            'residual_basis':basis.float(),'residual_eigenvalues':values.float(),
            'semantic_span_rank':semantic_rank,'residual_rank':rank,'fallback_concepts':fallback,
            'maximum_semantic_cosine':float((directions@semantic_bank['directions'].T).abs().max()),
            'construction_seed':seed}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='cub',choices=['cifar10','cifar100','cub'])
    p.add_argument('--rank',type=int,default=64);p.add_argument('--seed',type=int,default=707);args=p.parse_args()
    torch.set_num_threads(8);directory=ROOT/'artifacts'/args.dataset
    source=directory/'train_target.pt';data=torch.load(source,weights_only=True)
    pseudo_path=directory/'train_concept_labels.pt';targets=torch.load(pseudo_path,weights_only=True)
    semantic_path=directory/'joint_feature_v5/directions.pt';semantic=torch.load(semantic_path,weights_only=False)
    concept_bank=torch.load(directory/'concept_bank.pt',weights_only=False)
    baseline=torch.load(ROOT/f'outputs/{args.dataset}/baseline/seed42/corrected_best_validation.pt',weights_only=False)
    bank=build_residual_directions(data['features'],data['labels'],targets,semantic,concept_bank['class_sets'],len(baseline['bias']),args.rank,args.seed)
    protocol={'dataset':args.dataset,'rank_requested':args.rank,'seed':args.seed,'source_sha256':digest(__file__),
              'training_features_and_labels_sha256':digest(source),'training_pseudo_sha256':digest(pseudo_path),
              'semantic_bank_sha256':digest(semantic_path),'concept_bank_sha256':digest(directory/'concept_bank.pt'),
              'selection_split':'train','val_used':False,'test_used':False,
              'hypothesis':'Within-class feature variation outside mean-concept contrasts may regularize CBL while reducing semantic disruption.',
              'definition':'Training feature class residuals, projected outside concept mean span; rank-64 covariance. A fixed seeded draw from each positive-concept covariance provides a unit feature direction. Original sharing support is retained for latent coefficients.',
              'limits':'Orthogonal to training mean contrasts only; no guarantee of nonlinear CBL or native attribute invariance. Training class labels used for covariance construction; no class labels required at inference.'}
    bank['protocol']=protocol;output=directory/'joint_feature_residual_v6/directions.pt';atomic_save(bank,output)
    record={k:bank[k] for k in ['semantic_span_rank','residual_rank','maximum_semantic_cosine','construction_seed']}
    record.update(phase='prepared',concepts=len(bank['directions']),shared_class_directions=int(bank['valid_groups'].sum()),
                  fallback_concepts=int(bank['fallback_concepts'].sum()),protocol=protocol,artifact_sha256=digest(output))
    atomic_json(record,output.parent/'status.json');print(record,flush=True)

if __name__=='__main__':main()
