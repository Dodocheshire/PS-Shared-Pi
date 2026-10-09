"""Train-observed signed class-sharing factors with unchanged support/row energy."""
import json
import torch
from torch.nn import functional as F
from .common import ROOT,atomic_json,atomic_save,digest
from .compact_outputs import load_compact_checkpoint
from .protected_concept_forward import remapped_bank


def estimate_factor(covariance,factor):
    if covariance.shape!=(len(factor),len(factor)) or not torch.isfinite(covariance).all():
        raise ValueError('Invalid observed concept covariance')
    result=torch.zeros_like(factor,dtype=torch.float64)
    for group in range(factor.shape[1]):
        ids=factor[:,group].ne(0).nonzero().flatten()
        if not len(ids):continue
        local=covariance[ids][:,ids].double()
        eigenvalues,eigenvectors=torch.linalg.eigh(local)
        if eigenvalues[-1]<=1e-12:
            result[ids,group]=factor[ids,group].double();continue
        vector=eigenvectors[:,-1]
        if vector[vector.abs().argmax()]<0:vector=-vector
        result[ids,group]=vector*eigenvalues[-1].clamp_min(0).sqrt()
    missing=result.square().sum(1)==0
    result[missing]=factor[missing].double()
    desired=factor.double().square().sum(1)
    norm=result.square().sum(1)
    result*=torch.where(norm>0,(desired/norm.clamp_min(1e-30)).sqrt(),torch.zeros_like(norm))[:,None]
    result=result.to(factor.dtype)
    if result[factor==0].ne(0).any():raise RuntimeError('Signed geometry exceeds sharing support')
    if not torch.allclose(result.square().sum(1),factor.square().sum(1),atol=2e-7,rtol=2e-7):
        raise RuntimeError('Initial group variance per coordinate changed')
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    baseline=ROOT/'outputs/cub/baseline/seed42/corrected_best_validation.pt'
    compact=ROOT/'outputs/cub/compact_baselines/seed42/model.pt'
    cbl,pack=load_compact_checkpoint(compact)
    original=torch.load(baseline,weights_only=False,map_location='cpu')
    bank=remapped_bank(original,pack);positive=bank['factor'].clone()
    directory=ROOT/'artifacts/cub';train=torch.load(directory/'train_target.pt',weights_only=True)
    views=[];view_hashes=[]
    for view in range(2):
        path=directory/f'augmentation_pairs_v13/view{view}.pt';item=torch.load(path,weights_only=False,map_location='cpu')
        if not torch.equal(item['indices'],train['indices']) or item['protocol']['split']!='train':
            raise RuntimeError('Observed training view alignment differs')
        if item['protocol']['val_images_opened'] or item['protocol']['test_images_opened']:
            raise RuntimeError('Observed geometry uses held-out images')
        views.append(item['features']);view_hashes.append(digest(path))
    def concepts(features):
        return torch.cat([(cbl(x)-pack['train_mean'])/pack['train_std'] for x in features.split(2048)])
    clean=concepts(train['features'])
    residual=torch.cat([concepts(x)-clean for x in views]).double()
    residual-=residual.mean(0);residual*=bank['active_mask']
    observed=residual.T@residual/len(residual)
    signed=estimate_factor(observed,positive)
    bank['mean_factor']=positive;bank['factor']=signed;bank['group_directions']=signed.T.contiguous()
    support=(positive.ne(0).float()@positive.ne(0).float().T)>0
    support.fill_diagonal_(False)
    measured=observed[support];relevant=measured.abs()>1e-8
    def comparison(factor):
        values=(factor.double()@factor.double().T)[support]
        coefficient=(measured@values/values.square().sum().clamp_min(1e-30)).clamp_min(0)
        return {'offdiagonal_cosine':float(F.cosine_similarity(measured,values,dim=0)),
                'nonnegative_fit_relative_error':float((measured-coefficient*values).norm()/measured.norm().clamp_min(1e-30)),
                'fit_coefficient_diagnostic_only':float(coefficient)}
    protocol={'iteration':21,'baseline_sha256':digest(baseline),'compact_checkpoint_sha256':digest(compact),
              'view_sha256':view_hashes,'source_sha256':digest(__file__),'split':'train','rows':len(train['labels']),
              'view_pairs':2,'validation_used':False,'test_used':False,
              'rule':'top observed covariance eigenvector within each original sharing support; row normalization preserves original group variance; original factor retained for mean decoder',
              'uses_true_class_for_conditioning':False}
    folder=directory/'signed_shared_geometry_v21';folder.mkdir(exist_ok=True)
    guard=folder/'protocol.json'
    if guard.exists() and json.loads(guard.read_text())!=protocol:raise RuntimeError('Signed geometry protocol changed')
    if (folder/'bank.pt').exists():raise RuntimeError('Frozen geometry already exists; do not overwrite')
    atomic_json(protocol,guard);atomic_save({'bank':bank,'protocol':protocol},folder/'bank.pt')
    diagnostic={**protocol,'bank_sha256':digest(folder/'bank.pt'),'active_coordinates':int(bank['active_mask'].sum()),
                'negative_observed_offdiagonal_fraction':float((measured[relevant]<0).double().mean()),
                'original_positive_factor':comparison(positive),'observed_signed_factor':comparison(signed),
                'max_row_variance_delta':float((positive.square().sum(1)-signed.square().sum(1)).abs().max()),
                'negative_factor_loadings':int((signed<0).sum()),'mean_decoder_unchanged':True,
                'scope':'Training covariance approximation, not validation performance or noise-gain evidence; covariance is PSD by construction and only original connected concepts share noise.'}
    atomic_json(diagnostic,folder/'diagnostic.json')
    atomic_json(diagnostic,ROOT.parent/'Essay/PS-Shared/report/signed_geometry_diagnostic.json')
    print(json.dumps({k:diagnostic[k] for k in ['negative_observed_offdiagonal_fraction','original_positive_factor','observed_signed_factor','max_row_variance_delta']},indent=2),flush=True)


if __name__=='__main__':main()
