"""Noise directions estimated from paired weak views of training images."""
import torch
from torch.nn import functional as F
from .common import ROOT, atomic_json, atomic_save, digest
from .noise import sharing_factor


def build_augmentation_directions(features, views, targets, reference, class_sets, class_count, rank=64, seed=707):
    if features.ndim != 2 or views.ndim != 3 or tuple(views.shape[1:]) != tuple(features.shape):
        raise ValueError('Paired view and original feature shapes differ')
    if targets.ndim != 2 or len(targets) != len(features) or rank < 1:
        raise ValueError('Training target rows or requested rank invalid')
    if not torch.isfinite(features).all() or not torch.isfinite(views).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError('Finite paired features and binary original training targets required')
    std = reference['feature_std']
    denominator = torch.where(std > 1e-6, std, torch.ones_like(std)).double()
    differences = ((views.double()-features.double().unsqueeze(0))/denominator).flatten(0, 1)
    average = differences.mean(0)
    centered = differences-average
    covariance = centered.T@centered/max(len(centered)-1, 1)
    values, vectors = torch.linalg.eigh(covariance)
    available = int((values > values.max().clamp_min(1e-12)*1e-8).sum())
    rank = min(rank, available)
    if rank < 1:
        raise ValueError('No measured nuisance variance')
    trace = values.clamp_min(0).sum()
    values, basis = values[-rank:].flip(0), vectors[:, -rank:].flip(1)
    anchors = basis.abs().argmax(0)
    basis *= basis[anchors, torch.arange(rank)].sign().unsqueeze(0)
    coordinates = centered@basis
    expanded = targets.double().repeat(len(views), 1)
    positive = expanded.sum(0)
    first = expanded.T@coordinates/positive.clamp_min(1).unsqueeze(1)
    outer = (coordinates.unsqueeze(2)*coordinates.unsqueeze(1)).flatten(1)
    second = (expanded.T@outer/positive.clamp_min(1).unsqueeze(1)).reshape(len(positive), rank, rank)
    local = second-first.unsqueeze(2)*first.unsqueeze(1)
    local = .5*(local+local.transpose(1, 2))
    fallback = (targets.sum(0) < 2) | (local.diagonal(dim1=1, dim2=2).sum(1) <= 1e-10)
    local[fallback] = torch.diag(values)
    eigen, eigenvectors = torch.linalg.eigh(local)
    random = torch.randn(len(positive), rank, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
    projected = (eigenvectors.transpose(1, 2)@random.unsqueeze(2)).squeeze(2)
    coefficients = (eigenvectors@(eigen.clamp_min(0).sqrt()*projected).unsqueeze(2)).squeeze(2)
    directions = F.normalize(coefficients@basis.T, dim=1).float()
    factor = sharing_factor(class_sets, targets.shape[1], class_count)
    grouped = factor.T@directions
    norm = grouped.norm(dim=1); usable = norm > 1e-8
    factor = factor/torch.where(usable, norm, torch.ones_like(norm)).unsqueeze(0)
    factor[:, ~usable] = 0
    return {'feature_mean':reference['feature_mean'].clone(), 'feature_std':std.clone(),
            'directions':directions, 'factor':factor, 'group_directions':factor.T@directions,
            'positive_count':targets.sum(0), 'valid_concepts':directions.norm(dim=1)>1e-8,
            'valid_groups':usable, 'nuisance_basis':basis.float(), 'nuisance_eigenvalues':values.float(),
            'nuisance_rank':rank, 'nuisance_variance_fraction':float(values.sum()/trace),
            'observed_white_difference_rms':float(differences.square().mean().sqrt()),
            'observed_average_white_difference':average.float(), 'fallback_concepts':fallback,
            'maximum_semantic_cosine':float((directions@reference['directions'].T).abs().max()),
            'construction_seed':seed}


def main():
    torch.set_num_threads(8); directory=ROOT/'artifacts/cub'
    source=directory/'train_target.pt'; data=torch.load(source,weights_only=True)
    cache=directory/'augmentation_pairs_v13'
    status=json_import(cache/'status.json'); protocol=json_import(cache/'protocol.json')
    if status['phase']!='complete' or status['debug'] or protocol['split']!='train':
        raise RuntimeError('Full training augmentation cache required')
    assert not protocol['val_images_opened'] and not protocol['test_images_opened']
    views=[]
    for view,sha in enumerate(status['view_sha256']):
        path=cache/f'view{view}.pt'; assert digest(path)==sha
        pack=torch.load(path,weights_only=False)
        assert pack['protocol']==protocol
        assert torch.equal(pack['indices'],data['indices']) and torch.equal(pack['labels'],data['labels'])
        views.append(pack['features'])
    target_path=directory/'train_concept_labels.pt'; targets=torch.load(target_path,weights_only=True)
    semantic_path=directory/'joint_feature_v5/directions.pt'; semantic=torch.load(semantic_path,weights_only=False)
    bank_path=directory/'concept_bank.pt'; concept_bank=torch.load(bank_path,weights_only=False)
    baseline_path=ROOT/'outputs/cub/baseline/seed42/corrected_best_validation.pt'
    baseline=torch.load(baseline_path,weights_only=False)
    result=build_augmentation_directions(data['features'],torch.stack(views),targets,semantic,concept_bank['class_sets'],len(baseline['bias']))
    result['protocol']={'dataset':'cub','selection_split':'train','val_used':False,'test_used':False,
                        'source_sha256':digest(__file__),'original_features_sha256':digest(source),
                        'original_pseudo_sha256':digest(target_path),'view_protocol_sha256':digest(cache/'protocol.json'),
                        'view_sha256':status['view_sha256'],'semantic_normalization_sha256':digest(semantic_path),
                        'concept_bank_sha256':digest(bank_path),'rank_requested':64,'seed':707,
                        'definition':'Centered standardized view-minus-original changes; rank64 covariance; fixed draw from each positive-original-concept conditional covariance; original sharing support retained',
                        'limits':'Weak-view observations are correlated within images. Gaussian direction noise is an approximation, not exact empirical augmentation or guaranteed attribute invariance. No inference class labels.'}
    output=directory/'joint_feature_augmented_v13/directions.pt'; atomic_save(result,output)
    record={key:result[key] for key in ['nuisance_rank','nuisance_variance_fraction','observed_white_difference_rms','maximum_semantic_cosine','construction_seed']}
    record.update(phase='prepared',concepts=len(result['directions']),shared_class_directions=int(result['valid_groups'].sum()),
                  fallback_concepts=int(result['fallback_concepts'].sum()),protocol=result['protocol'],artifact_sha256=digest(output))
    atomic_json(record,output.parent/'status.json');print(record,flush=True)


def json_import(path):
    import json
    return json.loads(path.read_text())


if __name__=='__main__':main()
