"""Remove exactly unused classifier outputs while protecting semantic audit channels."""
import copy
import torch
from torch import nn


def retained_concepts(weight, protected=()):
    if weight.ndim!=2 or not torch.isfinite(weight).all():
        raise ValueError('Expected finite class-by-concept weights')
    protected=list(protected)
    if any(i<0 or i>=weight.shape[1] for i in protected):
        raise ValueError('Protected concept ID outside source bank')
    keep=weight.ne(0).any(0)
    if protected:
        keep[torch.tensor(protected,device=weight.device)]=True
    ids=keep.nonzero().flatten()
    if not len(ids):
        raise ValueError('No concept outputs remain; this would be a bias-only classifier')
    return ids


def compact_cbl(cbl, ids, source_concept_count):
    model=copy.deepcopy(cbl)
    if not isinstance(model.cbl[-1],nn.Linear) or model.cbl[-1].out_features!=source_concept_count:
        raise ValueError('Compaction requires independent last-linear concept outputs')
    if ids.ndim!=1 or not len(ids) or len(ids.unique())!=len(ids):
        raise ValueError('Retained IDs must be a nonempty unique vector')
    if ids.min()<0 or ids.max()>=source_concept_count:
        raise ValueError('Retained output outside original concept bank')
    source=model.cbl[-1]
    replacement=nn.Linear(source.in_features,len(ids),bias=source.bias is not None,
                          device=source.weight.device,dtype=source.weight.dtype)
    with torch.no_grad():
        replacement.weight.copy_(source.weight[ids])
        if source.bias is not None:replacement.bias.copy_(source.bias[ids])
    model.cbl[-1]=replacement
    return model


def remap_attributes(protocol, ids):
    result=copy.deepcopy(protocol)
    lookup={old:new for new,old in enumerate(ids.tolist())}
    for attribute in result['evaluated_cohort'].values():
        if not all(i in lookup for i in attribute['concept_ids']):
            raise ValueError('Compaction removed a protected attribute response')
        attribute['concept_ids']=[lookup[i] for i in attribute['concept_ids']]
    return result


def load_compact_checkpoint(path):
    from .common import official_module
    pack=torch.load(path,weights_only=False,map_location='cpu')
    config=pack['source_config'];count=pack['original_concept_count']
    CBL=official_module('model.cbl').ConceptBottleneckLayer
    source=CBL(pack['input_dimension'],count,config['cbl_layer_num'],config['cbl_bias'])
    model=compact_cbl(source,pack['retained_original_ids'],count)
    model.load_state_dict(pack['cbl_state']);model.eval()
    if len(pack['concepts'])!=model.cbl[-1].out_features or pack['weight'].shape[1]!=len(pack['concepts']):
        raise ValueError('Compact checkpoint concept/head shapes disagree')
    return model,pack
