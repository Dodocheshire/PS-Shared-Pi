import torch
import pytest
from torch import nn
from ps_pi.compact_outputs import compact_cbl, remap_attributes, retained_concepts


def test_exact_nonzero_and_protected_channels_are_retained():
    weight=torch.tensor([[0.,1.,0.,0.,0.],[0.,0.,0.,1e-9,0.]])
    ids=retained_concepts(weight,[2])
    assert ids.tolist()==[1,2,3]
    with pytest.raises(ValueError,match='outside'):
        retained_concepts(weight,[5])


def test_compact_last_layer_preserves_shared_hidden_and_protected_responses():
    torch.manual_seed(17)
    source=nn.Module();source.cbl=nn.Sequential(nn.Linear(3,7),nn.ReLU(),nn.Linear(7,5))
    weight=torch.tensor([[0.,2.,0.,0.,0.],[0.,0.,0.,1.,0.]])
    ids=retained_concepts(weight,[2])
    compact=compact_cbl(source,ids,5)
    assert compact.cbl[0].out_features==7 and compact.cbl[-1].in_features==7
    x=torch.randn(11,3);full=source.cbl(x);small=compact.cbl(x)
    assert torch.allclose(small,full[:,ids],atol=1e-7)
    assert torch.allclose(full@weight.T,small@weight[:,ids].T,atol=1e-7)
    assert source.cbl[-1].out_features==5
    protocol={'evaluated_cohort':{'0':{'concept_ids':[2],'threshold':.4}}}
    remapped=remap_attributes(protocol,ids)
    assert remapped['evaluated_cohort']['0']['concept_ids']==[1]
    assert remapped['evaluated_cohort']['0']['threshold']==.4
    assert torch.allclose(full[:,2].sigmoid(),small[:,1].sigmoid(),atol=1e-7)
    with pytest.raises(ValueError,match='protected'):
        remap_attributes(protocol,torch.tensor([1,3]))
