import torch
import pytest
from ps_pi.training_views import select_training_views


def test_image_view_and_conditioning_stay_paired_under_shuffled_duplicate_ids():
    identifiers=(100*torch.arange(3)[:,None]+torch.arange(4)[None,:]).float().unsqueeze(-1)
    contexts=torch.cat([identifiers,identifiers*2],-1)
    ids=torch.tensor([2,0,2,1]);views=torch.tensor([0,2,1,1])
    features,selected=select_training_views(identifiers,contexts,ids,views)
    assert features[:,0].tolist()==[2,200,102,101]
    assert torch.equal(selected[:,0],features[:,0]) and torch.equal(selected[:,1],features[:,0]*2)
    with pytest.raises(ValueError,match='pools'):
        select_training_views(identifiers,contexts[:2],ids,views)
