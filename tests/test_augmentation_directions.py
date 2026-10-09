import torch
import pytest
from ps_pi.directions import build_directions
from ps_pi.augmentation_directions import build_augmentation_directions


def fixture():
    generator=torch.Generator().manual_seed(115)
    x=torch.randn(30,8,generator=generator)
    targets=torch.stack([x[:,0]>0,x[:,1]>0],1).float()
    sets={0:[0],1:[0,1]};reference=build_directions(x,targets,sets,2)
    changes=torch.zeros(2,30,8);changes[:,:,3:5]=torch.randn(2,30,2,generator=generator)
    return x,x.unsqueeze(0)+changes,targets,reference,sets


def test_measured_noise_stays_in_observed_span_and_preserves_sharing():
    x,views,targets,reference,sets=fixture()
    bank=build_augmentation_directions(x,views,targets,reference,sets,2,rank=2)
    assert torch.allclose(bank['directions'].norm(dim=1),torch.ones(2),atol=1e-6)
    assert bank['directions'][:,[0,1,2,5,6,7]].abs().max()<1e-6
    assert torch.equal(bank['factor'][0],torch.zeros(2))
    assert torch.allclose(bank['group_directions'],bank['factor'].T@bank['directions'])
    assert bank['nuisance_variance_fraction']>1-1e-6


def test_paired_translation_invariance_reproducibility_and_zero_variance_rejection():
    x,views,targets,reference,sets=fixture()
    a=build_augmentation_directions(x,views,targets,reference,sets,2,rank=2)
    b=build_augmentation_directions(x+4,views+4,targets,reference,sets,2,rank=2)
    c=build_augmentation_directions(x,views,targets,reference,sets,2,rank=2)
    assert torch.allclose(a['directions'],b['directions'],atol=1e-5)
    assert torch.equal(a['directions'],c['directions'])
    with pytest.raises(ValueError,match='variance'):
        build_augmentation_directions(x,x.unsqueeze(0).expand(2,-1,-1),targets,reference,sets,2)
