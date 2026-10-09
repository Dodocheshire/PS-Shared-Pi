import torch
from ps_pi.directions import build_directions,expected_feature_energy

def fixture():
    torch.manual_seed(19)
    x=torch.randn(100,6);s=(torch.randn(100,4)>0).float()
    return x,s,{0:{0,1},1:{1,2},2:{2},3:{0}}

def test_training_feature_directions_are_translation_invariant_and_unit_normalized():
    x,s,groups=fixture();a=build_directions(x,s,groups,3);b=build_directions(x+3,s,groups,3)
    assert torch.allclose(a['directions'],b['directions'],atol=1e-5)
    assert torch.allclose(a['directions'].norm(dim=1),torch.ones(4),atol=1e-6)
    assert torch.allclose(a['group_directions'][a['valid_groups']].norm(dim=1),torch.ones(3),atol=1e-6)
    assert torch.equal(a['factor'][2],torch.zeros(3))

def test_analytic_projected_noise_energy_matches_independent_monte_carlo():
    x,s,groups=fixture();bank=build_directions(x,s,groups,3)
    mean=torch.tensor([[.1,-.2,.3,-.1]]);diag=torch.tensor([[.2,.1,.3,.2]]);group=torch.tensor([[.1,.2,.15]])
    deterministic,variance=expected_feature_energy(mean,diag,group,bank)
    torch.manual_seed(31)
    samples=(mean+torch.randn(40000,4)*diag)@bank['directions']+(torch.randn(40000,3)*group)@bank['group_directions']
    measured=samples.square().mean()
    assert torch.allclose(measured,deterministic+variance,rtol=.025)

def test_constant_or_empty_concepts_produce_zero_feature_directions():
    x,s,groups=fixture();s[:,2]=0;s[:,3]=1;bank=build_directions(x,s,groups,3)
    assert torch.equal(bank['directions'][2:],torch.zeros(2,6))
    assert not bank['valid_concepts'][2:].any()
