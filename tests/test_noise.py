import inspect
import torch
from torch.nn import functional as F
from ps_pi.noise import ConceptNoise,sharing_factor,marginal_nll

def fixture(mode='shared_noise'):
    return ConceptNoise(5,3,{0:{0,1},1:{1,2},2:{2},3:{0},4:{0,1}},mode,hidden=8)

def test_sharing_covariance_psd_private_and_disjoint_support():
    b=sharing_factor({0:{0,1},1:{2,3},2:{0},3:{1,2}},4,4)
    covariance=b@torch.diag(torch.tensor([.1,.2,.3,.4]))@b.T
    assert torch.linalg.eigvalsh(covariance).min()>-1e-7
    assert covariance[0,1]==0
    assert torch.equal(covariance[2],torch.zeros(4))
    assert covariance[0,3]>0

def test_forward_has_no_ground_truth_label_input_and_initial_mean_is_zero():
    model=fixture();z=torch.randn(9,5)
    assert list(inspect.signature(model.forward).parameters)==['concepts']
    mean,diag,groups=model(z)
    assert torch.equal(mean,torch.zeros_like(mean))
    assert torch.allclose(diag,torch.full_like(diag,.03))
    assert torch.allclose(groups,torch.full_like(groups,.03))

def test_zero_perturbation_exactly_restores_input_after_training_like_update():
    model=fixture();z=torch.randn(9,5)
    with torch.no_grad():model.network[-1].bias.add_(.2)
    draws=model.draw(z,8,mean_multiplier=0,scale_multiplier=0)
    assert torch.equal(draws[0],z)

def test_antithetic_mean_and_repeated_seed_reproduce():
    model=fixture();z=torch.randn(9,5)
    gen=torch.Generator().manual_seed(17)
    first=model.draw(z,20,gen)
    second=model.draw(z,20,torch.Generator().manual_seed(17))
    assert torch.equal(first,second)
    assert torch.allclose(first.mean(0),z,atol=2e-7)

def test_mc_marginal_nll_matches_probability_mixture_and_allows_scale_gradient():
    model=fixture();z=torch.randn(7,5);head=torch.randn(3,5);labels=torch.tensor([0,1,2,0,1,2,0])
    logits=F.linear(model.draw(z,10,torch.Generator().manual_seed(2)),head)
    actual=marginal_nll(logits,labels)
    expected=-logits.softmax(-1).mean(0)[torch.arange(7),labels].log().mean()
    assert torch.allclose(actual,expected,atol=1e-6)
    actual.backward()
    assert model.network[-1].bias.grad[5:10].abs().sum()>0
    assert model.network[-1].bias.grad[10:].abs().sum()>0

def test_mean_and_capacity_controls_are_deterministic():
    z=torch.randn(9,5)
    for mode in ['mean_only','adapter_matched']:
        model=fixture(mode)
        assert torch.equal(model.draw(z,64),z.unsqueeze(0))
        assert torch.equal(model.draw(z,64),model.draw(z,64))
