import torch
from torch import nn
from ps_pi.joint_noise import FeatureNoise
from ps_pi.expected_concepts import expected_concepts

class CBL(nn.Module):
    def __init__(self):
        super().__init__();self.cbl=nn.Sequential(nn.Linear(7,5),nn.ReLU(),nn.Linear(5,5))
    def forward(self,x):return self.cbl(x)

def fixture(mode):
    torch.manual_seed(831)
    bank={'directions':torch.nn.functional.normalize(torch.randn(5,7),dim=1),
          'group_directions':torch.nn.functional.normalize(torch.randn(2,7),dim=1),
          'factor':torch.randn(5,2),'feature_std':torch.rand(1,7)+.5}
    noise=FeatureNoise(bank,2,mode)
    with torch.no_grad():noise.network[-1].bias[:5].fill_(.3)
    return CBL(),noise,torch.randn(3,7),torch.randn(3,5)

def test_analytic_expectation_matches_large_antithetic_mc():
    torch.set_num_threads(4)
    for mode in ['shared_noise','fixed_isotropic']:
        cbl,noise,x,context=fixture(mode)
        with torch.no_grad():
            analytic=expected_concepts(cbl,noise,x,context)
            draws=noise.draw(x,context,60000,torch.Generator().manual_seed(103))
            empirical=cbl(draws.reshape(-1,7)).reshape(60000,3,5).mean(0)
        assert torch.allclose(analytic,empirical,atol=5e-4,rtol=0)

def test_sampling_disabled_matches_actual_policy_and_zero_scale():
    for mode in ['clean','mean_only','adapter_matched','shared_no_sampling']:
        cbl,noise,x,context=fixture(mode)
        assert torch.allclose(expected_concepts(cbl,noise,x,context),cbl(noise.draw(x,context)[0]),atol=1e-6)
    cbl,noise,x,context=fixture('shared_noise')
    assert torch.allclose(expected_concepts(cbl,noise,x,context,scale_multiplier=0),cbl(noise.draw(x,context,scale_multiplier=0)[0]),atol=1e-6)

def test_gradients_remain_finite_for_a_zero_variance_hidden_unit():
    cbl,noise,x,context=fixture('shared_noise')
    with torch.no_grad():cbl.cbl[0].weight[0].zero_();cbl.cbl[0].bias[0]=0
    expected_concepts(cbl,noise,x,context).square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(cbl.parameters())+list(noise.parameters()))
