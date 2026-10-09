import torch
from ps_pi.bottleneck_noise import BottleneckNoise, identity_concept_bank
from ps_pi.joint_noise import FeatureNoise


def bank():
    return identity_concept_bank({0:{0,1},1:{0,1},2:{2,3},3:{3}},4,4)


def test_direct_concept_covariance_is_psd_and_has_class_sharing_support():
    noise=BottleneckNoise(bank(),4,'shared_noise');context=torch.zeros(1,4)
    covariance=noise.covariance(context)[0]
    assert torch.linalg.eigvalsh(covariance).min()>0
    assert covariance[0,1]>0 and covariance[0,2]==0 and covariance[2,3]==0
    draws=noise.draw(context,context,32768,torch.Generator().manual_seed(10))[:,0]
    assert torch.allclose(draws.mean(0),torch.zeros(4),atol=1e-6)
    assert torch.allclose(torch.cov(draws.T),covariance,atol=.0005,rtol=.04)
    _,energy=noise.energy(context)
    assert torch.allclose(energy,covariance.diag().mean())


def test_identity_implementation_matches_old_feature_sampler_and_fixed_trace():
    context=torch.randn(3,4);values=torch.randn(3,4)
    direct=BottleneckNoise(bank(),4,'shared_noise')
    old=FeatureNoise(bank(),4,'shared_noise');old.load_state_dict(direct.state_dict())
    a=direct.draw(values,context,6,torch.Generator().manual_seed(8))
    b=old.draw(values,context,6,torch.Generator().manual_seed(8))
    assert torch.allclose(a,b,atol=1e-7)
    fixed=BottleneckNoise(bank(),4,'fixed_isotropic')
    assert torch.allclose(fixed.covariance(context)[0].diag().sum(),direct.covariance(context)[0].diag().sum())
    disabled=BottleneckNoise(bank(),4,'shared_no_sampling')
    assert disabled.draw(values,context,6).shape==(1,3,4)
    assert not disabled.covariance(context).any()
