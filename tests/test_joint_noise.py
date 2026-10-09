import inspect
import torch
from ps_pi.directions import build_directions
from ps_pi.joint_noise import FeatureNoise

def fixture(mode='shared_noise'):
    torch.manual_seed(19);x=torch.randn(100,6);targets=(torch.randn(100,4)>0).float()
    bank=build_directions(x,targets,{0:{0,1},1:{1,2},2:{2},3:{0}},3)
    return x,bank,FeatureNoise(bank,3,mode,hidden=8)

def test_feature_noise_has_no_class_label_input_and_zero_noise_restores_features():
    x,bank,model=fixture();ctx=torch.randn(len(x),4)
    assert list(inspect.signature(model.forward).parameters)==['context']
    with torch.no_grad():model.network[-1].bias.add_(.2)
    draws=model.draw(x,ctx,8,mean_multiplier=0.,scale_multiplier=0.)
    assert torch.equal(draws[0],x)

def test_isotropic_control_matches_initial_structured_expected_feature_energy():
    x,bank,model=fixture();ctx=torch.randn(len(x),4)
    isotropic=FeatureNoise(bank,3,'fixed_isotropic',hidden=8)
    _,actual=model.energy(ctx);_,matched=isotropic.energy(ctx)
    assert torch.allclose(actual,matched,atol=1e-7)

def test_projection_noise_sampling_matches_analytic_energy_and_is_antithetic():
    x,bank,model=fixture();x=x[:1];ctx=torch.zeros(1,4)
    generator=torch.Generator().manual_seed(10)
    draws=model.draw(x,ctx,40000,generator)
    white=(draws-x.unsqueeze(0))/bank['feature_std']
    _,energy=model.energy(ctx)
    assert torch.allclose(white.square().mean(),energy,rtol=.025)
    assert torch.allclose(draws.mean(0),x,atol=1e-6)

def test_no_sampling_control_matches_generator_parameters_and_prior():
    x,bank,model=fixture();control=FeatureNoise(bank,3,'shared_no_sampling',hidden=8)
    control.load_state_dict(model.state_dict());ctx=torch.randn(len(x),4)
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in control.parameters())
    for left,right in zip(model(ctx),control(ctx)):assert torch.equal(left,right)
    assert torch.equal(model.scale_prior(ctx),control.scale_prior(ctx))
    assert torch.equal(control.draw(x,ctx,8)[0],x)


def test_scale_bounds_remain_positive_after_extreme_context_outputs():
    x,bank,model=fixture();ctx=torch.randn(len(x),4)
    with torch.no_grad():model.network[-1].bias.fill_(100.)
    _,diag,groups=model(ctx)
    assert diag.min()>=model.min_scale and diag.max()<=model.max_scale
    assert groups.min()>=model.min_scale and groups.max()<=model.max_scale
