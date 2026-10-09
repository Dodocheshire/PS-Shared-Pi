import torch
from torch.nn import functional as F
from ps_pi.adapt_predictive import marginal_concept_nll

def test_predictive_concept_likelihood_is_stable_and_matches_mc_probability():
    raw=torch.tensor([[[2.,-1.],[.1,3.]],[[1.,-3.],[-.4,2.]]],requires_grad=True)
    labels=torch.tensor([[1.,0.],[.3,.8]])
    actual=marginal_concept_nll(raw,labels)
    reference=F.binary_cross_entropy(raw.sigmoid().mean(0),labels)
    expected_loss=F.binary_cross_entropy_with_logits(raw,labels.unsqueeze(0).expand_as(raw))
    assert torch.allclose(actual,reference,atol=1e-6)
    assert actual<=expected_loss+1e-7
    actual.backward();assert raw.grad.abs().sum()>0

def test_extreme_concept_logits_have_finite_predictive_loss_and_gradient():
    raw=torch.tensor([[[1000.,-1000.]],[[999.,-999.]]],requires_grad=True)
    value=marginal_concept_nll(raw,torch.tensor([[0.,1.]]))
    assert torch.isfinite(value)
    value.backward();assert torch.isfinite(raw.grad).all()
