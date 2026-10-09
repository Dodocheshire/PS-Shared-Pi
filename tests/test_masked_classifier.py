import torch
from ps_pi.masked_classifier import MaskedClassifier

def test_optimizer_cannot_add_off_support_coefficients():
    weight=torch.tensor([[1.,0.,-1.],[0.,2.,0.]]);head=MaskedClassifier(weight,torch.zeros(2))
    optimizer=torch.optim.Adam(head.parameters(),lr=.1,weight_decay=.01)
    for _ in range(3):
        loss=head(torch.randn(8,3)).square().mean()+head.regularization(.0008)
        optimizer.zero_grad();loss.backward();optimizer.step()
    assert torch.equal(head.weight[~head.support],torch.zeros(3))
    assert torch.equal(head.effective_weight()[~head.support],torch.zeros(3))
    pack={};head.sync(pack)
    assert torch.equal(pack['weight'],head.effective_weight())
    assert not pack['weight'].requires_grad

def test_elastic_net_matches_official_formula_at_initialization():
    weight=torch.tensor([[1.,0.,-1.],[0.,2.,0.]]);head=MaskedClassifier(weight,torch.zeros(2))
    expected=.0008*.99*weight.abs().sum()+.5*.0008*.01*weight.square().sum()
    assert torch.allclose(head.regularization(.0008),expected)
