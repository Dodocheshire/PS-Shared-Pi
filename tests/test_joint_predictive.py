import torch
from torch.nn import functional as F
from ps_pi.joint_residual_predictive_train import marginal_task_nll

def test_task_nll_matches_actual_marginal_probability():
    torch.manual_seed(221);logits=torch.randn(6,9,4,dtype=torch.float64);labels=torch.arange(9)%4
    expected=-logits.softmax(-1).mean(0)[torch.arange(9),labels].log().mean()
    assert torch.allclose(marginal_task_nll(logits,labels),expected,atol=1e-12)
    repeated=logits[0].unsqueeze(0).expand(4,-1,-1)
    assert torch.allclose(marginal_task_nll(repeated,labels),F.cross_entropy(logits[0],labels),atol=1e-12)

def test_task_nll_has_finite_gradients_for_extreme_logits():
    logits=torch.tensor([[[1000.,-1000.],[-1000.,1000.]],[[999.,-999.],[-999.,999.]]],requires_grad=True)
    loss=marginal_task_nll(logits,torch.tensor([1,0]));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
