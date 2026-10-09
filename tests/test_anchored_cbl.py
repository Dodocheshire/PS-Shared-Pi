import torch
from torch import nn
from ps_pi.anchored_cbl import AnchoredCBL


def test_hidden_encoder_learns_while_native_responses_and_anchor_weights_stay_exact():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__();self.cbl=nn.Sequential(nn.Linear(3,5),nn.ReLU(),nn.Linear(5,4))
        def forward(self,x):return self.cbl(x)
    torch.manual_seed(37);original=Encoder();x=torch.randn(11,3)
    before=original(x).detach().clone();initial_hidden=original.cbl[0].weight.detach().clone()
    anchor=AnchoredCBL(original,{'evaluated_cohort':{'0':{'concept_ids':[0,2]}}})
    assert torch.equal(anchor(x),before)
    optimizer=torch.optim.Adam(anchor.parameters(),lr=.02,weight_decay=.1)
    for _ in range(20):
        optimizer.zero_grad();anchor(x).square().mean().backward();optimizer.step();anchor.restore()
    assert not torch.equal(anchor.adaptive.cbl[0].weight,initial_hidden)
    assert torch.equal(anchor.anchor.cbl[0].weight,initial_hidden)
    assert torch.equal(anchor(x)[:,[0,2]],before[:,[0,2]])
    assert not torch.equal(anchor(x)[:,[1,3]],before[:,[1,3]])
    assert all(p.grad is None for p in anchor.anchor.parameters())
