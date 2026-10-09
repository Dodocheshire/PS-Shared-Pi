import torch
from torch import nn
from ps_pi.bottleneck_noise import identity_concept_bank
from ps_pi.joint_noise import MODES
from ps_pi.protected_concepts import FrozenConceptChannels,protect_bank,ProtectedBottleneckNoise
from ps_pi.protected_concept_forward import sample_concepts,mean_concept_probability

PROTOCOL={'evaluated_cohort':{'0':{'concept_ids':[0,2]}}}


def test_protected_responses_survive_adam_weight_decay():
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.cbl=nn.Sequential(nn.Linear(3,5),nn.ReLU(),nn.Linear(5,4))
        def forward(self,x):return self.cbl(x)
    torch.manual_seed(19);model=Model();x=torch.randn(7,3);before=model(x).detach().clone()
    guard=FrozenConceptChannels(model,PROTOCOL)
    optimizer=torch.optim.Adam(model.parameters(),lr=.05,weight_decay=.1)
    for _ in range(15):
        optimizer.zero_grad();model(x).square().mean().backward();optimizer.step();guard.restore()
    after=model(x).detach()
    assert torch.equal(before[:,[0,2]],after[:,[0,2]])
    assert not torch.equal(before[:,[1,3]],after[:,[1,3]])


def test_all_controls_leave_protected_channels_exact_and_priors_finite():
    bank=protect_bank(identity_concept_bank({0:[0,1],1:[0,1],2:[1],3:[0,1]},4,2),PROTOCOL)
    model=nn.Linear(3,4);x=torch.randn(8,3)
    pack={'train_mean':torch.randn(1,4),'train_std':torch.tensor([[.31,7.,.73,2.1]])}
    for mode in MODES:
        noise=ProtectedBottleneckNoise(bank,2,mode)
        z,raw,context=sample_concepts(model,noise,x,pack,8,torch.Generator().manual_seed(4))
        original=model(x)
        assert torch.equal(raw[:,:,[0,2]],original[:,[0,2]].unsqueeze(0).expand(len(raw),-1,-1))
        assert torch.equal(mean_concept_probability(raw,noise)[:,[0,2]],original.sigmoid()[:,[0,2]])
        covariance=noise.covariance(context)
        assert not covariance[:,[0,2]].ne(0).any()
        assert not covariance[:,:,[0,2]].ne(0).any()
        assert torch.isfinite(noise.scale_prior(context))
        if mode in ['mean_only','adapter_matched','shared_no_sampling','clean']:
            assert torch.equal(raw[0],original)


def test_fixed_and_structured_noise_have_equal_initial_energy_on_active_channels():
    bank=protect_bank(identity_concept_bank({0:[0,1],1:[0,1],2:[1],3:[0,1]},4,2),PROTOCOL)
    ctx=torch.zeros(2,4)
    shared=ProtectedBottleneckNoise(bank,2,'shared_noise');fixed=ProtectedBottleneckNoise(bank,2,'fixed_isotropic')
    assert torch.allclose(shared.covariance(ctx).diagonal(dim1=-2,dim2=-1).sum(-1),fixed.covariance(ctx).diagonal(dim1=-2,dim2=-1).sum(-1))
    assert torch.allclose(shared.energy(ctx)[1],fixed.energy(ctx)[1])
