import torch
from torch import nn
from ps_pi.bottleneck_noise import BottleneckNoise, identity_concept_bank
from ps_pi.compact_concept_forward import sample_concepts, remapped_bank


def test_noise_follows_scaling_and_cbl_is_evaluated_once():
    class Counted(nn.Linear):
        calls=0
        def forward(self,x):
            self.calls+=1
            return super().forward(x)
    model=Counted(3,2)
    pack={'train_mean':torch.tensor([[2.,-3.]]),'train_std':torch.tensor([[.25,4.]])}
    noise=BottleneckNoise(identity_concept_bank({0:[0,1],1:[0,1]},2,2),2,'shared_noise')
    x=torch.randn(5,3,requires_grad=True)
    draws,raw,context=sample_concepts(model,noise,x,pack,8,torch.Generator().manual_seed(7))
    assert model.calls==1 and not context.requires_grad
    assert torch.allclose(draws.mean(0),context,atol=1e-6)
    assert torch.allclose(raw,draws*pack['train_std']+pack['train_mean'])
    raw.sum().backward()
    assert x.grad is not None and model.weight.grad is not None
    assert noise.network[-1].weight.grad is not None


def test_remapping_keeps_original_sharing_structure():
    original={'concepts':['a','b','c'],'class_sets':{0:[0,1],1:[1],2:[1,2]}}
    compact={'concepts':['a','c'],'retained_original_ids':torch.tensor([0,2]),'bias':torch.zeros(3)}
    bank=remapped_bank(original,compact)
    assert torch.equal(bank['factor'].ne(0),torch.tensor([[True,True,False],[False,True,True]]))
    clean=BottleneckNoise(bank,3,'clean');model=nn.Linear(3,2)
    x=torch.randn(4,3);pack={'train_mean':torch.ones(1,2),'train_std':torch.full((1,2),3.)}
    _,raw,_=sample_concepts(model,clean,x,pack,4)
    assert torch.equal(raw[0],model(x))
