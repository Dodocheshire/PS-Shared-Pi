"""Update the original sparse classifier coefficients without expanding their support."""
import torch
from torch import nn
from torch.nn import functional as F

class MaskedClassifier(nn.Module):
    def __init__(self,weight,bias):
        super().__init__()
        self.weight=nn.Parameter(weight.detach().clone());self.bias=nn.Parameter(bias.detach().clone())
        self.register_buffer('support',weight.ne(0).detach().clone())
        self.register_buffer('reference_weight',weight.detach().clone())
        self.register_buffer('reference_bias',bias.detach().clone())
    def effective_weight(self):return self.weight*self.support
    def forward(self,concepts):return F.linear(concepts,self.effective_weight(),self.bias)
    def regularization(self,lam,alpha=.99,anchor_weight=.01):
        weight=self.effective_weight()
        elastic=lam*alpha*weight.abs().sum()+.5*lam*(1-alpha)*weight.square().sum()
        delta=(weight-self.reference_weight)[self.support]
        anchor=(delta.square().mean() if len(delta) else weight.new_zeros(()))+(self.bias-self.reference_bias).square().mean()
        return elastic+anchor_weight*anchor
    @torch.no_grad()
    def sync(self,pack):
        pack['weight']=self.effective_weight().detach();pack['bias']=self.bias.detach()
