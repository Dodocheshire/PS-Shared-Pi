"""Stronger deterministic capacity control with independently nonlinear output branches."""
import math
import torch
from .noise import ConceptNoise as BaseNoise

class ConceptNoise(BaseNoise):
    def forward(self,concepts):
        if self.mode!='adapter_matched':return super().forward(concepts)
        if concepts.ndim!=2 or concepts.shape[1]!=self.concept_count:raise ValueError('Expected batch x concepts')
        first,second,groups=torch.split(self.network(concepts),[self.concept_count,self.concept_count,self.class_count],-1)
        # Independent nonlinear branches prevent the two extra linear heads collapsing
        # algebraically into a single mean head. Allocated architecture matches Pi.
        mean=self.max_mean*torch.tanh((first+torch.tanh(second)+torch.tanh(groups)@self.factor.T)/math.sqrt(3))
        return mean,torch.zeros_like(mean),torch.zeros_like(groups)
