"""Use the full deterministic-adapter mean decoder in every learned noise model."""
import math
import torch
from .joint_noise import LEARNED
from .protected_concepts import ProtectedBottleneckNoise


class AllOutputMeanNoise(ProtectedBottleneckNoise):
    def forward(self,context):
        if context.ndim!=2 or context.shape[1]!=self.concept_count:
            raise ValueError('Expected predicted concept context')
        first,second,group=torch.split(self.network(context),[self.concept_count,self.concept_count,self.class_count],-1)
        mean=torch.zeros_like(first);diag=torch.zeros_like(first);groups=torch.zeros_like(group)
        if self.mode=='mean_only':mean=self.max_mean*torch.tanh(first)
        elif self.mode in LEARNED:
            mean=self.max_mean*torch.tanh((first+torch.tanh(second)+torch.tanh(group)@self.factor.T)/math.sqrt(3))
        if self.mode in ['diag_noise','shared_noise','shared_no_sampling']:
            diag=self.min_scale+(self.max_scale-self.min_scale)*torch.sigmoid(second+self.scale_offset)
        if self.mode in ['shared_noise','shared_no_sampling']:
            groups=self.min_scale+(self.max_scale-self.min_scale)*torch.sigmoid(group+self.scale_offset)
        if self.mode=='fixed_structured':
            diag=torch.full_like(diag,self.initial_scale);groups=torch.full_like(groups,self.initial_scale)
        return mean*self.active_mask,diag*self.active_mask,groups
