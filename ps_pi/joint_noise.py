"""Conditional semantic feature perturbations for jointly learning the PS-CBM encoder."""
import math
import torch
from torch import nn
from .directions import expected_feature_energy

MODES=('clean','mean_only','adapter_matched','diag_noise','shared_noise','shared_no_sampling','fixed_structured','fixed_isotropic')
LEARNED=('mean_only','adapter_matched','diag_noise','shared_noise','shared_no_sampling')

class FeatureNoise(nn.Module):
    def __init__(self,bank,class_count,mode='shared_noise',hidden=128,max_mean=.1,initial_scale=.1,min_scale=.02,max_scale=.3):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        if not 0<min_scale<initial_scale<max_scale:raise ValueError('Expected positive bounded noise scales')
        self.mode=mode;self.max_mean=max_mean;self.initial_scale=initial_scale;self.min_scale=min_scale;self.max_scale=max_scale
        for name in ['directions','group_directions','factor','feature_std']:
            self.register_buffer(name,bank[name].clone())
        self.concept_count=self.directions.shape[0];self.class_count=class_count
        self.register_buffer('scale_offset',torch.tensor(math.log((initial_scale-min_scale)/(max_scale-initial_scale))))
        self.network=nn.Sequential(nn.Linear(self.concept_count,hidden),nn.GELU(),nn.Linear(hidden,2*self.concept_count+class_count))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        if mode not in LEARNED:self.network.requires_grad_(False)

    def forward(self,context):
        if context.ndim!=2 or context.shape[1]!=self.concept_count:raise ValueError('Expected predicted concept context only')
        first,second,group=torch.split(self.network(context),[self.concept_count,self.concept_count,self.class_count],-1)
        mean=torch.zeros_like(first);diag=torch.zeros_like(first);groups=torch.zeros_like(group)
        if self.mode in LEARNED:
            mean=self.max_mean*torch.tanh(first)
        if self.mode=='adapter_matched':
            mean=self.max_mean*torch.tanh((first+torch.tanh(second)+torch.tanh(group)@self.factor.T)/math.sqrt(3))
        if self.mode in ['diag_noise','shared_noise','shared_no_sampling']:
            diag=self.min_scale+(self.max_scale-self.min_scale)*torch.sigmoid(second+self.scale_offset)
        if self.mode in ['shared_noise','shared_no_sampling']:
            groups=self.min_scale+(self.max_scale-self.min_scale)*torch.sigmoid(group+self.scale_offset)
        if self.mode=='fixed_structured':
            diag=torch.full_like(diag,self.initial_scale);groups=torch.full_like(groups,self.initial_scale)
        return mean,diag,groups

    def draw(self,features,context,samples=4,generator=None,mean_multiplier=1.,scale_multiplier=1.):
        mean,diag,groups=self(context)
        shift=(mean@self.directions)*self.feature_std*mean_multiplier
        if self.mode in ['clean','mean_only','adapter_matched','shared_no_sampling'] or scale_multiplier==0:
            return (features+shift).unsqueeze(0)
        count=(samples+1)//2
        if self.mode=='fixed_isotropic':
            eps=torch.randn(count,*features.shape,device=features.device,generator=generator)
            eps=torch.cat([eps,-eps],0)[:samples]
            active=(self.feature_std!=0).sum().clamp_min(1)
            total=self.directions.square().sum()+self.group_directions.square().sum()
            level=self.initial_scale*torch.sqrt(total/active)
            delta=eps*level
        else:
            eps=torch.randn(count,*context.shape,device=features.device,generator=generator)
            eta=torch.randn(count,len(features),self.class_count,device=features.device,generator=generator)
            eps=torch.cat([eps,-eps],0)[:samples];eta=torch.cat([eta,-eta],0)[:samples]
            delta=(eps*diag.unsqueeze(0))@self.directions+(eta*groups.unsqueeze(0))@self.group_directions
        return features.unsqueeze(0)+shift.unsqueeze(0)+scale_multiplier*delta*self.feature_std

    def energy(self,context):
        mean,diag,groups=self(context)
        if self.mode=='fixed_isotropic':
            return context.new_zeros(()),self.initial_scale**2*(self.directions.square().sum()+self.group_directions.square().sum())/self.directions.shape[1]
        bank={'directions':self.directions,'group_directions':self.group_directions}
        return expected_feature_energy(mean,diag,groups,bank)

    def scale_prior(self,context):
        _,diag,groups=self(context)
        if self.mode not in ['diag_noise','shared_noise','shared_no_sampling']:return context.new_zeros(())
        ratio=diag.square()/self.initial_scale**2
        loss=.5*(ratio-ratio.log()-1).mean()
        if self.mode in ['shared_noise','shared_no_sampling']:
            ratio=groups.square()/self.initial_scale**2
            loss+=.5*(ratio-ratio.log()-1).mean()
        return loss
