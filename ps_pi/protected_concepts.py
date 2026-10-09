"""Restrict concept adaptation/noise while exactly preserving mapped attributes."""
import torch
from .bottleneck_noise import BottleneckNoise


class FrozenConceptChannels:
    def __init__(self,cbl,protocol):
        last=cbl.cbl[-1]
        self.ids=torch.tensor(sorted({i for a in protocol['evaluated_cohort'].values() for i in a['concept_ids']}),device=last.weight.device)
        if not len(self.ids) or len(self.ids)>=last.out_features:
            raise ValueError('Need both protected and adaptable concept outputs')
        self.layer=last
        self.weight=last.weight[self.ids].detach().clone()
        self.bias=None if last.bias is None else last.bias[self.ids].detach().clone()
        cbl.requires_grad_(False);last.requires_grad_(True)
        active=torch.ones(last.out_features,device=last.weight.device);active[self.ids]=0
        self.hooks=[last.weight.register_hook(lambda grad:grad*active[:,None])]
        if last.bias is not None:self.hooks.append(last.bias.register_hook(lambda grad:grad*active))

    @torch.no_grad()
    def restore(self):
        # Adam weight decay acts on the full matrix despite masked gradients.
        self.layer.weight[self.ids]=self.weight
        if self.bias is not None:self.layer.bias[self.ids]=self.bias


def protect_bank(bank,protocol):
    result={k:v.clone() for k,v in bank.items()}
    active=torch.ones(bank['factor'].shape[0])
    for attribute in protocol['evaluated_cohort'].values():active[attribute['concept_ids']]=0
    if not active.any():raise ValueError('No adaptable concept channels remain')
    result['active_mask']=active
    result['directions']=torch.diag(active)
    result['factor']*=active[:,None]
    result['group_directions']=result['factor'].T.contiguous()
    return result


class ProtectedBottleneckNoise(BottleneckNoise):
    def __init__(self,bank,class_count,*args,**kwargs):
        super().__init__(bank,class_count,*args,**kwargs)
        self.register_buffer('active_mask',bank['active_mask'].clone())

    def forward(self,context):
        mean,diag,groups=super().forward(context)
        return mean*self.active_mask,diag*self.active_mask,groups

    def isotropic_level(self):
        trace=self.active_mask.sum()+self.factor.square().sum()
        return self.initial_scale*torch.sqrt(trace/self.active_mask.sum())

    def draw(self,features,context,samples=4,generator=None,mean_multiplier=1.,scale_multiplier=1.):
        if self.mode!='fixed_isotropic' or scale_multiplier==0:
            return super().draw(features,context,samples,generator,mean_multiplier,scale_multiplier)
        if samples<1 or features.shape!=context.shape:raise ValueError('Invalid concept samples/shapes')
        count=(samples+1)//2
        eps=torch.randn(count,*features.shape,device=features.device,generator=generator)
        eps=torch.cat([eps,-eps],0)[:samples]
        return features.unsqueeze(0)+scale_multiplier*eps*self.isotropic_level()*self.active_mask

    def covariance(self,context):
        if self.mode=='fixed_isotropic':
            return torch.diag(self.active_mask).unsqueeze(0).expand(len(context),-1,-1)*self.isotropic_level().square()
        return super().covariance(context)

    def scale_prior(self,context):
        if self.mode not in ['diag_noise','shared_noise','shared_no_sampling']:return context.new_zeros(())
        _,diag,groups=self(context)
        ratio=diag[:,self.active_mask.bool()].square()/self.initial_scale**2
        loss=.5*(ratio-ratio.log()-1).mean()
        active_groups=self.factor.ne(0).any(0)
        if self.mode in ['shared_noise','shared_no_sampling'] and active_groups.any():
            ratio=groups[:,active_groups].square()/self.initial_scale**2
            loss+=.5*(ratio-ratio.log()-1).mean()
        return loss
