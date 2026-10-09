"""Class-sharing Gaussian noise directly in standardized concept coordinates."""
import torch
from .noise import sharing_factor
from .joint_noise import FeatureNoise


def identity_concept_bank(class_sets, concept_count, class_count):
    factor=sharing_factor(class_sets,concept_count,class_count)
    return {'directions':torch.eye(concept_count),'group_directions':factor.T,
            'factor':factor,'feature_std':torch.ones(1,concept_count)}


class BottleneckNoise(FeatureNoise):
    """Same allocated generator/controls; explicit covariance support after the CBL."""
    def draw(self,features,context,samples=4,generator=None,mean_multiplier=1.,scale_multiplier=1.):
        if samples<1:raise ValueError('At least one sample required')
        if features.shape!=context.shape or features.ndim!=2:
            raise ValueError('Concept values and conditioning must share batch/concept shape')
        mean,diag,groups=self(context)
        shift=mean*mean_multiplier
        if self.mode in ['clean','mean_only','adapter_matched','shared_no_sampling'] or scale_multiplier==0:
            return (features+shift).unsqueeze(0)
        count=(samples+1)//2
        eps=torch.randn(count,*features.shape,device=features.device,generator=generator)
        eps=torch.cat([eps,-eps],0)[:samples]
        if self.mode=='fixed_isotropic':
            trace=self.concept_count+self.factor.square().sum()
            delta=eps*self.initial_scale*torch.sqrt(trace/self.concept_count)
        else:
            eta=torch.randn(count,len(features),self.class_count,device=features.device,generator=generator)
            eta=torch.cat([eta,-eta],0)[:samples]
            delta=eps*diag.unsqueeze(0)+(eta*groups.unsqueeze(0))@self.factor.T
        return features.unsqueeze(0)+shift.unsqueeze(0)+scale_multiplier*delta

    def covariance(self,context):
        _,diag,groups=self(context)
        if self.mode in ['clean','mean_only','adapter_matched','shared_no_sampling']:
            return context.new_zeros(len(context),self.concept_count,self.concept_count)
        if self.mode=='fixed_isotropic':
            level=self.initial_scale**2*(self.concept_count+self.factor.square().sum())/self.concept_count
            return torch.eye(self.concept_count,device=context.device).unsqueeze(0).expand(len(context),-1,-1)*level
        return torch.diag_embed(diag.square())+torch.einsum('kc,bc,lc->bkl',self.factor,groups.square(),self.factor)
