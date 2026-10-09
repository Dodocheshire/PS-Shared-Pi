"""Label-free conditional concept noise with class-sharing covariance support."""
import math
import torch
from torch import nn
from torch.nn import functional as F

MODES=('mean_only','adapter_matched','diag_noise','shared_noise','fixed_noise')

def sharing_factor(class_sets, concept_count, class_count):
    factor=torch.zeros(concept_count,class_count)
    if set(class_sets)!=set(range(concept_count)):
        raise ValueError('Concept class-set keys must exactly match concept order')
    for j in range(concept_count):
        classes=sorted(class_sets[j])
        if not classes or min(classes)<0 or max(classes)>=class_count:
            raise ValueError('Invalid concept class membership')
        if len(classes)>1:factor[j,classes]=1/math.sqrt(len(classes))
    return factor

class ConceptNoise(nn.Module):
    """A conditional distribution. Ground-truth classes never enter forward."""
    def __init__(self, concept_count, class_count, class_sets, mode='shared_noise',
                 hidden=128, max_mean=.5, initial_scale=.03, max_scale=.3):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        if not 0<initial_scale<max_scale:raise ValueError('Scale initialization must be strictly inside its bound')
        self.mode=mode;self.concept_count=concept_count;self.class_count=class_count
        self.max_mean=max_mean;self.max_scale=max_scale;self.initial_scale=initial_scale
        self.register_buffer('factor',sharing_factor(class_sets,concept_count,class_count))
        self.register_buffer('scale_offset',torch.tensor(math.log(initial_scale/(max_scale-initial_scale))))
        # Same allocated architecture in every learned method; adapter activates all outputs.
        self.network=nn.Sequential(nn.Linear(concept_count,hidden),nn.GELU(),
                                   nn.Linear(hidden,2*concept_count+class_count))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        if mode=='fixed_noise':self.network.requires_grad_(False)

    def forward(self, concepts):
        if concepts.ndim!=2 or concepts.shape[1]!=self.concept_count:
            raise ValueError('Expected batch x concept tensor')
        out=self.network(concepts)
        first,second,groups=torch.split(out,[self.concept_count,self.concept_count,self.class_count],dim=-1)
        if self.mode=='adapter_matched':
            mean=self.max_mean*torch.tanh((first+second+groups@self.factor.T)/math.sqrt(3))
        else:mean=self.max_mean*torch.tanh(first)
        diag=torch.zeros_like(mean);group_scale=torch.zeros_like(groups)
        if self.mode in ['diag_noise','shared_noise']:
            diag=self.max_scale*torch.sigmoid(second+self.scale_offset)
        if self.mode=='shared_noise':
            group_scale=self.max_scale*torch.sigmoid(groups+self.scale_offset)
        if self.mode=='fixed_noise':
            mean=torch.zeros_like(mean);diag=torch.full_like(mean,self.initial_scale)
        return mean,diag,group_scale

    def draw(self,concepts,samples=8,generator=None,mean_multiplier=1.,scale_multiplier=1.,antithetic=True):
        mean,diag,groups=self(concepts)
        if scale_multiplier==0 or self.mode in ['mean_only','adapter_matched']:
            return (concepts+mean_multiplier*mean).unsqueeze(0)
        if samples<1:raise ValueError('At least one Monte Carlo sample is required')
        count=(samples+1)//2 if antithetic else samples
        eps=torch.randn(count,*concepts.shape,device=concepts.device,dtype=concepts.dtype,generator=generator)
        eta=torch.randn(count,len(concepts),self.class_count,device=concepts.device,dtype=concepts.dtype,generator=generator)
        if antithetic:
            eps=torch.cat([eps,-eps],dim=0)[:samples];eta=torch.cat([eta,-eta],dim=0)[:samples]
        delta=eps*diag.unsqueeze(0)+(eta*groups.unsqueeze(0))@self.factor.T
        return concepts.unsqueeze(0)+mean_multiplier*mean.unsqueeze(0)+scale_multiplier*delta

    def expected_energy(self,concepts):
        mean,diag,groups=self(concepts)
        variance=diag.square()+groups.square()@self.factor.square().T
        return mean.square().mean(),variance.mean()

def marginal_nll(logits,labels):
    """Negative log of MC mean probability, rather than mean cross-entropy."""
    if logits.ndim!=3:raise ValueError('Expected samples x batch x classes')
    log_probability=logits.log_softmax(-1)
    chosen=log_probability.gather(-1,labels.reshape(1,-1,1).expand(logits.shape[0],-1,1)).squeeze(-1)
    return -(torch.logsumexp(chosen,dim=0)-math.log(logits.shape[0])).mean()
