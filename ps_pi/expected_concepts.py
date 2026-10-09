"""Exact concept-logit expectation for a two-linear-layer CBL under Gaussian feature noise."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def expected_concepts(cbl,noise,features,context,scale_multiplier=1.,mean_multiplier=1.):
    layers=cbl.cbl
    if len(layers) not in [1,3] or not isinstance(layers[0],nn.Linear):
        raise ValueError('Expected affine or Linear-ReLU-Linear official CBL')
    mean,diag,groups=noise(context)
    shift=(mean@noise.directions)*noise.feature_std*mean_multiplier
    a=layers[0](features+shift)
    if len(layers)==1:return a
    if not isinstance(layers[1],nn.ReLU) or not isinstance(layers[2],nn.Linear):
        raise ValueError('Unsupported hidden activation')
    if scale_multiplier==0 or noise.mode in ['clean','mean_only','adapter_matched','shared_no_sampling']:
        return layers[2](a.relu())
    if noise.mode=='fixed_isotropic':
        active=(noise.feature_std!=0).sum().clamp_min(1)
        total=noise.directions.square().sum()+noise.group_directions.square().sum()
        level2=noise.initial_scale**2*total/active
        var=level2*(layers[0].weight*noise.feature_std).square().sum(1).unsqueeze(0).expand_as(a)
    else:
        projected=F.linear(noise.directions*noise.feature_std,layers[0].weight)
        projected_groups=F.linear(noise.group_directions*noise.feature_std,layers[0].weight)
        var=diag.square()@projected.square()+groups.square()@projected_groups.square()
    variance=var*scale_multiplier**2
    std=variance.clamp_min(1e-16).sqrt();ratio=a/std
    cdf=.5*(1+torch.erf(ratio/math.sqrt(2)))
    density=torch.exp(-.5*ratio.square())/math.sqrt(2*math.pi)
    hidden=(a*cdf+std*density).clamp_min(0)
    hidden=torch.where(variance>1e-16,hidden,a.relu())
    return layers[2](hidden)
