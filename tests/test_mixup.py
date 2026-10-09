from types import SimpleNamespace
import torch
from ps_pi.adapt_mixup import marginal_soft_nll,selection_key
from ps_pi.noise import marginal_nll
from ps_pi.noise_v2 import ConceptNoise

def test_soft_marginal_likelihood_matches_manual_mixture():
    logits=torch.tensor([[[1.,0.,-1.],[0.,2.,1.]],[[2.,1.,0.],[1.,0.,2.]]],requires_grad=True)
    targets=torch.tensor([[.3,.7,0.],[0.,.25,.75]])
    reference=-(targets*logits.softmax(-1).mean(0).log()).sum(-1).mean()
    value=marginal_soft_nll(logits,targets)
    assert torch.allclose(value,reference,atol=1e-6)
    value.backward();assert logits.grad.abs().sum()>0
    labels=torch.tensor([0,2])
    assert torch.allclose(marginal_soft_nll(logits,torch.nn.functional.one_hot(labels,3).float()),marginal_nll(logits,labels))

def test_accuracy_gain_with_worse_calibration_cannot_outrank_eligible_model():
    args=SimpleNamespace(nll_tolerance=0.,ece_tolerance=.002,concept_bce_tolerance=.002,drift_limit=.3)
    clean=dict(nll=.3,ece15=.02,concept_bce=.07)
    eligible=dict(nll=.29,ece15=.021,concept_bce=.071,concept_drift_rms=.2,correct=4500)
    bad=dict(nll=.31,ece15=.02,concept_bce=.07,concept_drift_rms=.1,correct=4600)
    assert selection_key(eligible,clean,args)>selection_key(bad,clean,args)
    for key,value in [('ece15',.03),('concept_bce',.08),('concept_drift_rms',.31)]:
        rejected={**eligible,key:value}
        assert not selection_key(rejected,clean,args)[0]

def test_stronger_capacity_control_uses_all_nonlinear_branches():
    groups={0:{0,1},1:{1,2},2:{2},3:{0},4:{0,1}}
    model=ConceptNoise(5,3,groups,'adapter_matched',hidden=8)
    pi=ConceptNoise(5,3,groups,'shared_noise',hidden=8)
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in pi.parameters())
    with torch.no_grad():model.network[-1].weight.normal_(0,.05)
    model(torch.randn(6,5))[0].sum().backward()
    gradient=model.network[-1].bias.grad
    assert gradient[:5].abs().sum()>0 and gradient[5:10].abs().sum()>0 and gradient[10:].abs().sum()>0
