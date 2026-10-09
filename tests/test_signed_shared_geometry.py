import torch
from ps_pi.bottleneck_noise import identity_concept_bank
from ps_pi.protected_concepts import protect_bank
from ps_pi.all_output_mean_noise import AllOutputMeanNoise
from ps_pi.signed_shared_noise import SignedSharedNoise
from ps_pi.signed_shared_geometry import estimate_factor


def test_observed_anticorrelation_preserves_support_and_coordinate_variance():
    bank=protect_bank(identity_concept_bank({i:[0,1] for i in range(4)},4,2),{'evaluated_cohort':{'0':{'concept_ids':[0]}}})
    vector=torch.tensor([0.,1.,-1.,1.],dtype=torch.float64)
    covariance=vector[:,None]*vector[None,:]+torch.eye(4,dtype=torch.float64)*.01
    signed=estimate_factor(covariance,bank['factor'])
    assert torch.allclose(signed.square().sum(1),bank['factor'].square().sum(1))
    assert not signed[bank['factor']==0].ne(0).any()
    assert (signed@signed.T)[1,2]<0
    assert torch.linalg.eigvalsh(signed@signed.T).min()>-1e-6


def test_covariance_change_keeps_strong_mean_decoder_and_parameter_count():
    bank=protect_bank(identity_concept_bank({i:[0,1] for i in range(4)},4,2),{'evaluated_cohort':{'0':{'concept_ids':[0]}}})
    reference=AllOutputMeanNoise(bank,2,'shared_noise')
    bank['mean_factor']=bank['factor'].clone();bank['factor'][2]*=-1;bank['group_directions']=bank['factor'].T
    changed=SignedSharedNoise(bank,2,'shared_noise')
    with torch.no_grad():reference.network[-1].weight.normal_();reference.network[-1].bias.normal_()
    changed.network.load_state_dict(reference.network.state_dict());context=torch.randn(7,4)
    assert torch.equal(changed(context)[0],reference(context)[0])
    assert sum(p.numel() for p in changed.parameters())==sum(p.numel() for p in reference.parameters())
    assert not torch.equal(changed.covariance(context),reference.covariance(context))
    assert not changed.covariance(context)[:,0].ne(0).any()
