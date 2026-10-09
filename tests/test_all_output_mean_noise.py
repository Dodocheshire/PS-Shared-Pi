import torch
from ps_pi.bottleneck_noise import identity_concept_bank
from ps_pi.protected_concepts import protect_bank,ProtectedBottleneckNoise
from ps_pi.all_output_mean_noise import AllOutputMeanNoise


def test_mean_decoder_matches_strong_adapter_without_extra_parameters():
    protocol={'evaluated_cohort':{'0':{'concept_ids':[0]}}}
    bank=protect_bank(identity_concept_bank({0:[0,1],1:[0,1],2:[1],3:[0,1]},4,2),protocol)
    reference=ProtectedBottleneckNoise(bank,2,'adapter_matched')
    with torch.no_grad():
        reference.network[-1].weight.normal_(0,.2);reference.network[-1].bias.normal_(0,.2)
    context=torch.randn(7,4)
    for mode in ['adapter_matched','shared_noise','shared_no_sampling','diag_noise']:
        noise=AllOutputMeanNoise(bank,2,mode);noise.load_state_dict(reference.state_dict())
        assert torch.equal(noise(context)[0],reference(context)[0])
        assert sum(p.numel() for p in noise.parameters())==sum(p.numel() for p in reference.parameters())
        assert not noise(context)[0][:,0].ne(0).any()


def test_all_mean_latents_participate_in_shared_model_and_scale_law_is_preserved():
    protocol={'evaluated_cohort':{'0':{'concept_ids':[0]}}}
    bank=protect_bank(identity_concept_bank({0:[0,1],1:[0,1],2:[1],3:[0,1]},4,2),protocol)
    noise=AllOutputMeanNoise(bank,2,'shared_noise');context=torch.randn(5,4)
    gradient=torch.autograd.grad(noise(context)[0].sum(),noise.network[-1].bias)[0]
    assert gradient[4:8].abs().sum()>0 and gradient[8:].abs().sum()>0
    old=ProtectedBottleneckNoise(bank,2,'shared_noise');old.load_state_dict(noise.state_dict())
    assert torch.equal(noise.covariance(context),old.covariance(context))
