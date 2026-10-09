"""Load immutable covariance geometry; keep the deterministic mean factor intact."""
import torch
from .common import ROOT
from .protected_concept_forward import remapped_bank as original_bank
from .protected_concept_forward import sample_concepts,mean_concept_probability


def remapped_bank(original,compact):
    expected=original_bank(original,compact)
    frozen=torch.load(ROOT/'artifacts/cub/signed_shared_geometry_v21/bank.pt',weights_only=False,map_location='cpu')
    bank=frozen['bank']
    if frozen['protocol']['baseline_sha256']!=compact['protocol']['baseline_sha256']:
        raise ValueError('Signed geometry baseline differs')
    for key in ['directions','active_mask','feature_std']:
        if not torch.equal(bank[key],expected[key]):raise ValueError('Signed geometry changed protected coordinates')
    if not torch.equal(bank['mean_factor'],expected['factor']):raise ValueError('Signed geometry changed mean decoder')
    if bank['factor'][expected['factor']==0].ne(0).any():raise ValueError('Sharing support changed')
    return bank
