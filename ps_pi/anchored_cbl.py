"""Train the unprotected concept encoder while retaining exact frozen native outputs."""
import copy
import torch
from torch import nn


class AnchoredCBL(nn.Module):
    def __init__(self,cbl,protocol):
        super().__init__()
        self.adaptive=cbl
        self.anchor=copy.deepcopy(cbl).eval().requires_grad_(False)
        self.adaptive.requires_grad_(True)
        ids=sorted({i for item in protocol['evaluated_cohort'].values() for i in item['concept_ids']})
        last=cbl.cbl[-1]
        if not ids or len(ids)>=last.out_features:raise ValueError('Need protected and adaptive channels')
        self.register_buffer('ids',torch.tensor(ids,device=last.weight.device))
        mask=torch.zeros(1,last.out_features,dtype=torch.bool,device=last.weight.device);mask[:,ids]=True
        self.register_buffer('protected_mask',mask)

    @property
    def cbl(self):return self.adaptive.cbl

    def forward(self,features):
        learned=self.adaptive(features)
        fixed=self.anchor(features)
        return torch.where(self.protected_mask,fixed,learned)

    @torch.no_grad()
    def restore(self):
        # These adaptive rows are unused; retain their original values despite decay.
        current=self.adaptive.cbl[-1];reference=self.anchor.cbl[-1]
        current.weight[self.ids]=reference.weight[self.ids]
        if current.bias is not None:current.bias[self.ids]=reference.bias[self.ids]
