import torch
from ps_pi.augmentation_pairs import AugmentedTraining


class Source:
    def __init__(self): self.accessed=[]
    def __getitem__(self,index):
        self.accessed.append(index)
        return torch.tensor(float(index)),index%3


def test_views_only_read_registered_training_indices_and_keep_rng_state():
    source=Source();transform=lambda image:image+torch.rand(())
    dataset=AugmentedTraining(source,[2,8,11],transform,view=0)
    state=torch.random.get_rng_state().clone()
    a,label=dataset[1];b,again=dataset[1]
    assert a==b and label==again==2
    assert source.accessed==[8,8]
    assert torch.equal(state,torch.random.get_rng_state())
    other=AugmentedTraining(source,[2,8,11],transform,view=1)
    assert other[1][0]!=a
