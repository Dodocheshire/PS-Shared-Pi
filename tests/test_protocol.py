"""Tests for errors that could invalidate the research comparison."""
import torch
from ps_pi.baseline import snapshot
from ps_pi.common import verify_official, official_module
from ps_pi.splits import official_indices

def test_saved_best_checkpoint_does_not_follow_optimizer_updates():
    model=torch.nn.Linear(3,2)
    mutable=model.state_dict(); saved=snapshot(model)
    original={k:v.clone() for k,v in saved.items()}
    with torch.no_grad():
        for parameter in model.parameters():parameter.add_(1)
    for key in saved:
        assert torch.equal(saved[key],original[key])
        assert not torch.equal(saved[key],mutable[key])

def test_official_source_hashes_and_cbl_dimensions():
    assert verify_official()=='749c6e3b044606c18d60ce07e3b32171f25ef563'
    CBL=official_module('model.cbl').ConceptBottleneckLayer
    model=CBL(2048,7,2,True)
    assert model(torch.randn(4,2048)).shape==(4,7)
    assert sum(p.numel() for p in model.parameters())==2048*7+7+7*7+7

def test_official_cifar_split_maps_labels_and_partitions():
    # Dataset integration is optional on machines without the native data files.
    from pathlib import Path
    import pytest
    from torchvision.datasets import CIFAR10
    root=Path('/home/pengcheng/VLG-CBM/datasets')
    if not (root/'cifar-10-batches-py').exists():pytest.skip('Native CIFAR data not present')
    train,test=CIFAR10(root,train=True),CIFAR10(root,train=False)
    indices,hashes=official_indices('cifar10',train,test)
    assert {k:len(v) for k,v in indices.items()}=={'train':45000,'val':5000,'test':10000}
    assert set(indices['train']).isdisjoint(indices['val'])
    assert set(indices['train'])|set(indices['val'])==set(range(50000))
    assert all(len(value)==64 for value in hashes.values())


def test_official_cifar100_and_cub_split_mapping():
    from pathlib import Path
    import pytest
    from torchvision.datasets import CIFAR100, ImageFolder
    root=Path('/home/pengcheng/VLG-CBM/datasets')
    if not (root/'cifar-100-python').exists() or not (root/'CUB/train').exists():
        pytest.skip('Native CIFAR-100/CUB datasets not present')
    # CIFAR-100 has one repeated filename across different fine classes.
    # Class-qualified filenames must preserve both images.
    cases=[('cifar100',CIFAR100(root,train=True),CIFAR100(root,train=False),45000,5000,10000),
           ('cub',ImageFolder(root/'CUB/train'),ImageFolder(root/'CUB/test'),3994,2000,5794)]
    for name,train,test,ntrain,nval,ntest in cases:
        indices,_=official_indices(name,train,test)
        assert [len(indices[k]) for k in ['train','val','test']]==[ntrain,nval,ntest]
        assert set(indices['train']).isdisjoint(indices['val'])
        assert set(indices['train'])|set(indices['val'])==set(range(len(train)))
