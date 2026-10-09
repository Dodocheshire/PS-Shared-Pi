import torch
from ps_pi.directions import build_directions
from ps_pi.residual_directions import build_residual_directions

def fixture():
    g=torch.Generator().manual_seed(36)
    x=torch.randn(40,12,generator=g);y=torch.arange(40)%2
    x[:,0]+=y*3
    targets=torch.stack([y==0,y==1,x[:,2]>0],1).float()
    class_sets={0:[0],1:[1],2:[0,1]}
    semantic=build_directions(x,targets,class_sets,2)
    return x,y,targets,class_sets,semantic

def test_residual_directions_are_unit_and_orthogonal_with_valid_sharing():
    x,y,t,sets,s=fixture();b=build_residual_directions(x,y,t,s,sets,2,rank=5)
    assert torch.allclose(b['directions'].norm(dim=1),torch.ones(3),atol=1e-6)
    assert (b['directions']@s['directions'].T).abs().max()<1e-5
    assert torch.allclose(b['group_directions'],b['factor'].T@b['directions'])
    assert torch.equal(b['factor'][:2],torch.zeros(2,2))
    assert b['residual_rank']==5

def test_construction_is_reproducible_and_translation_invariant():
    x,y,t,sets,s=fixture();a=build_residual_directions(x,y,t,s,sets,2,rank=5)
    b=build_residual_directions(x,y,t,s,sets,2,rank=5)
    assert torch.equal(a['directions'],b['directions'])
    shifted=x+4;s2=build_directions(shifted,t,sets,2)
    c=build_residual_directions(shifted,y,t,s2,sets,2,rank=5)
    assert torch.allclose(a['directions'],c['directions'],atol=1e-5)
