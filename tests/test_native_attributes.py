import numpy as np
import pytest
from ps_pi.native_attributes import text_mapping,threshold_from_training

def test_literal_mapping_excludes_local_marks_and_conjunctions():
    c=['black wings','white wing bars','black and white wings','gray breast','black cap','yellow throat','white belly patch']
    a=['has_wing_color::black','has_wing_color::white','has_breast_color::grey','has_crown_color::black','has_throat_color::yellow','has_belly_color::white']
    assert text_mapping(c,a)=={0:[0],2:[3],3:[4],4:[5]}

def test_known_training_only_threshold_and_ties():
    x=np.array([.1,.2,.7,.9,100.]);y=np.array([0,0,1,1,0],dtype=bool);known=np.array([1,1,1,1,0],dtype=bool)
    threshold=threshold_from_training(x,y,known)
    assert threshold==.7
    assert np.array_equal(x[:4]>=threshold,y[:4])
    assert threshold_from_training(np.ones(4),np.array([0,0,1,1],bool),np.ones(4,bool))>1.

def test_threshold_requires_both_known_classes():
    with pytest.raises(ValueError):threshold_from_training(np.ones(2),np.ones(2,bool),np.ones(2,bool))

def test_all_negative_threshold_survives_float32_application():
    import torch
    scores=np.ones(4,dtype=np.float32)
    threshold=threshold_from_training(scores,np.array([0,0,1,1],bool),np.ones(4,bool))
    assert not (torch.from_numpy(scores)>=threshold).any()
