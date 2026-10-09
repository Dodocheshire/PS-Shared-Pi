import numpy as np
import pytest
from ps_pi.crossfit import fold_plan

def test_each_crossfit_image_is_predicted_by_a_model_that_did_not_fit_it():
    labels=np.repeat(np.arange(4),15)
    splits=fold_plan(labels,3,310);held_counts=np.zeros(len(labels),dtype=int)
    for fit,held in splits:
        assert not set(fit)&set(held)
        assert set(fit)|set(held)==set(range(len(labels)))
        assert np.array_equal(np.bincount(labels[held]),np.full(4,5))
        held_counts[held]+=1
    assert np.array_equal(held_counts,np.ones(len(labels),dtype=int))
    with pytest.raises(ValueError):fold_plan(np.array([0,0,1]),3,310)
