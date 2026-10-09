import pytest
from ps_pi.calibrated_scale import scale_from_training_diagnostic

def test_scale_uses_train_sensitivity_and_preserves_bound_ratios():
    diagnostic={'split':'train','val_used':False,'test_used':False,
                'records':[{'variant':'semantic_noise','standardized_concept_drift_rms':.1},
                           {'variant':'residual_noise','standardized_concept_drift_rms':.025}]}
    result=scale_from_training_diagnostic(diagnostic)
    assert result['initial_scale']==pytest.approx(.4)
    assert result['min_scale']==pytest.approx(.08)
    assert result['max_scale']==pytest.approx(1.2)
    diagnostic['val_used']=True
    with pytest.raises(ValueError):scale_from_training_diagnostic(diagnostic)

def test_zero_residual_sensitivity_is_not_silently_calibrated():
    diagnostic={'split':'train','records':[{'variant':'semantic_noise','standardized_concept_drift_rms':.1},
                                         {'variant':'residual_noise','standardized_concept_drift_rms':0}]}
    with pytest.raises(ValueError):scale_from_training_diagnostic(diagnostic)
