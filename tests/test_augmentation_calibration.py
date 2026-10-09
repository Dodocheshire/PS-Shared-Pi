import pytest
from ps_pi.augmentation_calibration import measured_scale


def test_measured_calibration_preserves_data_and_enforces_training_only():
    diagnostic={'split':'train','val_used':False,'test_used':False,'records':[
        {'variant':'semantic_noise','standardized_concept_drift_rms':.1},
        {'variant':'measured_noise','standardized_concept_drift_rms':.05}]}
    result=measured_scale(diagnostic)
    assert result['initial_scale']==pytest.approx(.2)
    assert result['target_variant']=='measured_noise' and 'residual_drift' not in result
    assert diagnostic['records'][1]['variant']=='measured_noise'
    with pytest.raises(ValueError,match='training'):
        measured_scale(dict(diagnostic,split='val'))
