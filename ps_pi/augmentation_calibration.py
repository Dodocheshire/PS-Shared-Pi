"""Apply the established training sensitivity rule to measured-view directions."""
from .calibrated_scale import scale_from_training_diagnostic


def measured_scale(diagnostic):
    mapped=dict(diagnostic)
    mapped['records']=[dict(row,variant='residual_noise') if row['variant']=='measured_noise' else dict(row)
                       for row in diagnostic['records']]
    result=scale_from_training_diagnostic(mapped)
    result['measured_drift']=result.pop('residual_drift')
    result['target_variant']='measured_noise'
    result['rule']='reference scale times semantic/measured-noise training RMS concept drift'
    return result
