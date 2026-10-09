"""One scale derived from training-only concept sensitivity, without validation search."""
import math

def scale_from_training_diagnostic(diagnostic,reference_scale=.1):
    if diagnostic.get('split')!='train' or diagnostic.get('val_used') or diagnostic.get('test_used'):
        raise ValueError('Scale calibration must use only training rows')
    records={r['variant']:r for r in diagnostic['records']}
    reference=records['semantic_noise']['standardized_concept_drift_rms']
    residual=records['residual_noise']['standardized_concept_drift_rms']
    if not all(math.isfinite(v) and v>0 for v in [reference,residual,reference_scale]):
        raise ValueError('Positive finite training sensitivities required')
    initial=reference_scale*reference/residual
    return {'initial_scale':initial,'min_scale':.2*initial,'max_scale':3*initial,
            'reference_scale':reference_scale,'reference_drift':reference,'residual_drift':residual,
            'rule':'reference scale times semantic/residual training RMS concept drift',
            'selection_split':'train','val_used':False,'test_used':False,
            'limits':'First-order sensitivity calibration, not an exact nonlinear drift match or validation-selected optimum.'}
