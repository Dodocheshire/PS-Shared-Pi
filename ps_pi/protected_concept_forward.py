"""Residual reconstruction preserves zero concept perturbations bit for bit."""
from .compact_concept_forward import remapped_bank as original_bank
from .protected_concepts import protect_bank


def remapped_bank(original,compact):
    return protect_bank(original_bank(original,compact),compact['native_protocol'])


def sample_concepts(cbl,noise,features,pack,samples,generator=None,mean_multiplier=1.,scale_multiplier=1.):
    raw=cbl(features)
    z=(raw-pack['train_mean'])/pack['train_std']
    context=z.detach()
    draws=noise.draw(z,context,samples,generator,mean_multiplier,scale_multiplier)
    # Reconstruct as a residual; a zero shift cannot reorder sigmoid score ties.
    sampled_raw=raw.unsqueeze(0)+(draws-z.unsqueeze(0))*pack['train_std']
    return draws,sampled_raw,context


def mean_concept_probability(raw,noise):
    values=raw.sigmoid()
    result=values.mean(0)
    protected=~noise.active_mask.bool()
    # The protected coordinates are deterministic. Use their exact expectation,
    # avoiding summation roundoff from averaging identical MC copies.
    result[:,protected]=values[0,:,protected]
    return result
