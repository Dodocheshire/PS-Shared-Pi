"""One CBL evaluation followed by class-sharing concept-coordinate sampling."""
from .bottleneck_noise import identity_concept_bank


def remapped_bank(original, compact):
    ids=compact['retained_original_ids'].tolist()
    if [original['concepts'][i] for i in ids]!=compact['concepts']:
        raise ValueError('Compact concepts do not match their original IDs')
    sets={new:original['class_sets'][old] for new,old in enumerate(ids)}
    return identity_concept_bank(sets,len(ids),len(compact['bias']))


def sample_concepts(cbl, noise, features, pack, samples, generator=None,
                    mean_multiplier=1., scale_multiplier=1.):
    raw=cbl(features)
    z=(raw-pack['train_mean'])/pack['train_std']
    # Detach conditioning, but keep gradients through the perturbed clean z.
    context=z.detach()
    draws=noise.draw(z,context,samples,generator,mean_multiplier,scale_multiplier)
    if noise.mode=='clean' or (mean_multiplier==0 and scale_multiplier==0):
        sampled_raw=raw.unsqueeze(0)
    else:
        sampled_raw=draws*pack['train_std']+pack['train_mean']
    return draws,sampled_raw,context
