"""Train-only native-attribute supervision in the existing fixed semantic mapping."""
import torch
from torch.nn import functional as F


def training_attribute_masses(annotations, protocol, indices, concept_count):
    if protocol['threshold_split'] != 'train' or protocol['test_examples_evaluated']:
        raise ValueError('Native mapping and thresholds must be training-only')
    if not torch.equal(annotations['indices'].cpu(), indices.cpu()):
        raise ValueError('Native training image order differs from feature order')
    attributes = sorted(map(int, protocol['evaluated_cohort']))
    groups = [protocol['evaluated_cohort'][str(a)]['concept_ids'] for a in attributes]
    if any(not g or min(g) < 0 or max(g) >= concept_count for g in groups):
        raise ValueError('Native mapping contains invalid concept IDs')
    thresholds = torch.tensor([protocol['evaluated_cohort'][str(a)]['threshold'] for a in attributes])
    if not ((thresholds > 0) & (thresholds < 1)).all():
        raise ValueError('Native thresholds must be strictly inside (0, 1)')
    known = annotations['certainty'][:, attributes] >= protocol['certainty_minimum']
    truth = annotations['positive'][:, attributes]
    positives = (known & truth).sum(0)
    negatives = (known & ~truth).sum(0)
    if (positives == 0).any() or (negatives == 0).any():
        raise ValueError('Both known native classes required for each fixed attribute')
    total = positives + negatives
    # Equal aggregate weight for positive/negative classes within each attribute.
    positive_mass = (known & truth).float() * (total / (2 * positives))
    negative_mass = (known & ~truth).float() * (total / (2 * negatives))
    return groups, thresholds, positive_mass, negative_mass


def threshold_relative_logits(raw, groups, thresholds):
    if raw.ndim != 3 or len(groups) != len(thresholds):
        raise ValueError('Expected sampled raw concepts and fixed attribute groups')
    probability = raw.sigmoid().mean(0)
    scores = torch.stack([probability[:, group].amax(-1) for group in groups], -1)
    return torch.logit(scores.clamp(1e-7, 1 - 1e-7)) - torch.logit(thresholds)


def balanced_native_loss(raw, groups, thresholds, positive_mass, negative_mass):
    logits = threshold_relative_logits(raw, groups, thresholds)
    if logits.shape != positive_mass.shape or logits.shape != negative_mass.shape:
        raise ValueError('Native masses must match the current image/attribute batch')
    mass = (positive_mass + negative_mass).sum(0)
    usable = mass > 0
    if not usable.any():
        return logits.sum() * 0
    losses = (positive_mass * F.softplus(-logits) + negative_mass * F.softplus(logits)).sum(0)
    return (losses[usable] / mass[usable]).mean()
