import math
import pytest
import torch
from ps_pi.native_training import balanced_native_loss, threshold_relative_logits, training_attribute_masses


def fixture():
    annotations = {'indices': torch.tensor([8, 2, 9, 1]),
                   'certainty': torch.tensor([[3], [3], [3], [1]]),
                   'positive': torch.tensor([[True], [False], [False], [True]])}
    protocol = {'threshold_split': 'train', 'test_examples_evaluated': False,
                'certainty_minimum': 3, 'evaluated_cohort': {'0': {'concept_ids': [0], 'threshold': .2}}}
    return annotations, protocol


def test_train_only_cohort_alignment_and_class_balance():
    a, p = fixture()
    _, _, positive, negative = training_attribute_masses(a, p, a['indices'], 2)
    assert positive.sum() == negative.sum() == 1.5
    assert positive[-1] == negative[-1] == 0
    with pytest.raises(ValueError, match='order'):
        training_attribute_masses(a, p, a['indices'].flip(0), 2)
    p['threshold_split'] = 'val'
    with pytest.raises(ValueError, match='training-only'):
        training_attribute_masses(a, p, a['indices'], 2)


def test_known_mass_mixup_and_fixed_threshold_gradients():
    a, p = fixture()
    groups, thresholds, positive, negative = training_attribute_masses(a, p, a['indices'], 2)
    raw = torch.tensor([[[math.log(.2 / .8), 7.]] * 4], requires_grad=True)
    assert torch.allclose(threshold_relative_logits(raw, groups, thresholds), torch.zeros(4, 1), atol=1e-6)
    loss = balanced_native_loss(raw, groups, thresholds, positive, negative)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)
    loss.backward()
    assert raw.grad[0, 0, 0] < 0 and raw.grad[0, 1, 0] > 0
    assert raw.grad[0, 3, 0] == 0 and (raw.grad[:, :, 1] == 0).all()
    permutation = torch.tensor([3, 2, 1, 0])
    mix = .3
    # Mix known positive/negative evidence separately; unknown labels add no mass.
    mp = mix * positive + (1 - mix) * positive[permutation]
    mn = mix * negative + (1 - mix) * negative[permutation]
    assert mp.sum() == positive.sum() and mn.sum() == negative.sum()
    assert torch.isfinite(balanced_native_loss(raw.detach(), groups, thresholds, mp, mn))


def test_attribute_response_averages_samples_before_concept_maximum():
    probability = torch.tensor([[[.9, .1]], [[.1, .9]]])
    raw = torch.logit(probability)
    result = threshold_relative_logits(raw, [[0, 1]], torch.tensor([.5]))
    assert torch.allclose(result, torch.zeros(1, 1), atol=1e-6)


def test_native_quality_gate_rejects_accuracy_gain_with_attribute_regression():
    from types import SimpleNamespace
    from ps_pi.joint_native_training_train import selection_key
    args = SimpleNamespace(nll_tolerance=0., ece_tolerance=.002,
                           concept_bce_tolerance=0., drift_limit=.3)
    base = {'correct': 10, 'nll': 1., 'ece15': .1, 'concept_bce': .05,
            'concept_drift_rms': 0., 'native_balanced_accuracy': .67, 'native_auroc': .72}
    assert selection_key(base, base, args)[0]
    improved = dict(base, correct=11, nll=.9)
    assert selection_key(improved, base, args)[0]
    assert not selection_key(dict(improved, native_balanced_accuracy=.669), base, args)[0]
    assert not selection_key(dict(improved, native_auroc=.719), base, args)[0]
