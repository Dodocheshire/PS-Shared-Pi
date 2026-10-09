import torch
from ps_pi.robust_likelihood import expected_soft_task_nll
from ps_pi.mixup_likelihood import marginal_soft_task_nll


def test_expected_risk_obeys_jensen_and_matches_deterministic_prediction():
    logits = torch.tensor([[[2., -1.], [0., 3.]], [[-2., 4.], [3., -1.]]], dtype=torch.float64)
    targets = torch.tensor([[.3, .7], [.8, .2]], dtype=torch.float64)
    assert expected_soft_task_nll(logits, targets) > marginal_soft_task_nll(logits, targets)
    deterministic = logits[:1].expand(4, -1, -1)
    assert torch.allclose(expected_soft_task_nll(deterministic, targets), marginal_soft_task_nll(deterministic, targets))


def test_expected_risk_has_finite_gradients_for_extreme_soft_targets():
    logits = torch.tensor([[[1000., -1000.]], [[-1000., 1000.]]], requires_grad=True)
    loss = expected_soft_task_nll(logits, torch.tensor([[.25, .75]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
