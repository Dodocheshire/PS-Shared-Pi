import torch
from ps_pi.mixup_likelihood import marginal_soft_task_nll
from ps_pi.joint_residual_predictive_train import marginal_task_nll


def test_one_hot_and_interpolated_targets_preserve_marginal_likelihood():
    logits = torch.tensor([[[2., -1., .5], [0., 3., -2.]],
                           [[-2., 4., 1.], [3., -1., 0.]]], dtype=torch.float64)
    first = torch.tensor([0, 1]); second = torch.tensor([2, 0])
    a = torch.nn.functional.one_hot(first, 3).double()
    b = torch.nn.functional.one_hot(second, 3).double()
    assert torch.allclose(marginal_soft_task_nll(logits, a), marginal_task_nll(logits, first))
    expected = .3 * marginal_task_nll(logits, first) + .7 * marginal_task_nll(logits, second)
    assert torch.allclose(marginal_soft_task_nll(logits, .3*a+.7*b), expected)


def test_extreme_logits_and_zero_target_coordinates_have_finite_gradients():
    logits = torch.tensor([[[1000., -1000., 0.]], [[-1000., 1000., 0.]]], requires_grad=True)
    loss = marginal_soft_task_nll(logits, torch.tensor([[.25, .75, 0.]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
