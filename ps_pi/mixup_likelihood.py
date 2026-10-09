"""Proper marginal class likelihood for interpolated training targets."""
import math
import torch


def marginal_soft_task_nll(logits, targets):
    """Average class probabilities over draws before applying soft-target NLL."""
    log_probability = torch.logsumexp(logits.log_softmax(-1), 0) - math.log(len(logits))
    return -(targets * log_probability).sum(-1).mean()
