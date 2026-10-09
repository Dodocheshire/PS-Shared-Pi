"""Expected classification risk, distinct from marginal predictive likelihood."""


def expected_soft_task_nll(logits, targets):
    return -(targets.unsqueeze(0) * logits.log_softmax(-1)).sum(-1).mean()
