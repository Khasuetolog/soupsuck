"""Compose the differentiable head with detached TRL/Rewind statistics."""


def fused_head(hidden, weight, labels, num_items_in_batch=None):
    import torch
    from torch.nn import functional as F
    from soup_cli.utils.stream_loss_chunk import ChunkedStep
    from .kernels.cross_entropy import fused_linear_cross_entropy

    targets = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    token_loss, entropy, argmax, _ = fused_linear_cross_entropy(
        hidden.reshape(-1, hidden.shape[-1]).contiguous(), weight,
        targets.reshape(-1), ignore_index=-100,
    )
    denominator = (targets != -100).sum() if num_items_in_batch is None else num_items_in_batch
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.to(token_loss.device)
    loss = token_loss.sum() / denominator
    with torch.no_grad():
        valid = labels[:, 1:] != -100
        correct = ((argmax.reshape(labels.shape)[:, :-1] == labels[:, 1:]) & valid).sum()
        step = ChunkedStep(
            token_entropy=entropy.reshape(labels.shape).detach(),
            token_loss=token_loss.reshape(labels.shape).detach(),
            correct=correct, labelled=valid.sum(),
        )
    return loss, step
