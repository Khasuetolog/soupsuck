"""Compose the differentiable head with detached TRL/Rewind statistics."""


def fused_head(hidden, weight, labels, num_items_in_batch=None, *, packed_head=None):
    """Compute CE on all positions, or certified real positions of one batch.

    Shift in the original [B,S] layout BEFORE selecting real positions. The
    last real token of a short row predicts ignored padding, never the next
    row's first label. Keep the original denominator and scatter detached
    statistics (not the loss graph) for TRL/Rewind. A stale/unsupported packed
    payload falls back to the existing full-row path.
    """
    import torch
    from torch.nn import functional as F
    from soup_cli.utils.stream_loss_chunk import ChunkedStep
    from .kernels.cross_entropy import fused_linear_cross_entropy

    targets = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    packed = packed_head is not None and packed_head.matches(hidden, labels)
    if packed:
        indices = packed_head.batch.indices
        head_hidden = packed_head.hidden.reshape(-1, hidden.shape[-1]).contiguous()
        head_targets = targets.reshape(-1).index_select(0, indices)
    else:
        head_hidden = hidden.reshape(-1, hidden.shape[-1]).contiguous()
        head_targets = targets.reshape(-1)
    token_loss, entropy, argmax, _ = fused_linear_cross_entropy(
        head_hidden, weight, head_targets, ignore_index=-100,
    )
    denominator = (targets != -100).sum() if num_items_in_batch is None else num_items_in_batch
    if isinstance(denominator, torch.Tensor):
        denominator = denominator.to(token_loss.device)
    loss = token_loss.sum() / denominator
    with torch.no_grad():
        if packed:
            # Padding is ignored by loss/accuracy and entropy*attention_mask.
            # Detached vectors preserve the old shape without retaining a
            # differentiable [B*S] scatter or a second head invocation.
            def scatter(values):
                result = torch.zeros(labels.numel(), device=values.device, dtype=values.dtype)
                return result.index_copy_(0, indices, values.detach())
            token_loss, entropy, argmax = map(scatter, (token_loss, entropy, argmax))
        valid = labels[:, 1:] != -100
        correct = ((argmax.reshape(labels.shape)[:, :-1] == labels[:, 1:]) & valid).sum()
        step = ChunkedStep(
            token_entropy=entropy.reshape(labels.shape).detach(),
            token_loss=token_loss.reshape(labels.shape).detach(),
            correct=correct, labelled=valid.sum(),
        )
    return loss, step
