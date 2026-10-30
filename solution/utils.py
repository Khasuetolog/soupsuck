"""Runtime eligibility checks; no CUDA kernel imports at registration time."""

import os


def enabled():
    """Developer switch, independent of profiling or harness state."""
    return os.environ.get("SOUP_FUSED_DISABLE", "0").lower() not in {"1", "true", "yes"}


def supported_tensor(x):
    import torch

    return (
        isinstance(x, torch.Tensor) and x.is_cuda
        and x.dtype in (torch.float16, torch.float32) and x.is_contiguous()
        and torch.cuda.get_device_capability(x.device)[0] >= 7
    )


def qwen_model(model, cls):
    """Unwrap DDP/Accelerate and PEFT without relying on delegated attributes."""
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        if type(model) is cls:
            return model
        if hasattr(model, "get_base_model"):
            candidate = model.get_base_model()
        else:
            candidate = getattr(model, "module", None)
        if candidate is None or candidate is model:
            return None
        model = candidate
    return None


def head_eligible(model, labels=None, logits_to_keep=0, **kwargs):
    import torch

    if not (enabled() and model.training and isinstance(labels, torch.Tensor)
            and labels.is_cuda and labels.is_contiguous()):
        return False
    if labels.dtype != torch.int64 or labels.ndim != 2 or labels.numel() == 0:
        return False
    if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
        return False
    if any(k in kwargs for k in ("shift_labels", "softcap", "logit_softcap", "num_logits_to_keep")):
        return False
    if kwargs.get("return_dict") is False or kwargs.get("ignore_index", -100) != -100:
        return False
    config = model.config
    if (getattr(config, "model_type", None) != "qwen2"
            or getattr(config, "pretraining_tp", 1) != 1
            or getattr(config, "final_logit_softcapping", None)
            or getattr(config, "logit_softcap", None)
            or getattr(config, "return_dict", True) is False
            or getattr(config, "num_virtual_tokens", 0)):
        return False
    head = model.lm_head
    if (type(head) is not torch.nn.Linear or head.bias is not None
            or head.weight.requires_grad or not supported_tensor(head.weight)
            or head.weight.device != labels.device
            or head.weight.shape != (config.vocab_size, config.hidden_size)):
        return False
    # The kernel reproduces an fp16 linear including output rounding.
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda") == torch.float16
    return head.weight.dtype == torch.float16
