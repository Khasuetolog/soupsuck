"""Runtime eligibility checks; no CUDA kernel imports at registration time."""

import os


def enabled():
    """Developer switch, independent of profiling or harness state."""
    return os.environ.get("SOUP_FUSED_DISABLE", "0").lower() not in {"1", "true", "yes"}


def tensor_ineligibility(x, name):
    import torch

    if not isinstance(x, torch.Tensor):
        return f"{name} is not a torch.Tensor"
    if not x.is_cuda:
        return f"{name}.device={x.device} (requires CUDA)"
    if x.dtype not in (torch.float16, torch.float32):
        return f"{name}.dtype={x.dtype} (requires fp16/fp32)"
    if not x.is_contiguous():
        return f"{name} is noncontiguous, stride={x.stride()}"
    capability = torch.cuda.get_device_capability(x.device)
    if capability[0] < 7:
        return f"{name} device capability={capability} (requires SM>=7)"
    return None


def supported_tensor(x):
    return tensor_ineligibility(x, "tensor") is None


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


def parallelism_ineligibility(config):
    """A no-op ParallelismConfig does not shard tokens or model weights.

    accelerate/parallelism_config.py:167-204,274-289 defines these axes and
    their default size of one. Unknown or enabled configurations fall back.
    """
    if config is None:
        return None
    for axis in ("dp_replicate_size", "dp_shard_size", "tp_size", "cp_size", "sp_size"):
        size = getattr(config, axis, None)
        if size != 1:
            return f"parallelism_config.{axis}={size!r} (requires 1)"
    return None


def head_ineligibility(model, labels=None, logits_to_keep=0, *, _check_precision=True, **kwargs):
    """Return the first concrete refusal reason, or None.

    The trainer preflight skips only the precision check: Trainer's CUDA
    context is a nullcontext (transformers/trainer.py:2075-2080), and Accelerate
    enters autocast inside model.forward (accelerate/accelerator.py:1818-1830).
    The actual head forward always checks precision in that inner context.
    """
    import torch

    if not enabled():
        return "SOUP_FUSED_DISABLE is enabled"
    if not model.training:
        return "model.training=False"
    if not isinstance(labels, torch.Tensor):
        return "labels are absent or not a torch.Tensor"
    if not labels.is_cuda:
        return f"labels.device={labels.device} (requires CUDA)"
    if not labels.is_contiguous():
        return "labels are noncontiguous"
    if labels.dtype != torch.int64:
        return f"labels.dtype={labels.dtype} (requires int64)"
    if labels.ndim != 2 or labels.numel() == 0:
        return f"labels.shape={tuple(labels.shape)} (requires nonempty [B,S])"
    if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
        return f"unsupported logits_to_keep={logits_to_keep!r}"
    for key in ("shift_labels", "softcap", "logit_softcap", "num_logits_to_keep"):
        if key in kwargs:
            return f"unsupported forward option {key}"
    if kwargs.get("return_dict") is False:
        return "return_dict=False"
    if kwargs.get("ignore_index", -100) != -100:
        return f"ignore_index={kwargs['ignore_index']!r} (requires -100)"
    config = model.config
    if getattr(config, "model_type", None) != "qwen2":
        return f"config.model_type={getattr(config, 'model_type', None)!r}"
    if getattr(config, "pretraining_tp", 1) != 1:
        return f"config.pretraining_tp={config.pretraining_tp!r}"
    for key in ("final_logit_softcapping", "logit_softcap", "num_virtual_tokens"):
        if getattr(config, key, None):
            return f"unsupported config.{key}={getattr(config, key)!r}"
    if getattr(config, "return_dict", True) is False:
        return "config.return_dict=False"
    head = model.lm_head
    if type(head) is not torch.nn.Linear:
        return f"lm_head type is {type(head).__name__} (requires plain Linear)"
    if head.bias is not None:
        return "lm_head has a bias"
    weight = head.weight
    if weight.requires_grad:
        return "lm_head.weight.requires_grad=True"
    reason = tensor_ineligibility(weight, "lm_head.weight")
    if reason is not None:
        return reason
    if weight.device != labels.device:
        return f"lm_head.weight.device={weight.device} differs from labels.device={labels.device}"
    if weight.shape != (config.vocab_size, config.hidden_size):
        return (f"lm_head.weight.shape={tuple(weight.shape)} differs from "
                f"config shape={(config.vocab_size, config.hidden_size)}")
    if not _check_precision:
        return None
    # Authoritative check at the head, inside Accelerate's forward autocast.
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        return None if dtype == torch.float16 else f"CUDA autocast dtype={dtype} (requires fp16)"
    if weight.dtype != torch.float16:
        return f"CUDA autocast is disabled with lm_head.weight.dtype={weight.dtype}"
    return None


def head_eligible(model, labels=None, logits_to_keep=0, **kwargs):
    return head_ineligibility(model, labels, logits_to_keep, **kwargs) is None
