"""Qwen2 module-global RoPE patch with an independent eager fallback.

Pinned modeling_qwen2.py:105-138 defines rotate_half and the two muls/add;
:211 looks up apply_rotary_pos_emb in that module's globals at call time.
Qwen2RotaryEmbedding.forward:87-102 computes frequencies in fp32, then
casts cos/sin to x.dtype. Its x is the pre-decoder residual/embedding stream,
so in the target run cos/sin remain fp32 while q/k can already be fp16.

hub_kernels.py:568-572 returns an identity decorator when the optional hub
package is unavailable. The downloaded T4 probe import logs contain the hub
integration module but no imports of the optional package itself. We never
request/install a hub kernel. Even if that package is present elsewhere, our
patch replaces the decorated callable, and fallback uses plain PyTorch math
instead of invoking a possibly hub-wrapped original function.
"""

from functools import wraps
import importlib
from types import ModuleType

from .utils import enabled

_MARKER = "_solution_rope_original"


def _supported(q, k, cos, sin, unsqueeze_dim=1):
    import torch

    tensors = (q, k, cos, sin)
    if (type(unsqueeze_dim) is not int or unsqueeze_dim != 1
            or not all(isinstance(t, torch.Tensor) and t.is_cuda
                       and t.dtype in (torch.float16, torch.float32) for t in tensors)
            or not all(t.device == q.device for t in tensors)
            or q.ndim != 4 or k.ndim != 4 or cos.ndim != 3 or sin.ndim != 3
            or cos.dtype != sin.dtype or cos.requires_grad or sin.requires_grad):
        return False
    b, _, sequence, width = q.shape
    return (width > 0 and width % 2 == 0 and k.shape[0] == b and k.shape[2:] == q.shape[2:]
            and cos.shape == sin.shape and cos.shape[0] in (1, b)
            and cos.shape[1:] == (sequence, width)
            and all(t.stride(-1) == 1 for t in tensors)
            and torch.cuda.get_device_capability(q.device)[0] >= 7)


def _eager_rotary(q, k, cos, sin, unsqueeze_dim=1):
    import torch

    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    def rotate(x):
        first, second = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-second, first), dim=-1)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


def patch_qwen2_rope(module=None):
    """Register after Qwen2 import; registration imports neither torch nor Triton."""
    if module is None:
        try:
            module = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
        except (ImportError, AttributeError):
            return False
    if not isinstance(module, ModuleType):
        return False
    original = getattr(module, "apply_rotary_pos_emb", None)
    if not callable(original):
        return False
    if hasattr(original, _MARKER):
        return True

    @wraps(original)
    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if not enabled() or not _supported(q, k, cos, sin, unsqueeze_dim):
            return _eager_rotary(q, k, cos, sin, unsqueeze_dim)
        from .kernels.rope import fused_rope

        return fused_rope(q, k, cos, sin)

    setattr(apply_rotary_pos_emb, _MARKER, original)
    module.apply_rotary_pos_emb = apply_rotary_pos_emb
    return True
