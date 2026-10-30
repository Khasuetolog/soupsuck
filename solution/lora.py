"""Vanilla PEFT Linear4bit with fp32 dropout and a rounded scale/add epilogue.

Pinned PEFT source: peft/tuners/lora/bnb.py:513-552. The existing Dropout
object is called once after base_layer, in the same order as the original.
An incoming fp16 activation is promoted to fp32 before dropout; no random
kernel is replaced and no extra RNG call is added. This preserves masks
and generator advancement relative to aligned contiguous fp32 norm output.

PyTorch v2.11.0 aten/src/ATen/native/Dropout.cpp:118-127 selects native CUDA
dropout for ordinary training. cuda/Dropout.cu:188-212,224-243 selects vector
width from dtype, alignment, layout and numel; :360-388 sets launch geometry
and Philox offset from numel/device; :49-87 derives masks from seed/offset
and thread index, not activation values. Both compared dropout inputs are
contiguous fp32 with 16-byte alignment, on the same device, with the same
shape, p, seed and offset. Thus their dispatch and random masks agree.
Sources: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Dropout.cpp
https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu

Mask equivalence does NOT recover pre-rounding fp32 activation values:
dropout(x16.float()) can differ numerically from dropout(x32). Also bnb
0.50.2 nn/modules.py:580-590 returns the input dtype, so an fp16 input can
change the final add's dtype. These rounding effects require GPU/quality
validation; no bitwise end-to-end equivalence is claimed for fp16 input.
"""

from functools import wraps
import math
import numbers

from .utils import enabled, supported_tensor

_MARKER = "_solution_lora_4bit_original"


def _eligible(layer, x, kwargs):
    import torch

    if (not enabled() or not supported_tensor(x) or x.ndim == 0
            or x.data_ptr() % 16 or kwargs.get("adapter_names") is not None
            or kwargs.get("alora_offsets") is not None
            or layer.disable_adapters or layer.merged
            or len(layer.active_adapters) != 1):
        return False
    if (x.dtype == torch.float16 and hasattr(layer.base_layer, "compute_dtype")
            and layer.base_layer.compute_dtype != torch.float16):
        # The norm-to-fp16 optimization assumes the base linear itself computes
        # in fp16; do not silently extend it to fp32/bf16/auto bnb computation.
        return False
    adapter = layer.active_adapters[0]
    if (adapter not in layer.lora_A or adapter not in layer.lora_B
            or adapter in layer.lora_variant or getattr(layer, "use_dora", {}).get(adapter, False)):
        return False
    a, b = layer.lora_A[adapter], layer.lora_B[adapter]
    dropout, scaling = layer.lora_dropout[adapter], layer.scaling[adapter]
    if (type(a) is not torch.nn.Linear or type(b) is not torch.nn.Linear
            or type(dropout) not in (torch.nn.Dropout, torch.nn.Identity)
            or getattr(dropout, "inplace", False)
            or not supported_tensor(a.weight) or not supported_tensor(b.weight)
            or a.weight.device != x.device or b.weight.device != x.device
            or not isinstance(scaling, numbers.Real) or not math.isfinite(scaling)):
        return False
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda") == torch.float16
    # Outside AMP, match PEFT's explicit input conversion. For fp16 ingress
    # require fp32 A/B, so restoring fp32 dropout doesn't feed a half GEMM.
    if x.dtype == torch.float16:
        return (a.weight.dtype == b.weight.dtype == torch.float32
                and getattr(layer, "cast_input_dtype_enabled", True))
    return True


def _patch_class(cls, variant_kwarg_keys):
    original = cls.forward
    if hasattr(original, _MARKER):
        return

    @wraps(original)
    def forward(self, x, *args, **kwargs):
        import torch
        from .kernels.lora_epilogue import lora_epilogue

        if not _eligible(self, x, kwargs):
            return original(self, x, *args, **kwargs)
        self._check_forward_args(x, *args, **kwargs)
        kwargs.pop("adapter_names", None)
        for key in variant_kwarg_keys:
            kwargs.pop(key, None)
        result = self.base_layer(x, *args, **kwargs).clone()
        adapter = self.active_adapters[0]
        a, b = self.lora_A[adapter], self.lora_B[adapter]
        lora_input = x.float() if x.dtype == torch.float16 else x
        requires_conversion = not torch.is_autocast_enabled("cuda")
        if requires_conversion:
            lora_input = self._cast_input_dtype(lora_input, a.weight.dtype)
        lora_out = b(a(self.lora_dropout[adapter](lora_input)))
        return lora_epilogue(result, lora_out, self.scaling[adapter],
                             cast_to_result=requires_conversion)

    setattr(forward, _MARKER, original)
    cls.forward = forward


def patch_lora_4bit():
    """Register independently of install(); optional PEFT/bnb imports are lazy."""
    try:
        from peft.tuners.lora.bnb import Linear4bit, VARIANT_KWARG_KEYS
    except (ImportError, AttributeError):
        return
    if isinstance(Linear4bit, type):
        _patch_class(Linear4bit, tuple(VARIANT_KWARG_KEYS))
