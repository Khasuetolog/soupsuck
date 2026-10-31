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

Launch trim defaults on; only SOUP_EXP_LAUNCH_TRIM=0 disables it. Combined
with process precompile in wave2 d5dc84b: wall 67.69 vs 69.55 s, paired
delta -1.80 s (CI [-5.96,-1.60]), unchanged eval and passing bitwise gates.
"""

from functools import wraps
import math
import numbers
import os
import sys

from .utils import enabled, supported_tensor

_MARKER = "_solution_lora_4bit_original"


def _read_only_base_result(layer, result):
    """EXP-004: our functional epilogue never writes PEFT's defensive clone.

    Keep cloning for custom modules/hooks which might retain/mutate the base
    output while the adapter runs. The certified unpad Dropout binding only
    scatters its own input and does not mutate this result.
    """
    if os.environ.get("SOUP_EXP_LAUNCH_TRIM") == "0" or not result.is_contiguous():
        return False
    base = layer.base_layer
    if (type(base).__module__ != "bitsandbytes.nn.modules" or type(base).__name__ != "Linear4bit"
            or any(p.requires_grad for p in base.parameters())):
        return False
    adapter = layer.active_adapters[0]
    for module in (base, layer.lora_A[adapter], layer.lora_B[adapter], layer.lora_dropout[adapter]):
        if (module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                or module._backward_pre_hooks):
            return False
        if "forward" in module.__dict__ and not (
                type(module).__name__ == "Dropout"
                and hasattr(module.forward, "_solution_unpad_original")):
            return False
    return True


def _eligible(layer, x, kwargs, *, _input_dtype=None):
    import torch

    if (not enabled() or not supported_tensor(x) or x.ndim == 0
            or x.data_ptr() % 16 or kwargs.get("adapter_names") is not None
            or kwargs.get("alora_offsets") is not None
            or layer.disable_adapters or layer.merged
            or len(layer.active_adapters) != 1):
        return False
    input_dtype = x.dtype if _input_dtype is None else _input_dtype
    if (input_dtype == torch.float16 and hasattr(layer.base_layer, "compute_dtype")
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
    if input_dtype == torch.float16:
        return (a.weight.dtype == b.weight.dtype == torch.float32
                and getattr(layer, "cast_input_dtype_enabled", True))
    return True


def fp16_ingress_ready(layer, x):
    """Certify the actual registered vanilla bnb forward before changing norm dtype.

    Qwen2Attention calls q_proj/k_proj/v_proj without kwargs. Reject hooks or
    instance overrides that could inject mixed-adapter/variant arguments after
    this check. Prediction uses the future fp16 dtype without allocating a cast.
    """
    import torch

    module = sys.modules.get("peft.tuners.lora.bnb")
    cls = getattr(module, "Linear4bit", None)
    if (type(layer) is not cls or not hasattr(cls.forward, _MARKER)
            or "forward" in layer.__dict__
            or layer._forward_hooks or layer._forward_pre_hooks
            or layer._backward_hooks or layer._backward_pre_hooks):
        return False
    base = layer.base_layer
    if (type(base).__module__ != "bitsandbytes.nn.modules" or type(base).__name__ != "Linear4bit"
            or base.compute_dtype != torch.float16 or not base.compute_type_is_set
            or "forward" in base.__dict__ or base._forward_hooks or base._forward_pre_hooks
            or base._backward_hooks or base._backward_pre_hooks
            or getattr(base.weight, "quant_state", None) is None):
        return False
    return _eligible(layer, x, {}, _input_dtype=torch.float16)


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
        result = self.base_layer(x, *args, **kwargs)
        if not _read_only_base_result(self, result):
            result = result.clone()
        adapter = self.active_adapters[0]
        a, b = self.lora_A[adapter], self.lora_B[adapter]
        lora_input = x.float() if x.dtype == torch.float16 else x
        requires_conversion = not torch.is_autocast_enabled("cuda")
        if requires_conversion:
            lora_input = self._cast_input_dtype(lora_input, a.weight.dtype)
        dropped = self.lora_dropout[adapter](lora_input)
        if os.environ.get("SOUP_EXP_LORA_PAIR") == "1":
            from .lora_pair import eligible, linear_pair

            lora_out = linear_pair(dropped, a, b) if eligible(dropped, a, b) else b(a(dropped))
        else:
            lora_out = b(a(dropped))
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
