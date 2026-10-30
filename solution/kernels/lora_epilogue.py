"""Vanilla LoRA scale/add, preserving eager intermediate rounding.

PEFT bnb.py:549-552 multiplies B's output in its own dtype, optionally
casts the product to the base result dtype without autocast, then adds.
Under fp16 autocast, B outputs fp16, but bnb's base result follows x.dtype:
fp16 input gives fp16 output; fp32 input gives promoted fp32 output.
An fp32 fused multiply-add would lose the intermediate fp16 rounding.

Backward mirrors addition's cast to the LoRA branch dtype, followed by
the scaling multiply. No activations are needed for this linear epilogue.
Only first derivatives are supported; scaling is a Python scalar, not a
trainable tensor. No dropout or random-number generation occurs here.
"""

import math
import numbers

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def lora_scale_add_kernel(Base, Adapter, Out, Count, Scale,
                          CAST_TO_BASE: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < Count
    base = tl.load(Base + pos, valid, other=0.0).to(tl.float32)
    adapter = tl.load(Adapter + pos, valid, other=0.0).to(tl.float32)
    scaled = (adapter * Scale).to(Adapter.dtype.element_ty).to(tl.float32)
    if CAST_TO_BASE:
        scaled = scaled.to(Base.dtype.element_ty).to(tl.float32)
    tl.store(Out + pos, base + scaled, valid)


@triton.jit
def lora_scale_gradient_kernel(Grad, AdapterGrad, Count, Scale, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    grad = tl.load(Grad + pos, pos < Count, other=0.0)
    # AddBackward casts BEFORE MulBackward scales, when output was promoted.
    branch_grad = grad.to(AdapterGrad.dtype.element_ty).to(tl.float32)
    tl.store(AdapterGrad + pos, branch_grad * Scale, pos < Count)


class _LoRAEpilogue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, result, lora_out, scaling, cast_to_result):
        dtype = result.dtype if cast_to_result else torch.promote_types(result.dtype, lora_out.dtype)
        out = torch.empty(result.shape, device=result.device, dtype=dtype)
        if result.numel():
            lora_scale_add_kernel[(triton.cdiv(result.numel(), 512),)](
                result, lora_out, out, result.numel(), float(scaling),
                CAST_TO_BASE=cast_to_result, BLOCK=512, enable_fp_fusion=False,
            )
        ctx.result_dtype, ctx.adapter_dtype = result.dtype, lora_out.dtype
        ctx.scaling, ctx.shape = float(scaling), result.shape
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if grad is None:
            return None, None, None, None
        d_result = grad.to(ctx.result_dtype) if ctx.needs_input_grad[0] else None
        d_adapter = None
        if ctx.needs_input_grad[1]:
            grad = grad.contiguous()
            d_adapter = torch.empty(ctx.shape, device=grad.device, dtype=ctx.adapter_dtype)
            if grad.numel():
                lora_scale_gradient_kernel[(triton.cdiv(grad.numel(), 512),)](
                    grad, d_adapter, grad.numel(), ctx.scaling, BLOCK=512,
                    enable_fp_fusion=False,
                )
        return d_result, d_adapter, None, None


def lora_epilogue(result, lora_out, scaling, *, cast_to_result=False):
    """Return result + rounded(scaling * lora_out), with optional PEFT cast."""
    if (not result.is_cuda or result.device != lora_out.device
            or result.shape != lora_out.shape):
        raise ValueError("Expected equal-shaped result/LoRA tensors on one CUDA device")
    if result.dtype not in (torch.float16, torch.float32) or lora_out.dtype not in (torch.float16, torch.float32):
        raise ValueError("LoRA epilogue supports fp16/fp32 tensors")
    if not isinstance(scaling, numbers.Real) or not math.isfinite(scaling):
        raise ValueError("scaling must be a finite real scalar")
    if not isinstance(cast_to_result, bool):
        raise ValueError("cast_to_result must be bool")
    return _LoRAEpilogue.apply(result.contiguous(), lora_out.contiguous(), scaling, cast_to_result)
