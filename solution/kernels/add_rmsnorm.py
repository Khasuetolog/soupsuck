"""Residual addition and Qwen2 RMSNorm in one forward/backward row pass.

Let s=cast(residual+delta, promote(residual.dtype,delta.dtype)), r=rsqrt(
mean(s.float()**2)+eps), q=cast(s.float()*r,s.dtype), y=weight*q in
promote(s.dtype,weight.dtype). Return (s,y), optionally casting y to fp16.
All casts follow eager PyTorch ordering. Backward combines the two edges:
u=cast(g_y*weight,s.dtype), a=s.float()*r,
ds=g_s+cast(r*(u-a*mean(u*a)),s.dtype), then cast ds to each input dtype.
The optional y cast has a straight-through backward cast to y's original dtype.
Weight gradients reuse the original two-stage, non-atomic RMSNorm reduction;
their product rounding depends on y's ORIGINAL dtype, before optional fp16.

On sm_75, H=1536 uses BLOCK2048/four warps without autotuning. Row counts
and strides are runtime values. Only output s, the weight reference and one
fp32 rstd per row are saved; neither original input nor y needs to be saved.
For fp32 residual/weight/output and fp16 delta, forward traffic is about
18H+4 bytes per row instead of 22H+4 (4H saved by reusing the add in registers).
Backward fuses the norm gradient, gradient-edge sum and fp16 delta cast,
avoiding separate add/copy launches and a full intermediate norm-gradient.
These are tensor-traffic estimates; T4 speed and kernel share need measurement.

Ideas: standard RMSNorm Jacobian and residual/norm fusion used in Apex-style
normalization kernels. No external kernel code or libraries are copied.
"""

import math
import numbers

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from .rmsnorm import rmsnorm_weight_partials_kernel, rmsnorm_weight_finalize_kernel


@triton.jit
def add_rmsnorm_forward_kernel(R, D, W, S, Y, Rstd, W_STRIDE,
                               H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    residual = tl.load(R + row * H + cols, mask, other=0.0).to(tl.float32)
    delta = tl.load(D + row * H + cols, mask, other=0.0).to(tl.float32)
    s = (residual + delta).to(S.dtype.element_ty).to(tl.float32)
    r = tl.rsqrt(tl.sum(s * s, axis=0) / H + EPS)
    q = (s * r).to(S.dtype.element_ty).to(tl.float32)
    w = tl.load(W + cols * W_STRIDE, mask, other=0.0).to(tl.float32)
    tl.store(S + row * H + cols, s, mask)
    tl.store(Y + row * H + cols, q * w, mask)
    tl.store(Rstd + row, r)


@triton.jit
def add_rmsnorm_backward_kernel(S, W, Rstd, Gs, Gy, Dr, Dd,
                                GS_ROW, GS_COL, GY_ROW, GY_COL, W_STRIDE,
                                H: tl.constexpr, BLOCK: tl.constexpr,
                                HAS_GS: tl.constexpr, HAS_GY: tl.constexpr,
                                WRITE_R: tl.constexpr, WRITE_D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    if HAS_GY:
        s = tl.load(S + row * H + cols, mask, other=0.0).to(tl.float32)
        g = tl.load(Gy + row * GY_ROW + cols * GY_COL, mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols * W_STRIDE, mask, other=0.0).to(tl.float32)
        r = tl.load(Rstd + row)
        a = s * r
        u = (g * w).to(S.dtype.element_ty).to(tl.float32)
        projection = tl.sum(u * a, axis=0) / H
        ds = (r * (u - a * projection)).to(S.dtype.element_ty).to(tl.float32)
    else:
        ds = tl.full((BLOCK,), 0.0, tl.float32)
    if HAS_GS:
        gs = tl.load(Gs + row * GS_ROW + cols * GS_COL, mask, other=0.0).to(tl.float32)
        ds += gs
    ds = ds.to(S.dtype.element_ty).to(tl.float32)
    if WRITE_R:
        tl.store(Dr + row * H + cols, ds, mask)
    if WRITE_D:
        tl.store(Dd + row * H + cols, ds, mask)


class _AddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, delta, weight, eps, out_dtype):
        h = residual.shape[-1]
        rows = math.prod(residual.shape[:-1])
        sum_dtype = torch.promote_types(residual.dtype, delta.dtype)
        exact_dtype = torch.promote_types(sum_dtype, weight.dtype)
        new_residual = torch.empty(residual.shape, device=residual.device, dtype=sum_dtype)
        normed = torch.empty(residual.shape, device=residual.device, dtype=out_dtype or exact_dtype)
        rstd = torch.empty(rows, device=residual.device, dtype=torch.float32)
        block = triton.next_power_of_2(h)
        warps = 4 if block <= 4096 else 8
        if rows:
            add_rmsnorm_forward_kernel[(rows,)](
                residual, delta, weight, new_residual, normed, rstd, weight.stride(0),
                h, float(eps), block, num_warps=warps, enable_fp_fusion=False)
        ctx.save_for_backward(new_residual, weight, rstd)
        ctx.residual_dtype, ctx.delta_dtype = residual.dtype, delta.dtype
        ctx.exact_dtype, ctx.rows, ctx.block, ctx.warps = exact_dtype, rows, block, warps
        ctx.set_materialize_grads(False)
        return new_residual, normed

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_residual, grad_normed):
        s, weight, rstd = ctx.saved_tensors
        rows, h = ctx.rows, s.shape[-1]
        need_r, need_d, need_w = ctx.needs_input_grad[:3]
        dr = torch.empty(s.shape, device=s.device, dtype=ctx.residual_dtype) if need_r else None
        dd = torch.empty(s.shape, device=s.device, dtype=ctx.delta_dtype) if need_d else None
        dw = None
        if rows and (need_r or need_d):
            gs = grad_residual.reshape(rows, h) if grad_residual is not None else s.view(rows, h)
            gy = grad_normed.reshape(rows, h) if grad_normed is not None else s.view(rows, h)
            add_rmsnorm_backward_kernel[(rows,)](
                s, weight, rstd, gs, gy, dr if need_r else s, dd if need_d else s,
                gs.stride(0), gs.stride(1), gy.stride(0), gy.stride(1), weight.stride(0),
                h, ctx.block, grad_residual is not None, grad_normed is not None,
                need_r, need_d, num_warps=ctx.warps, enable_fp_fusion=False)
        if need_w:
            dw = torch.zeros(weight.shape, device=weight.device, dtype=weight.dtype)
            if rows and grad_normed is not None:
                gy = grad_normed.reshape(rows, h)
                groups = triton.cdiv(rows, 32)
                partial = torch.empty((groups, h), device=s.device, dtype=torch.float32)
                rmsnorm_weight_partials_kernel[(groups, triton.cdiv(h, 128))](
                    s, gy, rstd, partial, rows, h, gy.stride(0), gy.stride(1), h,
                    ctx.exact_dtype == torch.float16, ROWS=32, COLS=128,
                    num_warps=4, enable_fp_fusion=False)
                rmsnorm_weight_finalize_kernel[(triton.cdiv(h, 32),)](
                    partial, dw, groups, h, BLOCK_R=128, BLOCK_C=32,
                    num_warps=4, enable_fp_fusion=False)
        return dr, dd, dw, None, None


def add_rms_norm(residual, delta, weight, eps, out_dtype=None):
    """Return residual+delta and its Qwen2 RMSNorm; inputs must be contiguous CUDA.

    residual/delta/weight support fp16/fp32. Shapes of the two inputs must
    agree, and weight is [H]. out_dtype is None (exact eager promotion) or
    torch.float16 (cast the exact result). Only first derivatives are supported.
    """
    if residual.shape != delta.shape or residual.ndim < 1 or residual.shape[-1] == 0:
        raise ValueError("Expected matching residual/delta [...,H] with positive H")
    if weight.ndim != 1 or weight.shape[0] != residual.shape[-1]:
        raise ValueError("Expected weight[H]")
    if not residual.is_cuda or delta.device != residual.device or weight.device != residual.device:
        raise ValueError("All inputs must be on the same CUDA device")
    if not residual.is_contiguous() or not delta.is_contiguous():
        raise ValueError("residual and delta must be contiguous")
    if any(t.dtype not in (torch.float16, torch.float32) for t in (residual, delta, weight)):
        raise ValueError("Only fp16/fp32 inputs are supported")
    if out_dtype not in (None, torch.float16):
        raise ValueError("out_dtype must be None or torch.float16")
    if not isinstance(eps, numbers.Real) or not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite nonnegative scalar")
    with torch.cuda.device(residual.device):
        return _AddRMSNorm.apply(residual, delta, weight, eps, out_dtype)
