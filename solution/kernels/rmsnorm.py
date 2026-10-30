"""Qwen2 RMSNorm with explicit cast order and an optional weight gradient.

Let r=rsqrt(mean(x**2)+eps), a=x*r in fp32 and q=cast(a, x.dtype).
Forward returns weight*q in torch.promote_types(x.dtype, weight.dtype).
The cast's backward first gives u=cast(g*weight, x.dtype), promoted to fp32;
dx=r*(u-a*mean(u*a)). The Jacobian uses unrounded a, while dweight=sum(g*q)
uses rounded q. If the output is fp16, each g*q product is rounded to fp16
before the fp32 row reduction, matching PyTorch's multiply-before-broadcast-sum.

For sm_75, H=1536 uses one program per row, BLOCK=2048, four warps; this
keeps row statistics on chip without padding the actual allocation or autotune.
Eight warps are used only for BLOCK>4096. Weight partials cover 32 rows x
128 features; the final reduction covers all partials x 32 features. No atomics
or weight-gradient buffers/launches are used for frozen weights. Forward saves
only references to x and weight and one fp32 rstd per row; strided leading
dimensions are addressed directly, without flattening/copying the input.

Ignoring cache reuse, forward transfers H*(sizeof(x)+sizeof(weight)+sizeof(y))
+4 bytes per row; frozen backward transfers H*(sizeof(x)+sizeof(weight)
+sizeof(g)+sizeof(dx))+4. For all-fp32 H=1536 these are 18436 and 24580 bytes.
Trainable weights add H*(sizeof(x)+sizeof(g))+4 per row, plus writes/reads of
4*ceil(rows/32)*H bytes of partials and the final weight-gradient write.
These are tensor traffic estimates, not measured sm_75 bandwidth or speed.

Ideas: the standard RMSNorm Jacobian and row reductions used in Apex/Liger
RMSNorm. All kernels here are original; no external implementation is copied.
"""

import math
import numbers

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def rmsnorm_forward_kernel(X, W, Y, Rstd, H: tl.constexpr, EPS: tl.constexpr,
                           LEADING: tl.constexpr, X_STRIDES: tl.constexpr,
                           W_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    remaining = row
    offset = 0
    for dim in tl.static_range(len(LEADING) - 1, -1, -1):
        offset += (remaining % LEADING[dim]) * X_STRIDES[dim]
        remaining = remaining // LEADING[dim]
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + offset + cols, cols < H, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=0) / H + EPS)
    normalized = (x * r).to(X.dtype.element_ty).to(tl.float32)
    w = tl.load(W + cols * W_STRIDE, cols < H, other=0.0).to(tl.float32)
    tl.store(Y + row * H + cols, normalized * w, cols < H)
    tl.store(Rstd + row, r)


@triton.jit
def rmsnorm_input_gradient_kernel(X, W, G, Rstd, Dx, H: tl.constexpr,
                                  LEADING: tl.constexpr, X_STRIDES: tl.constexpr,
                                  G_STRIDES: tl.constexpr, G_LAST: tl.constexpr,
                                  W_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    remaining = row
    xoffset = 0
    goffset = 0
    for dim in tl.static_range(len(LEADING) - 1, -1, -1):
        index = remaining % LEADING[dim]
        xoffset += index * X_STRIDES[dim]
        goffset += index * G_STRIDES[dim]
        remaining = remaining // LEADING[dim]
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(X + xoffset + cols, mask, other=0.0).to(tl.float32)
    g = tl.load(G + goffset + cols * G_LAST, mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols * W_STRIDE, mask, other=0.0).to(tl.float32)
    r = tl.load(Rstd + row)
    a = x * r
    # This rounding belongs to the backward of forward's cast to x.dtype.
    u = (g * w).to(X.dtype.element_ty).to(tl.float32)
    projection = tl.sum(u * a, axis=0) / H
    dx = r * (u - a * projection)
    tl.store(Dx + row * H + cols, dx, mask)


@triton.jit
def rmsnorm_weight_partials_kernel(X, G, Rstd, Partial, N: tl.constexpr,
                                   H: tl.constexpr, LEADING: tl.constexpr,
                                   X_STRIDES: tl.constexpr, G_STRIDES: tl.constexpr,
                                   G_LAST: tl.constexpr, OUT_HALF: tl.constexpr,
                                   ROWS: tl.constexpr, COLS: tl.constexpr):
    group = tl.program_id(0)
    rows = group * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    remaining = rows
    xoffset = tl.full((ROWS,), 0, tl.int64)
    goffset = tl.full((ROWS,), 0, tl.int64)
    for dim in tl.static_range(len(LEADING) - 1, -1, -1):
        index = remaining % LEADING[dim]
        xoffset += index * X_STRIDES[dim]
        goffset += index * G_STRIDES[dim]
        remaining = remaining // LEADING[dim]
    mask = (rows[:, None] < N) & (cols[None, :] < H)
    x = tl.load(X + xoffset[:, None] + cols[None, :], mask, other=0.0).to(tl.float32)
    g = tl.load(G + goffset[:, None] + cols[None, :] * G_LAST, mask, other=0.0).to(tl.float32)
    r = tl.load(Rstd + rows, rows < N, other=0.0)
    q = (x * r[:, None]).to(X.dtype.element_ty).to(tl.float32)
    product = g * q
    if OUT_HALF:
        product = product.to(tl.float16).to(tl.float32)
    partial = tl.sum(product, axis=0)
    tl.store(Partial + group * H + cols, partial, cols < H)


@triton.jit
def rmsnorm_weight_finalize_kernel(Partial, Dw, H: tl.constexpr, GROUPS: tl.constexpr,
                                   BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    rows = tl.arange(0, BLOCK_R)
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    partial = tl.load(Partial + rows[:, None] * H + cols[None, :],
                      (rows[:, None] < GROUPS) & (cols[None, :] < H), other=0.0)
    result = tl.sum(partial, axis=0)
    tl.store(Dw + cols, result, cols < H)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        h = x.shape[-1]
        rows = math.prod(x.shape[:-1])
        block = triton.next_power_of_2(h)
        warps = 4 if block <= 4096 else 8
        out = torch.empty(x.shape, device=x.device, dtype=torch.promote_types(x.dtype, weight.dtype))
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        if rows:
            rmsnorm_forward_kernel[(rows,)](
                x, weight, out, rstd, h, float(eps), x.shape[:-1], x.stride()[:-1],
                weight.stride(0), block, num_warps=warps, enable_fp_fusion=False)
        ctx.save_for_backward(x, weight, rstd)
        ctx.rows, ctx.block, ctx.warps = rows, block, warps
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if grad is None:
            return None, None, None
        x, weight, rstd = ctx.saved_tensors
        h, rows = x.shape[-1], ctx.rows
        dx, dw = None, None
        if ctx.needs_input_grad[0]:
            dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
            if rows:
                rmsnorm_input_gradient_kernel[(rows,)](
                    x, weight, grad, rstd, dx, h, x.shape[:-1], x.stride()[:-1],
                    grad.stride()[:-1], grad.stride(-1), weight.stride(0), ctx.block,
                    num_warps=ctx.warps, enable_fp_fusion=False)
        if ctx.needs_input_grad[1]:
            dw = torch.empty(weight.shape, device=weight.device, dtype=weight.dtype)
            if rows:
                groups = triton.cdiv(rows, 32)
                partial = torch.empty((groups, h), device=x.device, dtype=torch.float32)
                rmsnorm_weight_partials_kernel[(groups, triton.cdiv(h, 128))](
                    x, grad, rstd, partial, rows, h, x.shape[:-1], x.stride()[:-1],
                    grad.stride()[:-1], grad.stride(-1), grad.dtype == torch.float16,
                    ROWS=32, COLS=128, num_warps=4, enable_fp_fusion=False)
                rmsnorm_weight_finalize_kernel[(triton.cdiv(h, 32),)](
                    partial, dw, h, groups, triton.next_power_of_2(groups), 32,
                    num_warps=4, enable_fp_fusion=False)
            else:
                dw.zero_()
        return dx, dw, None


def rms_norm(x, weight, eps):
    """Normalize the last contiguous dimension, preserving Qwen2 cast semantics.

    Supports fp16/fp32 CUDA input with arbitrary leading shapes/strides and
    a one-dimensional fp16/fp32 weight. Only first-order gradients are supported.
    Invalid inputs raise ValueError; callers choose any fallback explicitly.
    """
    if x.ndim < 1 or x.shape[-1] == 0 or weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("Expected x[...,H] and weight[H] with positive H")
    if not x.is_cuda or weight.device != x.device:
        raise ValueError("x and weight must be on the same CUDA device")
    if x.stride(-1) != 1:
        raise ValueError("The last input dimension must be contiguous")
    if x.dtype not in (torch.float16, torch.float32) or weight.dtype not in (torch.float16, torch.float32):
        raise ValueError("RMSNorm supports only fp16/fp32 input and weight")
    if not isinstance(eps, numbers.Real) or not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite nonnegative scalar")
    with torch.cuda.device(x.device):
        return _RMSNorm.apply(x, weight, eps)
