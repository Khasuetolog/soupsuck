"""Joint q/k RoPE, with explicit eager multiply/add rounding and adjoint.

Rows combine q heads and k heads, with four token rows per program. Strides
are runtime values, including transposed projection views and B=1 cos/sin.
Output is contiguous [B,heads,S,D]. S, batch and head counts are runtime;
only head-width tile and dtype specialize the generated kernel geometry.

For z1=q1*c1-q2*s1, z2=q2*c2+q1*s2, the adjoint is
dq1=g1*c1+g2*s2, dq2=g2*c2-g1*s1. Each forward product is rounded to
promote(q.dtype,cos.dtype) BEFORE addition, just as eager PyTorch. In backward
each product is cast to q.dtype BEFORE branch accumulation (MulBackward's
gradient cast); the final sum is also rounded on store. k follows the same
independent contract, including mixed q/k dtypes. FMA fusion is disabled.
No q/k activation is saved: backward needs cos/sin and shape/dtype metadata.
Cos/sin gradients and other layouts use the caller's eager fallback.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def rope_pair_forward_kernel(Q, K, C, S, Yq, Yk, Tokens, Batches, QHeads, KHeads, Half, CBatches,
                             QB, QH, QT, KB, KH, KT, CB, CT, SB, ST,
                             ROWS: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    column = tl.arange(0, WIDTH)
    heads = QHeads + KHeads
    batch = row // (heads * Tokens)
    head = (row // Tokens) % heads
    token = row % Tokens
    valid = (batch[:, None] < Batches) & (column[None, :] < Half)
    is_q = head[:, None] < QHeads
    qoffset = batch[:, None] * QB + head[:, None] * QH + token[:, None] * QT + column[None, :]
    koffset = batch[:, None] * KB + (head[:, None] - QHeads) * KH + token[:, None] * KT + column[None, :]
    cbatch = tl.where(CBatches == 1, 0, batch)
    coffset = cbatch[:, None] * CB + token[:, None] * CT + column[None, :]
    soffset = cbatch[:, None] * SB + token[:, None] * ST + column[None, :]
    c1 = tl.load(C + coffset, valid, other=0.0).to(tl.float32)
    c2 = tl.load(C + coffset + Half, valid, other=0.0).to(tl.float32)
    s1 = tl.load(S + soffset, valid, other=0.0).to(tl.float32)
    s2 = tl.load(S + soffset + Half, valid, other=0.0).to(tl.float32)
    q1 = tl.load(Q + qoffset, valid & is_q, other=0.0).to(tl.float32)
    q2 = tl.load(Q + qoffset + Half, valid & is_q, other=0.0).to(tl.float32)
    k1 = tl.load(K + koffset, valid & ~is_q, other=0.0).to(tl.float32)
    k2 = tl.load(K + koffset + Half, valid & ~is_q, other=0.0).to(tl.float32)
    qlo = (q1 * c1).to(Yq.dtype.element_ty).to(tl.float32) + ((-q2) * s1).to(Yq.dtype.element_ty).to(tl.float32)
    qhi = (q2 * c2).to(Yq.dtype.element_ty).to(tl.float32) + (q1 * s2).to(Yq.dtype.element_ty).to(tl.float32)
    klo = (k1 * c1).to(Yk.dtype.element_ty).to(tl.float32) + ((-k2) * s1).to(Yk.dtype.element_ty).to(tl.float32)
    khi = (k2 * c2).to(Yk.dtype.element_ty).to(tl.float32) + (k1 * s2).to(Yk.dtype.element_ty).to(tl.float32)
    qout = ((batch[:, None] * QHeads + head[:, None]) * Tokens + token[:, None]) * (2 * Half) + column[None, :]
    kout = ((batch[:, None] * KHeads + head[:, None] - QHeads) * Tokens + token[:, None]) * (2 * Half) + column[None, :]
    tl.store(Yq + qout, qlo, valid & is_q)
    tl.store(Yq + qout + Half, qhi, valid & is_q)
    tl.store(Yk + kout, klo, valid & ~is_q)
    tl.store(Yk + kout + Half, khi, valid & ~is_q)


@triton.jit
def rope_pair_backward_kernel(Gq, Gk, C, S, Dq, Dk, Tokens, Batches, QHeads, KHeads, Half, CBatches,
                              QB, QH, QT, QD, KB, KH, KT, KD, CB, CT, SB, ST,
                              NEED_Q: tl.constexpr, NEED_K: tl.constexpr,
                              ROWS: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    column = tl.arange(0, WIDTH)
    heads = QHeads + KHeads
    batch = row // (heads * Tokens)
    head = (row // Tokens) % heads
    token = row % Tokens
    valid = (batch[:, None] < Batches) & (column[None, :] < Half)
    is_q = head[:, None] < QHeads
    cbatch = tl.where(CBatches == 1, 0, batch)
    coffset = cbatch[:, None] * CB + token[:, None] * CT + column[None, :]
    soffset = cbatch[:, None] * SB + token[:, None] * ST + column[None, :]
    c1 = tl.load(C + coffset, valid, other=0.0).to(tl.float32)
    c2 = tl.load(C + coffset + Half, valid, other=0.0).to(tl.float32)
    s1 = tl.load(S + soffset, valid, other=0.0).to(tl.float32)
    s2 = tl.load(S + soffset + Half, valid, other=0.0).to(tl.float32)
    if NEED_Q:
        off = batch[:, None] * QB + head[:, None] * QH + token[:, None] * QT + column[None, :] * QD
        g1 = tl.load(Gq + off, valid & is_q, other=0.0).to(tl.float32)
        g2 = tl.load(Gq + off + Half * QD, valid & is_q, other=0.0).to(tl.float32)
        lo = (g1 * c1).to(Dq.dtype.element_ty).to(tl.float32) + (g2 * s2).to(Dq.dtype.element_ty).to(tl.float32)
        hi = (g2 * c2).to(Dq.dtype.element_ty).to(tl.float32) - (g1 * s1).to(Dq.dtype.element_ty).to(tl.float32)
        out = ((batch[:, None] * QHeads + head[:, None]) * Tokens + token[:, None]) * (2 * Half) + column[None, :]
        tl.store(Dq + out, lo, valid & is_q)
        tl.store(Dq + out + Half, hi, valid & is_q)
    if NEED_K:
        off = batch[:, None] * KB + (head[:, None] - QHeads) * KH + token[:, None] * KT + column[None, :] * KD
        g1 = tl.load(Gk + off, valid & ~is_q, other=0.0).to(tl.float32)
        g2 = tl.load(Gk + off + Half * KD, valid & ~is_q, other=0.0).to(tl.float32)
        lo = (g1 * c1).to(Dk.dtype.element_ty).to(tl.float32) + (g2 * s2).to(Dk.dtype.element_ty).to(tl.float32)
        hi = (g2 * c2).to(Dk.dtype.element_ty).to(tl.float32) - (g1 * s1).to(Dk.dtype.element_ty).to(tl.float32)
        out = ((batch[:, None] * KHeads + head[:, None] - QHeads) * Tokens + token[:, None]) * (2 * Half) + column[None, :]
        tl.store(Dk + out, lo, valid & ~is_q)
        tl.store(Dk + out + Half, hi, valid & ~is_q)


class _RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        b, hq, tokens, width = q.shape
        hk = k.shape[1]
        yq = torch.empty(q.shape, device=q.device, dtype=torch.promote_types(q.dtype, cos.dtype))
        yk = torch.empty(k.shape, device=k.device, dtype=torch.promote_types(k.dtype, cos.dtype))
        if b and tokens and hq + hk:
            rope_pair_forward_kernel[(triton.cdiv(b * (hq + hk) * tokens, 4),)](
                q, k, cos, sin, yq, yk, tokens, b, hq, hk, width // 2, cos.shape[0],
                *q.stride()[:3], *k.stride()[:3], *cos.stride()[:2], *sin.stride()[:2],
                ROWS=4, WIDTH=triton.next_power_of_2(width // 2), num_warps=4,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(cos, sin)
        ctx.qshape, ctx.kshape = q.shape, k.shape
        ctx.qdtype, ctx.kdtype = q.dtype, k.dtype
        ctx.set_materialize_grads(False)
        non_differentiable = tuple(out for out, required in zip((yq, yk), ctx.needs_input_grad[:2]) if not required)
        if non_differentiable:
            ctx.mark_non_differentiable(*non_differentiable)
        return yq, yk

    @staticmethod
    @once_differentiable
    def backward(ctx, gq, gk):
        cos, sin = ctx.saved_tensors
        b, hq, tokens, width = ctx.qshape
        hk = ctx.kshape[1]
        dq = torch.empty(ctx.qshape, device=cos.device, dtype=ctx.qdtype) if gq is not None and ctx.needs_input_grad[0] else None
        dk = torch.empty(ctx.kshape, device=cos.device, dtype=ctx.kdtype) if gk is not None and ctx.needs_input_grad[1] else None
        if (dq is not None or dk is not None) and b and tokens and hq + hk:
            rope_pair_backward_kernel[(triton.cdiv(b * (hq + hk) * tokens, 4),)](
                gq if gq is not None else cos, gk if gk is not None else cos, cos, sin,
                dq if dq is not None else cos, dk if dk is not None else cos,
                tokens, b, hq, hk, width // 2, cos.shape[0],
                *(gq.stride() if gq is not None else (0, 0, 0, 0)),
                *(gk.stride() if gk is not None else (0, 0, 0, 0)),
                *cos.stride()[:2], *sin.stride()[:2],
                NEED_Q=dq is not None, NEED_K=dk is not None,
                ROWS=4, WIDTH=triton.next_power_of_2(width // 2), num_warps=4,
                enable_fp_fusion=False,
            )
        return dq, dk, None, None


def fused_rope(q, k, cos, sin):
    """Rotate q/k on CUDA; cos/sin are detached broadcast [1|B,S,D]."""
    from ..rope import _supported

    if not _supported(q, k, cos, sin):
        raise ValueError("Expected CUDA q/k[B,H,S,even D] and detached cos/sin[1|B,S,D], last stride 1")
    with torch.cuda.device(q.device):
        return _RoPE.apply(q, k, cos, sin)
