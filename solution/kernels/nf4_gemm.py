"""EXP-002: tile-local nested NF4 decoding feeding fp16 tensor-core GEMM.

No dense weight is written. Decoding rounds each weight to fp16 before tl.dot,
matching the existing decoder's multiply/add/cast order (FMA disabled). GEMM
uses fp32 accumulators and rounds its result to fp16; cuBLAS may choose a
different reduction, so GEMM equality is toleranced, not asserted bitwise.

The production allowlist starts empty. Only paired, correctness-gated T4
measurements may populate it; the opt-in flag alone never approves a shape.
No autotuning, model-dependent timing, or compiler-cache reuse in training.
"""

from functools import wraps
import os

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from .nf4 import _table, _validate_state, dequantize_nf4

# (M, original weight N, original weight K, transpose) -> launch configuration.
# transpose=False: X[M,K] @ W[N,K].T; True: dY[M,N] @ W[N,K].
APPROVED_SHAPES = {}
_MARKER = "_solution_exp_nf4_gemm_original"
CONFIGS = (
    dict(BM=16, BN=32, BK=32, num_warps=4, num_stages=2),
    dict(BM=32, BN=32, BK=32, num_warps=4, num_stages=2),
    dict(BM=32, BN=64, BK=32, num_warps=4, num_stages=2),
    dict(BM=32, BN=64, BK=64, num_warps=4, num_stages=2),
)


@triton.jit
def nf4_tile_matmul_kernel(X, Packed, Abs, Scale2, Code2, Offset, Table, Bias, Y,
                          M, N, K, TRANSPOSE: tl.constexpr, HAS_BIAS: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    if TRANSPOSE:
        output_width, reduction = K, N
    else:
        output_width, reduction = N, K
    accumulator = tl.full((BM, BN), 0, tl.float32)
    offset = tl.load(Offset)
    for start in range(tl.cdiv(reduction, BK)):
        r = start * BK + inner
        x = tl.load(X + row[:, None] * reduction + r[None, :],
                    (row[:, None] < M) & (r[None, :] < reduction), other=0)
        if TRANSPOSE:
            index = r[:, None] * K + col[None, :]
        else:
            index = col[None, :] * K + r[:, None]
        valid = (r[:, None] < reduction) & (col[None, :] < output_width)
        packed_byte = tl.load(Packed + index // 2, valid, other=0).to(tl.int32)
        symbol = tl.where((index & 1) == 0, packed_byte >> 4, packed_byte & 15)
        block = index // 64
        scale_symbol = tl.load(Abs + block, valid, other=0).to(tl.int32)
        scale_code = tl.load(Code2 + scale_symbol, valid, other=0.0)
        scale_abs = tl.load(Scale2 + block // 256, valid, other=0.0)
        scale = scale_code * scale_abs
        scale = scale + offset
        value = tl.load(Table + symbol, valid, other=0.0)
        weight = tl.where(valid, value * scale, 0.0).to(tl.float16)
        accumulator = tl.dot(x, weight, accumulator)
    if HAS_BIAS:
        accumulator += tl.load(Bias + col, col < output_width, other=0.0)[None, :].to(tl.float32)
    tl.store(Y + row[:, None] * output_width + col[None, :], accumulator,
             (row[:, None] < M) & (col[None, :] < output_width))


def nf4_matmul(x, packed, state, *, transpose=False, bias=None, config=None):
    """Direct benchmark entry. Does not claim shape approval or supply autograd."""
    n, k = _validate_state(packed, state, torch.float16)
    if (not isinstance(x, torch.Tensor) or x.device != packed.device
            or x.dtype != torch.float16 or not x.is_contiguous() or x.ndim < 2
            or x.shape[-1] != (n if transpose else k)):
        raise ValueError("Expected contiguous CUDA fp16 X matching canonical nested NF4 weight")
    width = k if transpose else n
    if bias is not None and (transpose or not isinstance(bias, torch.Tensor)
            or bias.device != x.device or bias.dtype != torch.float16
            or bias.shape != (width,) or not bias.is_contiguous() or bias.requires_grad):
        raise ValueError("Only frozen contiguous fp16 forward bias is supported")
    m = x.numel() // x.shape[-1]
    out = torch.empty((*x.shape[:-1], width), device=x.device, dtype=torch.float16)
    if m:
        launch = CONFIGS[0] if config is None else config
        with torch.cuda.device(x.device):
            nf4_tile_matmul_kernel[(triton.cdiv(m, launch["BM"]), triton.cdiv(width, launch["BN"]))](
                x, packed, state.absmax, state.state2.absmax, state.state2.code,
                state.offset, _table(x.device), bias if bias is not None else x, out,
                m, n, k, TRANSPOSE=transpose, HAS_BIAS=bias is not None,
                **launch, enable_fp_fusion=False)
    return out


class _NF4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, state, bias, original, forward_config, backward_config):
        ctx.save_for_backward(packed)
        ctx.state, ctx.backward_config = state, backward_config
        if forward_config is None:
            return original(x, packed, state, bias=bias)
        return nf4_matmul(x, packed, state, bias=bias, config=forward_config)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (packed,) = ctx.saved_tensors
        if ctx.backward_config is None:
            dense = dequantize_nf4(packed.view(-1, 1), ctx.state)
            dx = torch.matmul(grad, dense)
        else:
            dx = nf4_matmul(grad.contiguous(), packed, ctx.state, transpose=True,
                            config=ctx.backward_config)
        return dx, None, None, None, None, None, None


def patch_nf4_matmul(bnb):
    """Called only for opt-in registration beside the existing NF4 installer."""
    original = bnb.matmul_4bit
    if hasattr(original, _MARKER):
        return

    @wraps(original)
    def matmul(x, packed, quant_state, out=None, bias=None):
        shape = getattr(quant_state, "shape", ())
        if (os.environ.get("SOUP_EXP_NF4_GEMM") != "1" or not APPROVED_SHAPES
                or len(shape) != 2 or not isinstance(x, torch.Tensor)
                or not x.is_cuda or x.dtype != torch.float16 or not x.is_contiguous()
                or x.ndim < 2 or not x.numel() or out is not None
                or packed.requires_grad or (bias is not None and bias.requires_grad)
                or torch.cuda.get_device_capability(x.device) != (7, 5)):
            return original(x, packed, quant_state, out=out, bias=bias)
        n, k = shape
        m = x.numel() // x.shape[-1]
        forward_config = APPROVED_SHAPES.get((m, n, k, False))
        backward_config = APPROVED_SHAPES.get((m, n, k, True))
        if x.shape[-1] != k or (forward_config is None and backward_config is None):
            return original(x, packed, quant_state, out=out, bias=bias)
        try:
            _validate_state(packed, quant_state, torch.float16)
        except ValueError:
            return original(x, packed, quant_state, out=out, bias=bias)
        if bias is not None and (bias.dtype != torch.float16 or bias.device != x.device
                or bias.shape != (n,) or not bias.is_contiguous()):
            return original(x, packed, quant_state, out=out, bias=bias)
        return _NF4Linear.apply(x, packed, quant_state, bias, original,
                                forward_config, backward_config)

    setattr(matmul, _MARKER, original)
    bnb.matmul_4bit = matmul
