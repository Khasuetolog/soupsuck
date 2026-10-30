"""Frozen linear head, cross entropy and metrics without full vocabulary logits.

Logits follow baseline AMP: z=float32(round_fp16(fp16(X)@fp16(W).T)).
For each row we maintain m=max(z), s=sum(exp(z-m)), t=sum(exp(z-m)*z).
When a vocabulary chunk raises m, both s and t are rescaled by exp(m_old-m).
Then lse=m+log(s), loss=lse-z_target, entropy=lse-t/s. Ignored targets
have zero loss and gradient, but still get entropy, argmax and lse.

On sm_75 we choose cuBLAS tensor-core GEMMs with fp16 operands and fp32
accumulation/output (torch 2.11 CUDA out_dtype), rather than a hand-tuned
tl.dot implementation: cuBLAS handles the 1536-wide reduction and row tails,
while our Triton kernels fuse the vocabulary statistics and softmax derivative.
This is an engineering choice, not a measured T4 speed claim. No bf16 or TF32
path is used. V chunks default to 4096, with 4-warps row reductions; limits
also cap each fp32 logit buffer at 64 MiB and each weight cast at 32 MiB.
At N=2048,H=1536: logits 32 MiB, cast weight 12 MiB; backward adds fp16 dZ
16 MiB, fp32 dX accumulator 12 MiB and GEMM result 12 MiB (about 84 MiB
of live temporary tensors, plus 6 MiB if hidden needs an fp32-to-fp16 cast;
excluding inputs, outputs and cuBLAS workspace).
Only hidden, frozen weight (a reference, not a copy), targets and lse survive
forward. Backward recomputes chunks and computes dZ=g*(p-onehot) in fp32.
Before casting dZ to fp16 we divide each row by max(abs(g),1e-30)/1024;
the cast operands are bounded by about 1024 even with GradScaler. After
fp32 accumulation we undo this scaling, protecting tiny probabilities and
mean-loss gradients while honoring arbitrary signed upstream gradients.

Ideas: Milakov & Gimelshein, Online normalizer calculation for softmax (2018);
Liger FusedLinearCrossEntropy; Wijmans et al., Cut Cross-Entropy (2024).
Implementation is original; no external kernel code or kernel library is used.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def vocabulary_statistics_kernel(Z, Targets, State, Best, Width, Start,
                                 IGNORE: tl.constexpr, FIRST: tl.constexpr,
                                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    z = tl.load(Z + row * Width + col, col < Width, other=-float("inf"))
    z = z.to(tl.float16).to(tl.float32)
    tile_max = tl.max(z, axis=0)
    tile_arg = tl.min(tl.where((col < Width) & (z == tile_max),
                               col + Start, 2147483647), axis=0)
    if FIRST:
        old_max = -float("inf")
        old_s = 0.0
        old_t = 0.0
        target_z = 0.0
        old_arg = 2147483647
    else:
        old_max = tl.load(State + row * 4)
        old_s = tl.load(State + row * 4 + 1)
        old_t = tl.load(State + row * 4 + 2)
        target_z = tl.load(State + row * 4 + 3)
        old_arg = tl.load(Best + row).to(tl.int32)
    new_max = tl.maximum(old_max, tile_max)
    rescale = tl.exp(old_max - new_max)
    p = tl.exp(z - new_max)
    total_s = old_s * rescale + tl.sum(p, axis=0)
    # Mask before multiplication: exp(-inf) * -inf would be NaN.
    total_t = old_t * rescale + tl.sum(p * tl.where(col < Width, z, 0.0), axis=0)
    target = tl.load(Targets + row)
    found = (col < Width) & (target == col + Start) & (target != IGNORE)
    target_z += tl.sum(tl.where(found, z, 0.0), axis=0)
    best = tl.where(tile_max > old_max, tile_arg,
                    tl.where(tile_max == old_max, tl.minimum(old_arg, tile_arg), old_arg))
    tl.store(State + row * 4, new_max)
    tl.store(State + row * 4 + 1, total_s)
    tl.store(State + row * 4 + 2, total_t)
    tl.store(State + row * 4 + 3, target_z)
    tl.store(Best + row, best.to(tl.int64))


@triton.jit
def vocabulary_outputs_kernel(State, Targets, Loss, Entropy, Lse, N,
                              IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = rows < N
    maximum = tl.load(State + rows * 4, valid, other=0.0)
    normalizer = tl.load(State + rows * 4 + 1, valid, other=1.0)
    weighted = tl.load(State + rows * 4 + 2, valid, other=0.0)
    target_z = tl.load(State + rows * 4 + 3, valid, other=0.0)
    targets = tl.load(Targets + rows, valid, other=IGNORE)
    lse = maximum + tl.log(normalizer)
    tl.store(Lse + rows, lse, valid)
    tl.store(Entropy + rows, lse - weighted / normalizer, valid)
    tl.store(Loss + rows, tl.where(targets != IGNORE, lse - target_z, 0.0), valid)


@triton.jit
def vocabulary_derivative_kernel(Z, Targets, Lse, Grad, Gain, Dz, Width, Start,
                                 IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    target = tl.load(Targets + row)
    lse = tl.load(Lse + row)
    z = tl.load(Z + row * Width + cols, cols < Width, other=-float("inf"))
    z = z.to(tl.float16).to(tl.float32)
    g = tl.load(Grad + row)
    gain = tl.load(Gain + row)
    dz = (tl.exp(z - lse) - (cols + Start == target).to(tl.float32)) * g
    dz = dz / gain
    dz = tl.where(target != IGNORE, dz, 0.0)
    tl.store(Dz + row * Width + cols, dz, cols < Width)


def _chunk_size(n, h):
    limit = min(4096, (64 * 1024**2) // (4 * max(n, 1)),
                (32 * 1024**2) // (2 * h))
    if limit < 1:
        raise ValueError("A single vocabulary column exceeds the temporary memory budget")
    return 1 << (limit.bit_length() - 1)


class _FrozenLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, ignore_index):
        ctx.hidden_dtype = hidden.dtype
        hidden, targets = hidden.to(torch.float16).contiguous(), targets.contiguous()
        n, h = hidden.shape
        v = weight.shape[0]
        chunk = _chunk_size(n, h)
        loss = torch.empty(n, device=hidden.device, dtype=torch.float32)
        entropy, lse = torch.empty_like(loss), torch.empty_like(loss)
        best = torch.empty(n, device=hidden.device, dtype=torch.int64)
        if n:
            state = torch.empty((n, 4), device=hidden.device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", enabled=False):
                for start in range(0, v, chunk):
                    w = weight[start:start + chunk].to(torch.float16).contiguous()
                    z = torch.mm(hidden, w.t(), out_dtype=torch.float32)
                    width = w.shape[0]
                    vocabulary_statistics_kernel[(n,)](
                        z, targets, state, best, width, start, IGNORE=ignore_index,
                        FIRST=start == 0, BLOCK=triton.next_power_of_2(width), num_warps=4)
                    # Release before the next allocation, avoiding two live chunks.
                    del z, w
            vocabulary_outputs_kernel[(triton.cdiv(n, 256),)](
                state, targets, loss, entropy, lse, n, IGNORE=ignore_index, BLOCK=256)
        ctx.save_for_backward(hidden, weight, targets, lse)
        ctx.ignore_index, ctx.chunk = ignore_index, chunk
        ctx.mark_non_differentiable(entropy, best, lse)
        ctx.set_materialize_grads(False)
        return loss, entropy, best, lse

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, _grad_entropy, _grad_best, _grad_lse):
        if grad_loss is None or not ctx.needs_input_grad[0]:
            return None, None, None, None
        hidden, weight, targets, lse = ctx.saved_tensors
        n, _ = hidden.shape
        dx = torch.zeros_like(hidden, dtype=torch.float32)
        if n:
            with torch.autocast(device_type="cuda", enabled=False):
                grad = grad_loss.float().contiguous()
                gain = grad.abs().clamp_min_(1e-30).div_(1024.0)
                for start in range(0, weight.shape[0], ctx.chunk):
                    w = weight[start:start + ctx.chunk].to(torch.float16).contiguous()
                    z = torch.mm(hidden, w.t(), out_dtype=torch.float32)
                    width = w.shape[0]
                    dz = torch.empty_like(z, dtype=torch.float16)
                    vocabulary_derivative_kernel[(n, triton.cdiv(width, 256))](
                        z, targets, lse, grad, gain, dz, width, start,
                        IGNORE=ctx.ignore_index, BLOCK=256)
                    del z
                    contribution = torch.mm(dz, w, out_dtype=torch.float32)
                    dx.add_(contribution)
                    del dz, w, contribution
                dx.mul_(gain.reshape(n, 1))
        return dx.to(ctx.hidden_dtype), None, None, None


def fused_linear_cross_entropy(hidden, weight, targets, ignore_index=-100):
    """Return per-token loss, entropy, first argmax and lse for a frozen head.

    Inputs are CUDA hidden[N,H] fp16/fp32, weight[V,H] fp16/fp32 and targets[N]
    int64 on the same device. Targets are already shifted. Only loss has an
    autograd edge, and only first derivatives with respect to hidden are supported.
    """
    if weight.requires_grad:
        raise ValueError("fused_linear_cross_entropy requires frozen weight")
    if hidden.ndim != 2 or weight.ndim != 2 or targets.ndim != 1:
        raise ValueError("Expected hidden[N,H], weight[V,H], targets[N]")
    if hidden.shape[1] != weight.shape[1] or hidden.shape[0] != targets.shape[0]:
        raise ValueError("Incompatible hidden, weight and targets shapes")
    if weight.shape[0] == 0 or weight.shape[1] == 0:
        raise ValueError("Vocabulary and hidden dimensions must be positive")
    if hidden.dtype not in (torch.float16, torch.float32) or weight.dtype not in (torch.float16, torch.float32):
        raise ValueError("Expected fp16/fp32 hidden and weight")
    if targets.dtype != torch.int64:
        raise ValueError("Expected int64 targets")
    if not hidden.is_cuda or weight.device != hidden.device or targets.device != hidden.device:
        raise ValueError("Inputs must be on the same CUDA device")
    if not isinstance(ignore_index, int):
        raise ValueError("ignore_index must be an integer")
    with torch.cuda.device(hidden.device):
        torch._assert_async(((targets == ignore_index) |
                             ((targets >= 0) & (targets < weight.shape[0]))).all(),
                            "Target outside vocabulary")
        return _FrozenLinearCrossEntropy.apply(hidden, weight, targets, ignore_index)
