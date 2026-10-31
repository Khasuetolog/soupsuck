"""EXP-008: collapse two bias-free AMP LoRA Linear graphs into one edge.

Forward executes the same fp16 F.linear GEMMs as CUDA autocast. Backward
executes dZ=g@B, dB=g.T@Z, dA=dZ.T@X, dX=dZ@A, with a fp16 store after
each GEMM and a final cast to the original input/parameter dtypes. Dropout,
scale/add and RNG remain outside this boundary. TF32/backend flags are untouched.
Only tensors needed by requested gradients are saved, matching the two native
Linear backward dependencies; no persistent scratch/weight cache is added.
Speed/peak acceptance must be measured on T4. Only first derivatives supported.
"""

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def eligible(x, a, b):
    if (not torch.is_grad_enabled() or not x.is_cuda or not x.is_contiguous()
            or x.dtype not in (torch.float16, torch.float32)
            or not torch.is_autocast_enabled('cuda')
            or torch.get_autocast_dtype('cuda') != torch.float16):
        return False
    for layer in (a, b):
        if (type(layer) is not torch.nn.Linear or layer.bias is not None
                or layer.weight.device != x.device or not layer.weight.is_contiguous()
                or layer.weight.dtype not in (torch.float16, torch.float32)
                or 'forward' in layer.__dict__ or layer._forward_hooks or layer._forward_pre_hooks
                or layer._backward_hooks or layer._backward_pre_hooks):
            return False
    return (x.ndim >= 2 and a.out_features > 0 and b.out_features > 0
            and a.in_features == x.shape[-1] and a.out_features == b.in_features)


class _LinearPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b):
        need_x, need_a, need_b = ctx.needs_input_grad
        x16, a16, b16 = x.to(torch.float16), a.to(torch.float16), b.to(torch.float16)
        with torch.autocast('cuda', enabled=False):
            z = F.linear(x16, a16)
            y = F.linear(z, b16)
        ctx.save_for_backward(x16 if need_a else None, a16 if need_x else None,
                              z if need_b else None, b16 if need_x or need_a else None)
        ctx.shape, ctx.dtypes = x.shape, (x.dtype, a.dtype, b.dtype)
        ctx.set_materialize_grads(False)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if grad is None:
            return None, None, None
        x, a, z, b = ctx.saved_tensors
        need_x, need_a, need_b = ctx.needs_input_grad
        dx, da, db = None, None, None
        g = grad.reshape(-1, grad.shape[-1])
        with torch.autocast('cuda', enabled=False):
            dz = torch.mm(g, b) if need_x or need_a else None
            if need_b:
                db = torch.mm(g.t(), z.reshape(-1, z.shape[-1])).to(ctx.dtypes[2])
            if need_a:
                da = torch.mm(dz.t(), x.reshape(-1, x.shape[-1])).to(ctx.dtypes[1])
            if need_x:
                dx = torch.mm(dz, a).reshape(ctx.shape).to(ctx.dtypes[0])
        return dx, da, db


def linear_pair(x, a, b):
    """Use eligible() before this direct CUDA autocast-only entry point."""
    return _LinearPair.apply(x, a.weight, b.weight)
