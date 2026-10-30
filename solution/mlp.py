"""Frozen Qwen2MLP with original tiled SwiGLU and optional gate/up recomputation.

store saves two fp16 [N,I] tensors; recompute saves only fp16 [N,H]. At
N2048,I8960,H1536 these are 70 MiB versus 6 MiB per layer, saving 1792 MiB
over 28 layers, excluding transient buffers and projection graph metadata.
For fp32 input, the outer gradient/output retain their original dtype: bnb
0.50.2 Linear4bit.forward returns inp_dtype even when its compute dtype is fp16.
Only modules with confirmed fp16 arithmetic qualify, so compression is exact
for gate/up and for the values seen by the recomputed input projections.

The actual projection modules supply every linear backward. We retain local
autograd GradientEdges instead of activation Tensor leaves. A nonleaf input
proxy has only a scalar leaf anchor; frozen nn.Linear saves its weight, while
bnb 0.50.2 MatMul4Bit saves (None,B) and quant_state, not A (see its
autograd/_functions.py). The down input is an anchor.expand(...).clone() filled
by our kernel; its GradientEdge keeps scalar/shape metadata, not the full act.
This costs a transient initialization copy, but avoids rerunning down_proj in
backward and avoids keeping the full act alive across layers. Autograd stops at
that edge, so the scalar anchor's derivative is never needed. Projection
backward uses bnb's own dequantization policy; we do not dequantize weights or
replace bnb's derivative. recompute rebuilds gate/up graphs under the captured
forward autocast, adding exactly two projection forwards per layer.

No RNG operation is added; LoRA wrappers, trainable weights, custom projections
or hooks, and unsupported precision use the original forward. Higher-order
gradients are not supported. Retained first-order graphs can be replayed.
"""

from functools import wraps
import importlib
import os
from types import ModuleType

import torch
from torch.autograd.function import once_differentiable

from .utils import enabled, supported_tensor

_MARKER = "_solution_fused_original"


def _eligible(mlp, x):
    if not supported_tensor(x) or x.ndim < 2 or not x.numel():
        return False
    if getattr(getattr(mlp, "config", None), "hidden_act", "silu") != "silu":
        return False
    activation = getattr(mlp, "act_fn", None)
    if activation is not None and not (
            type(activation) is torch.nn.SiLU
            or (type(activation).__module__ == "transformers.activations"
                and type(activation).__name__ == "SiLUActivation")):
        return False
    modules = tuple(getattr(mlp, name, None) for name in ("gate_proj", "up_proj", "down_proj"))
    if len({type(module) for module in modules}) != 1:
        return False
    amp_half = torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
    if torch.is_autocast_enabled("cuda") and not amp_half:
        return False
    for module in modules:
        if not isinstance(module, torch.nn.Module) or any(p.requires_grad for p in module.parameters()):
            return False
        if (module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                or module._backward_pre_hooks or "forward" in module.__dict__):
            return False
        if type(module) is torch.nn.Linear:
            if not amp_half and not (x.dtype == module.weight.dtype == torch.float16):
                return False
        elif type(module).__module__ == "bitsandbytes.nn.modules" and type(module).__name__ == "Linear4bit":
            # Exact type excludes PEFT Linear4bit wrappers and subclass overrides.
            from bitsandbytes.nn import Linear4bit
            if (type(module) is not Linear4bit or module.compute_dtype != torch.float16
                    or not module.compute_type_is_set or getattr(module.weight, "quant_state", None) is None):
                return False
        else:
            return False
        if module.weight.device != x.device:
            return False
    gate, up, down = modules
    return (gate.in_features == up.in_features == down.out_features == x.shape[-1]
            and gate.out_features == up.out_features == down.in_features)


def _input_proxy(x):
    # AddBackward saves shape/alpha, not the value/storage of the detached x.
    anchor = torch.zeros((), device=x.device, dtype=x.dtype, requires_grad=True)
    return x.detach() + anchor


def _activation_buffer(gate):
    # CloneBackward saves no clone output; the only Tensor leaf is one scalar.
    anchor = torch.zeros((), device=gate.device, dtype=gate.dtype, requires_grad=True)
    return anchor.expand(gate.shape).clone()


class _FrozenMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mlp, recompute):
        from .kernels.swiglu_fused import swiglu_fused_forward

        ctx.mlp, ctx.recompute, ctx.input_dtype = mlp, recompute, x.dtype
        ctx.amp = torch.is_autocast_enabled("cuda")
        ctx.amp_dtype = torch.get_autocast_dtype("cuda")
        edge = torch.autograd.graph.get_gradient_edge
        if recompute:
            gate, up = mlp.gate_proj(x), mlp.up_proj(x)
            ctx.save_for_backward(x.detach().to(torch.float16))
        else:
            with torch.enable_grad():
                proxy = _input_proxy(x)
                gate, up = mlp.gate_proj(proxy), mlp.up_proj(proxy)
                ctx.projection_edges = ((edge(gate), edge(up)), edge(proxy))
            ctx.save_for_backward(gate.detach().to(torch.float16), up.detach().to(torch.float16))
        ctx.gate_dtype, ctx.up_dtype = gate.dtype, up.dtype
        with torch.enable_grad():
            act = _activation_buffer(gate)
            swiglu_fused_forward(gate, up, out=act)
            output = mlp.down_proj(act)
            ctx.down_edges = (edge(output), edge(act))
        return output.detach()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        from .kernels.swiglu_fused import swiglu_fused_backward

        output_edge, act_edge = ctx.down_edges
        # Standard backward precision, outside forward's autocast region.
        with torch.autocast(device_type="cuda", enabled=False):
            dact, = torch.autograd.grad(output_edge, act_edge, grad, retain_graph=True)
        if ctx.recompute:
            (x_half,) = ctx.saved_tensors
            with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=ctx.amp,
                                                     dtype=ctx.amp_dtype):
                proxy = _input_proxy(x_half.to(ctx.input_dtype))
                gate, up = ctx.mlp.gate_proj(proxy), ctx.mlp.up_proj(proxy)
                edge = torch.autograd.graph.get_gradient_edge
                projection_edges = ((edge(gate), edge(up)), edge(proxy))
        else:
            gate, up = ctx.saved_tensors
            projection_edges = ctx.projection_edges
        dgate, dup = swiglu_fused_backward(gate, up, dact, ctx.gate_dtype, ctx.up_dtype)
        outputs, input_edge = projection_edges
        with torch.autocast(device_type="cuda", enabled=False):
            dx, = torch.autograd.grad(outputs, input_edge, (dgate, dup), retain_graph=True)
        return dx.to(ctx.input_dtype), None, None


def fused_qwen2_mlp(mlp, x, recompute=None):
    """Run an eligible frozen MLP; direct callers receive ValueError otherwise."""
    if not _eligible(mlp, x):
        raise ValueError("Expected frozen direct fp16 nn.Linear/Linear4bit projections on CUDA")
    if recompute is None:
        recompute = os.environ.get("SOUP_MLP_RECOMPUTE") == "1"
    if not torch.is_grad_enabled() or not x.requires_grad:
        from .kernels.swiglu_fused import swiglu_fused_forward

        gate, up = mlp.gate_proj(x), mlp.up_proj(x)
        return mlp.down_proj(swiglu_fused_forward(gate, up))
    return _FrozenMLP.apply(x, mlp, bool(recompute))


def patch_qwen2_mlp_fused():
    """Install once on the real Qwen2MLP; registration belongs to install()."""
    try:
        module = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    except (ImportError, AttributeError):
        return False
    if not isinstance(module, ModuleType):
        return False
    cls = getattr(module, "Qwen2MLP", None)
    if not isinstance(cls, type) or not issubclass(cls, torch.nn.Module) or cls.__module__ != module.__name__:
        return False
    original = cls.forward
    if hasattr(original, _MARKER):
        return True

    @wraps(original)
    def forward(self, x):
        if not enabled() or type(self) is not cls or not _eligible(self, x):
            return original(self, x)
        return fused_qwen2_mlp(self, x)

    setattr(forward, _MARKER, original)
    cls.forward = forward
    return True
