"""Frozen Qwen2MLP with tiled SwiGLU, optional recomputation and async offload.

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

Offload is enabled by default; only SOUP_MLP_OFFLOAD=0 disables it. It
transfers compressed gate/up to persistent pinned CPU
slots. One reusable CUDA upload pair bounds in-flight D2H storage; two CUDA
download pairs prefetch the current/next backward layer. Events protect every
overwrite; record_stream protects cross-stream allocator lifetime. No host
synchronize is used. Pools own buffers only; weak ticket links retain no graph.
Normal backward releases host leases immediately, while retain_graph=True
keeps CPU data until the final replay or context destruction. Overlapping
replay of the same ticket is rejected; distinct tickets may run concurrently.
The keep-graph query is verified in torch 2.11 torch/csrc/autograd/init.cpp;
older versions
conservatively retain CPU leases until context destruction.

At the task's maximum N and without partial recompute, offload reserves
28*70=1960 MiB pinned host memory and three CUDA pairs (210 MiB), instead of
1960 MiB CUDA saved activations.
Each step transfers 3920 MiB over PCIe, plus an upload copy/cast on the GPU.
Overlap is best-effort: if PCIe falls behind, a GPU event wait bounds memory
but may slow forward. The default follows the full T4 run 39b5050 with
SOUP_MLP_OFFLOAD=1: all 10 checks passed, speedup 1.238x and memory ratio
3.810x versus Soup, peak 3765 MiB, eval loss +0.001%. The no-offload run
e0c50a2 had peak 4795 MiB.
prepare_mlp_offload(model,max_rows=2048) can reserve all buffers before
training; lazy first use rounds actual N to a power of two and reserves only
non-recomputed layers. Reservations keep their high-water mark. Only a
larger row bound or extra concurrently retained graphs expand capacity; serial
steps within the reservation do not allocate pinned/staging buffers again.

SOUP_MLP_RECOMPUTE_LAYERS=k selects the first k layers in forward depth order.
All Qwen MLP projection shapes are equal, so any k save the same bytes and
add the same GEMM work; this deterministic choice makes no unmeasured claim
that one layer is cheaper. Explicit preparation assigns depth indices; normal
Qwen forward assigns them lazily using its shared config. Full recompute
overrides partial selection; recomputed layers do not offload. Recompute
switches default off; explicit SOUP_MLP_OFFLOAD=0 selects store instead.

EXP-007: SOUP_EXP_OFFLOAD_TAIL=k (default 0) keeps the last k depth-ordered
MLPs in GPU store mode. Full/partial recompute still takes precedence. When
k>0, offloaded layers use three download pairs for current + two predecessors;
stored tail backward starts those prefetches before its projection backward.
At N2048/I8960 this adds 70*k MiB saved GPU data plus one 70 MiB download pair
when any layer still offloads. The transfer reduction is 70*k MiB D2H
and 70*k MiB H2D. Packed rows scale actual activation/transfer bytes linearly.
The single upload buffer still waits for its prior D2H before overwrite;
removing that dependency would corrupt data or require another upload buffer.
No speed claim is made until paired T4 measurements.

No RNG operation is added; LoRA wrappers, trainable weights, custom projections
or hooks, and unsupported precision use the original forward. Higher-order
gradients are not supported. Retained first-order graphs can be replayed.
"""

from functools import wraps
import importlib
import os
import sys
import threading
import weakref
from types import ModuleType

import torch
from torch.autograd.function import once_differentiable

from .utils import enabled, supported_tensor

_MARKER = "_solution_fused_original"
_OFFLOAD_POOLS = {}
_RETIRED_POOLS = []
_LAYER_ORDERS = {}
_MODE_LOCK = threading.RLock()


class _HostSlot:
    def __init__(self, capacity, width):
        self.gate = torch.empty((capacity, width), dtype=torch.float16, pin_memory=True)
        self.up = torch.empty_like(self.gate, pin_memory=True)
        self.produced = torch.cuda.Event()
        self.written = torch.cuda.Event()
        self.read = torch.cuda.Event()
        self.has_read = False
        self.has_write = False
        self.leased = False


class _DeviceSlot:
    def __init__(self, capacity, width, device):
        self.gate = torch.empty((capacity, width), device=device, dtype=torch.float16)
        self.up = torch.empty_like(self.gate)
        self.ready = torch.cuda.Event()
        self.used = torch.cuda.Event()
        self.has_use = False
        self.owner = None  # weak reference: no activation graph retained globally
        self.in_use = False


class _OffloadPool:
    def __init__(self, device, width, capacity, slots, lookahead=1):
        self.device, self.width, self.capacity = device, width, capacity
        self.lookahead = lookahead
        self.lock = threading.RLock()
        with torch.cuda.device(device):
            self.d2h = torch.cuda.Stream(device=device)
            self.h2d = torch.cuda.Stream(device=device)
            self.host = [_HostSlot(capacity, width) for _ in range(slots)]
            self.upload_gate = torch.empty((capacity, width), device=device, dtype=torch.float16)
            self.upload_up = torch.empty_like(self.upload_gate)
            self.upload_free = torch.cuda.Event()
            # Includes prior uses of storage recycled on this allocation stream.
            self.upload_free.record(torch.cuda.current_stream(device))
            with torch.cuda.stream(self.h2d):
                self.staging = [_DeviceSlot(capacity, width, device) for _ in range(lookahead + 1)]
        self.last = None
        self.host_allocations = slots
        self.prefetches = 0

    def reserve(self, count):
        with self.lock:
            for _ in range(max(0, count - len(self.host))):
                self.host.append(_HostSlot(self.capacity, self.width))
                self.host_allocations += 1

    def store(self, gate, up):
        with self.lock, torch.cuda.device(self.device):
            slot = next((s for s in self.host if not s.leased), None)
            if slot is None:
                # Capacity expansion is only for extra concurrently retained
                # graphs; ordinary serial steps reuse the preallocated slots.
                self.reserve(len(self.host) + 1)
                slot = self.host[-1]
            slot.leased = True
            previous = self.last() if self.last is not None else None
            if previous is not None and previous.closed:
                previous = None
            ticket = _OffloadTicket(self, slot, gate.shape, previous)
            self.last = weakref.ref(ticket)
            stream = torch.cuda.current_stream(self.device)
            rows = gate.numel() // self.width
            # Bounded upload storage: never retain one CUDA pair per pending
            # D2H transfer. The current GPU stream waits only if DMA is behind.
            stream.wait_event(self.upload_free)
            self.upload_gate[:rows].view(gate.shape).copy_(gate.detach(), non_blocking=True)
            self.upload_up[:rows].view(up.shape).copy_(up.detach(), non_blocking=True)
            slot.produced.record(stream)
            with torch.cuda.stream(self.d2h):
                self.d2h.wait_event(slot.produced)
                if slot.has_read:
                    self.d2h.wait_event(slot.read)
                slot.gate[:rows].copy_(self.upload_gate[:rows], non_blocking=True)
                slot.up[:rows].copy_(self.upload_up[:rows], non_blocking=True)
                slot.written.record(self.d2h)
                slot.has_write = True
                self.upload_free.record(self.d2h)
            self.upload_gate.record_stream(stream)
            self.upload_up.record_stream(stream)
            self.upload_gate.record_stream(self.d2h)
            self.upload_up.record_stream(self.d2h)
            return ticket

    def prefetch(self, ticket, required=False):
        with self.lock, torch.cuda.device(self.device):
            if ticket.closed:
                return
            if ticket.device_slot is not None:
                if required:
                    ticket.device_slot.in_use = True
                return
            slot = next((s for s in self.staging if s.owner is None or s.owner() is None), None)
            if slot is None and required:
                slot = next((s for s in self.staging if not s.in_use), None)
                if slot is not None:
                    old = slot.owner()
                    if old is not None:
                        old.device_slot = None  # discard speculative prefetch, not CPU data
            if slot is None:
                if not required:
                    return
                # Independent concurrent backward streams may require more than
                # two slots. Never evict a buffer currently read by a kernel.
                with torch.cuda.stream(self.h2d):
                    slot = _DeviceSlot(self.capacity, self.width, self.device)
                self.staging.append(slot)
            slot.owner = weakref.ref(ticket)
            slot.in_use = required
            ticket.device_slot = slot
            rows = ticket.rows
            with torch.cuda.stream(self.h2d):
                self.h2d.wait_event(ticket.host.written)
                if slot.has_use:
                    self.h2d.wait_event(slot.used)
                slot.gate[:rows].copy_(ticket.host.gate[:rows], non_blocking=True)
                slot.up[:rows].copy_(ticket.host.up[:rows], non_blocking=True)
                slot.ready.record(self.h2d)
                ticket.host.read.record(self.h2d)
                ticket.host.has_read = True
            slot.gate.record_stream(self.h2d)
            slot.up.record_stream(self.h2d)
            self.prefetches += 1

    def prefetch_chain(self, ticket, count):
        """Best-effort lookahead only; required readers are never evicted."""
        with self.lock:
            for _ in range(count):
                if ticket is None or ticket.closed:
                    break
                self.prefetch(ticket)
                ticket = ticket.previous() if ticket.previous is not None else None


class _OffloadTicket:
    def __init__(self, pool, host, shape, previous):
        self.pool, self.host, self.shape = pool, host, tuple(shape)
        self.rows = 1
        for dimension in self.shape[:-1]:
            self.rows *= dimension
        self.previous = weakref.ref(previous) if previous is not None else None
        self.device_slot = None
        self.compute_stream = None
        self.closed = False
        self.backward_active = False

    def begin_backward(self):
        with self.pool.lock:
            if self.closed:
                raise RuntimeError("Offloaded backward graph was freed; use retain_graph=True to replay")
            if self.backward_active:
                raise RuntimeError("Concurrent replay of one offloaded MLP ticket is unsupported")
            self.backward_active = True
            ready = False
            try:
                self.pool.prefetch(self, required=True)
                self.device_slot.in_use = True
                self.compute_stream = torch.cuda.current_stream(self.pool.device)
                previous = self.previous() if self.previous is not None else None
                self.pool.prefetch_chain(previous, self.pool.lookahead)
                ready = True
            finally:
                if not ready:
                    self.release_device()
                    self.backward_active = False

    def end_backward(self, keep_graph):
        """Release this invocation even on error; retained CPU data stays replayable."""
        with self.pool.lock:
            self.release_device()
            self.backward_active = False
            if not keep_graph:
                self.close()

    def tensors(self):
        slot = self.device_slot
        self.compute_stream.wait_event(slot.ready)
        slot.gate.record_stream(self.compute_stream)
        slot.up.record_stream(self.compute_stream)
        return (slot.gate[:self.rows].view(self.shape), slot.up[:self.rows].view(self.shape))

    def release_device(self):
        with self.pool.lock:
            slot = self.device_slot
            if slot is None:
                return
            if self.compute_stream is not None:
                slot.used.record(self.compute_stream)
                slot.has_use = True
            slot.owner, slot.in_use = None, False
            self.device_slot = None
            self.compute_stream = None

    def close(self):
        with self.pool.lock:
            if not self.closed:
                self.release_device()
                self.host.leased = False
                self.closed = True

    def __del__(self, _finalizing=sys.is_finalizing):
        if not _finalizing():
            self.close()


def _pool_for(mlp, x, reserve=None, max_rows=None):
    width = mlp.gate_proj.out_features
    rows = x.numel() // x.shape[-1]
    key = (x.device, width)
    total = getattr(getattr(mlp, "config", None), "num_hidden_layers", 1)
    tail = _tail_count(mlp)
    if tail:
        total = getattr(mlp, "_solution_mlp_layer_count", total)
    lookahead = 2 if tail else 1
    count = reserve if reserve is not None else max(1, total - _partial_count(mlp) - tail)
    with _MODE_LOCK:
        # A resized pool cannot disappear while DMA still touches pinned memory.
        # Query is nonblocking; there is no host synchronization or polling loop.
        _RETIRED_POOLS[:] = [old for old in _RETIRED_POOLS if any(
            s.leased or (s.has_write and not s.written.query()) or (s.has_read and not s.read.query())
            for s in old.host) or any(s.has_use and not s.used.query() for s in old.staging)]
        pool = _OFFLOAD_POOLS.get(key)
        if (pool is None or pool.capacity < rows or (max_rows is not None and pool.capacity < max_rows)
                or pool.lookahead != lookahead):
            if pool is not None:
                _RETIRED_POOLS.append(pool)
            # Grow geometrically with packed T, rather than allocating B*S=2048
            # on the first short batch. Explicit preparation remains an upper bound.
            capacity = (max(rows, max_rows) if max_rows is not None
                        else 1 << (max(rows, 1) - 1).bit_length())
            pool = _OffloadPool(x.device, width, capacity, count, lookahead=lookahead)
            _OFFLOAD_POOLS[key] = pool
        else:
            pool.reserve(count)
        return pool


def prepare_mlp_offload(model, max_rows=2048):
    """Optional pre-training reservation; serial steps <=max_rows allocate no pool buffers.

    Lazy first use reserves only non-recomputed layers and rounds actual rows
    up to a power of two. Explicit preparation supports fixed bounds and
    assigns deterministic layer indices even to small test MLPs with distinct configs.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    layers = [m for m in model.modules() if all(hasattr(m, name) for name in
               ("gate_proj", "up_proj", "down_proj"))]
    groups = {}
    for index, layer in enumerate(layers):
        layer._solution_mlp_layer_index = index
        layer._solution_mlp_layer_count = len(layers)
        key = (layer.gate_proj.weight.device, layer.gate_proj.out_features)
        groups.setdefault(key, []).append(layer)
    for members in groups.values():
        offloaded = [layer for layer in members if os.environ.get("SOUP_MLP_RECOMPUTE") != "1"
                     and not _partial_recompute(layer) and not _tail_store(layer)]
        if not offloaded:
            continue
        layer = offloaded[0]
        # A shape-only empty input does not allocate an activation table.
        probe = torch.empty((0, layer.gate_proj.in_features), device=layer.gate_proj.weight.device)
        _pool_for(layer, probe, reserve=len(offloaded), max_rows=max_rows)


def _partial_count(mlp):
    value = os.environ.get("SOUP_MLP_RECOMPUTE_LAYERS", "0")
    count = int(value)
    config = getattr(mlp, "config", None)
    total = getattr(config, "num_hidden_layers", None)
    if count < 0 or (total is not None and count > total):
        raise ValueError("SOUP_MLP_RECOMPUTE_LAYERS must be between 0 and the layer count")
    return count


def _partial_recompute(mlp):
    count = _partial_count(mlp)
    if not count:
        return False
    return _layer_index(mlp) < count


def _layer_index(mlp):
    config = getattr(mlp, "config", None)
    with _MODE_LOCK:
        if not hasattr(mlp, "_solution_mlp_layer_index"):
            # Real Qwen2 layers share one config and first run in forward depth
            # order. Weak keys retain no modules/graphs; explicit prepare handles
            # independent configs and nonstandard execution orders.
            for key in list(_LAYER_ORDERS):
                if not _LAYER_ORDERS[key]:
                    del _LAYER_ORDERS[key]
            group = _LAYER_ORDERS.setdefault(id(config), weakref.WeakKeyDictionary())
            group[mlp] = len(group)
            mlp._solution_mlp_layer_index = group[mlp]
        return mlp._solution_mlp_layer_index


def _tail_count(mlp):
    value = os.environ.get("SOUP_EXP_OFFLOAD_TAIL", "0")
    try:
        count = int(value)
    except ValueError:
        raise ValueError("SOUP_EXP_OFFLOAD_TAIL must be a nonnegative integer") from None
    if count == 0:
        return 0
    total = getattr(mlp, "_solution_mlp_layer_count",
                    getattr(getattr(mlp, "config", None), "num_hidden_layers", None))
    if total is None or count < 0 or count > total:
        raise ValueError("SOUP_EXP_OFFLOAD_TAIL requires 0<=k<=layer count; use prepare_mlp_offload for unknown depth")
    return count


def _tail_store(mlp):
    count = _tail_count(mlp)
    if not count:
        return False
    total = getattr(mlp, "_solution_mlp_layer_count",
                    getattr(getattr(mlp, "config", None), "num_hidden_layers", None))
    return _layer_index(mlp) >= total - count


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
    def forward(ctx, x, mlp, recompute, offload, tail_store):
        from .kernels.swiglu_fused import swiglu_fused_forward

        ctx.mlp, ctx.recompute, ctx.input_dtype = mlp, recompute, x.dtype
        ctx.amp = torch.is_autocast_enabled("cuda")
        ctx.amp_dtype = torch.get_autocast_dtype("cuda")
        if tail_store:
            pool = _OFFLOAD_POOLS.get((x.device, mlp.gate_proj.out_features))
            ctx.tail_previous = pool.last if pool is not None else None
        edge = torch.autograd.graph.get_gradient_edge
        if recompute:
            gate, up = mlp.gate_proj(x), mlp.up_proj(x)
            ctx.save_for_backward(x.detach().to(torch.float16))
        else:
            with torch.enable_grad():
                proxy = _input_proxy(x)
                gate, up = mlp.gate_proj(proxy), mlp.up_proj(proxy)
                ctx.projection_edges = ((edge(gate), edge(up)), edge(proxy))
            if offload:
                ctx.offloaded = _pool_for(mlp, x).store(gate, up)
                ctx.save_for_backward()
            else:
                saved_gate, saved_up = gate.detach().to(torch.float16), up.detach().to(torch.float16)
                ctx.save_for_backward(saved_gate, saved_up)
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

        offloaded = getattr(ctx, "offloaded", None)
        tail_previous = getattr(ctx, "tail_previous", None)
        if tail_previous is not None:
            ticket = tail_previous()
            if ticket is not None:
                ticket.pool.prefetch_chain(ticket, ticket.pool.lookahead)
        # Verified in torch 2.11 torch/csrc/autograd/init.cpp. Query OUTER task
        # before local autograd.grad(retain_graph=True) creates nested tasks.
        if offloaded is not None:
            keep_query = getattr(torch._C._autograd, "_get_current_graph_task_keep_graph", None)
            keep_graph = bool(keep_query()) if keep_query is not None else True
            offloaded.begin_backward()
        try:
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
            elif offloaded is not None:
                gate, up = offloaded.tensors()
                projection_edges = ctx.projection_edges
            else:
                gate, up = ctx.saved_tensors
                projection_edges = ctx.projection_edges
            dgate, dup = swiglu_fused_backward(gate, up, dact, ctx.gate_dtype, ctx.up_dtype)
            if offloaded is not None:
                offloaded.release_device()  # event after the LAST read of gate/up
            outputs, input_edge = projection_edges
            with torch.autocast(device_type="cuda", enabled=False):
                dx, = torch.autograd.grad(outputs, input_edge, (dgate, dup), retain_graph=True)
            return dx.to(ctx.input_dtype), None, None, None, None
        finally:
            if offloaded is not None:
                offloaded.end_backward(keep_graph)


def fused_qwen2_mlp(mlp, x, recompute=None, offload=None):
    """Run an eligible frozen MLP; direct callers receive ValueError otherwise."""
    if not _eligible(mlp, x):
        raise ValueError("Expected frozen direct fp16 nn.Linear/Linear4bit projections on CUDA")
    if recompute is None:
        recompute = os.environ.get("SOUP_MLP_RECOMPUTE") == "1" or _partial_recompute(mlp)
    if offload is None:
        offload = os.environ.get("SOUP_MLP_OFFLOAD") != "0"
    offload = bool(offload) and not recompute
    tail_store = offload and _tail_store(mlp)
    offload = offload and not tail_store
    if not torch.is_grad_enabled() or not x.requires_grad:
        from .kernels.swiglu_fused import swiglu_fused_forward

        gate, up = mlp.gate_proj(x), mlp.up_proj(x)
        return mlp.down_proj(swiglu_fused_forward(gate, up))
    return _FrozenMLP.apply(x, mlp, bool(recompute), offload, tail_store)


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
