"""Frozen whole-MLP correctness, graph retention and recomputation memory tests."""

import gc
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
TOLERANCE = dict(rtol=2e-3, atol=2e-3)


@pytest.fixture(autouse=True)
def explicit_mode_environment(monkeypatch):
    monkeypatch.setenv("SOUP_EXP_PRECOMPILE", "0")
    monkeypatch.setenv("SOUP_EXP_LAUNCH_TRIM", "0")
    monkeypatch.setenv("SOUP_MLP_OFFLOAD", "0")  # store cases are explicit, independent of production default
    for key in ("SOUP_MLP_RECOMPUTE", "SOUP_MLP_RECOMPUTE_LAYERS"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def offload_runtime(monkeypatch):
    import solution.mlp as runtime

    monkeypatch.setattr(runtime, "_OFFLOAD_POOLS", {})
    monkeypatch.setattr(runtime, "_RETIRED_POOLS", [])
    monkeypatch.setattr(runtime, "_LAYER_ORDERS", {})
    yield runtime
    # Tests may discard a whole pool; production uses events and never does this.
    torch.cuda.synchronize()
    gc.collect()


class FrozenMLP(torch.nn.Module):
    def __init__(self, hidden=64, intermediate=1001, dtype=torch.float16, bnb=False):
        super().__init__()
        self.config = SimpleNamespace(hidden_act="silu")
        if bnb:
            bitsandbytes = pytest.importorskip("bitsandbytes")

            def linear(inputs, outputs):
                return bitsandbytes.nn.Linear4bit(inputs, outputs, bias=False,
                                                  compute_dtype=torch.float16,
                                                  quant_type="nf4", compress_statistics=True)
        else:
            def linear(inputs, outputs):
                return torch.nn.Linear(inputs, outputs, bias=False, dtype=dtype)
        self.gate_proj = linear(hidden, intermediate)
        self.up_proj = linear(hidden, intermediate)
        self.down_proj = linear(intermediate, hidden)
        self.requires_grad_(False)
        self.cuda()

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("shape", [(2, 5, 64), (3, 77, 64)])
@pytest.mark.parametrize("amp", [False, True])
def test_frozen_linear_matches_pytorch(recompute, shape, amp):
    from solution.mlp import fused_qwen2_mlp

    torch.manual_seed(777)
    mlp = FrozenMLP(dtype=torch.float32 if amp else torch.float16)
    x = torch.randn(shape, device="cuda", dtype=torch.float32 if amp else torch.float16,
                    requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        expected = mlp(reference_x)
        actual = fused_qwen2_mlp(mlp, x, recompute)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, **TOLERANCE)
    upstream = torch.randn_like(actual)
    expected.backward(upstream)
    actual.backward(upstream)
    assert x.grad.dtype == x.dtype
    torch.testing.assert_close(x.grad, reference_x.grad, **TOLERANCE)
    relative_error = (x.grad.float() - reference_x.grad.float()).norm() / reference_x.grad.float().norm()
    assert relative_error.item() <= 1e-2
    assert all(p.grad is None for p in mlp.parameters())


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_bnb_linear4bit_matches_native_autograd(recompute, input_dtype):
    from solution.mlp import fused_qwen2_mlp

    torch.manual_seed(888)
    mlp = FrozenMLP(hidden=64, intermediate=256, bnb=True)
    x = torch.randn((2, 5, 64), device="cuda", dtype=input_dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    # Fixed bnb compute_dtype=fp16; its outer output still follows input dtype.
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        expected = mlp(reference_x)
        actual = fused_qwen2_mlp(mlp, x, recompute)
    assert actual.dtype == expected.dtype == input_dtype
    torch.testing.assert_close(actual, expected, **TOLERANCE)
    upstream = torch.randn_like(actual)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(x.grad, reference_x.grad, **TOLERANCE)
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("recompute", [False, True])
def test_projection_call_counts_and_retained_backward(recompute):
    from solution.mlp import fused_qwen2_mlp

    mlp = FrozenMLP()
    x = torch.randn((2, 5, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    # Profile module calls without hooks (hooks intentionally disable the fast path).
    calls = []
    original_linear = F.linear

    def counted_linear(*args, **kwargs):
        calls.append(args[1].data_ptr())
        return original_linear(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(F, "linear", counted_linear)
        output = fused_qwen2_mlp(mlp, x, recompute)
        assert len(calls) == 3
        upstream = torch.randn_like(output)
        output.backward(upstream, retain_graph=True)
        first = x.grad.clone()
        assert len(calls) == (5 if recompute else 3)
        x.grad = None
        output.backward(upstream)
        torch.testing.assert_close(x.grad, first, rtol=0, atol=0)
        assert len(calls) == (7 if recompute else 3)


@pytest.mark.parametrize("recompute", [False, True])
def test_five_sequence_lengths_reuse_tile_variants(recompute):
    from solution.mlp import fused_qwen2_mlp
    from solution.kernels.swiglu_fused import swiglu_tile_engine

    device = torch.cuda.current_device()

    def variants():
        entry = swiglu_tile_engine.device_caches.get(device)
        return len(entry[0]) if entry is not None else 0

    mlp = FrozenMLP(intermediate=256)
    old = variants()
    for sequence in (17, 32, 77, 128, 257):
        x = torch.randn((4, sequence, 64), device="cuda", dtype=torch.float16, requires_grad=True)
        out = fused_qwen2_mlp(mlp, x, recompute)
        out.backward(torch.randn_like(out))
    torch.cuda.synchronize()
    # One forward and one backward flag; runtime N alignment may double these.
    assert variants() - old <= 4


def test_recompute_reduces_peak_and_live_activation_memory():
    from solution.mlp import fused_qwen2_mlp

    # Multiple layers are essential: one layer has the same transient gate/up
    # working set in either mode; recompute saves the across-layer live tensors.
    torch.manual_seed(999)
    layers = torch.nn.ModuleList([FrozenMLP(intermediate=4096) for _ in range(4)])
    warm = torch.randn((2, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    for mode in (False, True):
        fused_qwen2_mlp(layers[0], warm, mode).sum().backward()
    del warm

    def measure(mode):
        gc.collect()
        x = torch.randn((4, 128, 64), device="cuda", dtype=torch.float16, requires_grad=True)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = x
        for layer in layers:
            output = fused_qwen2_mlp(layer, output, mode)
        torch.cuda.synchronize()
        live = torch.cuda.memory_allocated() - baseline
        output.sum().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        del output, x
        gc.collect()
        return live, peak

    store_live, store_peak = measure(False)
    recompute_live, recompute_peak = measure(True)
    print(f"MLP store live/peak={store_live}/{store_peak}; recompute={recompute_live}/{recompute_peak}")
    assert recompute_live < store_live
    assert recompute_peak < store_peak


def test_unsupported_projections_refused_and_patch_falls_back(monkeypatch):
    from solution.mlp import fused_qwen2_mlp, patch_qwen2_mlp_fused

    mlp = FrozenMLP()
    x = torch.randn((2, 5, 64), device="cuda", dtype=torch.float16)
    mlp.gate_proj.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        fused_qwen2_mlp(mlp, x)
    mlp.requires_grad_(False)

    class WrappedLinear(torch.nn.Linear):
        """Stand-in for a projection wrapper: only exact direct types qualify."""

    mlp.gate_proj = WrappedLinear(64, 1001, bias=False, dtype=torch.float16, device="cuda")
    mlp.gate_proj.requires_grad_(False)
    with pytest.raises(ValueError, match="direct"):
        fused_qwen2_mlp(mlp, x)
    pytest.importorskip("transformers")
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    monkeypatch.setattr(Qwen2MLP, "forward", Qwen2MLP.forward)
    original = Qwen2MLP.forward
    layer = Qwen2MLP(Qwen2Config(hidden_size=64, intermediate_size=128, hidden_act="silu")).cuda()
    # Trainable direct projections cannot use frozen backward: original is called.
    assert patch_qwen2_mlp_fused()
    patched = Qwen2MLP.forward
    assert patch_qwen2_mlp_fused() and Qwen2MLP.forward is patched
    values = x.float()
    torch.testing.assert_close(layer(values), original(layer, values))


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("bnb", [False, True])
def test_offload_matches_store_bits_and_retained_backward(offload_runtime, amp, bnb):
    runtime = offload_runtime
    torch.manual_seed(231)
    layer = FrozenMLP(dtype=torch.float32 if amp else torch.float16,
                      intermediate=256 if bnb else 1001, bnb=bnb)
    runtime.prepare_mlp_offload(layer, max_rows=231)
    dtype = torch.float32 if amp else torch.float16
    x = torch.randn((3, 77, 64), device="cuda", dtype=dtype, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        expected = runtime.fused_qwen2_mlp(layer, x, recompute=False, offload=False)
        actual = runtime.fused_qwen2_mlp(layer, other, recompute=False, offload=True)
    assert torch.equal(actual, expected)
    upstream = torch.randn_like(actual)
    expected.backward(upstream)
    actual.backward(upstream, retain_graph=True)
    assert torch.equal(other.grad, x.grad)
    pool = next(iter(runtime._OFFLOAD_POOLS.values()))
    assert any(slot.leased for slot in pool.host)
    other.grad = None
    actual.backward(upstream)
    assert torch.equal(other.grad, x.grad)
    assert not any(slot.leased for slot in pool.host)
    assert len(pool.staging) == 2


def test_offload_reuses_pinned_buffers_across_different_lengths(offload_runtime):
    runtime = offload_runtime
    layer = FrozenMLP(intermediate=1001)
    runtime.prepare_mlp_offload(layer, max_rows=4 * 257)
    pool = next(iter(runtime._OFFLOAD_POOLS.values()))
    pointers = [(s.gate.data_ptr(), s.up.data_ptr()) for s in pool.host]
    allocations = pool.host_allocations
    for sequence in (17, 257, 32, 128, 77, 17):
        x = torch.randn((4, sequence, 64), device="cuda", dtype=torch.float16, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        expected = runtime.fused_qwen2_mlp(layer, x, recompute=False, offload=False)
        actual = runtime.fused_qwen2_mlp(layer, other, recompute=False, offload=True)
        upstream = torch.randn_like(actual)
        expected.backward(upstream)
        actual.backward(upstream)
        assert torch.equal(actual, expected) and torch.equal(other.grad, x.grad)
        assert not any(slot.leased for slot in pool.host)
        assert pool.host_allocations == allocations
        assert [(s.gate.data_ptr(), s.up.data_ptr()) for s in pool.host] == pointers
    assert len(pool.staging) == 2


@pytest.mark.parametrize("count", [0, 1, 2, 4])
@pytest.mark.parametrize("offload", [False, True])
def test_partial_recompute_exact_layer_count_and_gradients(offload_runtime, monkeypatch, count, offload):
    runtime = offload_runtime
    torch.manual_seed(232)
    layers = torch.nn.ModuleList([FrozenMLP(intermediate=256) for _ in range(4)])
    shared = SimpleNamespace(hidden_act="silu", num_hidden_layers=4)
    for layer in layers:
        layer.config = shared
    x = torch.randn((2, 5, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    expected = x
    for layer in layers:
        expected = expected + runtime.fused_qwen2_mlp(layer, expected, recompute=False, offload=False)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    calls = []
    original_linear = F.linear

    def counted(*args, **kwargs):
        calls.append(args[1].data_ptr())
        return original_linear(*args, **kwargs)

    monkeypatch.setenv("SOUP_MLP_RECOMPUTE_LAYERS", str(count))
    monkeypatch.setenv("SOUP_MLP_OFFLOAD", "1" if offload else "0")
    monkeypatch.setattr(F, "linear", counted)
    actual = other
    for layer in layers:
        actual = actual + runtime.fused_qwen2_mlp(layer, actual)
    actual.backward(upstream)
    torch.testing.assert_close(actual, expected, **TOLERANCE)
    torch.testing.assert_close(other.grad, x.grad, **TOLERANCE)
    assert len(calls) == 3 * len(layers) + 2 * count
    assert bool(runtime._OFFLOAD_POOLS) == (offload and count < len(layers))


def test_offload_prefetch_and_multilayer_peak(offload_runtime):
    runtime = offload_runtime
    torch.manual_seed(233)
    layers = torch.nn.ModuleList([FrozenMLP(intermediate=4096) for _ in range(8)])
    runtime.prepare_mlp_offload(layers, max_rows=512)
    pool = next(iter(runtime._OFFLOAD_POOLS.values()))
    pinned_allocations = pool.host_allocations
    staging_bytes = 2 * (pool.upload_gate.numel() + pool.upload_up.numel())
    staging_bytes += sum(2 * (s.gate.numel() + s.up.numel()) for s in pool.staging)
    warm = torch.randn((4, 128, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    for recompute, offload in ((False, False), (True, False), (False, True)):
        runtime.fused_qwen2_mlp(layers[0], warm, recompute, offload).sum().backward()
    del warm

    def measure(recompute, offload):
        gc.collect()
        x = torch.randn((4, 128, 64), device="cuda", dtype=torch.float16, requires_grad=True)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        # Pools were prewarmed for all modes; count their FULL CUDA reservation
        # in offload's peak, rather than hiding persistent staging in baseline.
        if offload:
            baseline -= staging_bytes
        torch.cuda.reset_peak_memory_stats()
        out = x
        for layer in layers:
            out = out + runtime.fused_qwen2_mlp(layer, out, recompute, offload)
        out.sum().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        del out, x
        gc.collect()
        return peak

    store = measure(False, False)
    recompute = measure(True, False)
    offload = measure(False, True)
    print(f"MLP peaks bytes: store={store}, recompute={recompute}, offload={offload}; "
          f"offload CUDA window={staging_bytes}")
    assert offload < 0.75 * store
    # Offload keeps a bounded DMA window instead of per-layer saved tensors.
    # The fixed window is relatively large for tiny H=64; include it explicitly.
    assert offload <= recompute + staging_bytes + 2 * 1024**2
    assert pool.prefetches >= len(layers)
    assert pool.host_allocations == pinned_allocations
    assert not any(slot.leased for slot in pool.host)
    assert len(pool.staging) == 2


def test_offload_explicit_store_and_recompute_priority(offload_runtime, monkeypatch):
    runtime = offload_runtime
    layer = FrozenMLP(intermediate=256)
    x = torch.randn((2, 5, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    runtime.fused_qwen2_mlp(layer, x).sum().backward()
    assert not runtime._OFFLOAD_POOLS
    monkeypatch.setenv("SOUP_MLP_OFFLOAD", "1")
    monkeypatch.setenv("SOUP_MLP_RECOMPUTE", "1")
    x.grad = None
    runtime.fused_qwen2_mlp(layer, x).sum().backward()
    assert not runtime._OFFLOAD_POOLS  # full recompute takes precedence
    monkeypatch.delenv("SOUP_MLP_RECOMPUTE")
    x.grad = None
    runtime.fused_qwen2_mlp(layer, x).sum().backward()
    assert runtime._OFFLOAD_POOLS


@pytest.mark.parametrize("setting", [None, "0", "1", "other"])
def test_offload_default_and_literal_zero_opt_out(offload_runtime, monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("SOUP_MLP_OFFLOAD", raising=False)
    else:
        monkeypatch.setenv("SOUP_MLP_OFFLOAD", setting)
    layer = FrozenMLP(intermediate=256)
    x = torch.randn((2, 5, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    offload_runtime.fused_qwen2_mlp(layer, x).sum().backward()
    assert bool(offload_runtime._OFFLOAD_POOLS) == (setting != "0")


def test_offload_cross_stream_producer_and_consumer(offload_runtime):
    runtime = offload_runtime
    layer = FrozenMLP(intermediate=1001)
    runtime.prepare_mlp_offload(layer, max_rows=231)
    x = torch.randn((3, 77, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    reference = runtime.fused_qwen2_mlp(layer, x, False, False)
    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        actual = runtime.fused_qwen2_mlp(layer, other, False, True)
        actual.backward(upstream)
    torch.cuda.current_stream().wait_stream(producer)
    assert torch.equal(actual, reference)
    assert torch.equal(other.grad, x.grad)


def test_offload_ticket_rejects_overlapping_replay(offload_runtime):
    """Reject a second reader before it can alter the first reader's slot."""
    from concurrent.futures import ThreadPoolExecutor

    runtime = offload_runtime
    layer = FrozenMLP(intermediate=256)
    x = torch.randn((1, 17, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    output = runtime.fused_qwen2_mlp(layer, x, False, True)
    ticket = output.grad_fn.offloaded
    ticket.begin_backward()
    first_slot, first_stream = ticket.device_slot, ticket.compute_stream
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(ticket.begin_backward)
            with pytest.raises(RuntimeError, match="Concurrent replay"):
                future.result()
        assert ticket.device_slot is first_slot and ticket.compute_stream == first_stream
        assert first_slot.in_use and ticket.backward_active
    finally:
        ticket.end_backward(keep_graph=True)
    assert not ticket.backward_active and ticket.device_slot is None
    # A rejected concurrent reader does not consume/free the retained graph.
    output.sum().backward(retain_graph=True)
    gradient = x.grad.clone()
    x.grad = None
    output.sum().backward()
    assert torch.equal(x.grad, gradient)
    assert ticket.closed and not ticket.host.leased


@pytest.mark.parametrize("amp,bnb", [(False, False), (True, False), (True, True)])
def test_packed_variable_tokens_partial_offload_and_replay(offload_runtime, monkeypatch, amp, bnb):
    runtime = offload_runtime
    layers = torch.nn.ModuleList([FrozenMLP(intermediate=256, bnb=bnb,
                                           dtype=torch.float32 if amp else torch.float16)
                                  for _ in range(3)])
    shared = SimpleNamespace(hidden_act="silu", num_hidden_layers=3)
    for layer in layers:
        layer.config = shared
    monkeypatch.setenv("SOUP_MLP_RECOMPUTE_LAYERS", "1")
    monkeypatch.setenv("SOUP_MLP_OFFLOAD", "1")
    stable_pointers = None
    for tokens in (17, 123, 31, 257, 8, 65, 17):
        x = torch.randn((1, tokens, 64), device="cuda",
                        dtype=torch.float32 if amp else torch.float16, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            expected, actual = x, other
            for layer in layers:
                expected = expected + runtime.fused_qwen2_mlp(layer, expected, False, False)
                actual = actual + runtime.fused_qwen2_mlp(layer, actual)
        upstream = torch.randn_like(expected)
        expected.backward(upstream)
        actual.backward(upstream, retain_graph=True)
        torch.testing.assert_close(actual, expected, **TOLERANCE)
        torch.testing.assert_close(other.grad, x.grad, **TOLERANCE)
        other.grad = None
        actual.backward(upstream)
        torch.testing.assert_close(other.grad, x.grad, **TOLERANCE)
        pool = next(iter(runtime._OFFLOAD_POOLS.values()))
        assert pool.capacity >= tokens and pool.capacity <= 512
        assert len(pool.host) == 2  # three layers minus one recomputed layer
        assert not any(s.leased for s in pool.host)
        assert len(pool.staging) == 2
        if tokens == 17 and stable_pointers is None:
            assert pool.capacity == 32  # no implicit 2048-row reservation
        if tokens == 257:
            stable_pointers = [(s.gate.data_ptr(), s.up.data_ptr()) for s in pool.host]
        elif stable_pointers is not None:
            assert stable_pointers == [(s.gate.data_ptr(), s.up.data_ptr()) for s in pool.host]
        del actual, expected, x, other
    torch.cuda.synchronize()
    gc.collect()
    # A later access collects retired smaller pools after pending DMA completes.
    runtime._pool_for(layers[-1], torch.empty((1, 1, 64), device="cuda"))
    assert not runtime._RETIRED_POOLS


def test_offload_discard_graph_before_backward_reuses_host(offload_runtime):
    runtime = offload_runtime
    layer = FrozenMLP(intermediate=256)
    runtime.prepare_mlp_offload(layer, max_rows=97)
    pool = next(iter(runtime._OFFLOAD_POOLS.values()))
    pointers = [(s.gate.data_ptr(), s.up.data_ptr()) for s in pool.host]
    for tokens in (17, 97, 31, 65, 17):
        x = torch.randn((1, tokens, 64), device="cuda", dtype=torch.float16, requires_grad=True)
        output = runtime.fused_qwen2_mlp(layer, x, False, True)
        assert any(s.leased for s in pool.host)
        del output  # intentionally do not wait for D2H before context destruction
        gc.collect()
        assert not any(s.leased for s in pool.host)
    assert pool.host_allocations == 1
    assert pointers == [(s.gate.data_ptr(), s.up.data_ptr()) for s in pool.host]
    reference = runtime.fused_qwen2_mlp(layer, x, False, False)
    other = x.detach().clone().requires_grad_()
    actual = runtime.fused_qwen2_mlp(layer, other, False, True)
    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    actual.backward(upstream)
    assert torch.equal(actual, reference) and torch.equal(other.grad, x.grad)


def test_offload_out_of_order_graphs_evict_speculative_staging(offload_runtime):
    runtime = offload_runtime
    layer = FrozenMLP(intermediate=256)
    layer.config.num_hidden_layers = 3
    runtime.prepare_mlp_offload(layer, max_rows=65)
    inputs, outputs, references = [], [], []
    for tokens in (17, 31, 65):
        x = torch.randn((1, tokens, 64), device="cuda", dtype=torch.float16, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        references.append(runtime.fused_qwen2_mlp(layer, x, False, False))
        outputs.append(runtime.fused_qwen2_mlp(layer, other, False, True))
        inputs.append((x, other))
    pool = next(iter(runtime._OFFLOAD_POOLS.values()))
    pool.prefetch(outputs[0].grad_fn.offloaded)
    pool.prefetch(outputs[1].grad_fn.offloaded)
    assert all(slot.owner is not None for slot in pool.staging)
    # Graph 2 requires eviction; graphs 0/1 still retain their CPU leases.
    for index in (2, 0, 1):
        upstream = torch.randn_like(outputs[index])
        references[index].backward(upstream)
        outputs[index].backward(upstream)
        x, other = inputs[index]
        assert torch.equal(outputs[index], references[index])
        assert torch.equal(other.grad, x.grad)
    assert not any(s.leased for s in pool.host)
    assert len(pool.staging) == 2


def test_offload_backward_error_releases_reader_and_retains_cpu_data(offload_runtime, monkeypatch):
    runtime = offload_runtime
    from solution.kernels import swiglu_fused

    layer = FrozenMLP(intermediate=256)
    x = torch.randn((1, 17, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    output = runtime.fused_qwen2_mlp(layer, x, False, True)
    ticket = output.grad_fn.offloaded
    native = swiglu_fused.swiglu_fused_backward
    def fail(*args, **kwargs):
        raise RuntimeError("injected before activation backward")
    with monkeypatch.context() as patch:
        patch.setattr(swiglu_fused, "swiglu_fused_backward", fail)
        with pytest.raises(RuntimeError, match="injected"):
            output.sum().backward(retain_graph=True)
    assert not ticket.backward_active and ticket.device_slot is None
    assert ticket.host.leased and not ticket.closed
    assert swiglu_fused.swiglu_fused_backward is native
    output.sum().backward()
    assert ticket.closed and not ticket.host.leased


def test_offload_growth_preserves_an_older_live_graph(offload_runtime):
    runtime = offload_runtime
    layer = FrozenMLP(intermediate=256)
    graphs = []
    for tokens in (17, 97):
        x = torch.randn((1, tokens, 64), device="cuda", dtype=torch.float16, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        reference = runtime.fused_qwen2_mlp(layer, x, False, False)
        actual = runtime.fused_qwen2_mlp(layer, other, False, True)
        graphs.append((x, other, reference, actual))
    assert len(runtime._RETIRED_POOLS) == 1
    old = runtime._RETIRED_POOLS[0]
    assert old.capacity == 32 and any(s.leased for s in old.host)
    current = next(iter(runtime._OFFLOAD_POOLS.values()))
    assert current.capacity == 128
    for x, other, reference, actual in reversed(graphs):
        upstream = torch.randn_like(actual)
        reference.backward(upstream)
        actual.backward(upstream)
        assert torch.equal(actual, reference) and torch.equal(other.grad, x.grad)
    torch.cuda.synchronize()
    runtime._pool_for(layer, x)
    assert not runtime._RETIRED_POOLS
    assert not any(s.leased for s in current.host)


def test_prepare_reserves_only_nonrecomputed_layers(offload_runtime, monkeypatch):
    runtime = offload_runtime
    layers = torch.nn.ModuleList([FrozenMLP(intermediate=256) for _ in range(4)])
    monkeypatch.setenv("SOUP_MLP_RECOMPUTE_LAYERS", "3")
    runtime.prepare_mlp_offload(layers, max_rows=33)
    pool = next(iter(runtime._OFFLOAD_POOLS.values()))
    assert pool.capacity == 33 and len(pool.host) == 1
    assert [runtime._partial_recompute(layer) for layer in layers] == [True, True, True, False]
    runtime._OFFLOAD_POOLS.clear()
    monkeypatch.setenv("SOUP_MLP_RECOMPUTE", "1")
    runtime.prepare_mlp_offload(layers, max_rows=33)
    assert not runtime._OFFLOAD_POOLS
