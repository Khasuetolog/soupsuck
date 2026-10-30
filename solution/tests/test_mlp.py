"""Frozen whole-MLP correctness, graph retention and recomputation memory tests."""

import gc
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
TOLERANCE = dict(rtol=2e-3, atol=2e-3)


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
