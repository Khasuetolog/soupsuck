"""NF4 decoder: exact fp16 bits against the unpatched bnb 0.50.2 CUDA kernel."""

import copy
import importlib

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _original(function):
    while hasattr(function, "_solution_fused_original"):
        function = function._solution_fused_original
    return function


@pytest.fixture
def native_bnb(monkeypatch):
    bnb = pytest.importorskip("bitsandbytes")
    if bnb.__version__ != "0.50.2":
        pytest.skip("NF4 format/backend acceptance is pinned to bitsandbytes 0.50.2")
    backend = importlib.import_module("bitsandbytes.backends.cuda.ops")
    # Integration tests may already have installed the solution. Restore BOTH
    # entries while computing oracles so a wrapper cannot compare against itself.
    monkeypatch.setattr(bnb.functional, "dequantize_4bit", _original(bnb.functional.dequantize_4bit))
    monkeypatch.setattr(backend, "_dequantize_4bit_impl", _original(backend._dequantize_4bit_impl))
    return bnb, backend


def _quantized(bnb, shape, quant_type="nf4"):
    torch.manual_seed(2026)
    values = torch.randn(shape, device="cuda", dtype=torch.float16)
    return bnb.functional.quantize_4bit(values, quant_type=quant_type,
                                        compress_statistics=True, blocksize=64)


def _assert_bits(actual, expected):
    assert actual.dtype == expected.dtype == torch.float16
    assert actual.shape == expected.shape
    # Floating torch.equal treats +0 and -0 as equal; integer views check bits.
    assert torch.equal(actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16))


@pytest.mark.parametrize("shape", [
    (1536, 1536), (8960, 1536), (1536, 8960), (256, 1536),
    (7, 13), (3, 65), (17, 127), (1, 1), (1, 3),
])
def test_nested_nf4_all_bits_match(native_bnb, shape):
    from solution.kernels.nf4 import dequantize_nf4

    bnb, _ = native_bnb
    packed, state = _quantized(bnb, shape)
    expected = bnb.functional.dequantize_4bit(packed, state)
    _assert_bits(dequantize_nf4(packed, state), expected)


def test_packed_transpose_matches_bnb_shim(native_bnb):
    from solution.kernels.nf4 import dequantize_nf4

    bnb, _ = native_bnb
    packed, state = _quantized(bnb, (256, 1536))
    transposed = packed.t()
    expected = bnb.functional.dequantize_4bit(transposed, state)
    assert expected.shape == (1536, 256)
    _assert_bits(dequantize_nf4(transposed, state), expected)


def test_zero_weight_sign_bits(native_bnb):
    from solution.kernels.nf4 import dequantize_nf4

    bnb, _ = native_bnb
    packed, state = bnb.functional.quantize_4bit(
        torch.zeros((17, 127), device="cuda", dtype=torch.float16),
        quant_type="nf4", blocksize=64, compress_statistics=True)
    _assert_bits(dequantize_nf4(packed, state), bnb.functional.dequantize_4bit(packed, state))


def test_linear_matches_forced_large_m_bnb_fallback(native_bnb):
    from solution.kernels.nf4 import dequantize_nf4

    bnb, _ = native_bnb
    packed, state = _quantized(bnb, (256, 1536))
    # bnb 0.50.2 CUDA uses its dequant+linear fallback for M > 1536.
    activations = torch.randn((1537, 1536), device="cuda", dtype=torch.float16)
    with torch.autocast(device_type="cuda", enabled=False):
        expected = bnb.matmul_4bit(activations, packed.t(), quant_state=state)
        actual = F.linear(activations, dequantize_nf4(packed, state))
    _assert_bits(actual, expected)


def test_patch_covers_functional_dispatcher_forward_and_backward(native_bnb, monkeypatch):
    from solution.nf4_patch import patch_bnb_dequant
    from solution.kernels import nf4

    bnb, backend = native_bnb
    packed, state = _quantized(bnb, (256, 64))
    expected_weight = bnb.functional.dequantize_4bit(packed, state)
    packed_fp4, fp4_state = _quantized(bnb, (17, 128), quant_type="fp4")
    expected_fp4 = bnb.functional.dequantize_4bit(packed_fp4, fp4_state)
    scales = bnb.functional.dequantize_blockwise(state.absmax, state.state2) + state.offset
    calls = []
    engine = nf4.nf4_decode_kernel

    class LaunchSpy:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(kwargs["NESTED"])
                return engine[grid](*args, **kwargs)
            return launch

    monkeypatch.setattr(nf4, "nf4_decode_kernel", LaunchSpy())
    assert patch_bnb_dequant()
    f, raw = bnb.functional.dequantize_4bit, backend._dequantize_4bit_impl
    assert patch_bnb_dequant()
    assert bnb.functional.dequantize_4bit is f and backend._dequantize_4bit_impl is raw
    _assert_bits(f(packed, state), expected_weight)
    assert calls == [True]
    out = torch.empty_like(expected_weight)
    returned = f(packed, state, out=out)
    assert returned.data_ptr() == out.data_ptr()
    _assert_bits(out, expected_weight)
    calls.clear()
    dispatched = torch.ops.bitsandbytes.dequantize_4bit.default(
        packed, scales, 64, "nf4", state.shape, torch.float16)
    _assert_bits(dispatched, expected_weight)
    assert calls == [False]
    calls.clear()
    x = torch.randn((1537, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    with torch.autocast(device_type="cuda", enabled=False):
        expected = F.linear(reference_x, expected_weight)
        actual = bnb.matmul_4bit(x, packed.t(), quant_state=state)
    _assert_bits(actual, expected)
    assert calls == [False]  # native GEMM fallback called the backend helper
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    _assert_bits(x.grad, reference_x.grad)
    assert calls == [False, True]  # MatMul4Bit.backward called functional
    calls.clear()
    _assert_bits(f(packed_fp4, fp4_state), expected_fp4)
    assert calls == []  # unsupported format stays on the saved originals


def test_unsupported_states_raise_before_launch(native_bnb):
    from solution.kernels.nf4 import dequantize_nf4

    bnb, _ = native_bnb
    packed, state = _quantized(bnb, (17, 128))
    with pytest.raises(ValueError, match="fp16"):
        dequantize_nf4(packed, state, torch.float32)
    for field, value in (("quant_type", "fp4"), ("blocksize", 128), ("nested", False)):
        changed = copy.copy(state)
        setattr(changed, field, value)
        with pytest.raises(ValueError):
            dequantize_nf4(packed, changed)
    changed = copy.copy(state)
    changed.state2 = copy.copy(state.state2)
    changed.state2.blocksize = 64
    with pytest.raises(ValueError, match="state2"):
        dequantize_nf4(packed, changed)


def test_print_cuda_event_speed_against_bnb(native_bnb):
    from solution.kernels.nf4 import dequantize_nf4

    bnb, _ = native_bnb
    packed, state = _quantized(bnb, (8960, 1536))
    native = lambda: bnb.functional.dequantize_4bit(packed, state)
    ours = lambda: dequantize_nf4(packed, state)
    _assert_bits(ours(), native())

    def milliseconds(function):
        for _ in range(3):
            function()
        start, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(20):
            function()
        finish.record()
        finish.synchronize()
        return start.elapsed_time(finish) / 20

    print(f"NF4 (8960x1536) ms: native={milliseconds(native):.6f}, ours={milliseconds(ours):.6f}")
