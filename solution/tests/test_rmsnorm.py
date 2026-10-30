"""Qwen2 RMSNorm forward/backward acceptance tests (run on CUDA/T4)."""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# Fixed tolerances; the mixed-dtype path includes fp16 rounding in backward.
FP32_TOL = dict(rtol=1e-5, atol=1e-5)
FP16_TOL = dict(rtol=2e-3, atol=2e-3)


def _assert_half_store_roundoff(actual, expected):
    """One adjacent half value on at most 1/1000 random entries, fixed a priori.

    Half-output and fp32-output kernels have different store types and can
    compile to different arithmetic instructions. A last-bit fp32 product/
    rsqrt/reduction difference at a half midpoint can select the neighboring
    half value. This sparse-midpoint criterion is separate from the strict
    controlled-row cast test below; it does not allow a dense half-ULP bias.
    """
    assert actual.dtype == expected.dtype == torch.float16
    a, e = actual.detach(), expected.detach()
    assert torch.isfinite(a).all() and torch.isfinite(e).all()
    above = torch.nextafter(e, torch.full_like(e, float("inf"))).float()
    below = torch.nextafter(e, torch.full_like(e, -float("inf"))).float()
    upper, lower = (above - e.float()).abs(), (e.float() - below).abs()
    upper = torch.where(torch.isfinite(upper), upper, lower)
    lower = torch.where(torch.isfinite(lower), lower, upper)
    ulp = torch.maximum(upper, lower)
    error = (a.float() - e.float()).abs()
    assert (error <= ulp).all(), f"half-store error exceeds one ULP: max_abs={error.max().item()}"
    changed = int((a != e).sum())
    assert changed * 1000 <= a.numel(), f"half-store rounding changed {changed}/{a.numel()} (>1/1000)"


def _qwen2_reference(x, weight, eps):
    input_dtype = x.dtype
    values = x.to(torch.float32)
    variance = values.pow(2).mean(-1, keepdim=True)
    values = values * torch.rsqrt(variance + eps)
    return weight * values.to(input_dtype)


def _check(x, weight, eps=1e-6, grad=None):
    from solution.kernels import rms_norm

    xr = x.detach().clone().requires_grad_(x.requires_grad)
    wr = weight.detach().clone().requires_grad_(weight.requires_grad)
    expected = _qwen2_reference(xr, wr, eps)
    actual = rms_norm(x, weight, eps)
    tol = FP32_TOL if x.dtype == weight.dtype == torch.float32 else FP16_TOL
    assert actual.shape == x.shape
    assert actual.dtype == torch.promote_types(x.dtype, weight.dtype)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, **tol)
    if grad is None:
        grad = torch.randn_like(actual)
    expected.backward(grad)
    actual.backward(grad)
    if x.requires_grad:
        assert x.grad.dtype == x.dtype
        assert torch.isfinite(x.grad).all()
        torch.testing.assert_close(x.grad, xr.grad, **tol)
    if weight.requires_grad:
        assert weight.grad.dtype == weight.dtype
        assert torch.isfinite(weight.grad).all()
        torch.testing.assert_close(weight.grad, wr.grad, **tol)
    else:
        assert weight.grad is None
    return actual


@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1536), (2, 5, 1000)])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("train_weight", [False, True])
def test_matches_qwen2_forward_and_gradients(shape, input_dtype, weight_dtype, train_weight):
    torch.manual_seed(456)
    x = torch.randn(shape, device="cuda", dtype=input_dtype, requires_grad=True)
    weight = torch.randn(shape[-1], device="cuda", dtype=weight_dtype, requires_grad=train_weight)
    _check(x, weight)


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
def test_large_values_stay_finite(input_dtype, weight_dtype):
    torch.manual_seed(789)
    x = (torch.randn((3, 77, 1536), device="cuda", dtype=input_dtype) * 1000).requires_grad_()
    weight = torch.randn(1536, device="cuda", dtype=weight_dtype, requires_grad=True)
    _check(x, weight, eps=1e-5)


def test_repeated_calls_accumulate_gradients():
    from solution.kernels import rms_norm

    torch.manual_seed(123)
    x = torch.randn((2, 5, 1000), device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(1000, device="cuda", dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    wr = weight.detach().clone().requires_grad_()
    first, second = rms_norm(x, weight, 1e-6), rms_norm(x, weight, 1e-6)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    grad_first, grad_second = torch.randn_like(first), torch.randn_like(first)
    first.backward(grad_first)
    second.backward(grad_second)
    _qwen2_reference(xr, wr, 1e-6).backward(grad_first + grad_second)
    torch.testing.assert_close(x.grad, xr.grad, **FP32_TOL)
    torch.testing.assert_close(weight.grad, wr.grad, **FP32_TOL)


@pytest.mark.parametrize("shape", [(1536,), (2, 1, 3, 1000), (0, 3, 1000)])
def test_arbitrary_leading_shapes(shape):
    torch.manual_seed(234)
    x = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32, requires_grad=True)
    _check(x, weight)


def test_runtime_strided_2d_rows_without_input_copy():
    torch.manual_seed(321)
    x = torch.randn((12, 1000), device="cuda")[::2].detach().requires_grad_()
    weight = torch.randn(2000, device="cuda")[::2].detach().requires_grad_()
    assert x.stride(-1) == 1 and not x.is_contiguous()
    assert weight.stride(0) == 2
    _check(x, weight)


def test_changing_sequence_length_reuses_compiled_variants():
    from solution.kernels import rms_norm
    from solution.kernels.rmsnorm import (
        rmsnorm_forward_kernel,
        rmsnorm_input_gradient_kernel,
        rmsnorm_weight_partials_kernel,
        rmsnorm_weight_finalize_kernel,
    )

    # Verified against Triton v3.6.0 runtime/jit.py: create_binder returns
    # (kernel_cache, key_cache, target, backend, binder), stored per device.
    # https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/runtime/jit.py
    device = torch.cuda.current_device()
    kernels = (rmsnorm_forward_kernel, rmsnorm_input_gradient_kernel,
               rmsnorm_weight_partials_kernel, rmsnorm_weight_finalize_kernel)

    def variant_count(kernel):
        entry = kernel.device_caches.get(device)
        return len(entry[0]) if entry is not None else 0

    before = {kernel.__name__: variant_count(kernel) for kernel in kernels}
    weight = torch.ones(1536, device="cuda", requires_grad=True)
    for sequence in (17, 32, 77, 128, 257):
        x = torch.randn((4, sequence, 1536), device="cuda", requires_grad=True)
        # Random dense upstream keeps its stride specialization the same.
        out = rms_norm(x, weight, 2e-5)
        out.backward(torch.randn_like(out))
        weight.grad = None
    torch.cuda.synchronize()
    for kernel in kernels:
        growth = variant_count(kernel) - before[kernel.__name__]
        # Runtime integer alignment may produce divisible-by-16 and generic variants.
        assert growth <= 2, f"{kernel.__name__} compiled {growth} variants for five S values"


def test_broadcast_upstream_and_weight_gradient_only():
    torch.manual_seed(345)
    x = torch.randn((2, 5, 1000), device="cuda")
    weight = torch.randn(1000, device="cuda", requires_grad=True)
    grad = torch.tensor(0.25, device="cuda").expand_as(x)
    assert grad.stride(-1) == 0
    _check(x, weight, grad=grad)


def test_zero_rows_and_zero_input():
    x = torch.zeros((2, 5, 1000), device="cuda", requires_grad=True)
    weight = torch.ones(1000, device="cuda", requires_grad=True)
    _check(x, weight)


def test_half_cast_occurs_before_weight_multiply():
    from solution.kernels import rms_norm

    # Small fixed rows keep the half-rounded normalization away from midpoints,
    # so a different fp32 reduction order cannot change this cast-order check.
    x = torch.tensor([[1, 1, 2, 2], [3, 3, 4, 4]], device="cuda", dtype=torch.float16)
    weight = torch.tensor([0.1, 0.9, 1.7, 3.0], device="cuda", dtype=torch.float32)
    actual = rms_norm(x, weight, 1e-6)
    expected = _qwen2_reference(x, weight, 1e-6)
    values = x.float()
    other_order = weight * (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6))
    assert actual.dtype == torch.float32
    assert (expected != other_order).any()
    # fp32 output contains the exact product of rounded fp16 q and fp32 weight.
    torch.testing.assert_close(actual, expected, **FP32_TOL)


def test_invalid_layout_and_cpu_raise_value_error():
    from solution.kernels import rms_norm

    with pytest.raises(ValueError, match="CUDA"):
        rms_norm(torch.ones(2, 5), torch.ones(5), 1e-6)
    x = torch.randn((2, 5, 1000), device="cuda")[..., ::2]
    with pytest.raises(ValueError, match="contiguous"):
        rms_norm(x, torch.ones(500, device="cuda"), 1e-6)
    leading = torch.randn((3, 4, 1000), device="cuda").transpose(0, 1)
    with pytest.raises(ValueError, match="2-D"):
        rms_norm(leading, torch.ones(1000, device="cuda"), 1e-6)


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("train_weight", [False, True])
def test_fused_output_half_rounding_and_gradients(input_dtype, weight_dtype, train_weight):
    from solution.kernels.rmsnorm import rms_norm

    torch.manual_seed(692)
    x = torch.randn(2, 17, 1536, device="cuda", dtype=input_dtype, requires_grad=True)
    weight = torch.randn(1536, device="cuda", dtype=weight_dtype, requires_grad=train_weight)
    xr = x.detach().clone().requires_grad_()
    wr = weight.detach().clone().requires_grad_(train_weight)
    actual = rms_norm(x, weight, 1e-6, out_dtype=torch.float16)
    # Changing the store dtype may change final code generation near a half
    # midpoint; constrain both ULP distance and frequency on random inputs.
    exact = rms_norm(x.detach(), weight.detach(), 1e-6)
    _assert_half_store_roundoff(actual, exact.half())
    reference = _qwen2_reference(xr, wr, 1e-6).half()
    _assert_half_store_roundoff(actual, reference)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    reference.backward(grad)
    assert x.grad.dtype == input_dtype
    torch.testing.assert_close(x.grad, xr.grad, **FP16_TOL)
    if train_weight:
        torch.testing.assert_close(weight.grad, wr.grad, **FP16_TOL)
    else:
        assert weight.grad is None


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16])
def test_fused_half_output_bitwise_qwen_reference(input_dtype):
    from solution.kernels.rmsnorm import rms_norm

    # Exact row variances and half-representable gamma keep this cast check
    # independent of fp32 reduction-tree differences near half midpoints.
    x = torch.tensor([[1, -1, 1, -1] * 20, [2, -2, 2, -2] * 20],
                     device="cuda", dtype=input_dtype)
    weight = torch.tensor([0.5, 1, 1.5, 2] * 20, device="cuda", dtype=torch.float32)
    actual = rms_norm(x, weight, 1e-6, out_dtype=torch.float16)
    own_reference = rms_norm(x, weight, 1e-6).half()
    eager_reference = _qwen2_reference(x, weight, 1e-6).half()
    assert torch.equal(actual.view(torch.int16), own_reference.view(torch.int16))
    assert torch.equal(actual.view(torch.int16), eager_reference.view(torch.int16))


def test_half_output_sequence_lengths_reuse_variants():
    from solution.kernels.rmsnorm import (rms_norm, rmsnorm_forward_kernel,
                                        rmsnorm_input_gradient_kernel, rmsnorm_weight_partials_kernel,
                                        rmsnorm_weight_finalize_kernel)

    device = torch.cuda.current_device()
    kernels = (rmsnorm_forward_kernel, rmsnorm_input_gradient_kernel,
               rmsnorm_weight_partials_kernel, rmsnorm_weight_finalize_kernel)
    def count(kernel):
        entry = kernel.device_caches.get(device)
        return len(entry[0]) if entry is not None else 0
    before = {kernel.__name__: count(kernel) for kernel in kernels}
    weight = torch.ones(1536, device="cuda", requires_grad=True)
    for sequence in (17, 32, 77, 128, 257):
        x = torch.randn(4, sequence, 1536, device="cuda", requires_grad=True)
        out = rms_norm(x, weight, 3e-5, out_dtype=torch.float16)
        out.backward(torch.randn_like(out))
        weight.grad = None
    torch.cuda.synchronize()
    for kernel in kernels:
        assert count(kernel) - before[kernel.__name__] <= 2, kernel.__name__
