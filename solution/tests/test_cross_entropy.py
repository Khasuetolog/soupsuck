"""CUDA acceptance tests; AMP/fp32 references are row-chunked to bound memory.

Run: python -m pytest solution/tests/test_cross_entropy.py -q
Real-shape tests are intentionally enabled by default, including memory checking.
"""

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# Fixed before GPU execution; never adapt tolerances to observed errors.
METRIC_RTOL = 2e-3
METRIC_ATOL = 2e-3
GRAD_RTOL = 1e-2
GRAD_ATOL = 2e-3
ARGMAX_AGREEMENT = 0.999


@pytest.fixture(autouse=True)
def fp32_reference_without_tf32():
    if not torch.cuda.is_available():
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    previous_reduction = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous_reduction


def _inputs(n, h, v, weight_dtype, ignored="some", scale=1.0, ignore_index=-100,
            hidden_dtype=torch.float16):
    torch.manual_seed(123)
    x = (torch.randn(n, h, device="cuda", dtype=hidden_dtype) * scale).requires_grad_()
    w = torch.randn(v, h, device="cuda", dtype=weight_dtype) / math.sqrt(h)
    y = torch.randint(v, (n,), device="cuda")
    if ignored == "some":
        y[::3] = ignore_index
    elif ignored == "all":
        y.fill_(ignore_index)
    return x, w, y


def _reference(x, w, y, upstream, ignore_index=-100, pure_fp32=False):
    """Baseline AMP (fp16 logits then fp32 CE); optional pure fp32 comparison."""
    losses, entropies, indices, normalizers, gradients = [], [], [], [], []
    with torch.autocast(device_type="cuda", enabled=False):
        # Each graph is consumed immediately, not retained for the next block.
        for start in range(0, x.shape[0], 16):
            xr = x[start:start + 16].detach().float().requires_grad_()
            if pure_fp32:
                logits = xr @ w.float().t()
            else:
                logits = F.linear(xr.to(torch.float16), w.to(torch.float16)).float()
            logp = F.log_softmax(logits, dim=-1)
            loss = F.cross_entropy(logits, y[start:start + 16], reduction="none",
                                   ignore_index=ignore_index)
            entropy = -(logp.exp() * logp).sum(-1)
            losses.append(loss.detach())
            entropies.append(entropy.detach())
            indices.append(logits.detach().argmax(-1))
            normalizers.append(logits.detach().logsumexp(-1))
            loss.backward(upstream[start:start + 16].float())
            gradients.append(xr.grad.detach())
            del xr, logits, logp, loss, entropy
    return tuple(torch.cat(parts) for parts in
                 (losses, entropies, indices, normalizers, gradients))


def _compare(x, w, y, upstream, ignore_index=-100, pure_fp32=False):
    from solution.kernels import fused_linear_cross_entropy

    ref_loss, ref_entropy, ref_argmax, ref_lse, ref_dx = _reference(
        x, w, y, upstream, ignore_index, pure_fp32)
    loss, entropy, argmax, lse = fused_linear_cross_entropy(x, w, y, ignore_index)
    assert loss.dtype == entropy.dtype == lse.dtype == torch.float32
    assert argmax.dtype == torch.int64
    assert loss.shape == entropy.shape == argmax.shape == lse.shape == y.shape
    assert loss.requires_grad
    assert not entropy.requires_grad and not argmax.requires_grad and not lse.requires_grad
    for actual, expected in ((loss, ref_loss), (entropy, ref_entropy), (lse, ref_lse)):
        torch.testing.assert_close(actual, expected, rtol=METRIC_RTOL, atol=METRIC_ATOL)
        assert torch.isfinite(actual).all()
    assert (argmax == ref_argmax).float().mean().item() >= ARGMAX_AGREEMENT
    assert torch.count_nonzero(loss[y == ignore_index]).item() == 0
    loss.backward(upstream)
    assert x.grad.dtype == x.dtype
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    # Norm check prevents the absolute tolerance from hiding mean-loss underflow.
    error = torch.linalg.vector_norm(x.grad.float() - ref_dx)
    norm = torch.linalg.vector_norm(ref_dx)
    assert error.item() <= GRAD_RTOL * norm.item() + 1e-7
    assert torch.count_nonzero(x.grad[y == ignore_index]).item() == 0
    assert w.grad is None


@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("hidden_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("n,h,v,ignored,scale", [
    (77, 64, 1000, "some", 1.0),
    (1001, 96, 1000, "some", 1.0),
    (77, 64, 1000, "all", 1.0),
    (77, 64, 1000, "none", 20.0),
    (77, 80, 4103, "some", 1.0),  # vocabulary tail and a second chunk
    (1, 33, 7, "none", 1.0),
])
def test_forward_and_arbitrary_upstream(n, h, v, ignored, scale, weight_dtype, hidden_dtype):
    x, w, y = _inputs(n, h, v, weight_dtype, ignored, scale, hidden_dtype=hidden_dtype)
    upstream = torch.randn(n, device="cuda", dtype=torch.float32)
    _compare(x, w, y, upstream)


@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
def test_real_vocabulary_and_mean_sized_upstream(weight_dtype):
    x, w, y = _inputs(2048, 1536, 151936, weight_dtype, hidden_dtype=torch.float32)
    upstream = torch.randn(2048, device="cuda", dtype=torch.float32) / 2048
    _compare(x, w, y, upstream)


def test_separate_pure_fp32_reference():
    x, w, y = _inputs(77, 64, 1000, torch.float32, hidden_dtype=torch.float32)
    _compare(x, w, y, torch.randn(77, device="cuda"), pure_fp32=True)


def test_grad_scaler_sized_signed_upstream():
    x, w, y = _inputs(77, 64, 4103, torch.float32, hidden_dtype=torch.float32)
    upstream = torch.linspace(-1, 1, 77, device="cuda") * (65536.0 / 77)
    _compare(x, w, y, upstream)
    assert torch.isfinite(x.grad).all()


def test_upstream_exceeding_fp16_range_has_finite_fp32_gradient():
    from solution.kernels import fused_linear_cross_entropy

    x = torch.ones((1, 2), device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.tensor([[0.001, 0.001], [-0.001, -0.001]], device="cuda")
    y = torch.tensor([0], device="cuda")
    upstream = torch.tensor([1e8], device="cuda")
    # Analytic fp32 gradient of rounded logits: direct fp16 dZ would overflow.
    logits = F.linear(x.detach().half(), w.half()).float()
    dz = logits.softmax(-1) - F.one_hot(y, num_classes=2).float()
    expected = (dz * upstream[:, None]) @ w.half().float()
    fused_linear_cross_entropy(x, w, y)[0].backward(upstream)
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad, expected, rtol=GRAD_RTOL, atol=GRAD_ATOL)


def test_logit_rounding_precedes_argmax_and_statistics():
    from solution.kernels import fused_linear_cross_entropy

    x = torch.ones((1, 2), device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.tensor([[1.0, 0.0001], [1.0, 0.0002]], device="cuda")
    y = torch.tensor([1], device="cuda")
    pure_logits = x.detach() @ w.t()
    assert pure_logits.argmax(-1).item() == 1
    loss, entropy, argmax, lse = fused_linear_cross_entropy(x, w, y)
    assert argmax.item() == 0  # both logits round to 1.0 in fp16
    torch.testing.assert_close(loss, torch.full_like(loss, math.log(2)), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(entropy, loss.detach(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lse, loss.detach() + 1, rtol=1e-6, atol=1e-6)


def test_repeat_calls_and_first_argmax_on_ties():
    from solution.kernels import fused_linear_cross_entropy

    x, w, y = _inputs(77, 64, 4103, torch.float32)
    w.zero_()
    first = fused_linear_cross_entropy(x, w, y)
    second = fused_linear_cross_entropy(x, w, y)
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.count_nonzero(first[2]).item() == 0
    torch.testing.assert_close(first[1], torch.full_like(first[1], math.log(w.shape[0])),
                               rtol=METRIC_RTOL, atol=METRIC_ATOL)
    (first[0].sum() + second[0].sum()).backward()
    assert torch.count_nonzero(x.grad).item() == 0


def test_noncontiguous_inputs_and_custom_ignore_index():
    x, w, y = _inputs(77, 64, 1000, torch.float32, ignore_index=-7)
    x = x.detach().t().contiguous().t().requires_grad_()
    w = w.t().contiguous().t()
    y = y.repeat_interleave(2)[::2]
    _compare(x, w, y, torch.randn(77, device="cuda"), ignore_index=-7)


def test_trainable_weight_is_rejected():
    from solution.kernels import fused_linear_cross_entropy

    x, w, y = _inputs(2, 32, 1000, torch.float32)
    w.requires_grad_()
    with pytest.raises(ValueError, match="frozen"):
        fused_linear_cross_entropy(x, w, y)


def test_empty_rows():
    from solution.kernels import fused_linear_cross_entropy

    x, w, y = _inputs(0, 32, 1000, torch.float32)
    outputs = fused_linear_cross_entropy(x, w, y)
    assert all(out.shape == (0,) for out in outputs)
    outputs[0].sum().backward()
    assert x.grad.shape == x.shape


def test_peak_memory_real_shape():
    from solution.kernels import fused_linear_cross_entropy

    # Warm cuBLAS/Triton before recording allocations; input weight is excluded.
    wx, ww, wy = _inputs(2, 1536, 4096, torch.float32)
    fused_linear_cross_entropy(wx, ww, wy)[0].sum().backward()
    del wx, ww, wy
    x, w, y = _inputs(2048, 1536, 151936, torch.float32, hidden_dtype=torch.float32)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    loss, entropy, argmax, lse = fused_linear_cross_entropy(x, w, y)
    loss.mean().backward()
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - baseline
    budget = x.shape[0] * w.shape[0] * 2 // 4
    assert peak_delta <= budget, f"temporary peak {peak_delta:,} exceeds {budget:,} bytes"
    assert entropy.numel() == argmax.numel() == lse.numel() == x.shape[0]
