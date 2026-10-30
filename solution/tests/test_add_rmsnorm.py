"""Residual/norm fusion against eager Qwen2, including both autograd edges."""

import inspect

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

FP32_TOL = dict(rtol=1e-5, atol=1e-5)
FP16_TOL = dict(rtol=2e-3, atol=2e-3)
# Two fp32 reduction trees over at most 2048 tested rows have about 22
# addition levels in total; 32 also budgets product/normalization rounding.
# The standard absolute summation bound scales with sum(abs(terms)), not
# abs(sum(terms)), which can be arbitrarily small through cancellation.
DGAMMA_REDUCTION_FACTOR = 32


def _norm_reference(x, weight, eps):
    values = x.float()
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return weight * values.to(x.dtype)


def _reference(residual, delta, weight, eps, out_dtype):
    new = residual + delta
    normed = _norm_reference(new, weight, eps)
    return new, normed.to(out_dtype) if out_dtype is not None else normed


def _assert_dgamma(actual, expected, residual, delta, upstream, eps, tolerance):
    if tolerance is not FP32_TOL:
        torch.testing.assert_close(actual, expected, **tolerance)
        return
    with torch.no_grad():
        new = residual + delta
        values = new.float()
        q = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)).to(new.dtype)
        magnitude = (upstream * q).abs().reshape(-1, q.shape[-1]).sum(0)
        roundoff = DGAMMA_REDUCTION_FACTOR * torch.finfo(torch.float32).eps * magnitude
        error = (actual.float() - expected.float()).abs()
        budget = FP32_TOL["rtol"] * expected.float().abs() + roundoff
        assert (error <= budget).all(), (
            f"dgamma exceeds magnitude-scaled fp32 reduction budget by "
            f"{(error - budget).max().item():.8e}")


def _check(shape, delta_dtype, out_dtype, train_weight, residual_dtype=torch.float32,
           weight_dtype=torch.float32):
    from solution.kernels import add_rms_norm

    torch.manual_seed(101)
    residual = torch.randn(shape, device="cuda", dtype=residual_dtype, requires_grad=True)
    delta = torch.randn(shape, device="cuda", dtype=delta_dtype, requires_grad=True)
    weight = torch.randn(shape[-1], device="cuda", dtype=weight_dtype, requires_grad=train_weight)
    rr = residual.detach().clone().requires_grad_()
    dr = delta.detach().clone().requires_grad_()
    wr = weight.detach().clone().requires_grad_(train_weight)
    reference = _reference(rr, dr, wr, 1e-6, out_dtype)
    actual = add_rms_norm(residual, delta, weight, 1e-6, out_dtype)
    assert actual[0].dtype == torch.promote_types(residual_dtype, delta_dtype)
    assert actual[1].dtype == (out_dtype or torch.promote_types(actual[0].dtype, weight_dtype))
    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
    norm_tol = (FP32_TOL if actual[0].dtype == weight_dtype == actual[1].dtype == torch.float32
                else FP16_TOL)
    torch.testing.assert_close(actual[1], reference[1], **norm_tol)
    upstream = tuple(torch.randn_like(t) for t in actual)
    torch.autograd.backward(reference, upstream)
    torch.autograd.backward(actual, upstream)
    for value, ref in ((residual, rr), (delta, dr)):
        assert value.grad.dtype == value.dtype
        assert torch.isfinite(value.grad).all()
        tol = FP32_TOL if value.dtype == torch.float32 and norm_tol is FP32_TOL else FP16_TOL
        torch.testing.assert_close(value.grad, ref.grad, **tol)
    if train_weight:
        assert weight.grad.dtype == weight_dtype
        _assert_dgamma(weight.grad, wr.grad, rr, dr, upstream[1], 1e-6, norm_tol)
    else:
        assert weight.grad is None


@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1536), (2, 5, 1000)])
@pytest.mark.parametrize("delta_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("out_dtype", [None, torch.float16])
@pytest.mark.parametrize("train_weight", [False, True])
def test_both_outputs_and_gradients(shape, delta_dtype, out_dtype, train_weight):
    _check(shape, delta_dtype, out_dtype, train_weight)


@pytest.mark.parametrize("residual_dtype,delta_dtype,weight_dtype", [
    (torch.float16, torch.float16, torch.float16),
    (torch.float16, torch.float16, torch.float32),
    (torch.float16, torch.float32, torch.float16),
    (torch.float32, torch.float16, torch.float16),
])
@pytest.mark.parametrize("out_dtype", [None, torch.float16])
def test_dtype_promotion_and_cast_order(residual_dtype, delta_dtype, weight_dtype, out_dtype):
    _check((2, 5, 1000), delta_dtype, out_dtype, True, residual_dtype, weight_dtype)


@pytest.mark.parametrize("used_output", [0, 1])
def test_one_output_has_no_upstream(used_output):
    from solution.kernels import add_rms_norm

    torch.manual_seed(202)
    residual = torch.randn((2, 5, 1000), device="cuda", requires_grad=True)
    delta = torch.randn_like(residual, dtype=torch.float16, requires_grad=True)
    weight = torch.ones(1000, device="cuda", requires_grad=True)
    rr = residual.detach().clone().requires_grad_()
    dr = delta.detach().clone().requires_grad_()
    wr = weight.detach().clone().requires_grad_()
    actual = add_rms_norm(residual, delta, weight, 1e-6)
    reference = _reference(rr, dr, wr, 1e-6, None)
    upstream = torch.randn_like(actual[used_output])
    actual[used_output].backward(upstream)
    reference[used_output].backward(upstream)
    torch.testing.assert_close(residual.grad, rr.grad, **FP32_TOL)
    torch.testing.assert_close(delta.grad, dr.grad, **FP16_TOL)
    if used_output == 0:
        assert torch.count_nonzero(weight.grad).item() == 0
    else:
        _assert_dgamma(weight.grad, wr.grad, rr, dr, upstream, 1e-6, FP32_TOL)


def test_batch_lengths_reuse_compiled_variants():
    from solution.kernels import add_rms_norm
    from solution.kernels.add_rmsnorm import add_rmsnorm_forward_kernel, add_rmsnorm_backward_kernel

    device = torch.cuda.current_device()
    kernels = (add_rmsnorm_forward_kernel, add_rmsnorm_backward_kernel)

    def count(kernel):
        entry = kernel.device_caches.get(device)  # Triton 3.6.0 runtime/jit.py
        return len(entry[0]) if entry is not None else 0

    before = [count(kernel) for kernel in kernels]
    weight = torch.ones(1536, device="cuda")
    for sequence in (17, 32, 77, 128, 257):
        residual = torch.randn((4, sequence, 1536), device="cuda", requires_grad=True)
        delta = torch.randn_like(residual, dtype=torch.float16, requires_grad=True)
        outputs = add_rms_norm(residual, delta, weight, 2e-5)
        torch.autograd.backward(outputs, tuple(torch.randn_like(t) for t in outputs))
    torch.cuda.synchronize()
    for kernel, old in zip(kernels, before):
        assert count(kernel) - old <= 2


def test_invalid_layout_and_output_dtype():
    from solution.kernels import add_rms_norm

    residual = torch.ones((2, 5, 1000), device="cuda")
    delta, weight = torch.ones_like(residual), torch.ones(1000, device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        add_rms_norm(residual.transpose(0, 1), delta.transpose(0, 1), weight, 1e-6)
    with pytest.raises(ValueError, match="out_dtype"):
        add_rms_norm(residual, delta, weight, 1e-6, torch.float32)


@pytest.mark.parametrize("half_output", [False, True])
def test_decoder_patch_signature_kwargs_and_idempotency(monkeypatch, half_output):
    pytest.importorskip("transformers")
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    from solution.decoder import patch_qwen2_decoder

    class Attention(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            self.seen = kwargs
            return (hidden_states * 0.125).half(), None

    class MLP(torch.nn.Module):
        def forward(self, hidden):
            self.seen_dtype = hidden.dtype
            return (hidden * 0.25).half()

    monkeypatch.delenv("SOUP_FUSED_DISABLE", raising=False)
    monkeypatch.setenv("SOUP_FUSED_NORM_FP16", "1" if half_output else "0")
    # Restore class state after this test; the integration module owns installation.
    monkeypatch.setattr(Qwen2DecoderLayer, "forward", Qwen2DecoderLayer.forward)
    original = getattr(Qwen2DecoderLayer.forward, "_solution_fused_original", Qwen2DecoderLayer.forward)
    config = Qwen2Config(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=1)
    layer = Qwen2DecoderLayer(config, layer_idx=0).cuda()
    layer.self_attn, layer.mlp = Attention(), MLP()
    layer.input_layernorm.weight.requires_grad_(False)
    layer.post_attention_layernorm.weight.requires_grad_(False)
    assert patch_qwen2_decoder()
    patched = Qwen2DecoderLayer.forward
    assert patch_qwen2_decoder() and Qwen2DecoderLayer.forward is patched
    assert inspect.signature(patched) == inspect.signature(original)
    hidden = torch.randn((2, 5, 32), device="cuda", requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_()
    delta = (_norm_reference(ref_hidden, layer.input_layernorm.weight,
                             layer.input_layernorm.variance_epsilon) * 0.125).half()
    residual = ref_hidden + delta
    normed = _norm_reference(residual, layer.post_attention_layernorm.weight,
                             layer.post_attention_layernorm.variance_epsilon)
    # The substituted MLP is arbitrary and has no certified fp16 projection
    # consumers. The env switch is a request, not permission to narrow it.
    expected = residual + (normed * 0.25).half()
    arguments = dict(attention_mask=torch.ones((2, 5), device="cuda"),
                     position_ids=torch.arange(5, device="cuda")[None, :],
                     past_key_values=object(), use_cache=True,
                     position_embeddings=(torch.ones(1, device="cuda"), torch.zeros(1, device="cuda")),
                     custom_argument=object())
    actual = layer(hidden, **arguments)
    for key, value in arguments.items():
        assert layer.self_attn.seen[key] is value
    assert layer.mlp.seen_dtype == torch.float32
    torch.testing.assert_close(actual, expected, **FP16_TOL)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, **FP16_TOL)
