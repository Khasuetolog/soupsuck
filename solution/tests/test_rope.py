"""Joint RoPE: strict eager forward bits, q/k VJP, strides and JIT reuse."""

from types import ModuleType

import pytest


@pytest.fixture(autouse=True)
def explicit_legacy_environment(monkeypatch):
    monkeypatch.setenv("SOUP_EXP_PRECOMPILE", "0")
    monkeypatch.setenv("SOUP_EXP_LAUNCH_TRIM", "0")


@pytest.fixture
def cuda_runtime():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("triton")
    return torch


def test_patch_module_global_lookup_idempotence_and_no_hub_fallback(monkeypatch):
    import solution.rope as patch

    module = ModuleType("fake_qwen_rope")
    exec("def apply_rotary_pos_emb(*args, **kwargs):\n    raise AssertionError('hub must not run')\n"
         "def attention(q, k, c, s):\n    return apply_rotary_pos_emb(q, k, c, s)\n", module.__dict__)
    original = module.apply_rotary_pos_emb
    monkeypatch.setattr(patch, "_supported", lambda *args: False)
    monkeypatch.setattr(patch, "_eager_rotary", lambda *args: ("eager-q", "eager-k"))
    assert patch.patch_qwen2_rope(module)
    installed = module.apply_rotary_pos_emb
    assert patch.patch_qwen2_rope(module) and module.apply_rotary_pos_emb is installed
    assert getattr(installed, patch._MARKER) is original
    assert module.attention(None, None, None, None) == ("eager-q", "eager-k")


def _reference(torch, q, k, cos, sin, unsqueeze_dim=1):
    c, s = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    def rotate(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)
    return q * c + rotate(q) * s, k * c + rotate(k) * s


def _bits(torch, tensor):
    return tensor.detach().contiguous().view(torch.int16 if tensor.dtype == torch.float16 else torch.int32)


@pytest.mark.parametrize("shape", [(4, 12, 2, 512, 128, 1, True),
                                    (3, 5, 2, 17, 66, 3, True),
                                    (1, 3, 1, 7, 10, 1, False)])
@pytest.mark.parametrize("dtypes", [("float16", "float16", "float16"),
                                     ("float16", "float16", "float32"),
                                     ("float32", "float32", "float32"),
                                     ("float32", "float16", "float16")])
def test_forward_bits_and_qk_gradients(cuda_runtime, shape, dtypes):
    torch = cuda_runtime
    from solution.kernels.rope import fused_rope

    torch.manual_seed(347)
    batch, hq, hk, sequence, dim, cbatch, transposed = shape
    qtype, ktype, ctype = (getattr(torch, name) for name in dtypes)
    def tensor(heads, dtype):
        if transposed:
            return torch.randn(batch, sequence, heads, dim, device="cuda", dtype=dtype).transpose(1, 2).detach().requires_grad_()
        return torch.randn(batch, heads, sequence, dim, device="cuda", dtype=dtype, requires_grad=True)
    q, k = tensor(hq, qtype), tensor(hk, ktype)
    qr, kr = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
    # Independent halves exercise the actual transpose derivative, without
    # relying on Qwen's repeated cos/sin halves.
    cos = torch.randn(cbatch, sequence, dim, device="cuda", dtype=ctype)
    sin = torch.randn_like(cos)
    expected = _reference(torch, qr, kr, cos, sin)
    actual = fused_rope(q, k, cos, sin)
    gradients = [torch.randn_like(out) for out in actual]
    for got, ref in zip(actual, expected):
        assert got.dtype == ref.dtype
        assert torch.equal(_bits(torch, got), _bits(torch, ref))
    torch.autograd.backward(actual, gradients)
    torch.autograd.backward(expected, gradients)
    for got, ref in ((q.grad, qr.grad), (k.grad, kr.grad)):
        tolerance = dict(rtol=2e-3, atol=2e-3) if got.dtype == torch.float16 else dict(rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(got, ref, **tolerance)


def test_broadcast_and_strided_cos_upstream(cuda_runtime):
    torch = cuda_runtime
    from solution.kernels.rope import fused_rope

    q = torch.randn(2, 3, 13, 128, device="cuda", requires_grad=True)
    k = torch.randn(2, 1, 13, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    qr, kr = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
    cos = torch.randn(1, 26, 128, device="cuda")[:, ::2]
    sin = torch.randn_like(cos).expand(2, -1, -1)
    cos = cos.expand(2, -1, -1)
    expected, actual = _reference(torch, qr, kr, cos, sin), fused_rope(q, k, cos, sin)
    for got, ref in zip(actual, expected):
        assert torch.equal(_bits(torch, got), _bits(torch, ref))
    gradients = [torch.tensor(0.25, device="cuda").expand_as(out) for out in actual]
    torch.autograd.backward(actual, gradients)
    torch.autograd.backward(expected, gradients)
    torch.testing.assert_close(q.grad, qr.grad, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(k.grad, kr.grad, rtol=2e-3, atol=2e-3)


def test_optional_gradient_and_empty_sequence(cuda_runtime):
    torch = cuda_runtime
    from solution.kernels.rope import fused_rope

    for sequence in (0, 7):
        q = torch.randn(2, 3, sequence, 128, device="cuda")
        k = torch.randn(2, 1, sequence, 128, device="cuda", requires_grad=True)
        cos, sin = torch.ones(1, sequence, 128, device="cuda"), torch.zeros(1, sequence, 128, device="cuda")
        qo, ko = fused_rope(q, k, cos, sin)
        assert not qo.requires_grad and ko.requires_grad
        ko.sum().backward()
        torch.testing.assert_close(k.grad, torch.ones_like(k), rtol=0, atol=0)


def test_sequence_length_does_not_recompile(cuda_runtime):
    torch = cuda_runtime
    from solution.kernels.rope import fused_rope, rope_pair_forward_kernel, rope_pair_backward_kernel

    device = torch.cuda.current_device()
    kernels = (rope_pair_forward_kernel, rope_pair_backward_kernel)
    def count(kernel):
        entry = kernel.device_caches.get(device)
        return len(entry[0]) if entry is not None else 0
    warmed = None
    for index, sequence in enumerate((17, 32, 77, 128, 257)):
        q = torch.randn(2, sequence, 3, 128, device="cuda", dtype=torch.float16).transpose(1, 2).detach().requires_grad_()
        k = torch.randn(2, sequence, 1, 128, device="cuda", dtype=torch.float16).transpose(1, 2).detach().requires_grad_()
        cos = torch.randn(1, sequence, 128, device="cuda")
        sin = torch.randn_like(cos)
        outputs = fused_rope(q, k, cos, sin)
        torch.autograd.backward(outputs, [torch.randn_like(out) for out in outputs])
        torch.cuda.synchronize()
        if index == 1:
            warmed = tuple(count(kernel) for kernel in kernels)
        if index > 1:
            assert tuple(count(kernel) for kernel in kernels) == warmed


def test_qwen_module_registration_and_eager_fallback(cuda_runtime):
    torch = cuda_runtime
    pytest.importorskip("transformers")
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    from solution.rope import patch_qwen2_rope, _MARKER
    from transformers.integrations import hub_kernels
    import solution

    solution.apply()
    installed = qwen.apply_rotary_pos_emb
    assert hasattr(installed, _MARKER)
    patch_qwen2_rope(qwen)
    solution.apply()
    assert qwen.apply_rotary_pos_emb is installed
    if not hub_kernels.is_kernels_available():
        def ordinary_function():
            return "ordinary"
        assert hub_kernels.use_kernel_forward_from_hub("rotary_pos_emb")(ordinary_function) is ordinary_function
    # CPU/unsupported unsqueeze and cos/sin gradients stay in eager PyTorch.
    q = torch.randn(2, 7, 3, 10, requires_grad=True)
    k = torch.randn(2, 7, 1, 10, requires_grad=True)
    cos = torch.randn(1, 7, 10, requires_grad=True)
    sin = torch.randn_like(cos, requires_grad=True)
    expected, actual = _reference(torch, q, k, cos, sin, 2), qwen.apply_rotary_pos_emb(q, k, cos, sin, 2)
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
    sum(out.sum() for out in actual).backward()
    assert cos.grad is not None and sin.grad is not None
