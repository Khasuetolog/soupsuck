"""CUDA acceptance: exact epilogue/RNG checks and bounded fp16 ingress error."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def explicit_legacy_environment(monkeypatch):
    monkeypatch.setenv("SOUP_EXP_PRECOMPILE", "0")
    monkeypatch.setenv("SOUP_EXP_LAUNCH_TRIM", "0")


@pytest.mark.parametrize("base_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("adapter_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("cast_to_result", [False, True])
def test_epilogue_rounding_promotion_and_gradients(base_dtype, adapter_dtype, cast_to_result):
    from solution.kernels.lora_epilogue import lora_epilogue

    torch.manual_seed(175)
    result = torch.randn(3, 7, 79, device="cuda", dtype=base_dtype).requires_grad_()
    adapter = torch.randn_like(result, dtype=adapter_dtype).requires_grad_()
    rr = result.detach().clone().requires_grad_()
    ar = adapter.detach().clone().requires_grad_()
    scaling = 1.3
    scaled = ar * scaling
    if cast_to_result:
        scaled = scaled.to(rr.dtype)
    expected = rr + scaled
    actual = lora_epilogue(result, adapter, scaling, cast_to_result=cast_to_result)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(result.grad, rr.grad, rtol=0, atol=0)
    torch.testing.assert_close(adapter.grad, ar.grad, rtol=0, atol=0)


def test_epilogue_scaled_half_must_round_before_add():
    from solution.kernels.lora_epilogue import lora_epilogue

    adapter = torch.linspace(-2, 2, 2049, device="cuda", dtype=torch.float16)
    scaling = 1.3
    # Cancel the rounded product so an incorrectly fused multiply-add is visible.
    result = -(adapter * scaling)
    actual = lora_epilogue(result, adapter, scaling)
    assert torch.count_nonzero(actual) == 0


def _linear_shim():
    """Dense fallback with the same vanilla PEFT control flow, without bnb."""
    class DenseLoRA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(80, 48, bias=False)
            self.base_layer.weight.requires_grad_(False)
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(80, 4, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 48, bias=False)})
            self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(0.05)})
            self.scaling = {"default": 2.0}
            self.active_adapters = ["default"]
            self.lora_variant, self.use_dora = {}, {}
            self.disable_adapters, self.merged = False, False

        def _check_forward_args(self, x, *args, **kwargs):
            assert not args and not kwargs

        def _cast_input_dtype(self, x, dtype):
            return x.to(dtype)

        def forward(self, x, *args, **kwargs):
            self._check_forward_args(x, *args, **kwargs)
            result = self.base_layer(x).clone()
            if self.disable_adapters or self.merged:
                return result
            a, b = self.lora_A["default"], self.lora_B["default"]
            convert = not torch.is_autocast_enabled("cuda")
            if convert:
                x = self._cast_input_dtype(x, a.weight.dtype)
            output = b(a(self.lora_dropout["default"](x))) * self.scaling["default"]
            if convert:
                output = output.to(result.dtype)
            return result + output

    return DenseLoRA().cuda()


def _layer(backend):
    if backend == "linear":
        return _linear_shim()
    # With bnb installed on CUDA this path is mandatory: import or CUDA errors
    # must fail the test, never silently substitute the dense fixture.
    bnb = pytest.importorskip("bitsandbytes")
    peft = pytest.importorskip("peft")
    from peft.tuners.lora.bnb import Linear4bit

    base = bnb.nn.Linear4bit(80, 48, bias=False, compute_dtype=torch.float16,
                            compress_statistics=True, quant_type="nf4").cuda()
    config = peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                             target_modules=["q_proj", "v_proj"])
    layer = Linear4bit(base, "default", config=config, r=4, lora_alpha=8).cuda()
    assert layer.base_layer.weight.quant_state is not None
    return layer


@pytest.mark.parametrize("backend", ["linear", "bnb"])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_lora_same_dropout_rng_and_close_forward_backward(backend, training, input_dtype):
    from solution.lora import _MARKER, _patch_class, patch_lora_4bit

    torch.manual_seed(567)
    layer = _layer(backend)
    # Both A/B need nonzero gradients; standard LoRA initializes B at zero.
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(0, 0.04)
    layer.train(training)
    cls = type(layer)
    original = getattr(cls.forward, _MARKER, cls.forward)
    if backend == "bnb":
        patch_lora_4bit()
        patched = cls.forward
        patch_lora_4bit()
    else:
        _patch_class(cls, ("alora_offsets",))
        patched = cls.forward
        _patch_class(cls, ("alora_offsets",))
    assert cls.forward is patched and getattr(patched, _MARKER) is original

    # Nonzero values make dropout masks directly observable. Include ragged N.
    values = torch.rand(3, 11, 80, device="cuda", dtype=torch.float32) + 0.2
    x_ref = values.detach().clone().requires_grad_()
    x = values.to(input_dtype).detach().requires_grad_()
    gradient = torch.randn(3, 11, 48, device="cuda", dtype=torch.float16)
    captures = []
    hook = layer.lora_dropout["default"].register_forward_hook(
        lambda module, args, output: captures.append((args[0].dtype, output.detach().clone()))
    )
    rng = torch.cuda.get_rng_state()
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            expected = original(layer, x_ref)
        rng_ref = torch.cuda.get_rng_state()
        expected.backward(gradient.to(expected.dtype))
        expected_grads = {name: p.grad.detach().clone() for name, p in layer.named_parameters()
                          if p.requires_grad and p.grad is not None}
        expected_dx = x_ref.grad.detach().clone()
        layer.zero_grad(set_to_none=True)
        torch.cuda.set_rng_state(rng)
        with torch.autocast("cuda", dtype=torch.float16):
            actual = layer(x)
        rng_actual = torch.cuda.get_rng_state()
        actual.backward(gradient.to(actual.dtype))
    finally:
        hook.remove()
    assert len(captures) == 2
    assert captures[0][0] == captures[1][0] == torch.float32
    assert torch.equal(captures[0][1] != 0, captures[1][1] != 0)
    assert torch.equal(rng_ref, rng_actual)
    assert torch.equal(rng, rng_ref) is (not training)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=4e-3, atol=3e-3)
    if input_dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, p in layer.named_parameters():
        if name not in expected_grads:
            assert p.grad is None
            continue
        ref = expected_grads[name]
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad.float(), ref.float(), rtol=1e-2, atol=4e-3, msg=name)
        error, norm = torch.linalg.vector_norm(p.grad.float() - ref.float()), torch.linalg.vector_norm(ref.float())
        assert error <= 1e-2 * norm + 1e-6, name
    torch.testing.assert_close(x.grad.float(), expected_dx, rtol=1e-2, atol=4e-3)
    assert torch.linalg.vector_norm(x.grad.float() - expected_dx) <= 1e-2 * torch.linalg.vector_norm(expected_dx) + 1e-6


@pytest.mark.parametrize("shape", [(3, 11, 80), (2, 7, 79), (4, 13, 1536)])
def test_native_dropout_fp32_restoration_matches_mask_and_generator(shape):
    # Exact guarantee separately from the toleranced activation/gradient check.
    x32 = torch.rand(shape, device="cuda", dtype=torch.float32) + 0.2
    x16 = x32.half()
    dropout = torch.nn.Dropout(0.05).train()
    initial = torch.cuda.get_rng_state()
    ref = dropout(x32)
    after_ref = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(initial)
    actual = dropout(x16.float())
    assert torch.equal(ref != 0, actual != 0)
    assert torch.equal(after_ref, torch.cuda.get_rng_state())


def test_lora_disabled_uses_original_without_dropout():
    from solution.lora import _patch_class

    layer = _linear_shim()
    cls = type(layer)
    original = cls.forward
    _patch_class(cls, ())
    layer.disable_adapters = True
    x = torch.randn(2, 7, 80, device="cuda")
    initial = torch.cuda.get_rng_state()
    with torch.autocast("cuda", dtype=torch.float16):
        expected, actual = original(layer, x), layer(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(initial, torch.cuda.get_rng_state())
