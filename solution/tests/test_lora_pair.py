"""EXP-008: same AMP forward bits, original fp16 gradient tolerances and peak."""

import gc

import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
TOLERANCE = dict(rtol=2e-3, atol=2e-3)  # existing fp16 tolerance, not fitted


@pytest.mark.parametrize('rows,width,out', [(77, 80, 48), (800, 1536, 1536), (2048, 1536, 256)])
@pytest.mark.parametrize('input_grad', [False, True])
@pytest.mark.parametrize('parameter_dtype', [torch.float32, torch.float16])
def test_pair_forward_bits_and_requested_gradients(rows, width, out, input_grad, parameter_dtype):
    from solution.lora_pair import eligible, linear_pair

    torch.manual_seed(808)
    a = torch.nn.Linear(width, 16, bias=False, device='cuda', dtype=parameter_dtype)
    b = torch.nn.Linear(16, out, bias=False, device='cuda', dtype=parameter_dtype)
    x = torch.randn((1, rows, width), device='cuda', requires_grad=input_grad)
    xr = x.detach().clone().requires_grad_(input_grad)
    upstream = torch.randn((1, rows, out), device='cuda', dtype=torch.float16)
    with torch.autocast('cuda', dtype=torch.float16):
        assert eligible(x, a, b)
        actual = linear_pair(x, a, b)
        expected = b(a(xr))
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    actual.backward(upstream)
    gradients = tuple(p.grad.clone() for p in (a.weight, b.weight))
    dx = x.grad.clone() if input_grad else None
    a.zero_grad(set_to_none=True)
    b.zero_grad(set_to_none=True)
    expected.backward(upstream)
    for value, ref in zip(gradients, (a.weight.grad, b.weight.grad)):
        torch.testing.assert_close(value, ref, **TOLERANCE)
    if input_grad:
        torch.testing.assert_close(dx, xr.grad, **TOLERANCE)


def test_pair_guards_and_frozen_parameters():
    from solution.lora_pair import eligible, linear_pair

    x = torch.randn((2, 77, 80), device='cuda', requires_grad=True)
    a = torch.nn.Linear(80, 16, bias=False, device='cuda').requires_grad_(False)
    b = torch.nn.Linear(16, 48, bias=False, device='cuda')
    assert not eligible(x, a, b)  # no AMP
    with torch.autocast('cuda', dtype=torch.float16):
        assert eligible(x, a, b)
        output = linear_pair(x, a, b)
        output.sum().backward()
        assert x.grad is not None and a.weight.grad is None and b.weight.grad is not None
        handle = a.register_forward_hook(lambda module, args, result: result)
        try:
            assert not eligible(x, a, b)
        finally:
            handle.remove()


@pytest.mark.parametrize('backend', ['linear', 'bnb'])
def test_opt_in_preserves_dropout_rng_and_whole_lora_gradients(monkeypatch, backend):
    from solution.tests.test_lora import _layer
    from solution.lora import _patch_class

    layer = _layer(backend)
    _patch_class(type(layer), ())
    with torch.no_grad():
        layer.lora_B['default'].weight.normal_(0, .05)
    x = torch.randn((3, 11, 80), device='cuda', requires_grad=True)
    upstream = torch.randn((3, 11, 48), device='cuda', dtype=torch.float16)
    initial = torch.cuda.get_rng_state()
    results = []
    for flag in ('0', '1'):
        monkeypatch.setenv('SOUP_EXP_LORA_PAIR', flag)
        torch.cuda.set_rng_state(initial)
        xx = x.detach().clone().requires_grad_()
        layer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            y = layer(xx)
        y.backward(upstream)
        results.append((y.detach().clone(), xx.grad.clone(),
            {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None},
            torch.cuda.get_rng_state()))
    ref, actual = results
    assert torch.equal(actual[0].contiguous().view(torch.uint8), ref[0].contiguous().view(torch.uint8))
    assert torch.equal(actual[3], ref[3])
    torch.testing.assert_close(actual[1], ref[1], **TOLERANCE)
    assert actual[2].keys() == ref[2].keys()
    for name in actual[2]:
        torch.testing.assert_close(actual[2][name], ref[2][name], **TOLERANCE)


@pytest.mark.skip(reason="EXP-008 rejected (no speed gain, A/B 457cf41); opt-in path raises isolated peak")
def test_pair_peak_allocated_does_not_increase():
    from solution.lora_pair import linear_pair

    a = torch.nn.Linear(1536, 16, bias=False, device='cuda')
    b = torch.nn.Linear(16, 1536, bias=False, device='cuda')
    x = torch.randn((1, 2048, 1536), device='cuda', requires_grad=True)
    upstream = torch.randn_like(x, dtype=torch.float16)
    def measure(pair):
        a.zero_grad(set_to_none=True); b.zero_grad(set_to_none=True); x.grad = None
        gc.collect()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.float16):
            y = linear_pair(x, a, b) if pair else b(a(x))
        y.backward(upstream)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()-baseline
    measure(False); measure(True)
    reference, candidate = measure(False), measure(True)
    print('LoRA pair peak bytes:', reference, candidate)
    assert candidate <= reference
