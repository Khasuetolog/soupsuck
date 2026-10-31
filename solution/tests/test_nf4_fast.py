"""EXP-006: strict baseline/packed-word equality, routing and output storage.

CPU routing tests run without torch/Triton; CUDA tests need only torch/Triton,
not bnb. Existing test_nf4.py also exercises both variants against native bnb.
"""

import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('flag,pointer,fast', [
    (None, 64, False), ('0', 64, False), ('true', 64, False), ('process', 64, False),
    ('1', 64, True), ('1', 68, True), ('1', 65, False), ('1', 66, False),
])
def test_fast_plan_requires_explicit_flag_and_word_alignment(monkeypatch, flag, pointer, fast):
    # Execute the actual pure dispatcher, without importing CUDA-only modules.
    path = Path(__file__).resolve().parents[1] / 'kernels/nf4.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_decode_plan')
    slow, candidate = object(), object()
    scope = {'os': os, 'nf4_decode_kernel': slow, 'nf4_decode_fast_kernel': candidate}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    if flag is None:
        monkeypatch.delenv('SOUP_EXP_NF4_FAST', raising=False)
    else:
        monkeypatch.setenv('SOUP_EXP_NF4_FAST', flag)

    def data_ptr():
        assert flag == '1', 'default path should not query pointer alignment'
        return pointer

    kernel, tile = scope['_decode_plan'](SimpleNamespace(data_ptr=data_ptr))
    assert kernel is (candidate if fast else slow)
    assert tile == (4096 if fast else 2048)


@pytest.mark.parametrize('flag,pointer,block', [(None, 64, 2048), ('0', 64, 2048),
                                             ('1', 64, 4096), ('1', 65, 2048)])
@pytest.mark.parametrize('nested', [False, True])
def test_actual_launch_routes_to_separate_kernels_with_literal_options(monkeypatch, flag, pointer, block, nested):
    path = Path(__file__).resolve().parents[1] / 'kernels/nf4.py'
    tree = ast.parse(path.read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in ('_decode_plan', '_launch_decode')]
    calls = []

    class Kernel:
        def __init__(self, name):
            self.name = name
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((self.name, grid, args, kwargs))

    scope = dict(os=os, triton=SimpleNamespace(cdiv=lambda n, b: (n + b - 1) // b),
                 nf4_decode_kernel=Kernel('baseline'), nf4_decode_fast_kernel=Kernel('fast'),
                 _table=lambda device: 'format-table')
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), scope)
    if flag is None:
        monkeypatch.delenv('SOUP_EXP_NF4_FAST', raising=False)
    else:
        monkeypatch.setenv('SOUP_EXP_NF4_FAST', flag)
    packed = SimpleNamespace(data_ptr=lambda: pointer, device='cuda:0')
    out = SimpleNamespace(numel=lambda: 4097)
    scope['_launch_decode'](packed, 'abs', 'scale2', 'code2', 'offset', out, nested)
    assert calls == [('fast' if block == 4096 else 'baseline', ((4097 + block - 1) // block,),
                      (packed, 'abs', 'scale2', 'code2', 'offset', 'format-table', out, 4097),
                      dict(NESTED=nested, BLOCK=block, num_warps=8 if block == 4096 else 4,
                           enable_fp_fusion=False))]


@pytest.fixture
def gpu():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('needs CUDA for the actual Triton bit comparison')
    pytest.importorskip('triton')
    from solution.kernels import nf4
    return torch, nf4


def _inputs(torch, shape, offset):
    n = math.prod(shape)
    blocks = (n + 63) // 64
    # Every byte value / nibble pair, repeated without padding the physical
    # allocation: partial word loads must not overread the last 1/2/3 bytes.
    packed = (torch.arange((n + 1) // 2, device='cuda', dtype=torch.int64) % 256).to(torch.uint8).view(-1, 1)
    qabs = (torch.arange(blocks, device='cuda', dtype=torch.int64) % 256).to(torch.uint8)
    second = SimpleNamespace(blocksize=256, dtype=torch.float32, nested=False,
                             code=torch.linspace(-1, 1, 256, device='cuda', dtype=torch.float32),
                             absmax=1.125 + 0.375 * torch.arange((blocks + 255) // 256,
                                                               device='cuda', dtype=torch.float32))
    state = SimpleNamespace(quant_type='nf4', blocksize=64, dtype=torch.float16,
                            nested=True, shape=shape, absmax=qabs, state2=second,
                            offset=torch.tensor([offset], device='cuda', dtype=torch.float32))
    return packed, state


def _bits(torch, left, right):
    assert left.shape == right.shape
    assert left.dtype == right.dtype == torch.float16
    assert torch.equal(left.contiguous().view(torch.int16), right.contiguous().view(torch.int16))


@pytest.mark.parametrize('shape', [
    (1, 1), (1, 2), (1, 3), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
    (3, 21), (1, 64), (1, 65), (7, 13),
    (1, 4095), (1, 4096), (1, 4097), (1, 16383), (1, 16384), (1, 16385),
    (1536, 1536), (8960, 1536), (1536, 8960), (256, 1536),
])
@pytest.mark.parametrize('offset', [0.0, 0.0625])
def test_fast_nested_bits_match_current_decode(gpu, monkeypatch, shape, offset):
    torch, nf4 = gpu
    packed, state = _inputs(torch, shape, offset)
    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '0')
    expected = nf4.dequantize_nf4(packed, state)
    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '1')
    kernel, _ = nf4._decode_plan(packed)
    assert kernel is nf4.nf4_decode_fast_kernel
    _bits(torch, nf4.dequantize_nf4(packed, state), expected)
    # Confirm the transposed packed shim also preserves the exact existing view.
    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '0')
    expected_t = nf4.dequantize_nf4(packed.t(), state)
    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '1')
    _bits(torch, nf4.dequantize_nf4(packed.t(), state), expected_t)


@pytest.mark.parametrize('n', [1, 2, 3, 5, 6, 7, 8, 9, 63, 64, 65, 4095, 4096, 4097, 16385])
def test_fast_flat_bits_and_no_output_overwrite(gpu, monkeypatch, n):
    torch, nf4 = gpu
    packed, state = _inputs(torch, (1, n), 0.0)
    blocks = (n + 63) // 64
    # Signed scales exercise the sign of NF4's zero code as well as tails.
    scales = torch.linspace(-2, 2, blocks, device='cuda', dtype=torch.float32)
    reference = torch.empty((1, n), device='cuda', dtype=torch.float16)
    guarded = torch.full((n + 2,), 123.0, device='cuda', dtype=torch.float16)
    out = guarded[1:n + 1].view(1, n)  # contiguous but not vector-store aligned
    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '0')
    nf4._dequantize_flat_into(packed, scales, reference)
    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '1')
    assert nf4._dequantize_flat_into(packed, scales, out).data_ptr() == out.data_ptr()
    _bits(torch, out, reference)
    assert guarded[0].item() == 123.0 and guarded[-1].item() == 123.0


def test_unaligned_packed_view_uses_current_decoder(gpu, monkeypatch):
    torch, nf4 = gpu
    packed, state = _inputs(torch, (17, 127), 0.0625)
    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '0')
    expected = nf4.dequantize_nf4(packed, state)
    storage = torch.empty((packed.numel() + 1,), dtype=torch.uint8, device='cuda')
    source = storage[1:].view(packed.shape)
    source.copy_(packed)
    assert source.data_ptr() % 4 != 0
    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '1')
    assert nf4._decode_plan(source)[0] is nf4.nf4_decode_kernel
    _bits(torch, nf4.dequantize_nf4(source, state), expected)


@pytest.mark.parametrize('nested', [False, True])
def test_fast_reuses_out_with_no_additional_tensor_memory(gpu, monkeypatch, nested):
    torch, nf4 = gpu
    packed, state = _inputs(torch, (256, 1536), 0.0625)
    scales = torch.full((packed.numel() // 32,), 1.25, device='cuda', dtype=torch.float32)
    out = torch.empty(state.shape, device='cuda', dtype=torch.float16)

    def decode():
        return nf4.dequantize_nf4(packed, state, out=out) if nested else nf4._dequantize_flat_into(packed, scales, out)

    # Warm both kernels/table before measuring live tensor allocations; timing
    # tests remain separate, and official training still uses empty caches.
    for flag in ('0', '1'):
        monkeypatch.setenv('SOUP_EXP_NF4_FAST', flag)
        decode()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '1')
    result = decode()
    torch.cuda.synchronize()
    assert result.data_ptr() == out.data_ptr()
    assert torch.cuda.memory_allocated() == before
    assert torch.cuda.max_memory_allocated() == before

    def no_temporary(*args, **kwargs):
        raise AssertionError('out= fast decoder allocated an extra tensor')
    monkeypatch.setattr(torch, 'empty', no_temporary)
    monkeypatch.setattr(torch, 'empty_like', no_temporary)
    assert decode().data_ptr() == out.data_ptr()
