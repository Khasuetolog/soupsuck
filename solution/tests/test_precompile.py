"""EXP-001 opt-in, host ownership, real cache hits and unchanged kernel bits."""

import ast
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_default_apply_keeps_preparation_and_heavy_imports_cold():
    code = '''
import os, sys
os.environ.pop('SOUP_EXP_PRECOMPILE', None)
import solution
solution.apply()
solution.apply()
assert not any(n in sys.modules for n in
    ('solution.precompile', 'torch', 'triton', 'transformers', 'soup_cli'))
'''
    subprocess.run([sys.executable, '-c', code], check=True,
                   cwd=Path(__file__).resolve().parents[2])


def test_plans_cover_production_kernels_and_bind_real_signatures():
    from solution.precompile import plans

    found = set()
    root = Path(__file__).resolve().parents[2]
    expected = set()
    # EXP-002 may add a separate opt-in kernel concurrently; EXP-001 prepares
    # the seven production modules in its task brief, not unrelated experiments.
    for stem in ('cross_entropy', 'rmsnorm', 'add_rmsnorm', 'swiglu_fused',
                 'lora_epilogue', 'nf4', 'rope'):
        file = root / 'solution/kernels' / (stem + '.py')
        for node in ast.parse(file.read_text()).body:
            if isinstance(node, ast.FunctionDef) and any(
                    isinstance(d, ast.Attribute) and d.attr == 'jit' for d in node.decorator_list):
                expected.add(node.name)
    for plan in plans():
        tree = ast.parse((root / (plan.module.replace('.', '/') + '.py')).read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == plan.kernel)
        # inspect.Signature equivalent without importing torch/Triton modules.
        names = [a.arg for a in fn.args.args]
        assert len(plan.args) <= len(names)
        assert set(names[len(plan.args):]) == set(plan.options).intersection(names)
        assert not set(names[:len(plan.args)]).intersection(plan.options)
        assert plan.options['num_warps'] == 4
        found.add(plan.kernel)
    assert found == expected
    assert any(p.kernel == 'vocabulary_statistics_kernel' and p.args[4] == 384
               and p.args[-1] == 512 for p in plans())


def test_start_is_literal_opt_in_and_idempotent(monkeypatch):
    import solution.precompile as prep

    monkeypatch.setattr(prep, '_STATE', dict(started=False, compiled=0, pinned_pairs=0,
                                          errors=[], finished=[]))
    monkeypatch.setattr(prep, '_THREADS', [])
    called = []
    monkeypatch.setattr(prep, '_compile', lambda: called.append('compile'))
    monkeypatch.setattr(prep, '_pinned', lambda: called.append('pinned'))
    for value in ('0', 'true', ''):
        monkeypatch.setenv('SOUP_EXP_PRECOMPILE', value)
        assert not prep.start()
    monkeypatch.setenv('SOUP_EXP_PRECOMPILE', '1')
    assert prep.start()
    assert not prep.start()
    state = prep.wait(5)
    assert not state['alive'] and not state['errors']
    assert sorted(called) == ['compile', 'pinned']


def test_pinned_pairs_are_consumed_once_with_original_shape_policy(monkeypatch):
    import solution
    import solution.precompile as prep

    class Buffer:
        def __init__(self, capacity, width, owner=None):
            self.shape = (capacity, width)
            self.owner = owner if owner is not None else object()
        def __getitem__(self, sl):
            return Buffer(sl.stop, self.shape[1], self.owner)
    class Slot:
        def __init__(self, capacity, width):
            self.gate, self.up = Buffer(capacity, width), Buffer(capacity, width)
            self.leased, self.has_write, self.has_read = False, False, False
    fake = ModuleType('solution.mlp')
    fake._HostSlot = Slot
    monkeypatch.setitem(sys.modules, 'solution.mlp', fake)
    monkeypatch.setattr(solution, 'mlp', fake, raising=False)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(device=lambda _: nullcontext())))
    monkeypatch.setattr(prep, '_PINNED', [])
    monkeypatch.setattr(prep, '_STATE', dict(pinned_pairs=0))
    monkeypatch.setenv('SOUP_MLP_OFFLOAD', '1')
    monkeypatch.setenv('SOUP_MLP_RECOMPUTE', '0')
    monkeypatch.setenv('SOUP_MLP_RECOMPUTE_LAYERS', '26')
    prep._pinned()
    assert len(prep._PINNED) == 2
    first, second = fake._HostSlot(1024, 8960), fake._HostSlot(512, 8960)
    assert first.gate.shape == (1024, 8960) and second.up.shape == (512, 8960)
    assert first.gate.owner is not second.gate.owner
    assert not prep._PINNED
    assert not first.leased and not first.has_read and not first.has_write
    # No available reservation / larger or unrelated model: original allocator.
    assert fake._HostSlot(4096, 1000).gate.shape == (4096, 1000)


def test_warmup_real_kernel_cache_hits_and_exact_outputs(monkeypatch):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    triton = pytest.importorskip('triton')
    import solution.precompile as prep
    from tools.kaggle.bench_kernels import cases, bitwise

    monkeypatch.setenv('SOUP_EXP_PRECOMPILE', '1')
    monkeypatch.setenv('SOUP_MLP_OFFLOAD', '0')
    monkeypatch.setattr(prep, '_STATE', dict(started=False, compiled=0, pinned_pairs=0,
                                          errors=[], finished=[]))
    monkeypatch.setattr(prep, '_THREADS', [])
    assert prep.start()
    state = prep.wait(180)
    assert not state['alive'] and not state['errors'], state
    assert state['compiled'] == len(prep.plans())
    device = torch.cuda.current_device()
    seen = set()
    for case in cases(torch, triton, (800,)):
        name = case.kernel.__name__
        if name in seen:
            continue
        seen.add(name)
        # The offline sweep forces FMA off for every case. CE production
        # leaves its default True; test the exact production option here.
        config = dict(case.baseline)
        fusion = name.startswith('vocabulary_')
        def launch():
            if case.reset is not None:
                case.reset()
            case.kernel[case.grid(config)](*case.args, **(case.fixed or {}), **config,
                                          enable_fp_fusion=fusion)
        cache = case.kernel.device_caches[device][0]
        before = set(cache)
        launch()
        torch.cuda.synchronize()
        assert set(cache) == before, f'precompile missed first real {case.key}'
        expected = tuple(t.clone() for t in case.outputs)
        # Compare against an ordinary first call with no in-process entry.
        # The current run's disk cache is allowed, but no binary is launched
        # until this real call. warmup must restore the same in-process keys.
        cache.clear()
        launch()
        torch.cuda.synchronize()
        assert bitwise(torch, case.outputs, expected) == [0] * len(expected)
        cache.clear()
        for plan in prep.plans():
            if plan.kernel == name:
                prep.warmup_plan(plan)
        before = set(cache)
        launch()
        torch.cuda.synchronize()
        assert set(cache) == before
        assert bitwise(torch, case.outputs, expected) == [0] * len(expected)
    assert seen == {p.kernel for p in prep.plans()}


def test_prepared_pinned_storage_preserves_offload_outputs_and_gradients(monkeypatch):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    import solution.precompile as prep
    import solution.mlp as mlp
    from solution.tests.test_mlp import FrozenMLP

    torch.cuda.synchronize()
    monkeypatch.setattr(prep, '_PINNED', [])
    monkeypatch.setattr(prep, '_STATE', dict(pinned_pairs=0))
    monkeypatch.setattr(mlp, '_OFFLOAD_POOLS', {})
    monkeypatch.setattr(mlp, '_RETIRED_POOLS', [])
    monkeypatch.setattr(mlp, '_LAYER_ORDERS', {})
    monkeypatch.setattr(mlp, '_HostSlot', mlp._HostSlot)  # restore factory on exit
    monkeypatch.setenv('SOUP_MLP_OFFLOAD', '1')
    monkeypatch.setenv('SOUP_MLP_RECOMPUTE', '0')
    monkeypatch.setenv('SOUP_MLP_RECOMPUTE_LAYERS', '27')
    prep._pinned()  # just one 70 MiB host pair for this isolated test
    monkeypatch.setenv('SOUP_MLP_RECOMPUTE_LAYERS', '0')
    assert len(prep._PINNED) == 1
    owner = prep._PINNED[0].gate
    layer = FrozenMLP(hidden=64, intermediate=8960).cuda()
    try:
        for rows in (10, 77, 5):
            x = torch.randn(rows, 64, device='cuda', dtype=torch.float16, requires_grad=True)
            ref_x = x.detach().clone().requires_grad_()
            expected = mlp.fused_qwen2_mlp(layer, ref_x, recompute=False, offload=False)
            actual = mlp.fused_qwen2_mlp(layer, x, recompute=False, offload=True)
            grad = torch.randn_like(actual)
            expected.backward(grad)
            actual.backward(grad)
            torch.cuda.synchronize()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            assert torch.equal(x.grad.view(torch.int16), ref_x.grad.view(torch.int16))
            if rows == 10:
                pool = next(iter(mlp._OFFLOAD_POOLS.values()))
                assert pool.capacity == 16  # existing geometric capacity policy
                assert pool.host[0].gate.data_ptr() == owner.data_ptr()
        assert not prep._PINNED
    finally:
        torch.cuda.synchronize()
