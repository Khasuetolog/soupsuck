"""Default process preparation, explicit opt-out, ownership and cache bits."""

import ast
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def explicit_legacy_environment(monkeypatch):
    monkeypatch.setenv('SOUP_EXP_PRECOMPILE', '0')
    monkeypatch.setenv('SOUP_EXP_PARENT_CACHE', '0')
    monkeypatch.setenv('SOUP_EXP_CPU_LAUNCHERS', '0')
    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '0')
    monkeypatch.setenv('SOUP_EXP_LAUNCH_TRIM', '0')


def test_explicit_opt_out_keeps_preparation_and_heavy_imports_cold():
    code = '''
import os, sys
os.environ['SOUP_EXP_PRECOMPILE'] = '0'
import solution
solution.apply()
solution.apply()
assert not any(n in sys.modules for n in
    ('solution.precompile', 'torch', 'triton', 'transformers', 'soup_cli'))
'''
    subprocess.run([sys.executable, '-c', code], check=True,
                   cwd=Path(__file__).resolve().parents[2])


@pytest.mark.parametrize('fast', [False, True])
def test_plans_cover_production_kernels_and_bind_real_signatures(monkeypatch, fast):
    from solution.precompile import plans

    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '1' if fast else '0')
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
                if node.name != 'nf4_decode_fast_kernel' or fast:
                    expected.add(node.name)
    for plan in plans():
        tree = ast.parse((root / (plan.module.replace('.', '/') + '.py')).read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == plan.kernel)
        # inspect.Signature equivalent without importing torch/Triton modules.
        names = [a.arg for a in fn.args.args]
        assert len(plan.args) <= len(names)
        assert set(names[len(plan.args):]) == set(plan.options).intersection(names)
        assert not set(names[:len(plan.args)]).intersection(plan.options)
        assert plan.options.get('num_warps', 4) == (8 if plan.kernel == 'nf4_decode_fast_kernel' else 4)
        found.add(plan.kernel)
    assert found == expected
    assert any(p.kernel == 'vocabulary_statistics_kernel' and p.args[4] == 384
               and p.args[-1] == 512 for p in plans())


@pytest.mark.parametrize('fast', [False, True])
def test_compiler_options_match_production_presence_and_insertion_order(monkeypatch, fast):
    from solution.precompile import plans

    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '1' if fast else '0')
    for plan in plans():
        keys = tuple(k for k in plan.options if k in ('num_warps', 'enable_fp_fusion'))
        if plan.kernel == 'vocabulary_statistics_kernel':
            assert keys == ('num_warps',)  # implicit backend fusion default
        elif plan.kernel.startswith('vocabulary_'):
            assert keys == ()  # derivative/output omit BOTH compiler defaults
        elif plan.kernel.startswith('lora_scale_'):
            assert keys == ('enable_fp_fusion',)  # implicit backend warp default
        else:
            assert keys == ('num_warps', 'enable_fp_fusion')


def test_fast_nf4_preparation_is_opt_in_and_keeps_baseline_descriptors(monkeypatch):
    from solution.precompile import plans, _launcher_options

    baseline = plans()
    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '1')
    candidate = plans()
    added = [p for p in candidate if p.kernel == 'nf4_decode_fast_kernel']
    assert [p for p in candidate if p.kernel != 'nf4_decode_fast_kernel'] == baseline
    assert len(added) == 2
    assert {p.args[-2:] for p in added} == {(False, 4096), (True, 4096)}
    assert all(p.options == dict(num_warps=8, enable_fp_fusion=False) for p in added)
    assert _launcher_options('nf4', 'nf4_decode_kernel') == (('num_warps', 4), ('enable_fp_fusion', False))
    assert _launcher_options('nf4', 'nf4_decode_fast_kernel') == (('num_warps', 8), ('enable_fp_fusion', False))


def test_thread_mode_and_literal_zero_opt_out_are_idempotent(monkeypatch):
    import solution.precompile as prep

    monkeypatch.setattr(prep, '_STATE', dict(started=False, compiled=0, pinned_pairs=0,
                                          errors=[], finished=[]))
    monkeypatch.setattr(prep, '_THREADS', [])
    called = []
    monkeypatch.setattr(prep, '_compile', lambda: called.append('compile'))
    monkeypatch.setattr(prep, '_pinned', lambda: called.append('pinned'))
    monkeypatch.setenv('SOUP_EXP_PRECOMPILE', '0')
    assert not prep.start()
    monkeypatch.setenv('SOUP_EXP_PRECOMPILE', '1')
    assert prep.start()
    assert not prep.start()
    state = prep.wait(5)
    assert not state['alive'] and not state['errors']
    assert sorted(called) == ['compile', 'pinned']


@pytest.mark.parametrize('setting', [None, 'process', 'other', ''])
def test_default_process_requires_run_cache_and_uses_spawn_inherited_env(monkeypatch, tmp_path, setting):
    import solution.precompile as prep
    import multiprocessing

    monkeypatch.setattr(prep, '_STATE', dict(started=False, compiled=0, pinned_pairs=0,
                                          errors=[], finished=[]))
    monkeypatch.setattr(prep, '_THREADS', [])
    monkeypatch.setattr(prep, '_PROCESS', None)
    monkeypatch.setattr(prep, '_PROCESS_RESULT', None)
    if setting is None:
        monkeypatch.delenv('SOUP_EXP_PRECOMPILE')
    else:
        monkeypatch.setenv('SOUP_EXP_PRECOMPILE', setting)
    monkeypatch.delenv('TRITON_CACHE_DIR', raising=False)
    assert not prep.start()
    assert 'TRITON_CACHE_DIR' in prep.status()['errors'][0]
    seen = []
    class End:
        def close(self):
            seen.append('closed')
        def poll(self):
            return False
    class Process:
        def __init__(self, **kwargs):
            seen.append(kwargs)
            self.pid, self.exitcode, self.name = 42, None, kwargs['name']
        def start(self):
            seen.append(dict(TRITON_CACHE_DIR=__import__('os').environ['TRITON_CACHE_DIR']))
        def is_alive(self):
            return False
        def join(self, timeout=None):
            return None
    def context(method):
        assert method == 'spawn'
        return SimpleNamespace(Pipe=lambda duplex: (End(), End()), Process=Process)
    monkeypatch.setattr(multiprocessing, 'get_context', context)
    monkeypatch.setattr(prep, '_pinned', lambda: None)
    cache_dir = str(tmp_path / 'this-run' / 'triton_cache')
    monkeypatch.setenv('TRITON_CACHE_DIR', cache_dir)
    assert prep.start()
    assert not prep.start()
    prep.wait(5)
    spec = next(v for v in seen if isinstance(v, dict) and 'target' in v)
    assert spec['target'] is prep._process_compile and spec['daemon'] is False
    assert spec['args'][1] == cache_dir
    assert dict(TRITON_CACHE_DIR=cache_dir) in seen
    assert not Path(cache_dir).exists()  # start never creates or substitutes a cache


@pytest.mark.parametrize('setting', [None, '0', '1', 'process', 'other', ''])
def test_install_prepares_by_default_and_only_zero_disables(monkeypatch, setting):
    import solution.patch as patch

    calls = []
    fake = ModuleType('solution.precompile')
    fake.start = lambda: calls.append('start')
    monkeypatch.setitem(sys.modules, 'solution.precompile', fake)
    monkeypatch.setattr(patch, '_TARGET_MODULES', set())
    if setting is None:
        monkeypatch.delenv('SOUP_EXP_PRECOMPILE')
    else:
        monkeypatch.setenv('SOUP_EXP_PRECOMPILE', setting)
    patch.install()
    assert len(calls) == (0 if setting == '0' else 1)


def test_process_child_failure_report_is_bounded_and_checks_cache_identity(monkeypatch):
    import pickle
    import solution.precompile as prep

    messages = []
    send = SimpleNamespace(send=messages.append, close=lambda: None)
    monkeypatch.setenv('TRITON_CACHE_DIR', '/run/current/triton_cache')
    def forbidden():
        raise AssertionError('must reject a different cache before compilation')
    monkeypatch.setattr(prep, '_offline_compile', forbidden)
    with pytest.raises(SystemExit):
        prep._process_compile(send, '/run/another/triton_cache')
    assert 'exact run TRITON_CACHE_DIR' in messages[-1]['errors'][0]
    monkeypatch.setattr(prep, '_offline_compile', lambda: dict(
        compiled=0, errors=['failure ' + 'x'*10000] * 62, cuda_initialized=False))
    with pytest.raises(SystemExit):
        prep._process_compile(send, '/run/current/triton_cache')
    assert len(pickle.dumps(messages[-1])) < 4096


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


@pytest.mark.parametrize('mode,cpu_launchers,fast,parent_cache',
                         [('1', False, False, False), ('process', False, False, False),
                          ('process', True, False, False), ('process', False, True, False),
                          ('process', True, True, False), ('process', True, False, True)],
                         ids=['thread', 'spawn-disk-cache', 'spawn-cpu-launchers',
                              'spawn-fast-nf4', 'spawn-fast-nf4-cpu-launchers', 'spawn-parent-cache'])
def test_warmup_real_kernel_cache_hits_and_exact_outputs(
        monkeypatch, tmp_path, mode, cpu_launchers, fast, parent_cache):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    triton = pytest.importorskip('triton')
    import importlib
    import solution.precompile as prep
    from triton.runtime.jit import compute_cache_key
    from solution.kernels.cross_entropy import fused_linear_cross_entropy
    from solution.kernels.rmsnorm import rms_norm
    from solution.kernels.add_rmsnorm import add_rms_norm
    from solution.kernels.swiglu_fused import swiglu_fused_forward, swiglu_fused_backward
    from solution.kernels.lora_epilogue import lora_epilogue
    from solution.kernels.nf4 import dequantize_nf4, _dequantize_flat_into
    from solution.kernels.rope import fused_rope

    monkeypatch.setenv('SOUP_EXP_PRECOMPILE', mode)
    monkeypatch.setenv('SOUP_EXP_CPU_LAUNCHERS', '1' if cpu_launchers else '0')
    monkeypatch.setenv('SOUP_EXP_NF4_FAST', '1' if fast else '0')
    monkeypatch.setenv('SOUP_EXP_PARENT_CACHE', '1' if parent_cache else '0')
    monkeypatch.setenv('SOUP_MLP_OFFLOAD', '0')
    monkeypatch.setattr(prep, '_STATE', dict(started=False, compiled=0, pinned_pairs=0,
                                          errors=[], finished=[]))
    monkeypatch.setattr(prep, '_THREADS', [])
    monkeypatch.setattr(prep, '_PROCESS', None)
    monkeypatch.setattr(prep, '_PROCESS_RESULT', None)
    device = torch.cuda.current_device()
    kernels = {p.kernel: getattr(importlib.import_module(p.module), p.kernel) for p in prep.plans()}
    if mode == 'process':
        assert torch.cuda.get_device_capability() == (7, 5), 'offline target is sm75 (T4)'
        cache_dir = tmp_path / 'this-test-run' / 'triton_cache'
        assert not cache_dir.exists()
        monkeypatch.setenv('TRITON_CACHE_DIR', str(cache_dir))
        # Triton env knobs may have been materialized by preceding tests.
        monkeypatch.setattr(triton.knobs.cache, 'dir', str(cache_dir))
        for kernel in kernels.values():
            kernel.device_caches[device][0].clear()
    before_allocated = torch.cuda.memory_allocated()
    assert prep.start()
    state = prep.wait(180)
    assert not state['alive'] and not state['errors'], state
    assert state['compiled'] == len(prep.plans())
    if mode == 'process':
        assert state['process_exitcode'] == 0, state
        assert state['child_cuda_initialized'] is False, state
        assert state['child_driver_touched'] is False, state
        if cpu_launchers:
            assert state['launchers_prepared'] == len(prep.plans()), state
            assert state['launcher_prepare_s'] >= 0, state
            assert torch.cuda.memory_allocated() == before_allocated
            if parent_cache:
                assert state['parent_cache_prepared'] == len(prep.plans()), state
                assert state['parent_cache_skipped'] == 0, state
                assert all(k.device_caches[device][0] for k in kernels.values())
                # Native warmup only: no CUDA module handles or launches.
                assert all(binary.module is None and binary.function is None
                           for k in kernels.values() for binary in k.device_caches[device][0].values())
            else:
                assert all(not k.device_caches[device][0] for k in kernels.values())
    seen = set()
    disk_hits, compilation_misses = set(), []
    active = [True]
    native_misses = []
    if cpu_launchers:
        from triton.runtime import build as native_build

        original_build = native_build._build
        def count_native_build(name, *args, **kwargs):
            if active[0] and name == '__triton_launcher':
                native_misses.append(name)
            return original_build(name, *args, **kwargs)
        monkeypatch.setattr(native_build, '_build', count_native_build)
    touched = set()
    case_name = ['']
    previous_listener = triton.knobs.compilation.listener
    def compilation_listener(**event):
        if active[0] and mode == 'process':
            name = event['src'].name
            if name in kernels:
                if event['cache_hit']:
                    disk_hits.add(name)
                else:
                    compilation_misses.append((case_name[0], name))
                assert event['cache_hit'], f'actual compiler miss: {case_name[0]} / {name}'
        if previous_listener is not None:
            previous_listener(**event)
    monkeypatch.setattr(triton.knobs.compilation, 'listener', compilation_listener)

    def check_hook(name, kernel):
        def hook(*args, **kwargs):
            if not active[0]:
                return
            cache, key_cache, _, _, binder = kernel.device_caches[device]
            _, specialization, options = binder(*args, **kwargs)
            key = compute_cache_key(key_cache, specialization, options)
            # Hook runs inside real JIT.run AFTER debug/instrumentation kwargs
            # are added, and BEFORE cache lookup. No hand-coded D/==1 guesses.
            assert (mode == 'process' and not parent_cache) or key in cache, (f'precompile missed {case_name[0]} / {name}: '
                                  f'specialization={specialization!r}, options={options!r}; '
                                  f'prepared_keys={tuple(cache)!r}')
            seen.add(name)
            touched.add(name)
        return hook

    for name, kernel in kernels.items():
        monkeypatch.setattr(kernel, 'pre_run_hooks',
                            list(kernel.pre_run_hooks) + [check_hook(name, kernel)])

    def compare(name, run):
        case_name[0] = name
        touched.clear()
        prepared = {k: dict(v.device_caches[device][0]) for k, v in kernels.items()}
        actual = tuple(v.detach().clone() for v in run())
        torch.cuda.synchronize()
        called = set(touched)
        assert called, name
        active[0] = False
        try:
            for k in called:
                kernels[k].device_caches[device][0].clear()
            expected = run()  # unchanged production path, ordinary first JIT call
            torch.cuda.synchronize()
            for got, ref in zip(actual, expected, strict=True):
                assert torch.equal(got.contiguous().view(torch.uint8),
                                   ref.detach().contiguous().view(torch.uint8)), name
        finally:
            for k in called:
                cache = kernels[k].device_caches[device][0]
                cache.clear()
                cache.update(prepared[k])
            active[0] = True

    def random(*shape, dtype=torch.float32):
        return torch.randn(shape, device='cuda', dtype=dtype)

    # Aligned, nonaligned and maximum training row counts. Every launch is
    # issued by its production wrapper, including omitted default options and
    # exact dummy-pointer dtypes; no kernel is excluded after its first case.
    for n in (800, 801, 2048):
        x, w, delta = random(n, 1536), random(1536), random(n, 1536, dtype=torch.float16)
        for out_dtype in (None, torch.float16):
            gy = random(n, 1536, dtype=out_dtype or torch.float32)
            gs = random(n, 1536)
            def norm_run():
                xx, ww = x.clone().requires_grad_(), w.clone().requires_grad_()
                y = rms_norm(xx, ww, 1e-6, out_dtype)
                y.backward(gy)
                return y, xx.grad, ww.grad
            compare(f'rms/N{n}/{out_dtype}', norm_run)
            def add_run():
                xx, dd, ww = (x.clone().requires_grad_(), delta.clone().requires_grad_(),
                              w.clone().requires_grad_())
                s, y = add_rms_norm(xx, dd, ww, 1e-6, out_dtype)
                torch.autograd.backward((s, y), (gs, gy))
                return s, y, xx.grad, dd.grad, ww.grad
            compare(f'add-rms/N{n}/{out_dtype}', add_run)
        def first_add_run():
            dd = delta.clone().requires_grad_()
            s, y = add_rms_norm(x, dd, w, 1e-6, torch.float16)
            torch.autograd.backward((s, y), (gs, gy))
            return s, y, dd.grad
        compare(f'add-rms-first-frozen-residual/N{n}', first_add_run)
        for dtype in (torch.float16, torch.float32):
            g, u, da = (random(n, 8960, dtype=dtype) for _ in range(3))
            def swiglu_run():
                y = swiglu_fused_forward(g, u)
                dg, du = swiglu_fused_backward(g.half(), u.half(), da, dtype, dtype)
                return y, dg, du
            compare(f'swiglu/N{n}/{dtype}', swiglu_run)
            base, adapter = random(n, 1536, dtype=dtype), random(n, 1536, dtype=torch.float16)
            upstream = random(n, 1536, dtype=dtype)
            for cast in (False, True):
                def lora_run():
                    bb, aa = base.clone().requires_grad_(), adapter.clone().requires_grad_()
                    y = lora_epilogue(bb, aa, 2.0, cast_to_result=cast)
                    y.backward(upstream)
                    return y, bb.grad, aa.grad
                compare(f'lora/N{n}/{dtype}/cast{cast}', lora_run)
        for b in (1, 4):
            for dtype, ctype in ((torch.float16, torch.float32),
                                 (torch.float16, torch.float16),
                                 (torch.float32, torch.float32)):
                q = random(b, n, 12, 128, dtype=dtype).transpose(1, 2)
                k = random(b, n, 2, 128, dtype=dtype).transpose(1, 2)
                c, s = random(b, n, 128, dtype=ctype), random(b, n, 128, dtype=ctype)
                gdtype = torch.promote_types(dtype, ctype)
                gq, gk = random(*q.shape, dtype=gdtype), random(*k.shape, dtype=gdtype)
                def rope_run():
                    qq, kk = q.detach().requires_grad_(), k.detach().requires_grad_()
                    yq, yk = fused_rope(qq, kk, c, s)
                    torch.autograd.backward((yq, yk), (gq, gk))
                    return yq, yk, qq.grad, kk.grad
                compare(f'rope/B{b}/T{n}/{dtype}/{ctype}', rope_run)
        head = random(151936, 1536, dtype=torch.float16)
        targets = torch.arange(n, device='cuda') % 151936
        targets[::3] = -100
        upstream_ce = random(n) / n
        def ce_run():
            xx = x.clone().requires_grad_()
            outputs = fused_linear_cross_entropy(xx, head, targets)
            outputs[0].backward(upstream_ce)
            return (*outputs, xx.grad)
        compare(f'CE/N{n}/V151936/full+tail+compact-bwd', ce_run)
        del head

    import bitsandbytes as bnb
    for shape in ((1536, 1536), (8960, 1536), (1536, 8960), (256, 1536)):
        packed, quant = bnb.functional.quantize_4bit(random(*shape, dtype=torch.float16),
            quant_type='nf4', compress_statistics=True, blocksize=64)
        flat = bnb.functional.dequantize_blockwise(quant.absmax, quant.state2) + quant.offset
        def nf4_run():
            nested = dequantize_nf4(packed, quant)
            out = torch.empty(shape, device='cuda', dtype=torch.float16)
            return nested, _dequantize_flat_into(packed, flat, out)
        compare(f'NF4/{shape}/nested+flat', nf4_run)
        if fast:
            # Baseline descriptors remain available for normal decode/fallback;
            # exercise both independently named JITs with aligned pointers.
            with monkeypatch.context() as patch:
                patch.setenv('SOUP_EXP_NF4_FAST', '0')
                compare(f'NF4/{shape}/baseline-nested+flat', nf4_run)
    assert seen == set(kernels)
    if mode == 'process':
        assert not compilation_misses
        if not parent_cache:
            assert disk_hits == set(kernels), 'every kernel must load a child-built disk artifact'
        if cpu_launchers:
            assert not native_misses, 'production launch rebuilt a child-prepared CPU launcher'


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
