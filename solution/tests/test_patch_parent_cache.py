"""EXP-013 parent JIT cache lifecycle and first-production-call races (CPU)."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


class FakeJIT:
    def __init__(self, run=None, module='solution.kernels.fixture'):
        self.fn = SimpleNamespace(__module__=module)
        self.execute = run or (lambda *args, **kwargs: 'result')

    def run(self, *args, **kwargs):
        return self.execute(*args, **kwargs)


def test_parent_gate_preserves_calls_and_skips_a_used_jit():
    from solution.precompile import _ParentCacheGate

    original, calls = FakeJIT.run, []
    gate = _ParentCacheGate(FakeJIT)
    kernel = FakeJIT(lambda *a, **kw: calls.append((a, kw)) or 17)
    external = FakeJIT(module='other.kernels')
    try:
        assert gate.warm(kernel, lambda: kernel.run(3, warmup=True))
        assert kernel.run(4, warmup=False, grid=(2,)) == 17
        assert not gate.warm(kernel, lambda: pytest.fail('used cache must not be republished'))
        assert external.run(warmup=False) == 'result'
        assert external not in gate.states
        assert calls == [((3,), dict(warmup=True)), ((4,), dict(warmup=False, grid=(2,)))]
    finally:
        gate.close()
    assert FakeJIT.run is original


def test_parent_warmup_and_first_real_call_cannot_publish_concurrently():
    from solution.precompile import _ParentCacheGate

    started, release, attempted = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def execute(*, warmup):
        if warmup:
            calls.append('warm begin')
            started.set()
            assert release.wait(5)
            calls.append('warm end')
        else:
            calls.append('real')
        return 'ok'

    gate = _ParentCacheGate(FakeJIT)
    kernel = FakeJIT(execute)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            warm = pool.submit(gate.warm, kernel, lambda: kernel.run(warmup=True))
            assert started.wait(5)
            def real_call():
                attempted.set()
                return kernel.run(warmup=False)
            real = pool.submit(real_call)
            assert attempted.wait(5)
            assert calls == ['warm begin']
            release.set()
            assert warm.result(timeout=5) and real.result(timeout=5) == 'ok'
        assert calls == ['warm begin', 'warm end', 'real']
        assert not gate.warm(kernel, lambda: pytest.fail('late warmup after production'))
    finally:
        release.set()
        gate.close()


def test_gate_cleanup_preserves_a_later_instrumenter():
    from solution.precompile import _ParentCacheGate

    original = FakeJIT.run
    gate = _ParentCacheGate(FakeJIT)
    previous = FakeJIT.run
    def instrument(kernel, *args, **kwargs):
        return previous(kernel, *args, **kwargs)
    FakeJIT.run = instrument
    try:
        gate.close()
        assert FakeJIT.run is instrument
        assert FakeJIT().run(warmup=False) == 'result'
    finally:
        FakeJIT.run = original


@pytest.fixture
def runtime(monkeypatch):
    import solution.precompile as prep

    monkeypatch.setenv('SOUP_EXP_PARENT_CACHE', '0')
    monkeypatch.setattr(prep, '_STATE', dict(started=False, compiled=0, pinned_pairs=0,
                                          errors=[], finished=[]))
    monkeypatch.setattr(prep, '_THREADS', [])
    monkeypatch.setattr(prep, '_PROCESS', None)
    monkeypatch.setattr(prep, '_PROCESS_RESULT', None)
    return prep


@pytest.mark.parametrize('failure', [None, 'child', 'cache', 'warmup'])
def test_parent_cache_waits_checks_identity_and_always_restores_gate(monkeypatch, runtime, failure):
    prep = runtime
    original = FakeJIT.run
    gate = prep._ParentCacheGate(FakeJIT)
    seen = []
    monkeypatch.setenv('TRITON_CACHE_DIR', 'different' if failure == 'cache' else 'this-run')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(device=lambda d: nullcontext())))
    module = ModuleType('exp013_fake')
    module.first, module.used = FakeJIT(), FakeJIT()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    plans = [prep.Plan(module.__name__, key, (), {}) for key in ('first', 'used')]
    monkeypatch.setattr(prep, 'plans', lambda: plans)
    module.used.run(warmup=False)

    def warm(plan):
        assert seen == ['joined']
        if failure == 'warmup':
            raise RuntimeError('cache load failed')
        return getattr(module, plan.kernel).run(warmup=True)

    monkeypatch.setattr(prep, 'warmup_plan', warm)
    process = SimpleNamespace(exitcode=1 if failure == 'child' else 0,
                              join=lambda: seen.append('joined'))
    if failure:
        with pytest.raises(RuntimeError, match='exitcode|TRITON_CACHE_DIR|cache load failed'):
            prep._parent_cache(process, 'this-run', gate, 7)
    else:
        prep._parent_cache(process, 'this-run', gate, 7)
        assert prep._STATE['parent_cache_prepared'] == 1
        assert prep._STATE['parent_cache_skipped'] == 1
        assert prep._STATE['parent_cache_prepare_s'] >= 0
    assert seen == ['joined'] and FakeJIT.run is original


@pytest.mark.parametrize('mode,flag,initialized,expected', [
    ('process', None, True, False), ('process', '0', True, False), ('process', 'other', True, False),
    ('process', '1', True, True), ('process', '1', False, False), ('1', '1', True, False),
])
def test_parent_cache_start_is_opt_in_and_inherits_existing_device(
        monkeypatch, tmp_path, runtime, mode, flag, initialized, expected):
    import multiprocessing

    prep = runtime
    monkeypatch.setenv('SOUP_EXP_PRECOMPILE', mode)
    if flag is None:
        monkeypatch.delenv('SOUP_EXP_PARENT_CACHE')
    else:
        monkeypatch.setenv('SOUP_EXP_PARENT_CACHE', flag)
    cache = str(tmp_path/'run-cache')
    monkeypatch.setenv('TRITON_CACHE_DIR', cache)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: initialized, current_device=lambda: 7)))
    monkeypatch.setitem(sys.modules, 'triton.runtime.jit', SimpleNamespace(JITFunction=FakeJIT))
    monkeypatch.setattr(FakeJIT, 'run', FakeJIT.run)  # restore on unexpected failure
    threads, called = [], []

    class End:
        def close(self): pass
        def poll(self): return False
    class Process:
        def __init__(self, **kwargs):
            self.exitcode = None
        def start(self): pass
    class Thread:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            threads.append(self)
        def start(self): pass

    def context(method):
        assert method == 'spawn'
        return SimpleNamespace(Pipe=lambda duplex: (End(), End()), Process=Process)
    def parent(process, directory, gate, device):
        called.append((directory, device))
        gate.close()

    monkeypatch.setattr(multiprocessing, 'get_context', context)
    monkeypatch.setattr(threading, 'Thread', Thread)
    monkeypatch.setattr(prep, '_parent_cache', parent)
    assert prep.start() and not prep.start()
    worker = [t for t in threads if t.name == 'soup-precompile-parent-cache']
    assert bool(worker) is expected
    if worker:
        assert len(worker) == 1 and worker[0].daemon is False
        worker[0].target(*worker[0].args)
        assert called == [(cache, 7)]
    if flag == '1' and mode == 'process' and not initialized:
        assert 'not initialized' in prep._STATE['parent_cache_skip_reason']


@pytest.mark.parametrize('failure', ['process', 'thread'])
def test_start_failure_restores_the_jit_gate(monkeypatch, tmp_path, runtime, failure):
    import multiprocessing

    prep = runtime
    original = FakeJIT.run
    monkeypatch.setenv('SOUP_EXP_PRECOMPILE', 'process')
    monkeypatch.setenv('SOUP_EXP_PARENT_CACHE', '1')
    monkeypatch.setenv('TRITON_CACHE_DIR', str(tmp_path))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: True, current_device=lambda: 0)))
    monkeypatch.setitem(sys.modules, 'triton.runtime.jit', SimpleNamespace(JITFunction=FakeJIT))
    closed = []

    class End:
        def close(self): closed.append(self)
    class Process:
        def __init__(self, **kwargs): pass
        def start(self):
            if failure == 'process':
                raise RuntimeError('start failed')
    class Thread:
        def __init__(self, **kwargs): pass
        def start(self): raise RuntimeError('start failed')

    monkeypatch.setattr(multiprocessing, 'get_context', lambda method: SimpleNamespace(
        Pipe=lambda duplex: (End(), End()), Process=Process))
    monkeypatch.setattr(threading, 'Thread', Thread)
    with pytest.raises(RuntimeError, match='start failed'):
        prep.start()
    assert FakeJIT.run is original
    if failure == 'process':
        assert len(closed) == 2 and not prep._STATE['started']
