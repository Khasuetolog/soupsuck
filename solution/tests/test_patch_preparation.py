"""EXP-010 CPU-only native launcher preparation and measured-run lifecycle."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def isolated_preparation(monkeypatch):
    import solution.precompile as prep

    monkeypatch.setenv('SOUP_EXP_PRECOMPILE', '0')
    monkeypatch.setenv('SOUP_EXP_PARENT_CACHE', '0')
    monkeypatch.setenv('SOUP_EXP_CPU_LAUNCHERS', '0')
    monkeypatch.setattr(prep, '_STATE', dict(started=False, compiled=0, pinned_pairs=0,
                                          errors=[], finished=[]))
    monkeypatch.setattr(prep, '_THREADS', [])
    monkeypatch.setattr(prep, '_PROCESS', None)
    monkeypatch.setattr(prep, '_PROCESS_RESULT', None)
    yield prep


@pytest.mark.parametrize('flag,selected', [(None, True), ('0', False), ('other', True), ('1', True)])
def test_child_prepares_launchers_unless_disabled(monkeypatch, isolated_preparation, flag, selected):
    prep = isolated_preparation
    if flag is None:
        monkeypatch.delenv('SOUP_EXP_CPU_LAUNCHERS')
    else:
        monkeypatch.setenv('SOUP_EXP_CPU_LAUNCHERS', flag)
    monkeypatch.setenv('TRITON_CACHE_DIR', 'this-run')
    called, sent = [], []

    def compile_child(**kwargs):
        called.append(kwargs)
        return dict(compiled=3, launchers_prepared=3, binary_prepare_s=0.1,
                    launcher_prepare_s=0.2, errors=[], cuda_initialized=False, driver_touched=False)

    monkeypatch.setattr(prep, '_offline_compile', compile_child)
    end = SimpleNamespace(send=lambda value: sent.append(value), close=lambda: None)
    prep._process_compile(end, 'this-run')
    assert called == ([dict(prepare_launchers=True)] if selected else [{}])
    assert len(sent) == 1 and not sent[0]['errors']


def test_child_cache_identity_and_launcher_failure_are_reported(monkeypatch, isolated_preparation):
    prep = isolated_preparation
    monkeypatch.setenv('SOUP_EXP_CPU_LAUNCHERS', '1')
    monkeypatch.setenv('TRITON_CACHE_DIR', 'this-run')
    called, sent = [], []
    monkeypatch.setattr(prep, '_offline_compile', lambda **kw: called.append(kw))
    end = SimpleNamespace(send=lambda value: sent.append(value), close=lambda: None)
    with pytest.raises(SystemExit) as failure:
        prep._process_compile(end, 'different-run')
    assert failure.value.code == 1 and not called
    assert 'TRITON_CACHE_DIR' in sent[0]['errors'][0]
    sent.clear()

    def fail(**kwargs):
        raise RuntimeError('native compiler unavailable')

    monkeypatch.setattr(prep, '_offline_compile', fail)
    with pytest.raises(SystemExit) as failure:
        prep._process_compile(end, 'this-run')
    assert failure.value.code == 1
    assert sent[0]['errors'] == ['RuntimeError: native compiler unavailable']


@pytest.mark.parametrize('prepare', [False, True])
def test_offline_builds_cpu_glue_without_touching_runtime_handles(
        monkeypatch, isolated_preparation, prepare):
    prep = isolated_preparation
    events = []
    plan = prep.Plan('exp010_fake_kernel', 'kernel', (17,), {'num_warps': 4})
    monkeypatch.setattr(prep, 'plans', lambda: [plan])

    class Compiled:
        src, metadata = object(), object()
        @property
        def run(self):
            raise AssertionError('GPU handles/launches must remain lazy')

    compiled = Compiled()

    def compile_binary(source, target, options):
        events.append(('binary', tuple(options)))
        return compiled

    def launcher(source, metadata):
        assert source is compiled.src and metadata is compiled.metadata
        events.append('cpu-launcher')

    fake_kernel = SimpleNamespace(debug=False, signature=None, params=[],
        _pack_args=lambda *args: (SimpleNamespace(num_warps=4), {}, {}, {}))
    driver = SimpleNamespace(_active=None, _default=None)
    modules = {
        'torch': SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)),
        'triton': SimpleNamespace(__version__='3.6.0', compile=compile_binary,
            knobs=SimpleNamespace(runtime=SimpleNamespace(debug=False),
                                  compilation=SimpleNamespace(instrumentation_mode=''))),
        'triton.backends.compiler': SimpleNamespace(GPUTarget=lambda *a: object()),
        'triton.backends.nvidia.driver': SimpleNamespace(CudaLauncher=launcher),
        'triton.compiler.compiler': SimpleNamespace(ASTSource=lambda *a: object(), make_backend=lambda t: object()),
        'triton.runtime.jit': SimpleNamespace(MockTensor=lambda t: t,
            create_function_from_signature=lambda *a: lambda *args, **kwargs: ({}, [], {})),
        'triton.runtime.driver': SimpleNamespace(driver=driver),
        'exp010_fake_kernel': SimpleNamespace(kernel=fake_kernel),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    result = prep._offline_compile(prepare_launchers=prepare)
    assert result['compiled'] == 1 and not result['errors']
    assert events == [('binary', ('num_warps',))] + (['cpu-launcher'] if prepare else [])
    if prepare:
        assert result['launchers_prepared'] == 1 and result['launcher_prepare_s'] >= 0
    else:
        assert 'launchers_prepared' not in result
    driver._active = object()
    result = prep._offline_compile(prepare_launchers=prepare)
    assert result['errors'] == ['Offline compiler unexpectedly accessed an active Triton driver']


def test_launcher_telemetry_reaches_parent_status(monkeypatch, isolated_preparation):
    prep = isolated_preparation
    result = dict(compiled=3, errors=[], cuda_initialized=False, driver_touched=False,
                  launchers_prepared=3, binary_prepare_s=0.1, launcher_prepare_s=0.2)
    monkeypatch.setattr(prep, '_PROCESS_RESULT',
                        SimpleNamespace(poll=lambda: True, recv=lambda: result, close=lambda: None))
    state = prep.status()
    assert state['launchers_prepared'] == 3 and state['launcher_prepare_s'] == 0.2
    assert state['child_cuda_initialized'] is False and state['child_driver_touched'] is False


def test_explicit_precompile_disable_stays_cold_with_launcher_flag():
    code = '''
import os, sys
os.environ['SOUP_EXP_PRECOMPILE'] = '0'
os.environ['SOUP_EXP_CPU_LAUNCHERS'] = '1'
import solution
solution.apply()
assert not any(n in sys.modules for n in ('torch', 'triton', 'solution.precompile'))
'''
    subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[2], check=True)
