#!/usr/bin/env python3
"""Separate startup/save/exit probe of the official solution train subprocess.

Stdlib-only observer; never imported by solution or an ordinary harness run.
--self-test exercises the shim and parser without torch, models or a GPU.
"""

import argparse
import atexit
from collections import defaultdict
from functools import wraps
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
OWNER = 'SOUP_START_PROBE_OWNER_PID'
EVENTS = 'SOUP_START_PROBE_EVENTS'
MARKER = '_soup_start_probe_wrapper'
ROOT_IMPORTS = {'torch', 'transformers', 'trl', 'peft', 'bitsandbytes', 'soup_cli'}
# Only modules actually imported by the official command are intercepted.
TARGETS = ROOT_IMPORTS | {
    'solution', 'torch.cuda', 'soup_cli.cli', 'soup_cli.trainer.sft',
    'transformers.modeling_utils', 'transformers.models.auto.auto_factory',
    'transformers.trainer', 'transformers.quantizers.base',
    'transformers.quantizers.quantizer_bnb_4bit', 'transformers.tokenization_utils_base',
    'transformers.integrations.bitsandbytes',
    'trl.trainer.sft_trainer', 'peft.mapping_func', 'peft.utils.other',
    'peft.peft_model', 'bitsandbytes.functional', 'bitsandbytes.nn.modules',
}


class Recorder:
    def __init__(self, path):
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self.lock = threading.RLock()
        self.serial = 0

    def emit(self, kind, phase, *, at=None, **details):
        # Observer failures must not change an observed function's outcome.
        try:
            with self.lock:
                self.serial += 1
                observed_ns = time.perf_counter_ns()
                event_ns = at if at is not None else observed_ns
                value = dict(kind=kind, phase=phase, ns=event_ns,
                             unix_ns=time.time_ns()-(observed_ns-event_ns), pid=os.getpid(), tid=threading.get_ident(),
                             serial=self.serial, **details)
                os.write(self.fd, (json.dumps(value, separators=(',', ':')) + '\n').encode())
                return self.serial
        except (OSError, ValueError):
            return None

    def wrap(self, owner, name, phase, *, after=None, predicate=None):
        # Read __dict__ to preserve class/static method descriptors and avoid
        # resolving Transformers/TRL lazy attributes prematurely.
        raw = vars(owner).get(name)
        descriptor = type(raw) if isinstance(raw, (classmethod, staticmethod)) else None
        function = raw.__func__ if descriptor else raw
        if not callable(function) or getattr(function, MARKER, False):
            return

        @wraps(function)
        def observed(*args, **kwargs):
            if predicate is not None and not predicate():
                return function(*args, **kwargs)
            span = self.emit('begin', phase)
            error = None
            try:
                value = function(*args, **kwargs)
                if after is not None:
                    try:
                        after(args, value)
                    except Exception as exc:
                        self.emit('observer_error', phase, error=type(exc).__name__)
                return value
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                self.emit('end', phase, span=span, error=error)

        setattr(observed, MARKER, True)
        setattr(owner, name, descriptor(observed) if descriptor else observed)
        self.emit('hook', phase, target=getattr(owner, '__name__', '?') + '.' + name)

    def patch_module(self, name, module):
        def method(class_name, attribute, phase, **kwargs):
            cls = vars(module).get(class_name)
            if isinstance(cls, type):
                self.wrap(cls, attribute, phase, **kwargs)

        if name == 'solution':
            self.wrap(module, 'apply', 'solution_apply')
        elif name == 'torch.cuda':
            self.wrap(module, '_lazy_init', 'cuda_context_init',
                      predicate=lambda: not module.is_initialized())
        elif name == 'transformers.modeling_utils':
            method('PreTrainedModel', 'from_pretrained', 'model_from_pretrained')
        elif name == 'transformers.models.auto.auto_factory':
            method('_BaseAutoModelClass', 'from_pretrained', 'model_from_pretrained')
        elif name == 'transformers.quantizers.base':
            for attr in ('preprocess_model', 'postprocess_model'):
                method('HfQuantizer', attr, 'quantization_setup')
        elif name == 'transformers.quantizers.quantizer_bnb_4bit':
            method('Bnb4BitHfQuantizer', 'create_quantized_param', 'quantization')
        elif name == 'transformers.integrations.bitsandbytes':
            method('Bnb4bitQuantize', 'convert', 'quantization')
        elif name == 'bitsandbytes.functional':
            for attr in ('quantize_4bit', 'quantize_blockwise'):
                self.wrap(module, attr, 'quantization')
        elif name == 'bitsandbytes.nn.modules':
            method('Params4bit', '_quantize', 'quantization')
        elif name == 'peft.mapping_func':
            self.wrap(module, 'get_peft_model', 'peft_wrap')
        elif name == 'peft.utils.other':
            self.wrap(module, 'prepare_model_for_kbit_training', 'kbit_prepare')
        elif name == 'peft.peft_model':
            method('PeftModel', 'save_pretrained', 'adapter_save')
        elif name == 'transformers.tokenization_utils_base':
            method('PreTrainedTokenizerBase', 'save_pretrained', 'tokenizer_save')
        elif name == 'transformers.trainer':
            method('Trainer', '__init__', 'trainer_init', after=self.add_callback)
            method('Trainer', 'train', 'trainer_train')
            method('Trainer', 'save_model', 'save_model')
            method('Trainer', 'get_train_dataloader', 'dataloader_build')
        elif name == 'trl.trainer.sft_trainer':
            method('SFTTrainer', '__init__', 'trainer_init')
            method('SFTTrainer', '_prepare_dataset', 'dataset_prepare')
        elif name == 'soup_cli.trainer.sft':
            method('SFTTrainerWrapper', 'setup', 'soup_setup')
            method('SFTTrainerWrapper', 'train', 'soup_train_and_save')

    def add_callback(self, args, result):
        base = vars(sys.modules['transformers.trainer_callback'])['TrainerCallback']
        recorder = self

        class ProbeCallback(base):
            def on_step_begin(self, train_args, state, control, **kwargs):
                self.span = recorder.emit('begin', 'train_step', step=state.global_step + 1)

            def on_step_end(self, train_args, state, control, **kwargs):
                recorder.emit('end', 'train_step', span=self.span, step=state.global_step)

            def on_train_end(self, train_args, state, control, **kwargs):
                recorder.emit('point', 'trainer_on_train_end', step=state.global_step)

        args[0].add_callback(ProbeCallback())


class ObservedLoader:
    def __init__(self, original, name, recorder):
        self.original, self.name, self.recorder = original, name, recorder

    def __getattr__(self, name):
        return getattr(self.original, name)

    def create_module(self, spec):
        create = getattr(self.original, 'create_module', None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        phase = 'import.' + self.name
        span = self.recorder.emit('begin', phase)
        error = None
        try:
            self.original.exec_module(module)
            # Some packages replace themselves with a LazyModule in exec_module.
            try:
                self.recorder.patch_module(self.name, sys.modules[self.name])
            except Exception as exc:
                self.recorder.emit('observer_error', phase, error=type(exc).__name__)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            self.recorder.emit('end', phase, span=span, error=error)


class ObservedFinder:
    def __init__(self, recorder):
        self.recorder = recorder

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        for finder in tuple(sys.meta_path):
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if hasattr(spec.loader, 'exec_module'):
                    spec.loader = ObservedLoader(spec.loader, fullname, self.recorder)
                return spec
        return None


def install_shim(entered_ns):
    """Called only in the instrumented train interpreter, never compiler children."""
    recorder = Recorder(os.environ[EVENTS])
    recorder.emit('point', 'interpreter_sitecustomize', at=entered_ns)
    # CPython's import profiler remains enabled in PyConfig; multiprocessing
    # builds child flags from this dict. Do not mix child importtime into ours.
    sys._xoptions.pop('importtime', None)
    sys.meta_path.insert(0, ObservedFinder(recorder))
    recorder.emit('point', 'shim_ready')
    atexit.register(recorder.emit, 'point', 'atexit_late_callback')


def shim_source():
    return '''import time
_start_probe_entry = time.perf_counter_ns()
import os, sys
if os.environ.get('SOUP_START_PROBE_EVENTS') and not os.environ.get('SOUP_START_PROBE_OWNER_PID'):
    os.environ['SOUP_START_PROBE_OWNER_PID'] = str(os.getpid())
    from tools.kaggle.start_probe import install_shim
    install_shim(_start_probe_entry)
# Preserve any sitecustomize the environment would ordinarily have loaded.
# Compiler children skip our observer but still execute the original module.
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
_start_probe_dir = os.path.dirname(os.path.abspath(__file__))
_start_probe_paths = [p for p in sys.path if os.path.abspath(p) != _start_probe_dir]
_start_probe_spec = PathFinder.find_spec('sitecustomize', _start_probe_paths)
if _start_probe_spec is not None and _start_probe_spec.loader is not None:
    _start_probe_original = module_from_spec(_start_probe_spec)
    sys.modules['sitecustomize'] = _start_probe_original
    _start_probe_spec.loader.exec_module(_start_probe_original)
'''


def execute(command, environment, out, timeout):
    """Return parent timestamps around one subprocess; no evaluation or extra train."""
    events = out / 'events.jsonl'
    events.touch(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='soup-startprobe-shim-') as directory:
        Path(directory, 'sitecustomize.py').write_text(shim_source())
        env = dict(environment)
        env.pop(OWNER, None)
        env.pop('PYTHONPROFILEIMPORTTIME', None)  # use only our parent -X flag
        env[EVENTS] = str(events)
        env['PYTHONPATH'] = os.pathsep.join([directory, str(ROOT), env.get('PYTHONPATH', '')])
        timeline = Recorder(events)
        timeline.emit('point', 'process_launch_begin')
        with (out / 'train.log').open('w') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            timeline.emit('point', 'process_spawn_return', child_pid=proc.pid)
            timed_out = False
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait()
            timeline.emit('point', 'process_end_observed', child_pid=proc.pid,
                          exit_code=code, timed_out=timed_out)
            # Same process-group cleanup policy as the official run_logged.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    os.close(timeline.fd)
    return code


def union_ns(intervals):
    total, end = 0, None
    for left, right in sorted(intervals):
        if right < left:
            raise ValueError('negative phase interval')
        total += max(0, right - max(left, end if end is not None else left))
        end = max(end if end is not None else right, right)
    return total


def import_costs(log):
    rows = {}
    pattern = re.compile(r'^import time:\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\S.*?)\s*$')
    for line in log.splitlines():
        match = pattern.match(line)
        if match:
            own, cumulative, module = match.groups()
            row = rows.setdefault(module, dict(module=module, calls=0, self_us=0, max_cumulative_us=0))
            row['calls'] += 1
            row['self_us'] += int(own)
            row['max_cumulative_us'] = max(row['max_cumulative_us'], int(cumulative))
    return sorted(rows.values(), key=lambda r: (-r['self_us'], r['module']))[:30]


def summarize(out, *, command, commit, exit_code):
    events, malformed = [], []
    for number, line in enumerate((out / 'events.jsonl').read_text().splitlines(), 1):
        try:
            events.append(json.loads(line))
        except ValueError:
            malformed.append(number)
    spawned = next((e for e in events if e['phase'] == 'process_spawn_return'), {})
    pid = spawned.get('child_pid')
    child = [e for e in events if e['pid'] == pid]
    origin = next(e['ns'] for e in events if e['phase'] == 'process_launch_begin')
    by_id, spans, unclosed = {}, [], []
    for event in sorted(child, key=lambda e: e['ns']):
        if event['kind'] == 'begin':
            by_id[event['serial']] = event
        elif event['kind'] == 'end':
            begin = by_id.pop(event['span'], None)
            if begin is not None:
                spans.append(dict(phase=event['phase'], start_ns=begin['ns'], end_ns=event['ns'],
                                  tid=event['tid'], error=event.get('error'), step=begin.get('step')))
    for event in by_id.values():
        unclosed.append(dict(phase=event['phase'], start_ns=event['ns']))
    groups = defaultdict(list)
    for span in spans:
        groups[span['phase']].append(span)
    phases = []
    for phase, items in sorted(groups.items()):
        phases.append(dict(phase=phase, calls=len(items),
                           start_s=(min(s['start_ns'] for s in items)-origin)/1e9,
                           end_s=(max(s['end_ns'] for s in items)-origin)/1e9,
                           inclusive_sum_s=sum(s['end_ns']-s['start_ns'] for s in items)/1e9,
                           wall_union_s=union_ns([(s['start_ns'], s['end_ns']) for s in items])/1e9,
                           errors=sum(bool(s['error']) for s in items)))
    points = [dict(phase=e['phase'], offset_s=(e['ns']-origin)/1e9, pid=e['pid'])
              for e in sorted(events, key=lambda e: e['ns']) if e['kind'] == 'point']
    steps = sorted(groups.get('train_step', []), key=lambda s: s['start_ns'])
    for name, step, key in [('first_step_begin', steps[:1], 'start_ns'),
                            ('last_step_end', steps[-1:], 'end_ns')]:
        if step:
            points.append(dict(phase=name, offset_s=(step[0][key]-origin)/1e9, pid=pid))
    boundaries = {point['phase']: point['offset_s'] for point in points}
    apply = groups.get('solution_apply', [])
    if apply:
        boundaries['apply_begin'] = (min(s['start_ns'] for s in apply)-origin)/1e9
        boundaries['apply_end'] = (max(s['end_ns'] for s in apply)-origin)/1e9
    order = ['process_launch_begin', 'interpreter_sitecustomize', 'shim_ready',
             'apply_begin', 'apply_end', 'first_step_begin', 'last_step_end',
             'atexit_late_callback', 'process_end_observed']
    known = [name for name in order if name in boundaries]
    boundary_intervals = [dict(begin=a, end=b, wall_s=boundaries[b]-boundaries[a])
                          for a, b in zip(known, known[1:])]
    official_metrics, metrics_error = {}, None
    if (out/'metrics.json').exists():
        try:
            metrics = json.loads((out/'metrics.json').read_text())
            official_metrics = {key: metrics.get(key) for key in
                                ('apply_s', 'steps', 'train_loop_s', 'median_step_s')}
            official_metrics['first_step_s'] = next(iter(metrics.get('step_s', [])), None)
        except (ValueError, OSError) as exc:
            metrics_error = type(exc).__name__
    required = ['import.'+name for name in sorted(ROOT_IMPORTS)] + [
        'solution_apply', 'model_from_pretrained', 'quantization', 'peft_wrap',
        'kbit_prepare', 'trainer_init', 'train_step', 'save_model', 'cuda_context_init']
    missing = [phase for phase in required if phase not in groups]
    observer_errors = [e for e in child if e['kind'] == 'observer_error']
    value = dict(schema=1, measurement='separate_instrumented_probe_not_jury_timing',
                 commit=commit, command=command, exit_code=exit_code, child_pid=pid,
                 ok=exit_code == 0 and not (missing or malformed or unclosed or observer_errors or metrics_error),
                 phases=phases, timestamps=sorted(points, key=lambda e: e['offset_s']),
                 boundary_intervals=boundary_intervals, official_metrics=official_metrics,
                 official_metrics_error=metrics_error,
                 top_imports=import_costs((out / 'train.log').read_text(errors='replace')),
                 hooks=[dict(phase=e['phase'], target=e['target']) for e in child if e['kind'] == 'hook'],
                 steps=len(steps), missing_phases=missing, malformed_event_lines=malformed,
                 unclosed_spans=unclosed, observer_errors=observer_errors,
                 notes=[
                     'Times are CPU wall observations; no added CUDA synchronization/events/work.',
                     'Phase unions overlap; inclusive sums and cumulative imports are not additive.',
                     'Importtime self/cumulative costs include observer overhead; threaded imports can distort nesting.',
                     'Interpreter launch is bracketed by parent Popen and sitecustomize, not an exact Py_Initialize timer.',
                     'atexit_late_callback was registered early and executes late; it is not the start of shutdown.',
                     'Step callback timestamps precede the harness synchronization; compare official metrics.json.',
                     'Parent wait(timeout) observes exit with polling latency; process_end is an upper bound.',
                 ])
    (out / 'start_probe.json').write_text(json.dumps(value, indent=2) + '\n')
    print_table(value)
    return value


def print_table(value):
    print('Startup probe (instrumented, not jury timing), commit', value['commit'])
    print(f'{"phase":44s} {"calls":>6s} {"start":>9s} {"end":>9s} {"union s":>10s}')
    for row in value['phases']:
        print(f'{row["phase"]:44s} {row["calls"]:6d} {row["start_s"]:9.3f} '
              f'{row["end_s"]:9.3f} {row["wall_union_s"]:10.3f}')
    print('Boundary observations (seconds after parent launch begin):')
    for point in value['timestamps']:
        print(f'{point["offset_s"]:10.3f}  {point["phase"]}')
    print('Adjacent boundary intervals (partition of observed process wall):')
    for interval in value['boundary_intervals']:
        print(f'{interval["wall_s"]:10.3f}  {interval["begin"]} -> {interval["end"]}')
    if value['official_metrics']:
        print('Official callback metrics in this instrumented run:', json.dumps(value['official_metrics']))
    print('Top 30 imports by self time (cumulative is nested, do not sum):')
    for row in value['top_imports']:
        print(f'{row["self_us"]/1000:10.3f} ms self  {row["max_cumulative_us"]/1000:10.3f} ms cumulative  {row["module"]}')
    if not value['ok']:
        print('INCOMPLETE:', json.dumps({k: value[k] for k in
              ('exit_code', 'missing_phases', 'unclosed_spans', 'observer_errors', 'malformed_event_lines')}))


def self_test():
    """CPU-only acceptance checks kept in this owned tool, not root tests/."""
    import io
    from contextlib import redirect_stdout

    assert union_ns([(0, 10), (2, 3), (10, 12), (20, 22)]) == 14
    rows = import_costs('import time: self [us] | cumulative | imported package\n'
                        'import time: 12 | 100 |   pkg\nimport time: 8 | 20 | pkg\n'
                        'import time: 50 | 300 | other\nnoise\n')
    assert rows == [dict(module='other', calls=1, self_us=50, max_cumulative_us=300),
                    dict(module='pkg', calls=2, self_us=20, max_cumulative_us=100)]
    with tempfile.TemporaryDirectory(prefix='soup-startprobe-selftest-') as temporary:
        out = Path(temporary)
        fixture = out/'fixtures'
        fixture.mkdir()
        (fixture/'sitecustomize.py').write_text(
            "import os\nos.environ['START_PROBE_ORIGINAL_SITE_PID'] = str(os.getpid())\n")
        # Fake imports exercise the actual fresh-interpreter sitecustomize and
        # meta loader, including lazy-module replacement and classmethod binding.
        (fixture/'solution.py').write_text('def apply(): return 17\n')
        (fixture/'peft.py').write_text('''import sys
from types import ModuleType
class Lazy(ModuleType):
    def __getattr__(self, name): raise AssertionError('lazy attribute forced')
sys.modules[__name__] = Lazy(__name__)
''')
        (fixture/'torch').mkdir()
        (fixture/'torch/__init__.py').write_text('from . import cuda\n')
        (fixture/'torch/cuda.py').write_text('''ready = False
def is_initialized(): return ready
def _lazy_init():
    global ready
    ready = True
    return 19
''')
        (fixture/'transformers').mkdir()
        (fixture/'transformers/__init__.py').write_text('')
        (fixture/'transformers/trainer_callback.py').write_text('class TrainerCallback: pass\n')
        (fixture/'transformers/modeling_utils.py').write_text('''class PreTrainedModel:
    @classmethod
    def from_pretrained(cls, value): return cls, value
''')
        (fixture/'transformers/trainer.py').write_text('''from .trainer_callback import TrainerCallback
class Trainer:
    def __init__(self): self.callbacks = []
    def add_callback(self, callback): self.callbacks.append(callback)
    def save_model(self, value):
        if value == 'fail': raise ValueError('original failure')
        return value
''')
        script = out/'fixture_train.py'
        fixture_code = '''import os, sys
assert os.environ['START_PROBE_ORIGINAL_SITE_PID'] == str(os.getpid())
sys.path.insert(0, os.environ['START_PROBE_FIXTURE'])
import torch, solution, peft
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer import Trainer
from types import SimpleNamespace
assert solution.apply() == 17
assert torch.cuda._lazy_init() == 19
assert torch.cuda._lazy_init() == 19
class Child(PreTrainedModel): pass
assert Child.from_pretrained(23) == (Child, 23)
t = Trainer()
assert len(t.callbacks) == 1
for step in range(2):
    t.callbacks[0].on_step_begin(None, SimpleNamespace(global_step=step), None)
    t.callbacks[0].on_step_end(None, SimpleNamespace(global_step=step+1), None)
assert t.save_model('keep') == 'keep'
try: t.save_model('fail')
except ValueError as exc: assert str(exc) == 'original failure'
else: raise AssertionError('exception swallowed')
import fractions
import subprocess
subprocess.run([sys.executable, '-c', "import os; assert os.environ['SOUP_START_PROBE_OWNER_PID'] != str(os.getpid()); assert os.environ['START_PROBE_ORIGINAL_SITE_PID'] == str(os.getpid()); import decimal"], check=True)
from multiprocessing import get_context
p = get_context('spawn').Process(target=exec, args=("import sys, os, colorsys; assert 'importtime' not in sys._xoptions; assert os.environ['SOUP_START_PROBE_OWNER_PID'] != str(os.getpid())",))
p.start()
p.join(10)
assert p.exitcode == 0
'''
        script.write_text("if __name__ == '__main__':\n" +
                          ''.join('    '+line+'\n' for line in fixture_code.splitlines()))
        env = dict(os.environ, START_PROBE_FIXTURE=str(fixture),
                   PYTHONPATH=os.pathsep.join([str(fixture), os.environ.get('PYTHONPATH', '')]))
        command = [sys.executable, '-X', 'importtime', str(script)]
        code = execute(command, env, out, 30)
        assert code == 0, (out/'train.log').read_text()
        with redirect_stdout(io.StringIO()):
            result = summarize(out, command=command, commit='self-test', exit_code=code)
        phases = {row['phase']: row for row in result['phases']}
        assert phases['solution_apply']['calls'] == 1
        assert phases['import.peft']['calls'] == 1
        assert phases['cuda_context_init']['calls'] == 1
        assert phases['model_from_pretrained']['calls'] == 1
        assert phases['save_model']['calls'] == 2 and phases['save_model']['errors'] == 1
        assert result['steps'] == 2 and not result['unclosed_spans']
        assert abs(sum(r['wall_s'] for r in result['boundary_intervals']) -
                   next(p['offset_s'] for p in result['timestamps'] if p['phase'] == 'process_end_observed')) < 1e-8
        assert not result['observer_errors'] and not result['ok']  # missing real training phases
        points = {point['phase'] for point in result['timestamps']}
        assert {'first_step_begin', 'last_step_end', 'atexit_late_callback', 'process_end_observed'} <= points
        log = (out/'train.log').read_text()
        assert re.search(r'^import time:.*\| fractions$', log, re.M), 'parent importtime was disabled'
        assert not re.search(r'^import time:.*\| colorsys$', log, re.M), 'child polluted importtime table'
        assert all(event['pid'] in (os.getpid(), result['child_pid'])
                   for event in map(json.loads, (out/'events.jsonl').read_text().splitlines()))
    print('PASS: interval union, import ranking, fresh shim, lazy CUDA init, classmethod binding, '
          'callback steps, output/exception preservation, child isolation, original sitecustomize, atexit/exit timestamps')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'runs_startprobe')
    parser.add_argument('--commit', default='unrecorded')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT/'harness'))
    import run_task as official

    # Reuse official helpers: same cache warm-up, YAML edits, offline env,
    # fixed task seed, allocator and CUDA_VISIBLE_DEVICES=0. No eval run.
    official.warm_up(out)
    config = official.make_config(out, ROOT/'data/train.jsonl', official.load_task()['training']['seed'])
    env = dict(official.OFFLINE_ENV, SOUP_DB_PATH=str(out/'experiments.db'),
               TRITON_CACHE_DIR=str(out/'triton_cache'), TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache'))
    command = [sys.executable, '-X', 'importtime', str(ROOT/'harness/_train.py'),
               '--config', str(config), '--metrics', str(out/'metrics.json'), '--solution']
    (out/'probe_config.json').write_text(json.dumps(dict(
        commit=args.commit, command=command, config=str(config),
        environment={key: env[key] for key in (
            'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'PYTORCH_CUDA_ALLOC_CONF',
            'PYTHONUNBUFFERED', 'TOKENIZERS_PARALLELISM', 'SOUP_NO_AUDIT_LOG', 'HF_HUB_OFFLINE')},
        warmup='official harness.run_task.warm_up; outside instrumented train',
        evaluation=False, config_helper='harness.run_task.make_config',
    ), indent=2)+'\n')
    code = execute(command, env, out, args.timeout)
    result = summarize(out, command=command, commit=args.commit, exit_code=code)
    # Keep measurement logs, never export adapters/cache/model weights.
    import shutil
    for name in ('adapter', 'triton_cache', 'inductor_cache'):
        shutil.rmtree(out/name, ignore_errors=True)
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
