"""Native CPU/CUDA attribution without layerprof's Python autograd-node hooks.

Diagnostic only; not imported by solution or used for paired timings. Runs the
unchanged harness training entrypoint with its allocator/device policy. Requires
an explicit empty per-run TRITON_CACHE_DIR, inherited by process precompile.
"""

import argparse
import cProfile
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--first-step', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('TRITON_CACHE_DIR'):
        parser.error('set TRITON_CACHE_DIR to a fresh directory for this probe run')
    sys.path[:0] = [str(ROOT), str(ROOT / 'harness')]
    from common import ALLOC_CONF
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ALLOC_CONF
    import torch
    import transformers
    import yaml

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    config['output'] = str(args.out / 'adapter')
    config_path = args.out / 'task.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    python_profile = cProfile.Profile()
    first, last = (1, 1) if args.first_step else (9, 10)
    original_linear = torch.nn.Linear.forward
    @wraps(original_linear)
    def linear(self, x):
        if self.weight.requires_grad and (self.in_features == 16 or self.out_features == 16):
            name = 'exp008.lora.A' if self.out_features == 16 else 'exp008.lora.B'
            with torch.profiler.record_function(name):
                return original_linear(self, x)
        return original_linear(self, x)
    torch.nn.Linear.forward = linear  # class wrapper, no module/Tensor hooks

    def collect(prof):
        prof.export_chrome_trace(str(args.out / 'trace.json'))
        rows = []
        for event in prof.key_averages(group_by_input_shape=True):
            key = event.key
            family = ('casts' if key in ('aten::to', 'aten::_to_copy', 'aten::copy_') else
                      'host_sync' if 'Synchronize' in key or key in ('aten::item', 'aten::_local_scalar_dense') else
                      'sdpa' if 'scaled_dot_product' in key else
                      'optimizer' if 'Optimizer.step' in key else
                      'autograd' if 'autograd::' in key or 'Backward' in key else
                      'gemm' if key in ('aten::mm', 'aten::addmm', 'aten::bmm') else
                      'elementwise' if key in ('aten::add', 'aten::add_', 'aten::mul', 'aten::mul_',
                                               'aten::div', 'aten::sum', 'aten::pow') else 'other')
            rows.append(dict(key=key, family=family, calls=event.count,
                input_shapes=event.input_shapes, self_cpu_us=event.self_cpu_time_total,
                cpu_us=event.cpu_time_total, self_device_us=event.self_device_time_total,
                device_us=event.device_time_total))
        (args.out / 'operators.json').write_text(json.dumps(rows, indent=2))

    class Steps(transformers.TrainerCallback):
        def on_step_begin(self, train_args, state, control, **kwargs):
            if state.global_step + 1 == first:
                python_profile.enable()
        def on_step_end(self, train_args, state, control, **kwargs):
            if state.global_step == last:
                python_profile.disable()
            prof.step()
    original_init = transformers.Trainer.__init__
    @wraps(original_init)
    def init(self, *pos, **kw):
        bound = inspect.signature(original_init).bind(self, *pos, **kw)
        bound.arguments['callbacks'] = list(bound.arguments.get('callbacks') or ()) + [Steps()]
        return original_init(*bound.args, **bound.kwargs)
    transformers.Trainer.__init__ = init
    schedule = torch.profiler.schedule(wait=0 if args.first_step else 7,
        warmup=0 if args.first_step else 1, active=1 if args.first_step else 2, repeat=1)
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], schedule=schedule, record_shapes=True,
                profile_memory=True, on_trace_ready=collect) as prof:
            sys.argv = [str(ROOT / 'harness/_train.py'), '--solution', '--config', str(config_path),
                        '--metrics', str(args.out / 'metrics.json')]
            runpy.run_path(str(ROOT / 'harness/_train.py'), run_name='__main__')
    finally:
        python_profile.disable()
        python_profile.dump_stats(str(args.out / 'python.pstats'))
        transformers.Trainer.__init__ = original_init
        torch.nn.Linear.forward = original_linear


if __name__ == '__main__':
    main()
