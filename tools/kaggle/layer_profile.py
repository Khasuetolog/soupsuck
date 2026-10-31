#!/usr/bin/env python3
"""Standalone instrumented Soup training; never imported by solution or the jury.

The driver uses official config/warmup/process helpers. Only its worker process
installs wrappers. CUDA events measure regions, not hardware utilization: the
uncovered wall residual is NOT proof of GPU idle. No module hooks are added.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_META = {"family": "other", "phase": "forward", "layer": "global", "projection": "none"}
_PROBE_OWNER = '_layer_profile_owner'
_MISSING = object()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def interval_union(intervals):
    total, end = 0.0, None
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if end is None or left >= end:
            total += right - left
        elif right > end:
            total += right - end
        end = max(end, right) if end is not None else right
    return total


def exclusive_intervals(spans, wall_ms):
    """Assign each stream instant to its deepest timed child, never its parent.

    Stream totals may overlap each other. Return the cross-stream union
    separately so wall residuals never subtract nested or parallel time twice.
    """
    streams, all_intervals = defaultdict(list), []
    for index, span in enumerate(spans):
        left, right = max(0.0, span['start_ms']), min(wall_ms, span['end_ms'])
        if right > left:
            streams[span['stream']].append((left, right, index))
            all_intervals.append((left, right))
    exclusive = defaultdict(float)
    parent_categories = {'bnb.linear.forward.total', 'bnb.matmul.forward', 'bnb.matmul.backward',
                         'bnb.fallback.total', 'mlp.total.forward', 'mlp.total.backward',
                         'mlp.offload.store', 'mlp.offload.prefetch', 'mlp.offload.tensors'}
    for intervals in streams.values():
        boundaries = defaultdict(lambda: [[], []])
        for left, right, index in intervals:
            boundaries[left][1].append(index)
            boundaries[right][0].append(index)
        active, previous = set(), None
        for point in sorted(boundaries):
            if active and previous is not None:
                # Native autograd can run in another Python Context. Prefer a
                # specific operation to its aggregate even if depth resets.
                winner = max(active, key=lambda i: (spans[i]['category'] not in parent_categories,
                                                     spans[i]['depth'], i))
                exclusive[winner] += point - previous
            ended, started = boundaries[point]
            active.difference_update(ended)
            active.update(started)
            previous = point
    return dict(exclusive), interval_union(all_intervals)


def row_key(span):
    return tuple(span[k] for k in ('category', 'phase', 'projection', 'layer'))


def group_spans(spans, exclusive, *, detailed=True):
    groups = {}
    for index, span in enumerate(spans):
        key = row_key(span) if detailed else row_key(span)[:3]
        row = groups.setdefault(key, dict(category=key[0], phase=key[1], projection=key[2],
                                          layer=key[3] if detailed else 'all', calls=0,
                                          inclusive_ms=0.0, exclusive_stream_ms=0.0))
        row['calls'] += 1
        row['inclusive_ms'] += span['end_ms'] - span['start_ms']
        row['exclusive_stream_ms'] += exclusive.get(index, 0.0)
    return list(groups.values())


class Recorder:
    def __init__(self, torch):
        self.torch = torch
        self.active = False
        self.step = 0
        self.meta = ContextVar('layerprof_meta', default=DEFAULT_META)
        self.stack = ContextVar('layerprof_stack', default=())
        self.names, self.records, self.steps = {}, [], []
        self.node_handles, self.patches, self.optimizer_handles = [], [], []
        self.errors, self.installed = [], []
        self.shapes = defaultdict(int)
        self.instrumented = False

    @contextmanager
    def scope(self, **updates):
        token = self.meta.set(dict(self.meta.get(), **updates))
        try:
            yield
        finally:
            self.meta.reset(token)

    @contextmanager
    def span(self, category, stream=None):
        if not self.active:
            yield
            return
        torch = self.torch
        stream = stream if stream is not None else torch.cuda.current_stream()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stack = self.stack.get()
        record = dict(self.meta.get(), category=category, depth=len(stack),
                      stream=int(stream.cuda_stream), start=start, end=end)
        token = self.stack.set(stack + (category,))
        start.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            self.records.append(record)
            self.stack.reset(token)

    def patch(self, obj, name, wrapper, *, static=False):
        original = getattr(obj, name)
        owner = getattr(original, _PROBE_OWNER, None)
        if owner is self:
            return False
        if owner is not None:
            raise RuntimeError(f'{name} is already instrumented by another recorder; restore it first')
        # Preserve the actual descriptor/inheritance, not only its bound value.
        self.patches.append((obj, name, vars(obj).get(name, _MISSING)))
        setattr(wrapper, _PROBE_OWNER, self)
        setattr(obj, name, staticmethod(wrapper) if static else wrapper)
        self.installed.append(f"{getattr(obj, '__name__', type(obj).__name__)}.{name}")
        return True

    def wrap_function(self, obj, name, category, *, family=None):
        original = getattr(obj, name)
        recorder = self

        @wraps(original)
        def wrapped(*args, **kwargs):
            if family is not None and recorder.meta.get()['family'] != family:
                return original(*args, **kwargs)
            with recorder.span(category):
                return original(*args, **kwargs)
        self.patch(obj, name, wrapped)

    def discover(self, model):
        self.names = {id(module): name for name, module in model.named_modules()}

    def module_meta(self, module):
        path = self.names.get(id(module), 'unmapped')
        match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', path)
        projection = next((p for p in ('q', 'k', 'v', 'o', 'gate', 'up', 'down')
                           if re.search(r'(?:^|\.)' + p + r'_proj(?:\.|$)', path)), 'none')
        return dict(layer=match.group(1) if match else 'global', projection=projection)

    def graph_region(self, output, inputs, category, meta):
        """Tag native backward nodes up to exact input GradientEdges.

        No Tensor/module hooks or value copies; no graph/Tensor is retained in
        the recorder. RemovableHandles hold weak hook-dict references only.
        """
        if not self.active or not self.torch.is_grad_enabled() or output.grad_fn is None:
            return
        graph = self.torch.autograd.graph
        cuts = {graph.get_gradient_edge(x).node for x in inputs
                if isinstance(x, self.torch.Tensor) and x.requires_grad}
        seen, pending = set(), [output.grad_fn]
        while pending:
            node = pending.pop()
            if node is None or node in cuts or node in seen:
                continue
            seen.add(node)
            if len(seen) > 256:
                raise RuntimeError(f'native autograd boundary unexpectedly large: {category}')
            if 'AccumulateGrad' in node.name():
                continue
            pending.extend(next_node for next_node, _ in node.next_functions)
            pending_events = []

            def pre(_grad_outputs, pending_events=pending_events):
                # Autograd may invoke pre/post via different Python thread
                # contexts. Do not carry a ContextVar token between callbacks.
                if not self.active:
                    pending_events.append(None)
                    return
                stream = self.torch.cuda.current_stream()
                start = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)
                record = dict(meta, category=category, phase='backward',
                              depth=len(self.stack.get()) + 1,
                              stream=int(stream.cuda_stream), start=start, end=end)
                start.record(stream)
                pending_events.append((record, stream))

            def post(_grad_inputs, _grad_outputs, pending_events=pending_events):
                pending = pending_events.pop()
                if pending is not None:
                    record, stream = pending
                    record['end'].record(stream)
                    self.records.append(record)

            self.node_handles.extend((node.register_prehook(pre), node.register_hook(post)))

    def begin(self, step):
        self.step = step
        self.torch.cuda.synchronize()
        self.wall_start = time.perf_counter()
        self.records = []
        self.active = 8 <= step <= 24
        self.anchor = self.torch.cuda.Event(enable_timing=True) if self.active else None
        if self.anchor is not None:
            self.anchor.record()

    def finish(self):
        self.torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - self.wall_start) * 1000
        self.last_wall_ms = wall_ms
        self.active = False
        if 8 <= self.step <= 24:
            spans = []
            for record in self.records:
                row = {k: v for k, v in record.items() if k not in ('start', 'end')}
                row.update(start_ms=self.anchor.elapsed_time(record['start']),
                           end_ms=self.anchor.elapsed_time(record['end']))
                spans.append(row)
            exclusive, union_ms = exclusive_intervals(spans, wall_ms)
            stream_ms = sum(exclusive.values())
            self.steps.append(dict(step=self.step, wall_ms=wall_ms, union_event_span_ms=union_ms,
                                   exclusive_stream_sum_ms=stream_ms,
                                   raw_wall_minus_stream_sum_ms=wall_ms-stream_ms,
                                   unattributed_or_idle_ms=max(0.0, wall_ms-union_ms),
                                   gpu_idle_exact_ms=None,
                                   rows=group_spans(spans, exclusive),
                                   stream_ids=sorted({r['stream'] for r in spans})))
        for handle in self.node_handles:
            handle.remove()
        self.node_handles.clear()
        self.records.clear()

    def restore(self):
        self.active = False
        for handle in self.node_handles + self.optimizer_handles:
            handle.remove()
        self.node_handles.clear()
        self.optimizer_handles.clear()
        while self.patches:
            obj, name, descriptor = self.patches.pop()
            if descriptor is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, descriptor)
        self.instrumented = False


def materialize_mlp(rec, qwen, mlp):
    """Resolve solution.patch's identity-sensitive lazy trampoline first.

    Its CUDA branch checks cls.forward is the trampoline before installation,
    then dispatches via cls.forward. An outer probe wrapper breaks that check
    and causes wrapper -> trampoline -> wrapper recursion on the first batch.
    This invokes the same installer, without a dummy forward or CUDA warmup.
    """
    cls = qwen.Qwen2MLP
    forward = cls.forward
    if getattr(forward, _PROBE_OWNER, None) is not None:
        raise RuntimeError('Qwen2MLP is already instrumented; restore the owning recorder first')
    # wraps() deliberately reports Qwen2's module/name; inspect the actual code
    # object instead of decorator metadata (also covered by the CPU regression).
    code = getattr(forward, '__code__', None)
    if (code is not None and code.co_qualname == '_patch_lazy_mlp.<locals>.forward'
            and Path(code.co_filename).resolve() == ROOT / 'solution/patch.py'):
        original = getattr(forward, '_solution_fused_original')
        descriptor = vars(cls).get('forward', _MISSING)
        cls.forward = original
        try:
            if not mlp.patch_qwen2_mlp_fused():
                raise RuntimeError('failed to materialize the solution fused MLP')
        except BaseException:
            if descriptor is _MISSING:
                delattr(cls, 'forward')
            else:
                cls.forward = descriptor
            raise
        rec.patches.append((cls, 'forward', descriptor))
        rec.installed.append('Qwen2MLP.materialize_lazy')


def instrument(rec):
    """Install after solution's post-import patches, before constructing the model."""
    if rec.instrumented:
        return False
    torch = rec.torch
    bnb = importlib.import_module('bitsandbytes')
    backend = importlib.import_module('bitsandbytes.backends.cuda.ops')
    functions = importlib.import_module('bitsandbytes.autograd._functions')
    qwen = importlib.import_module('transformers.models.qwen2.modeling_qwen2')
    mlp = importlib.import_module('solution.mlp')
    materialize_mlp(rec, qwen, mlp)

    def module_wrapper(cls, family, category=None, *, lora=False):
        original = cls.forward

        @wraps(original)
        def forward(module, *args, **kwargs):
            path = rec.names.get(id(module), '')
            if lora and not any('.' + part + '.' in '.' + path + '.' for part in ('lora_A', 'lora_B')):
                return original(module, *args, **kwargs)
            role = ('A' if '.lora_A.' in '.' + path + '.' else 'B') if lora else None
            actual_category = 'lora.' + role + '.forward' if lora else category
            if rec.active and args and isinstance(args[0], torch.Tensor):
                shape_key = (family, role, rec.names.get(id(module), 'unmapped'),
                             tuple(args[0].shape), str(args[0].dtype))
                rec.shapes[shape_key] += 1
            with rec.scope(family=family, phase='forward', **rec.module_meta(module)):
                with rec.span(actual_category) if actual_category else null_span():
                    output = original(module, *args, **kwargs)
                if lora and rec.active:
                    rec.graph_region(output, (args[0], module.weight), 'lora.' + role + '.backward',
                                     dict(rec.meta.get()))
                return output
        rec.patch(cls, 'forward', forward)

    module_wrapper(bnb.nn.Linear4bit, 'bnb', 'bnb.linear.forward.total')
    module_wrapper(torch.nn.Linear, 'lora', lora=True)
    module_wrapper(qwen.Qwen2Attention, 'attention')
    module_wrapper(qwen.Qwen2DecoderLayer, 'decoder')
    module_wrapper(qwen.Qwen2MLP, 'mlp')
    module_wrapper(qwen.Qwen2RMSNorm, 'norm')

    def custom_function(cls, family, category):
        original_f, original_b = cls.forward, cls.backward

        @wraps(original_f)
        def forward(ctx, *args, **kwargs):
            ctx._layer_profile_meta = dict(rec.meta.get(), family=family)
            with rec.scope(**dict(ctx._layer_profile_meta, phase='forward')):
                with rec.span(category + '.forward'):
                    return original_f(ctx, *args, **kwargs)

        @wraps(original_b)
        def backward(ctx, *args, **kwargs):
            with rec.scope(**dict(ctx._layer_profile_meta, phase='backward')):
                with rec.span(category + '.backward'):
                    return original_b(ctx, *args, **kwargs)
        rec.patch(cls, 'forward', forward, static=True)
        rec.patch(cls, 'backward', backward, static=True)

    custom_function(functions.MatMul4Bit, 'bnb', 'bnb.matmul')
    custom_function(mlp._FrozenMLP, 'mlp', 'mlp.total')
    custom_function(importlib.import_module('solution.kernels.cross_entropy')._FrozenLinearCrossEntropy,
                    'ce', 'fused_ce')
    custom_function(importlib.import_module('solution.kernels.rmsnorm')._RMSNorm, 'norm', 'norm.rms')
    custom_function(importlib.import_module('solution.kernels.add_rmsnorm')._AddRMSNorm, 'norm', 'norm.add_rms')
    # Optional RoPE keeps its own backward separate from the SDPA core.
    custom_function(importlib.import_module('solution.kernels.rope')._RoPE, 'rope', 'rope')

    for name in ('_dequantize_4bit_impl', '_dequantize_blockwise_impl'):
        rec.wrap_function(backend, name, 'bnb.dequant', family='bnb')
    rec.wrap_function(bnb.functional, 'dequantize_4bit', 'bnb.dequant', family='bnb')
    rec.wrap_function(backend, '_dequant_linear_fallback', 'bnb.fallback.total', family='bnb')
    rec.wrap_function(backend, '_gemm_4bit_kernel_impl', 'bnb.fused_dequant_gemm', family='bnb')
    rec.wrap_function(torch.nn.functional, 'linear', 'bnb.gemm', family='bnb')
    rec.wrap_function(torch, 'matmul', 'bnb.dx_gemm', family='bnb')

    sdpa = torch.nn.functional.scaled_dot_product_attention

    @wraps(sdpa)
    def attention(*args, **kwargs):
        with rec.scope(family='attention'):
            with rec.span('attention.sdpa.forward'):
                output = sdpa(*args, **kwargs)
            if rec.active:
                rec.graph_region(output, args[:3], 'attention.sdpa.backward', dict(rec.meta.get()))
            return output
    rec.patch(torch.nn.functional, 'scaled_dot_product_attention', attention)

    swiglu = importlib.import_module('solution.kernels.swiglu_fused')
    rec.wrap_function(swiglu, 'swiglu_fused_forward', 'mlp.swiglu.forward')
    rec.wrap_function(swiglu, 'swiglu_fused_backward', 'mlp.swiglu.backward')

    store = mlp._OffloadPool.store

    @wraps(store)
    def storing(pool, *args, **kwargs):
        with rec.scope(family='offload'):
            with rec.span('mlp.offload.store'):
                ticket = store(pool, *args, **kwargs)
                ticket._layer_profile_meta = dict(rec.meta.get())
                return ticket
    rec.patch(mlp._OffloadPool, 'store', storing)
    for name in ('prefetch',):
        original = getattr(mlp._OffloadPool, name)

        @wraps(original)
        def prefetch(pool, ticket, *args, **kwargs):
            tag = getattr(ticket, '_layer_profile_meta', rec.meta.get())
            with rec.scope(family='offload', layer=tag['layer'], projection=tag['projection']):
                with rec.span('mlp.offload.prefetch'):
                    return original(pool, ticket, *args, **kwargs)
        rec.patch(mlp._OffloadPool, name, prefetch)
    rec.wrap_function(mlp._OffloadTicket, 'tensors', 'mlp.offload.tensors')

    wait = torch.cuda.Stream.wait_event

    @wraps(wait)
    def wait_event(stream, event):
        if rec.meta.get()['family'] not in ('offload', 'mlp'):
            return wait(stream, event)
        with rec.span('mlp.offload.event_wait', stream=stream):
            return wait(stream, event)
    rec.patch(torch.cuda.Stream, 'wait_event', wait_event)
    copy = torch.Tensor.copy_

    @wraps(copy)
    def copy_(destination, source, *args, **kwargs):
        if rec.meta.get()['family'] == 'offload' and destination.is_cuda != source.is_cuda:
            category = 'mlp.offload.H2D' if destination.is_cuda else 'mlp.offload.D2H'
            with rec.span(category):
                return copy(destination, source, *args, **kwargs)
        return copy(destination, source, *args, **kwargs)
    rec.patch(torch.Tensor, 'copy_', copy_)
    rec.instrumented = True
    return True


@contextmanager
def null_span():
    yield


def summary(rec):
    count = len(rec.steps)
    totals = {}
    total_wall = sum(step['wall_ms'] for step in rec.steps)
    for step in rec.steps:
        for row in step['rows']:
            key = row_key(row)
            target = totals.setdefault(key, dict(row, calls=0, inclusive_ms=0.0, exclusive_stream_ms=0.0))
            for metric in ('calls', 'inclusive_ms', 'exclusive_stream_ms'):
                target[metric] += row[metric]
    rows = list(totals.values())
    projection = {}
    for row in rows:
        key = (row['category'], row['phase'], row['projection'])
        target = projection.setdefault(key, dict(row, layer='all', calls=0, inclusive_ms=0.0, exclusive_stream_ms=0.0))
        for metric in ('calls', 'inclusive_ms', 'exclusive_stream_ms'):
            target[metric] += row[metric]
    for row in rows + list(projection.values()):
        row['ms_per_step'] = row['exclusive_stream_ms'] / count if count else None
        row['inclusive_ms_per_step'] = row['inclusive_ms'] / count if count else None
        row['percent_step_wall'] = 100 * row['exclusive_stream_ms'] / total_wall if total_wall else None
    return dict(measured_steps=count, mean_step_wall_ms=total_wall/count if count else None,
                mean_unattributed_or_idle_ms=sum(s['unattributed_or_idle_ms'] for s in rec.steps)/count if count else None,
                mean_union_event_span_ms=sum(s['union_event_span_ms'] for s in rec.steps)/count if count else None,
                mean_exclusive_stream_sum_ms=sum(s['exclusive_stream_sum_ms'] for s in rec.steps)/count if count else None,
                mean_raw_wall_minus_stream_sum_ms=sum(s['raw_wall_minus_stream_sum_ms'] for s in rec.steps)/count if count else None,
                observed_shapes=[dict(family=k[0], role=k[1], module=k[2], shape=k[3], dtype=k[4], calls=v)
                                 for k, v in rec.shapes.items()],
                per_layer=sorted(rows, key=lambda r: -r['exclusive_stream_ms']),
                by_projection=sorted(projection.values(), key=lambda r: -r['exclusive_stream_ms']))


def print_table(report):
    values = report.get('summary', {})
    print('Layer probe (instrumented, not a jury timing):', values.get('measured_steps'), 'steps')
    print('category | phase | projection | calls | inclusive ms/step | exclusive stream ms/step | % step wall')
    for row in values.get('by_projection', []):
        print(f"{row['category']} | {row['phase']} | {row['projection']} | {row['calls']} | "
              f"{row['inclusive_ms_per_step']:.4f} | {row['ms_per_step']:.4f} | {row['percent_step_wall']:.2f}%")
    print('mean step wall ms:', values.get('mean_step_wall_ms'))
    print('unattributed_or_idle ms (wall - cross-stream event union, NOT exact GPU idle):',
          values.get('mean_unattributed_or_idle_ms'))
    print('raw wall minus sum of stream times ms (can be negative from overlap):',
          values.get('mean_raw_wall_minus_stream_sum_ms'))
    print('Parallel stream percentages can sum above 100%; inclusive rows MUST NOT be added.')


def worker(args):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'harness'))
    from _train import TRAINER_ARGS, plain
    # --verbose is a ROOT Soup option, before train; its error handler would
    # otherwise swallow the Python traceback and only print a friendly line.
    sys.argv = ['soup', '--verbose', 'train', '--config', str(args.config), '--yes']
    sys.modules['unsloth'] = None
    sys.modules['unsloth_zoo'] = None
    report = dict(ok=False, probe=True, commit=args.commit, step_window=[8, 24], step_numbering='1-based, inclusive',
                  errors=[], soup_argv=list(sys.argv),
                  note='CUDA region timings include instrumentation/CPU launch gaps; not jury results')
    rec = None
    path = args.out / 'layer_profile.json'
    save(path, report)
    code = 1
    try:
        import torch
        torch.empty(1, device='cuda')
        import transformers
        import solution
        solution.apply()
        # Finish all post-import hooks before wrapping the selected functions.
        importlib.import_module('soup_cli.cli')
        importlib.import_module('trl')
        importlib.import_module('peft.tuners.lora.bnb')
        importlib.import_module('transformers.models.qwen2.modeling_qwen2')
        rec = Recorder(torch)
        instrument(rec)
        report.update(device=torch.cuda.get_device_name(0), torch=torch.__version__, installed=rec.installed,
                      env={k: os.environ[k] for k in (
                          'SOUP_EXP_PRECOMPILE', 'SOUP_EXP_LAUNCH_TRIM', 'SOUP_EXP_NF4_GEMM',
                          'SOUP_FUSED_DISABLE', 'SOUP_MLP_OFFLOAD', 'SOUP_MLP_RECOMPUTE',
                          'SOUP_MLP_RECOMPUTE_LAYERS', 'SOUP_UNPAD', 'SOUP_FUSED_NORM_FP16') if k in os.environ},
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'), all_step_wall_ms=[])

        class Probe(transformers.TrainerCallback):
            def on_train_begin(self, train_args, state, control, **kwargs):
                rec.discover(kwargs['model'])
                report['modules'] = rec.names.copy()
                optimizer = kwargs.get('optimizer')
                while optimizer is not None and getattr(optimizer, 'optimizer', None) is not None:
                    optimizer = optimizer.optimizer
                if optimizer is None:
                    raise RuntimeError('trainer optimizer unavailable')
                report['optimizer'] = type(optimizer).__module__ + '.' + type(optimizer).__name__
                contexts = []
                def pre(opt, call_args, call_kwargs):
                    ctx = rec.scope(family='optimizer', phase='step', layer='global', projection='none')
                    ctx.__enter__()
                    span = rec.span('optimizer.step')
                    span.__enter__()
                    contexts.append((ctx, span))
                def post(opt, call_args, call_kwargs):
                    ctx, span = contexts.pop()
                    span.__exit__(None, None, None)
                    ctx.__exit__(None, None, None)
                rec.optimizer_handles.extend((optimizer.register_step_pre_hook(pre), optimizer.register_step_post_hook(post)))

            def on_step_begin(self, train_args, state, control, **kwargs):
                rec.begin(int(state.global_step) + 1)

            def on_step_end(self, train_args, state, control, **kwargs):
                rec.finish()
                report['all_step_wall_ms'].append(rec.last_wall_ms)
                if 8 <= rec.step <= 24:
                    report.update(steps=rec.steps, summary=summary(rec))
                    save(path, report)

        original_init = transformers.Trainer.__init__
        @wraps(original_init)
        def init(trainer, *positional, **kwargs):
            original_init(trainer, *positional, **kwargs)
            rec.discover(trainer.model)
            trainer.add_callback(Probe())
            report['trainer'] = {'class':type(trainer).__name__,
                                 **{name:plain(getattr(trainer.args,name,None)) for name in TRAINER_ARGS}}
        rec.patch(transformers.Trainer, '__init__', init)
        try:
            from soup_cli.cli import run
            run()
            code = 0
        except SystemExit as exc:
            code = 0 if exc.code is None else exc.code if isinstance(exc.code,int) else 1
        report.update(exit_code=code, steps=rec.steps, summary=summary(rec))
        present = {row['category'] for row in report['summary']['by_projection']}
        required = {'bnb.linear.forward.total', 'bnb.matmul.backward', 'lora.A.forward', 'lora.A.backward',
                    'lora.B.forward', 'lora.B.backward', 'attention.sdpa.forward', 'attention.sdpa.backward',
                    'fused_ce.forward', 'fused_ce.backward', 'norm.rms.forward', 'norm.rms.backward',
                    'norm.add_rms.forward', 'norm.add_rms.backward', 'optimizer.step',
                    'bnb.dequant', 'bnb.dx_gemm', 'mlp.total.forward', 'mlp.total.backward',
                    'mlp.swiglu.forward', 'mlp.swiglu.backward'}
        if os.environ.get('SOUP_MLP_OFFLOAD') != '0' and os.environ.get('SOUP_MLP_RECOMPUTE') != '1':
            required.update(('mlp.offload.H2D', 'mlp.offload.D2H', 'mlp.offload.event_wait'))
        report['missing_categories'] = sorted(required - present)
        projections = {row['projection'] for row in report['summary']['by_projection']
                       if row['category'] == 'bnb.linear.forward.total'}
        report['missing_categories'].extend('bnb.forward:' + p for p in ('q','k','v','o','gate','up','down')
                                            if p not in projections)
        if not {'bnb.gemm', 'bnb.fused_dequant_gemm'} & present:
            report['missing_categories'].append('bnb.forward:GEMM_or_fused')
        report['ok'] = code == 0 and len(rec.steps) == 17 and not report['missing_categories']
        print_table(report)
    except Exception:
        report['errors'].append(traceback.format_exc())
        print(report['errors'][-1], file=sys.stderr)
    finally:
        if rec is not None:
            rec.restore()
        save(path, report)
    return 0 if report['ok'] else 1


def driver(args):
    sys.path.insert(0, str(ROOT / 'harness'))
    from common import OFFLINE_ENV, run_logged
    import run_task as official
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / 'layer_profile.json'
    if report_path.exists():
        raise ValueError('layer_profile.json already exists; choose a fresh output directory')
    run_dir = args.out / 'layerprof_run'
    run_dir.mkdir(exist_ok=False)
    task = official.load_task()
    config = official.make_config(run_dir, ROOT / 'data/train.jsonl', task['training']['seed'])
    crash_root = ROOT / '.soup-crashes'
    existing_crashes = set(crash_root.glob('*.crash'))
    try:
        official.warm_up(run_dir)
        with tempfile.TemporaryDirectory(prefix='soup-layerprof-cold-') as cache:
            environment = dict(OFFLINE_ENV, SOUP_DB_PATH=str(run_dir / 'experiments.db'),
                               TRITON_CACHE_DIR=str(Path(cache)/'triton'), TORCHINDUCTOR_CACHE_DIR=str(Path(cache)/'inductor'))
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--config', str(config),
                       '--out', str(args.out), '--commit', args.commit]
            code, wall, peak, start = run_logged(command, run_dir / 'train.log', args.timeout, environment)
        report = json.loads(report_path.read_text()) if report_path.exists() else {'ok':False}
        report.update(driver_exit_code=code, driver_wall_s=wall, driver_peak_mib=peak, driver_gpu_start_mib=start,
                      source_task=official.load_task(), command=command,
                      cache_policy='fresh temporary Triton/Inductor directories; removed after this probe')
        report['ok'] = bool(report.get('ok')) and code == 0
        if not report['ok']:
            # The notebook captures THIS process into layer_profile.log. Echo
            # the full worker output, not a tail, so Soup's verbose traceback
            # remains visible there as well as in layerprof_run/train.log.
            print('=== failed layerprof worker: full train.log / traceback ===', flush=True)
            print((run_dir / 'train.log').read_text(errors='replace')
                  if (run_dir / 'train.log').exists() else 'Worker log missing', flush=True)
        save(report_path, report)
        print_table(report)
        return 0 if report['ok'] else 1
    except (Exception, SystemExit):
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        save(report_path, {'ok':False, 'error':error, 'probe':True, 'commit':args.commit})
        return 1
    finally:
        for path in sorted(set(crash_root.glob('*.crash')) - existing_crashes):
            if path.is_file() and not path.is_symlink():
                target = args.out / 'layerprof_crashes' / path.name
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(path, target)
        shutil.rmtree(run_dir / 'adapter', ignore_errors=True)


def self_test():
    """CPU regression using the ACTUAL lazy/concrete installer source bodies.

    Fake pinned APIs replace CUDA math, but the solution's identity checks,
    markers and trampoline code execute unchanged. No torch import is needed.
    """
    import ast
    from types import ModuleType, SimpleNamespace
    from unittest.mock import patch

    def functions(path, names, namespace):
        tree = ast.parse(path.read_text(), filename=str(path))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert len(selected) == len(names)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)

    class Module:
        def forward(self, x):
            return x

    class Linear(Module):
        def forward(self, x):
            return x + 1

    class Linear4bit(Linear):
        def forward(self, x):
            return super().forward(x)

    class Input(int):
        is_cuda = True  # exercise the CUDA *branch*, on a plain CPU integer

    class Tensor:
        def copy_(self, source, *args, **kwargs):
            return self

    class Stream:
        def wait_event(self, event):
            pass

    class Function:
        @staticmethod
        def forward(ctx, x, *args, **kwargs):
            return x

        @staticmethod
        def backward(ctx, grad, *args, **kwargs):
            return grad

    class Ticket:
        def tensors(self):
            return None

    class Pool:
        def store(self, *args, **kwargs):
            return Ticket()

        def prefetch(self, ticket, *args, **kwargs):
            return None

    def identity(x, *args, **kwargs):
        return x

    torch = SimpleNamespace(nn=SimpleNamespace(Module=Module, Linear=Linear,
                                              functional=SimpleNamespace(linear=identity,
                                                                         scaled_dot_product_attention=identity)),
                            Tensor=Tensor, cuda=SimpleNamespace(Stream=Stream), matmul=identity)
    qwen = ModuleType('transformers.models.qwen2.modeling_qwen2')
    for name in ('Qwen2Attention', 'Qwen2DecoderLayer', 'Qwen2MLP', 'Qwen2RMSNorm'):
        setattr(qwen, name, type(name, (Module,), {'__module__': qwen.__name__}))
    qwen.Qwen2MLP.forward = lambda self, x: x + 10
    mlp = ModuleType('solution.mlp')
    calls = []
    concrete_namespace = dict(__name__='solution.mlp', ModuleType=ModuleType, torch=torch,
                              importlib=SimpleNamespace(import_module=lambda name: qwen), wraps=wraps,
                              _MARKER='_solution_fused_original', enabled=lambda: True,
                              _eligible=lambda module, x: True,
                              fused_qwen2_mlp=lambda module, x: calls.append('fused') or x + 20)
    functions(ROOT / 'solution/mlp.py', {'patch_qwen2_mlp_fused'}, concrete_namespace)
    mlp.patch_qwen2_mlp_fused = concrete_namespace['patch_qwen2_mlp_fused']
    lazy_namespace = dict(__name__='solution.patch', wraps=wraps, _MARKER='_solution_fused_original',
                          enabled=lambda: True)
    functions(ROOT / 'solution/patch.py', {'_mark', '_patch_lazy_mlp'}, lazy_namespace)
    lazy_namespace['_patch_lazy_mlp'](qwen.Qwen2MLP)
    lazy = qwen.Qwen2MLP.forward
    package = ModuleType('solution')
    package.__path__ = []
    with patch.dict(sys.modules, {'solution': package, 'solution.mlp': mlp}):
        # Prove the pre-fix chain fails: replacing cls.forward prevents the
        # lazy installer from seeing itself, and it redispatches to the probe.
        @wraps(lazy)
        def broken(module, x):
            return lazy(module, x)
        qwen.Qwen2MLP.forward = broken
        try:
            qwen.Qwen2MLP().forward(Input(2))
        except RecursionError:
            pass
        else:
            raise AssertionError('lazy trampoline regression was not reproduced')
        finally:
            qwen.Qwen2MLP.forward = lazy

        mlp._FrozenMLP = type('FrozenMLP', (Function,), {})
        mlp._OffloadPool, mlp._OffloadTicket = Pool, Ticket
        modules = {'bitsandbytes': SimpleNamespace(nn=SimpleNamespace(Linear4bit=Linear4bit),
                                                  functional=SimpleNamespace(dequantize_4bit=identity)),
                   'bitsandbytes.backends.cuda.ops': SimpleNamespace(**{name: identity for name in
                       ('_dequantize_4bit_impl', '_dequantize_blockwise_impl', '_dequant_linear_fallback', '_gemm_4bit_kernel_impl')}),
                   'bitsandbytes.autograd._functions': SimpleNamespace(MatMul4Bit=type('MatMul4Bit', (Function,), {})),
                   qwen.__name__: qwen, 'solution.mlp': mlp,
                   'solution.kernels.swiglu_fused': SimpleNamespace(swiglu_fused_forward=identity, swiglu_fused_backward=identity)}
        for module, name in (('cross_entropy', '_FrozenLinearCrossEntropy'), ('rmsnorm', '_RMSNorm'),
                             ('add_rmsnorm', '_AddRMSNorm'), ('rope', '_RoPE')):
            modules['solution.kernels.' + module] = SimpleNamespace(**{name: type(name, (Function,), {})})
        recorder = Recorder(torch)
        original_linear, original_bnb = Linear.forward, Linear4bit.forward
        with patch.object(importlib, 'import_module', side_effect=lambda name: modules[name]):
            try:
                assert instrument(recorder) is True
                installed = (qwen.Qwen2MLP.forward, Linear4bit.forward, Linear.forward)
                count = len(recorder.patches)
                assert instrument(recorder) is False
                assert installed == (qwen.Qwen2MLP.forward, Linear4bit.forward, Linear.forward)
                assert len(recorder.patches) == count
                lazy_namespace['_patch_lazy_mlp'](qwen.Qwen2MLP)
                assert mlp.patch_qwen2_mlp_fused() is True
                assert installed == (qwen.Qwen2MLP.forward, Linear4bit.forward, Linear.forward)
                assert qwen.Qwen2MLP().forward(Input(2)) == 22 and calls == ['fused']
                concrete_namespace['_eligible'] = lambda module, x: False
                assert qwen.Qwen2MLP().forward(Input(2)) == 12  # original fallback
                assert Linear4bit().forward(2) == 3  # preserved super/Linear chain
                assert qwen.Qwen2Attention().forward(2) == 2
                try:
                    instrument(Recorder(torch))
                except RuntimeError as exc:
                    assert 'already instrumented' in str(exc)
                else:
                    raise AssertionError('nested recorder installation accepted')
            finally:
                recorder.restore()
            recorder.restore()  # double cleanup is harmless
            assert qwen.Qwen2MLP.forward is lazy
            assert Linear.forward is original_linear and Linear4bit.forward is original_bnb
            # Reinstall after restoration, then exercise the same lazy chain.
            concrete_namespace['_eligible'] = lambda module, x: True
            try:
                assert instrument(recorder) is True
                assert qwen.Qwen2MLP().forward(Input(2)) == 22
            finally:
                recorder.restore()
    print('PASS: reproduced lazy recursion; double install, real solution installer/marker chain, '
          'fused/fallback calls, Linear4bit super chain, owner rejection, restore/reinstall (CPU only)')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true', help='CPU-only lazy-wrapper/idempotence regression')
    parser.add_argument('--commit', default='unknown')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--config', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.out is None:
        parser.error('--out is required unless --self-test')
    args.out = args.out.resolve()
    if args.worker and args.config is None:
        parser.error('--worker requires --config')
    try:
        return worker(args) if args.worker else driver(args)
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
