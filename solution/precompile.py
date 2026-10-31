"""Measured-run process preparation by default; no CUDA kernel launches.

Wave2 d5dc84b process+launch trim: wall 67.69 vs 69.55 s, paired delta
-1.80 s, CI [-5.96,-1.60], unchanged eval and passing bitwise gates.
Only SOUP_EXP_PRECOMPILE=0 disables preparation; =1 retains the thread mode.
Every other value, including an unset variable, selects process mode.

In the optional thread mode, Triton 3.6 runtime/jit.py:695-736 obtains the
current CUDA device/stream even for warmup, initializing CUDA before model
loading completes. warmup=True skips kernel.run; _do_compile:826-854
inserts the binary in the ordinary JITFunction.device_caches[device][0].
MockTensor supplies pointer dtype/alignment without allocating device tensors.
All files use the caller's current TRITON_CACHE_DIR; no persistent cache is
created, copied, restored or selected here. Compilation time counts in the run.

Plans cover Qwen2.5-1.5B signatures (H1536/I8960, sm75, fp16/fp32), not every
possible public API input. Runtime values represent ==1/alignment classes;
they are not constexpr batch sizes. New signatures compile normally. Preparing
a variant does not alter launch options, model values or tensor layouts.

Pinned pairs are allocated on a second thread and handed to the existing MLP
HostSlot factory as same-shape views. The pool still selects its own capacity,
streams, events and lifetime policy. Only unconsumed host reservations remain
here; no staging CUDA tensors are preallocated and no DMA is submitted.
Errors disable only the failed preparation operation and are reported; actual
training/kernel errors still propagate from the unchanged production path.

SOUP_EXP_PRECOMPILE=process uses a fresh multiprocessing spawn child and an
explicit sm75 target. It builds ASTSource via Triton's native argument binder
and _pack_args, then compiles into the INHERITED run TRITON_CACHE_DIR. No child
CUDA context, warmup/launch, tensors, pinned buffers or device cache are used.
The parent loads these binaries normally on first use. No cache directory is
created outside the measured run or carried between runs. Non-daemon children
are joined by multiprocessing at normal parent exit, so their time counts.

EXP-010: SOUP_EXP_CPU_LAUNCHERS=1 also constructs native CPU CudaLauncher
objects in the existing process compiler, using each binary's exact source
and metadata. The constructor builds/imports CPU glue only; it does not access
CompiledKernel.run/_init_handles or load cubins. CUDA and Triton driver state
must still remain uninitialized in the child. The normal first launch reads
the current run's native launcher cache and initializes its own GPU handles.
No parent JIT entries, tensors, streams, kernels or compiler options change.
"""

from dataclasses import dataclass
import ast
from functools import lru_cache
import importlib
import os
from pathlib import Path
import sys
import threading
import time


@dataclass(frozen=True)
class Plan:
    module: str
    kernel: str
    args: tuple
    options: dict


@lru_cache(None)
def _launcher_options(module, kernel):
    """Read ordered compiler keywords from the actual production launch sites.

    Triton 3.6 compute_cache_key uses str(options), including insertion order
    and explicitly passed defaults. In particular CE omits enable_fp_fusion,
    and LoRA omits num_warps. Backend-equivalent options are NOT cache-equal.
    Kernel constexpr kwargs are bound separately and aren't compiler options.
    """
    path = Path(__file__).parent / 'kernels' / (module + '.py')
    tree = ast.parse(path.read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == kernel)
    parameters = {p.arg for p in definition.args.args}
    sites = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == kernel):
            keywords = []
            for kw in node.keywords:
                if kw.arg is None:
                    raise RuntimeError(f'{kernel}: dynamic launch keywords need explicit support')
                if kw.arg not in parameters:
                    # Values like num_warps=warps are resolved by the known
                    # H1536/BLOCK2048 plan; literals come directly from source.
                    value = ast.literal_eval(kw.value) if isinstance(kw.value, ast.Constant) else None
                    keywords.append((kw.arg, value))
            sites.append(tuple(keywords))
    if not sites or any(site != sites[0] for site in sites):
        raise RuntimeError(f'{kernel}: no unique production compiler-option sequence')
    return sites[0]


def plans():
    """Shape-only descriptors; safe to enumerate without torch, Triton or CUDA."""
    f, h, i, u = '*float32', '*float16', '*int64', '*uint8'
    result = []

    def add(module, kernel, args, **options):
        options.setdefault('num_warps', 4)
        options.setdefault('enable_fp_fusion', False)
        compile_options = _launcher_options(module, kernel)
        compiler_names = {'num_warps', 'enable_fp_fusion'}
        # Preserve kernel kwargs (e.g. BACKWARD), then source compiler keyword
        # order. JIT's binder removes kernel kwargs before constructing its key.
        exact = {k: v for k, v in options.items() if k not in compiler_names}
        for key, literal in compile_options:
            exact[key] = options[key] if literal is None else literal
        result.append(Plan('solution.kernels.' + module, kernel, tuple(args), exact))

    # Frozen gamma in training; optional dgamma entries cover all own JITs too.
    for output in (f, h):
        add('rmsnorm', 'rmsnorm_forward_kernel',
            (f, f, output, f, 1536, 1, 1536, 1e-6, 2048))
        add('rmsnorm', 'rmsnorm_input_gradient_kernel',
            (f, f, output, f, f, 1536, 1536, 1, 1, 1536, 2048))
        add('add_rmsnorm', 'add_rmsnorm_forward_kernel',
            (f, h, f, f, output, f, 1, 1536, 1e-6, 2048))
        for need_r in (False, True):
            add('add_rmsnorm', 'add_rmsnorm_backward_kernel',
                (f, f, f, f, output, f, h, 1536, 1, 1536, 1, 1,
                 1536, 2048, True, True, need_r, True))

    for n in (800, 801):
        for grad_type in (h, f):
            add('rmsnorm', 'rmsnorm_weight_partials_kernel',
                (f, grad_type, f, f, n, 1536, 1536, 1, 1536, False, 32, 128))
        add('cross_entropy', 'vocabulary_outputs_kernel',
            (f, i, f, f, f, n, -100, 256))
        for dtype in (h, f):
            for backward in (False, True):
                source = h if backward else dtype
                # Production passes source as unused A during backward.
                add('swiglu_fused', 'swiglu_tile_engine',
                    (source, source, source if backward else dtype, dtype, dtype, dtype,
                     n, 8960, 8960, 1, 8960, 1, 8960, 1, 8960, 1, 8960, 1, 8960, 1),
                    BACKWARD=backward, ROWS=4, COLS=256)
        for batch in (1, 4):
            for qtype, ctype in ((h, f), (h, h), (f, f)):
                out = f if f in (qtype, ctype) else h
                prefix = (qtype, qtype, ctype, ctype, out, out,
                          n, batch, 12, 2, 64, batch)
                # Transposed projection view. Contiguous head-major strides
                # have the same alignment signature (all multiples of 16).
                strides = (12*n*128, 128, 12*128, 2*n*128, 128, 2*128,
                           n*128, 128, n*128, 128)
                add('rope', 'rope_pair_forward_kernel', prefix + strides, ROWS=4, WIDTH=64)
                gstrides = (12*n*128, 128, 12*128, 1, 2*n*128, 128, 2*128, 1,
                            n*128, 128, n*128, 128)
                add('rope', 'rope_pair_backward_kernel',
                    (out, out, ctype, ctype, qtype, qtype, n, batch, 12, 2, 64, batch)
                    + gstrides, NEED_Q=True, NEED_K=True, ROWS=4, WIDTH=64)

    for groups in (25, 64):
        add('rmsnorm', 'rmsnorm_weight_finalize_kernel',
            (f, f, groups, 1536, 128, 32))
    # N doesn't enter these row kernels' signature. Both vocabulary widths
    # (4096 and the 384 tail) and Start=0/4096 share alignment classes.
    for first in (False, True):
        add('cross_entropy', 'vocabulary_statistics_kernel',
            (h, i, f, i, 4096, 0 if first else 4096, -100, first, 4096))
    add('cross_entropy', 'vocabulary_statistics_kernel',
        (h, i, f, i, 151936 % 4096, 37*4096, -100, False, 512))
    add('cross_entropy', 'vocabulary_derivative_kernel',
        (h, i, f, f, f, h, 4096, 4096, -100, 256))
    for base in (h, f):
        for cast in (False, True):
            add('lora_epilogue', 'lora_scale_add_kernel',
                (base, h, base, 800*1536, 2.0, cast, 512))
        add('lora_epilogue', 'lora_scale_gradient_kernel', (base, h, 800*1536, 2.0, 512))
    for nested in (False, True):
        add('nf4', 'nf4_decode_kernel',
            (u, u if nested else f, f, f, f, f, h, 1536*8960, nested, 2048))
        if os.environ.get('SOUP_EXP_NF4_FAST') == '1':
            add('nf4', 'nf4_decode_fast_kernel',
                (u, u if nested else f, f, f, f, f, h, 1536*8960, nested, 4096), num_warps=8)
    return result


_LOCK = threading.RLock()
_THREADS = []
_PROCESS = None
_PROCESS_RESULT = None
_STATE = dict(started=False, compiled=0, pinned_pairs=0, errors=[], finished=[])
_PINNED = []


def status():
    """JSON-friendly diagnostics; no synchronization with the GPU."""
    with _LOCK:
        _collect_process()
        return {**_STATE, 'errors': list(_STATE['errors']),
                'finished': list(_STATE['finished']),
                'alive': [t.name for t in _THREADS if t.is_alive()]
                         + ([_PROCESS.name] if _PROCESS is not None and _PROCESS.is_alive() else []),
                'process_pid': _PROCESS.pid if _PROCESS is not None else None,
                'process_exitcode': _PROCESS.exitcode if _PROCESS is not None else None}


def wait(timeout=None):
    """Testing/diagnostics only; install and training never wait for preparation."""
    deadline = None if timeout is None else time.monotonic() + timeout
    if _PROCESS is not None:
        _PROCESS.join(None if deadline is None else max(0, deadline-time.monotonic()))
    for thread in list(_THREADS):
        thread.join(None if deadline is None else max(0, deadline-time.monotonic()))
    return status()


def _error(phase, exc):
    message = f'{phase}: {type(exc).__name__}: {exc}'
    with _LOCK:
        _STATE['errors'].append(message)
    print('[solution] EXP-001 preparation failed: ' + message, file=sys.stderr, flush=True)


def warmup_plan(plan):
    """Populate the production JIT cache using only dtype-shaped mock pointers."""
    import torch
    from triton.runtime.jit import MockTensor

    kernel = getattr(importlib.import_module(plan.module), plan.kernel)
    arguments = tuple(MockTensor(getattr(torch, a[1:]))
                      if isinstance(a, str) and a.startswith('*') else a
                      for a in plan.args)
    return kernel.warmup(*arguments, grid=(1,), **plan.options)


def _offline_compile(prepare_launchers=False):
    """Compile the SAME native specializations for sm75, without driver.active.

    Use explicit backend + create_function_from_signature instead of
    JITFunction.create_binder (which queries the active CUDA device).
    _pack_args and triton.compile are the same methods used by real JIT.run.
    CompiledKernel handles are lazy; never access .run/.function here.
    """
    import torch
    import triton
    from triton import knobs
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, make_backend
    from triton.runtime.jit import MockTensor, create_function_from_signature
    from triton.runtime.driver import driver

    if not triton.__version__.startswith('3.6.'):
        raise RuntimeError('Process preparation supports Triton 3.6.x only')
    if prepare_launchers:
        # Constructor builds/imports CPU glue only. CompiledKernel.run would
        # also load cubins, including unused variants, increasing code memory.
        from triton.backends.nvidia.driver import CudaLauncher
    target = GPUTarget('cuda', 75, 32)
    backend = make_backend(target)
    prepared, launchers, errors = 0, 0, []
    binary_seconds = launcher_seconds = 0.0
    for plan in plans():
        try:
            kernel = getattr(importlib.import_module(plan.module), plan.kernel)
            args = tuple(MockTensor(getattr(torch, a[1:]))
                         if isinstance(a, str) and a.startswith('*') else a for a in plan.args)
            kwargs = dict(plan.options)
            kwargs['debug'] = kwargs.get('debug', kernel.debug) or knobs.runtime.debug
            kwargs['instrumentation_mode'] = knobs.compilation.instrumentation_mode
            binder = create_function_from_signature(kernel.signature, kernel.params, backend)
            bound, specialization, options = binder(*args, **kwargs)
            options, signature, constants, attrs = kernel._pack_args(
                backend, kwargs, bound, specialization, options)
            if prepare_launchers:
                started = time.perf_counter()
            compiled = triton.compile(ASTSource(kernel, signature, constants, attrs),
                                      target=target, options=options.__dict__)
            if prepare_launchers:
                binary_seconds += time.perf_counter() - started
                started = time.perf_counter()
                CudaLauncher(compiled.src, compiled.metadata)
                launcher_seconds += time.perf_counter() - started
                launchers += 1
            prepared += 1
        except Exception as exc:
            errors.append(f'{plan.kernel}: {type(exc).__name__}: {exc}')
    initialized = torch.cuda.is_initialized()  # flag read; does not initialize CUDA
    driver_touched = driver._active is not None or driver._default is not None
    if initialized:
        errors.append('Offline compiler unexpectedly initialized torch CUDA')
    if driver_touched:
        errors.append('Offline compiler unexpectedly accessed an active Triton driver')
    result = dict(compiled=prepared, errors=errors, cuda_initialized=initialized,
                  driver_touched=driver_touched)
    if prepare_launchers:
        result['launchers_prepared'] = launchers
        result['binary_prepare_s'] = round(binary_seconds, 4)
        result['launcher_prepare_s'] = round(launcher_seconds, 4)
    return result


def _process_compile(send, cache_dir):
    """Spawn entrypoint; only inherits this measured run's cache environment."""
    result = dict(compiled=0, errors=[], cuda_initialized=False)
    try:
        if not cache_dir or os.environ.get('TRITON_CACHE_DIR') != cache_dir:
            raise RuntimeError('Child must inherit the exact run TRITON_CACHE_DIR')
        if os.environ.get('SOUP_EXP_CPU_LAUNCHERS') == '1':
            result = _offline_compile(prepare_launchers=True)
            print(f"[solution] EXP-010 CPU launchers prepared={result['launchers_prepared']} "
                  f"binary_s={result['binary_prepare_s']:.4f} "
                  f"launcher_s={result['launcher_prepare_s']:.4f} "
                  f"errors={len(result['errors'])}", file=sys.stderr, flush=True)
        else:
            result = _offline_compile()
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        # Parent never has to poll during training. Keep the single IPC packet
        # below PIPE_BUF so even failure cannot block child exit/parent atexit
        # join on a full pipe; full diagnostics go to the inherited train log.
        for error in result['errors']:
            print('[solution] EXP-005 compiler child: ' + error, file=sys.stderr, flush=True)
        errors = result['errors']
        result['errors'] = [e[:512] for e in errors[:4]]
        if len(errors) > 4:
            result['errors'].append(f'{len(errors)-4} additional errors; see inherited stderr')
        send.send(result)
        send.close()
    if result['errors']:
        raise SystemExit(1)


def _collect_process():
    """Nonblocking result transfer; caller holds _LOCK."""
    global _PROCESS_RESULT
    if _PROCESS_RESULT is None:
        return
    if _PROCESS_RESULT.poll():
        try:
            result = _PROCESS_RESULT.recv()
        except EOFError:
            result = dict(compiled=0, errors=['Compiler child exited without a result'])
        _PROCESS_RESULT.close()
        _PROCESS_RESULT = None
        _STATE['compiled'] = result['compiled']
        _STATE['child_cuda_initialized'] = result.get('cuda_initialized')
        _STATE['child_driver_touched'] = result.get('driver_touched')
        for key in ('launchers_prepared', 'binary_prepare_s', 'launcher_prepare_s'):
            if key in result:
                _STATE[key] = result[key]
        _STATE['errors'].extend(result['errors'])
        _STATE['finished'].append('compile-process')
        for message in result['errors']:
            print('[solution] EXP-005 preparation failed: ' + message, file=sys.stderr, flush=True)


def _compile():
    import torch
    import triton

    if not triton.__version__.startswith('3.6.'):
        raise RuntimeError('EXP-001 supports Triton 3.6.x only')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    with torch.cuda.device(0):
        for plan in plans():
            try:
                warmup_plan(plan)
                with _LOCK:
                    _STATE['compiled'] += 1
            except Exception as exc:
                _error(plan.kernel, exc)


def _pinned():
    import torch
    from . import mlp

    if os.environ.get('SOUP_MLP_OFFLOAD') == '0' or os.environ.get('SOUP_MLP_RECOMPUTE') == '1':
        return
    count = max(0, 28 - min(28, max(0, int(os.environ.get('SOUP_MLP_RECOMPUTE_LAYERS', '0')))))
    original = mlp._HostSlot

    class PreparedHostSlot(original):
        def __init__(self, capacity, width):
            with _LOCK:
                slot = _PINNED.pop() if width == 8960 and capacity <= 2048 and _PINNED else None
            if slot is None:
                super().__init__(capacity, width)
            else:
                self.__dict__.update(slot.__dict__)
                self.gate, self.up = slot.gate[:capacity], slot.up[:capacity]

    # Publish before filling: an early foreground pool can consume ready pairs
    # or allocate normally, never wait for the producer or share one pair twice.
    mlp._HostSlot = PreparedHostSlot
    with torch.cuda.device(0):
        for _ in range(count):
            slot = original(2048, 8960)
            with _LOCK:
                _PINNED.append(slot)
                _STATE['pinned_pairs'] += 1


def _run(name, function):
    try:
        function()
    except Exception as exc:
        _error(name, exc)
    finally:
        with _LOCK:
            _STATE['finished'].append(name)


def start():
    """Start once: process by default, literal 1 for thread, only 0 disables."""
    global _PROCESS, _PROCESS_RESULT
    mode = os.environ.get('SOUP_EXP_PRECOMPILE')
    if mode == '0':
        return False
    mode = '1' if mode == '1' else 'process'
    with _LOCK:
        if _STATE['started']:
            return False
        if mode == 'process' and not os.environ.get('TRITON_CACHE_DIR'):
            _error('process', RuntimeError('Explicit run TRITON_CACHE_DIR is required'))
            return False
        _STATE['started'] = True
        if mode == 'process':
            import multiprocessing

            context = multiprocessing.get_context('spawn')
            receive, send = context.Pipe(duplex=False)
            process = context.Process(target=_process_compile,
                args=(send, os.environ['TRITON_CACHE_DIR']), name='soup-precompile-process', daemon=False)
            try:
                process.start()
            except Exception:
                receive.close()
                send.close()
                _STATE['started'] = False
                raise
            send.close()
            _PROCESS, _PROCESS_RESULT = process, receive
        jobs = (('compile', _compile), ('pinned', _pinned)) if mode == '1' else (('pinned', _pinned),)
        for name, fn in jobs:
            thread = threading.Thread(target=_run, args=(name, fn),
                                      name='soup-precompile-' + name, daemon=True)
            _THREADS.append(thread)
            thread.start()
    return True
