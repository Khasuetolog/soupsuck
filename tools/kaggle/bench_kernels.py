"""Offline T4 launch sweep; never imported by solution or training.

Direct launches of all current @triton.jit entrypoints, synthetic data at real
shapes, no model/data download and no extra kernel libraries. Only candidates
whose EVERY output matches the current launch bit-for-bit are timed/ranked.
Reduction layouts can fail this strict gate even when mathematically correct;
they are rejected rather than granted looser tolerances. CE uses its actual
4096-column vocabulary chunk and 384-column tail, not a full [N,V] buffer.
Optional dgamma partials retain 32-row boundaries so outputs are comparable.

Each candidate compiles/warms outside timing, then gets 20 CUDA-event samples
(median/p10/p90). Stateful CE is reset BEFORE the start event each iteration.
Buffers and reference snapshots are allocated outside event intervals. Results
include every rejection/error and each case's fastest exact candidate, without
changing solution code or producing a training cache. A temporary Triton cache
is discarded on exit. No clocks, allocator settings or model settings change.

Usage: python tools/kaggle/bench_kernels.py --out /kaggle/working/out
EXP-011 sweeps geometry, warp counts and pipeline stages, including both NF4
engines. Production compiler defaults (including CE's implicit fusion) remain
the reference. Fast NF4 also must match the byte decoder. JSON includes case
results, rejection reasons and a table grouped by kernel; no configs are applied.
Local inspection without torch: --list and --self-check (stdlib only).
"""

import argparse
import ast
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
KERNELS = (
    "nf4_decode_kernel", "nf4_decode_fast_kernel", "swiglu_tile_engine",
    "rmsnorm_forward_kernel", "rmsnorm_input_gradient_kernel",
    "rmsnorm_weight_partials_kernel", "rmsnorm_weight_finalize_kernel",
    "add_rmsnorm_forward_kernel", "add_rmsnorm_backward_kernel",
    "vocabulary_statistics_kernel", "vocabulary_outputs_kernel", "vocabulary_derivative_kernel",
    "lora_scale_add_kernel", "lora_scale_gradient_kernel",
    "rope_pair_forward_kernel", "rope_pair_backward_kernel",
)


@dataclass
class Case:
    key: str
    kernel: object
    shape: dict
    args: tuple
    outputs: tuple
    baseline: dict
    candidates: list
    grid: object
    fixed: dict = None
    reset: object = None
    compiler: dict = None
    oracle: object = None

    def launch(self, config):
        return self.kernel[self.grid(config)](*self.args, **(self.fixed or {}),
                                               **(self.compiler or {}), **config)


def configurations(case, stages):
    """Keep the literal production baseline, then sweep geometry x pipeline depth."""
    result = [dict(case.baseline)]
    for geometry in case.candidates:
        for depth in stages:
            config = dict(geometry, num_stages=depth)
            if config not in result:
                result.append(config)
    return result


def flat_grid():
    return [dict(BLOCK=b, num_warps=w) for b, w in
            ((256, 4), (512, 4), (1024, 2), (1024, 4), (2048, 4), (4096, 4), (4096, 8))]


def row_grid(block):
    return [dict(BLOCK=b, num_warps=w) for b in (block, block * 2) for w in (2, 4, 8)]


def cases(torch, triton, rows, modules=None):
    def load(name):
        return modules[name] if modules is not None else importlib.import_module('solution.kernels.' + name)
    ce, norm, add, swiglu, nf4, lora, rope = map(load,
        ('cross_entropy', 'rmsnorm', 'add_rmsnorm', 'swiglu_fused', 'nf4', 'lora_epilogue', 'rope'))
    h, intermediate, vocab = 1536, 8960, 151936

    def random(*shape, dtype=torch.float32):
        return torch.randn(shape, device="cuda", dtype=dtype)

    def empty(*shape, dtype=torch.float32):
        return torch.empty(shape, device="cuda", dtype=dtype)

    def emit(key, kernel, shape, args, outputs, baseline, candidates, grid, fixed=None, reset=None,
             oracle=None):
        baseline = dict(baseline)
        name = kernel.__name__
        compiler = {} if name.startswith('vocabulary_') else dict(enable_fp_fusion=False)
        if name in ('vocabulary_outputs_kernel', 'vocabulary_derivative_kernel') or name.startswith('lora_'):
            baseline.pop('num_warps', None)  # these production launches leave Triton's default implicit
        return Case(key, kernel, shape, tuple(args), tuple(outputs), baseline, candidates, grid,
                    fixed, reset, compiler, oracle)

    for n in rows:
        x, weight = random(n, h), random(h)
        rstd = torch.rsqrt(x.square().mean(-1) + 1e-6)
        for out_type in (torch.float32, torch.float16):
            y = empty(n, h, dtype=out_type)
            stats = empty(n)
            yield emit(f"rms-fwd/N{n}/{out_type}", norm.rmsnorm_forward_kernel,
                       dict(N=n, H=h, x="fp32", weight="fp32", out=str(out_type)),
                       (x, weight, y, stats, h, 1, h, 1e-6), (y, stats),
                       dict(BLOCK=2048, num_warps=4), row_grid(2048), lambda c: (n,))
        grad, dx = random(n, h, dtype=torch.float16), empty(n, h)
        yield emit(f"rms-bwd/N{n}", norm.rmsnorm_input_gradient_kernel, dict(N=n, H=h, grad="fp16"),
                   (x, weight, grad, rstd, dx, h, h, 1, 1, h), (dx,),
                   dict(BLOCK=2048, num_warps=4), row_grid(2048), lambda c: (n,))
        groups = triton.cdiv(n, 32)
        partial, dw = empty(groups, h), empty(h)
        partial_configs = [dict(ROWS=32, COLS=c, num_warps=w) for c in (64, 128, 256) for w in (2, 4, 8)]
        yield emit(f"rms-dgamma-partial/N{n}", norm.rmsnorm_weight_partials_kernel,
                   dict(N=n, H=h, ROWS=32, note="fixed row boundaries; optional trainable gamma"),
                   (x, grad, rstd, partial, n, h, h, 1, h), (partial,),
                   dict(ROWS=32, COLS=128, num_warps=4), partial_configs,
                   lambda c: (groups, triton.cdiv(h, c["COLS"])), fixed=dict(OUT_HALF=False))
        # Stable, independent input for timing finalization (not candidate partials).
        partial.copy_(random(groups, h))
        final_configs = [dict(BLOCK_R=r, BLOCK_C=c, num_warps=4)
                         for r in (64, 128, 256) for c in (16, 32, 64)]
        yield emit(f"rms-dgamma-final/N{n}", norm.rmsnorm_weight_finalize_kernel, dict(N=n, H=h, groups=groups),
                   (partial, dw, groups, h), (dw,), dict(BLOCK_R=128, BLOCK_C=32, num_warps=4),
                   final_configs, lambda c: (triton.cdiv(h, c["BLOCK_C"]),))

        delta, gs = random(n, h, dtype=torch.float16), random(n, h)
        for out_type in (torch.float16, torch.float32):
            summed, y, stats = empty(n, h), empty(n, h, dtype=out_type), empty(n)
            yield emit(f"add-rms-fwd/N{n}/{out_type}", add.add_rmsnorm_forward_kernel,
                       dict(N=n, H=h, residual="fp32", delta="fp16", out=str(out_type)),
                       (x, delta, weight, summed, y, stats, 1, h, 1e-6), (summed, y, stats),
                       dict(BLOCK=2048, num_warps=4), row_grid(2048), lambda c: (n,))
        summed = x + delta
        inverse = torch.rsqrt(summed.square().mean(-1) + 1e-6)
        dr, dd = empty(n, h), empty(n, h, dtype=torch.float16)
        for write_r in (False, True):
            yield emit(f"add-rms-bwd/N{n}/residual_grad{write_r}", add.add_rmsnorm_backward_kernel,
                       dict(N=n, H=h, residual_grad=write_r),
                       (summed, weight, inverse, gs, grad, dr, dd, h, 1, h, 1, 1, h),
                       (dr, dd) if write_r else (dd,),
                       dict(BLOCK=2048, num_warps=4), row_grid(2048), lambda c: (n,),
                       fixed=dict(HAS_GS=True, HAS_GY=True, WRITE_R=write_r, WRITE_D=True))

        for dtype in (torch.float16, torch.float32):
            gate, up = random(n, intermediate, dtype=dtype), random(n, intermediate, dtype=dtype)
            act, da = empty(n, intermediate, dtype=dtype), random(n, intermediate, dtype=dtype)
            # store backward reads compressed half even when outputs were fp32.
            saved_gate, saved_up = gate.half(), up.half()
            dg, du = empty(n, intermediate, dtype=dtype), empty(n, intermediate, dtype=dtype)
            tile_configs = [dict(ROWS=r, COLS=c, num_warps=w) for r, c, w in
                            ((1, 256, 4), (2, 256, 4), (4, 128, 4), (4, 256, 4),
                             (8, 256, 4), (4, 512, 4), (4, 256, 8), (8, 128, 2))]
            for backward in (False, True):
                g, u = (saved_gate, saved_up) if backward else (gate, up)
                yield emit(f"swiglu/{'bwd' if backward else 'fwd'}/N{n}/{dtype}", swiglu.swiglu_tile_engine,
                           dict(N=n, I=intermediate, out=str(dtype), backward=backward),
                           (g, u, act, da, dg, du, n, intermediate,
                            intermediate, 1, intermediate, 1, intermediate, 1,
                            intermediate, 1, intermediate, 1, intermediate, 1),
                           (dg, du) if backward else (act,), dict(ROWS=4, COLS=256, num_warps=4),
                           tile_configs, lambda c: (triton.cdiv(n, c["ROWS"]),
                                                   triton.cdiv(intermediate, c["COLS"])),
                           fixed=dict(BACKWARD=backward))

        targets = torch.randint(0, vocab, (n,), device="cuda")
        targets[::3] = -100
        for width in (4096, vocab % 4096):
            logits = random(n, width, dtype=torch.float16)
            maximum, normalizer = random(n) + 3, random(n).abs() + 10
            seed = torch.stack((maximum, normalizer, normalizer * (maximum - 1), random(n)), dim=1)
            state = empty(n, 4)
            best = torch.empty(n, device="cuda", dtype=torch.int64)

            def reset_state(state=state, seed=seed, best=best):
                state.copy_(seed)
                best.fill_(23)

            for first in (True, False):
                if first and width != 4096:
                    continue  # the vocabulary tail is never the first chunk
                start = 0 if first else (4096 if width == 4096 else vocab - width)
                targets[1::3] = start  # exercise target-hit and ignored paths in every actual chunk
                block = triton.next_power_of_2(width)
                yield emit(f"ce-stats/N{n}/W{width}/first{first}", ce.vocabulary_statistics_kernel,
                           dict(N=n, V=vocab, chunk=width, start=start, first=first),
                           (logits, targets, state, best, width, start), (state, best),
                           dict(BLOCK=block, num_warps=4), row_grid(block), lambda c: (n,),
                           fixed=dict(IGNORE=-100, FIRST=first), reset=reset_state)
            dz = empty(n, width, dtype=torch.float16)
            lse = torch.full((n,), 10.0, device="cuda")
            g = random(n)
            gain = g.abs().clamp_min(1e-30) / 1024
            valid_targets = targets.clamp_min(0)  # production backward has compacted ignored rows
            yield emit(f"ce-derivative/N{n}/W{width}", ce.vocabulary_derivative_kernel,
                       dict(N=n, V=vocab, chunk=width, start=start),
                       (logits, valid_targets, lse, g, gain, dz, width, start), (dz,),
                       dict(BLOCK=256, num_warps=4), flat_grid(),
                       lambda c: (n, triton.cdiv(width, c["BLOCK"])), fixed=dict(IGNORE=-100))
        loss, entropy, lse = empty(n), empty(n), empty(n)
        state.copy_(seed)
        yield emit(f"ce-outputs/N{n}", ce.vocabulary_outputs_kernel, dict(N=n),
                   (state, targets, loss, entropy, lse, n), (loss, entropy, lse),
                   dict(BLOCK=256, num_warps=4), flat_grid(),
                   lambda c: (triton.cdiv(n, c["BLOCK"]),), fixed=dict(IGNORE=-100))

        for width in (1536, 256):
            for base_dtype in (torch.float32, torch.float16):
                base = random(n, width, dtype=base_dtype)
                adapter = random(n, width, dtype=torch.float16)
                out, upstream, branch_grad = empty(n, width, dtype=base_dtype), random(n, width, dtype=base_dtype), empty(n, width, dtype=torch.float16)
                size = n * width
                for backward in (False, True):
                    yield emit(f"lora/{'bwd' if backward else 'fwd'}/N{n}/H{width}/{base_dtype}",
                               lora.lora_scale_gradient_kernel if backward else lora.lora_scale_add_kernel,
                               dict(N=n, width=width, base=str(base_dtype), adapter="fp16", backward=backward),
                               (upstream, branch_grad, size, 2.0) if backward else (base, adapter, out, size, 2.0),
                               (branch_grad,) if backward else (out,), dict(BLOCK=512, num_warps=4), flat_grid(),
                               lambda c: (triton.cdiv(size, c["BLOCK"]),),
                               fixed={} if backward else dict(CAST_TO_BASE=False))

        layouts = [(1, n)] + ([(4, n // 4)] if n % 4 == 0 else [])
        for batch, tokens in layouts:
            for dtype, cdtype in ((torch.float32, torch.float32), (torch.float16, torch.float16),
                                 (torch.float16, torch.float32)):
                q = random(batch, tokens, 12, 128, dtype=dtype).transpose(1, 2)
                k = random(batch, tokens, 2, 128, dtype=dtype).transpose(1, 2)
                angles = random(batch, tokens, 64)
                cos = torch.cat((angles.cos(), angles.cos()), -1).to(cdtype)
                sin = torch.cat((angles.sin(), angles.sin()), -1).to(cdtype)
                out_dtype = torch.promote_types(dtype, cdtype)
                yq, yk = empty(batch, 12, tokens, 128, dtype=out_dtype), empty(batch, 2, tokens, 128, dtype=out_dtype)
                gq = random(batch, tokens, 12, 128, dtype=out_dtype).transpose(1, 2)
                gk = random(batch, tokens, 2, 128, dtype=out_dtype).transpose(1, 2)
                dq, dk = empty(batch, 12, tokens, 128, dtype=dtype), empty(batch, 2, tokens, 128, dtype=dtype)
                rope_configs = [dict(ROWS=r, WIDTH=w, num_warps=warps) for r, w, warps in
                                ((1, 64, 4), (2, 64, 4), (4, 64, 2), (4, 64, 4),
                                 (8, 64, 4), (16, 64, 8), (4, 128, 4))]
                for backward in (False, True):
                    prefix = (gq, gk, cos, sin, dq, dk) if backward else (q, k, cos, sin, yq, yk)
                    strides = (*gq.stride(), *gk.stride()) if backward else (*q.stride()[:3], *k.stride()[:3])
                    yield emit(f"rope/{'bwd' if backward else 'fwd'}/B{batch}/T{tokens}/{dtype}/{cdtype}",
                               rope.rope_pair_backward_kernel if backward else rope.rope_pair_forward_kernel,
                               dict(B=batch, T=tokens, q_heads=12, k_heads=2, D=128, qk=str(dtype), cos=str(cdtype)),
                               (*prefix, tokens, batch, 12, 2, 64, batch, *strides, *cos.stride()[:2], *sin.stride()[:2]),
                               (dq, dk) if backward else (yq, yk), dict(ROWS=4, WIDTH=64, num_warps=4),
                               rope_configs, lambda c: (triton.cdiv(14 * n, c["ROWS"]),),
                               fixed=dict(NEED_Q=True, NEED_K=True) if backward else {})

    for shape in ((1536, 1536), (8960, 1536), (1536, 8960), (256, 1536)):
        count = shape[0] * shape[1]
        blocks = triton.cdiv(count, 64)
        packed = torch.randint(0, 256, (triton.cdiv(count, 2),), device="cuda", dtype=torch.uint8)
        quant_abs = torch.randint(0, 256, (blocks,), device="cuda", dtype=torch.uint8)
        second = torch.rand((triton.cdiv(blocks, 256),), device="cuda") * 0.02 + 0.001
        code = torch.linspace(-1, 1, 256, device="cuda")
        offset = torch.full((), 0.06, device="cuda")
        flat_abs = code[quant_abs.long()] * second[torch.arange(blocks, device="cuda") // 256] + offset
        out = empty(*shape, dtype=torch.float16)
        for nested in (True, False):
            arguments = (packed, quant_abs if nested else flat_abs, second, code, offset,
                         nf4._table(out.device), out, count)
            yield emit(f"nf4/default/{shape[0]}x{shape[1]}/nested{nested}", nf4.nf4_decode_kernel,
                       dict(weight=shape, blocksize=64, nested=nested, state2_blocksize=256),
                       arguments, (out,),
                       dict(BLOCK=2048, num_warps=4), flat_grid(),
                       lambda c: (triton.cdiv(count, c["BLOCK"]),), fixed=dict(NESTED=nested))

            def byte_oracle(arguments=arguments, count=count, nested=nested):
                nf4.nf4_decode_kernel[(triton.cdiv(count, 2048),)](
                    *arguments, NESTED=nested, BLOCK=2048, num_warps=4, enable_fp_fusion=False)

            yield emit(f"nf4/fast/{shape[0]}x{shape[1]}/nested{nested}", nf4.nf4_decode_fast_kernel,
                       dict(weight=shape, blocksize=64, nested=nested, state2_blocksize=256,
                            constraint="BLOCK=4096 is statically required by the current fast JIT"),
                       arguments, (out,), dict(BLOCK=4096, num_warps=8),
                       [dict(BLOCK=4096, num_warps=w) for w in (2, 4, 8)],
                       lambda c: (triton.cdiv(count, c["BLOCK"]),), fixed=dict(NESTED=nested),
                       oracle=byte_oracle)


def bitwise(torch, outputs, expected):
    mismatches = []
    for actual, reference in zip(outputs, expected):
        # Byte views distinguish signed zero and NaN payloads as well as values.
        a, b = actual.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)
        mismatches.append(int(torch.count_nonzero(a != b).item()))
    return mismatches


def measure(torch, case, config, warmup, repeats):
    begin = time.perf_counter()
    for _ in range(warmup):
        if case.reset is not None:
            case.reset()
        case.launch(config)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - begin
    start, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        if case.reset is not None:
            case.reset()
        start.record()
        case.launch(config)
        finish.record()
        finish.synchronize()
        samples.append(start.elapsed_time(finish))
    ordered = sorted(samples)
    if not all(math.isfinite(ms) and ms > 0 for ms in samples):
        raise RuntimeError('CUDA event timing must be finite and positive')
    return dict(median_ms=statistics.median(samples), p10_ms=ordered[int((len(ordered) - 1) * .1)],
                p90_ms=ordered[int((len(ordered) - 1) * .9)], min_ms=min(samples),
                samples_ms=samples, warmup_s=warmup_s)


def resources(compiled):
    result = {}
    for key in ('n_regs', 'n_spills'):
        value = getattr(compiled, key, None)
        if value is not None:
            result[key] = int(value)
    metadata = getattr(compiled, 'metadata', None)
    for key in ('shared', 'global_scratch_size', 'profile_scratch_size'):
        value = getattr(metadata, key, None)
        if value is not None:
            result[key] = int(value)
    return result


def initialize_outputs(case):
    if case.reset is not None:
        case.reset()  # stateful CE reads the reset input/output state
    else:
        for output in case.outputs:
            output.fill_(float('nan') if output.is_floating_point() else -1)


def benchmark_case(torch, case, args, measure_fn=measure, bitwise_fn=bitwise):
    record = dict(case=case.key, kernel=case.kernel.__name__, shape=case.shape,
                  baseline=case.baseline, compiler_options=case.compiler or {}, candidates=[], ok=False,
                  reference='current byte decoder' if case.oracle is not None else 'current configuration')
    initialize_outputs(case)
    initial = time.perf_counter()
    if case.oracle is None:
        case.launch(case.baseline)
    else:
        case.oracle()
    torch.cuda.synchronize()
    record["baseline_first_call_s"] = time.perf_counter() - initial
    reference = tuple(t.detach().clone() for t in case.outputs)
    baseline_resources = None
    for config in configurations(case, args.stages):
        trial = dict(config=config, exact=False, eligible=False)
        begin = time.perf_counter()
        try:
            initialize_outputs(case)
            allocated = torch.cuda.memory_allocated()
            compiled = case.launch(config)
            torch.cuda.synchronize()
            trial["first_validation_call_s"] = time.perf_counter() - begin
            trial['resources'] = resources(compiled)
            trial['tensor_bytes_delta'] = torch.cuda.memory_allocated() - allocated
            if baseline_resources is None:
                baseline_resources = trial['resources']
            growth = [key for key in ('global_scratch_size', 'profile_scratch_size', 'n_spills')
                      if trial['resources'].get(key, 0) > baseline_resources.get(key, 0)]
            trial["mismatch_bytes"] = bitwise_fn(torch, case.outputs, reference)
            trial["exact"] = not any(trial["mismatch_bytes"])
            if not trial['exact']:
                trial["rejected"] = "bitwise mismatch against current configuration"
            elif growth or trial['tensor_bytes_delta'] > 0:
                trial['rejected'] = f"additional memory/spills: {growth}, tensor bytes={trial['tensor_bytes_delta']}"
            else:
                trial.update(measure_fn(torch, case, config, args.warmup, args.repeats))
                trial['post_timing_mismatch_bytes'] = bitwise_fn(torch, case.outputs, reference)
                trial['exact'] = not any(trial['post_timing_mismatch_bytes'])
                trial['eligible'] = trial['exact'] and math.isfinite(trial['median_ms']) and trial['median_ms'] > 0
                if not trial['eligible']:
                    trial['rejected'] = 'post-timing mismatch or invalid event timing'
        except Exception as exc:
            trial["error"] = f"{type(exc).__name__}: {exc}"[:2000]
            # A poisoned context cannot safely benchmark another candidate.
            fatal = ("illegal memory access", "device-side assert", "misaligned address", "launch failure")
            if any(word in str(exc).lower() for word in fatal):
                record["candidates"].append(trial)
                record["fatal_error"] = trial["error"]
                return record
        record["candidates"].append(trial)
        print(json.dumps(dict(case=case.key, **trial)), flush=True)
    eligible = [c for c in record["candidates"] if c.get("eligible")]
    baseline = next((c for c in eligible if c["config"] == case.baseline), None)
    if eligible and baseline:
        best = min(eligible, key=lambda c: c["median_ms"])
        record["best"] = dict(config=best["config"], median_ms=best["median_ms"],
                              baseline_ms=baseline["median_ms"], speedup=baseline["median_ms"] / best["median_ms"],
                              exact=True, resources=best.get('resources', {}),
                              separated_samples=best['p90_ms'] < baseline['p10_ms'])
        record["ok"] = True
    return record


def result_tables(records):
    """Per-shape winners plus one config only when it passes every case of a JIT."""
    table, grouped = [], {}
    lookup = {r['case']: r for r in records}
    for record in records:
        best = record.get('best', {})
        row = dict(kernel=record['kernel'], case=record['case'], shape=record.get('shape', {}),
                   ok=record.get('ok', False), baseline_config=record.get('baseline'),
                   best_config=best.get('config'), baseline_ms=best.get('baseline_ms'),
                   median_ms=best.get('median_ms'), speedup=best.get('speedup'),
                   exact=best.get('exact', False), separated_samples=best.get('separated_samples', False))
        if record['kernel'] == 'nf4_decode_fast_kernel':
            byte = lookup.get(record['case'].replace('/fast/', '/default/'), {}).get('best', {})
            if byte and best:
                row['default_decoder_ms'] = byte['baseline_ms']
                row['speedup_vs_byte_decoder'] = byte['baseline_ms'] / best['median_ms']
        table.append(row)
        grouped.setdefault(record['kernel'], []).append(record)
    summaries = {}
    for kernel, cases_for_kernel in grouped.items():
        common, timings = None, {}
        for case in cases_for_kernel:
            entries = {json.dumps(c['config'], sort_keys=True): c for c in case['candidates']
                       if c.get('eligible')}
            common = set(entries) if common is None else common & entries.keys()
            timings[case['case']] = entries
        summary = dict(cases=len(cases_for_kernel), all_cases_ok=all(c.get('ok') for c in cases_for_kernel),
                       common_config=None, policy='equal-weight geometric mean across all shapes/dtypes/phases')
        if common and summary['all_cases_ok']:
            def relative(key):
                return math.exp(statistics.mean(math.log(
                    timings[c['case']][key]['median_ms'] / c['best']['baseline_ms'])
                    for c in cases_for_kernel))
            winner = min(sorted(common), key=relative)
            summary.update(common_config=json.loads(winner), geometric_mean_speedup=1 / relative(winner))
        else:
            summary['reason'] = 'no shared eligible configuration or a failed baseline; inspect per-case table'
        summaries[kernel] = summary
    return table, summaries


def self_check():
    """Stdlib regressions for baselines, search, strict rejection and result selection."""
    from types import SimpleNamespace

    definitions = {}
    for path in (ROOT / 'solution/kernels').glob('*.py'):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in KERNELS:
                definitions[node.name] = node
    assert set(definitions) == set(KERNELS)
    calls = []
    class Kernel:
        __name__ = 'self_check_kernel'
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((grid, kwargs))
    case = Case('check', Kernel(), {}, (), (), dict(BLOCK=256),
                [dict(BLOCK=256, num_warps=4)], lambda c: (1,), compiler={})
    case.launch(case.baseline)
    assert calls == [((1,), dict(BLOCK=256))]  # no accidental fusion/warp/stage override
    configs = configurations(case, [1, 2, 3])
    assert configs[0] == case.baseline and len(configs) == 4
    assert {c['num_stages'] for c in configs[1:]} == {1, 2, 3}
    case.compiler = dict(enable_fp_fusion=False)
    case.launch(case.baseline)
    assert calls[-1][1]['enable_fp_fusion'] is False

    class Output:
        def __init__(self, value=1):
            self.value = value
        def detach(self):
            return self
        def clone(self):
            return Output(self.value)
        def is_floating_point(self):
            return True
        def fill_(self, value):
            self.value = value
    output = Output()
    class NumericCase:
        key, kernel, shape, fixed, compiler, reset, oracle = 'numeric', Kernel(), {}, None, {}, None, None
        outputs, baseline = (output,), dict(BLOCK=256)
        candidates = [dict(BLOCK=512), dict(BLOCK=1024), dict(BLOCK=2048), dict(BLOCK=4096)]
        def launch(self, config):
            output.value = 2 if config['BLOCK'] == 512 else 1
            return SimpleNamespace(n_spills=2 if config['BLOCK'] == 4096 else 0)
    fake = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, memory_allocated=lambda: 0))
    def timing(torch, case, config, warmup, repeats):
        ms = 0.5 if config['BLOCK'] == 1024 else 1.0
        if config['BLOCK'] == 2048:
            output.value = 2  # correctness must still hold after timing
        return dict(median_ms=ms, p10_ms=ms, p90_ms=ms)
    with redirect_stdout(io.StringIO()):
        record = benchmark_case(fake, NumericCase(), SimpleNamespace(stages=[1], warmup=1, repeats=1),
                                measure_fn=timing,
                                bitwise_fn=lambda torch, actual, expected: [int(actual[0].value != expected[0].value)])
    assert record['ok'] and record['best']['config']['BLOCK'] == 1024
    assert record['best']['speedup'] == 2
    assert next(c for c in record['candidates'] if c['config']['BLOCK'] == 512)['exact'] is False
    assert not next(c for c in record['candidates'] if c['config']['BLOCK'] == 2048)['eligible']
    assert 'memory/spills' in next(c for c in record['candidates'] if c['config']['BLOCK'] == 4096)['rejected']
    table, grouped = result_tables([record])
    assert table[0]['exact'] and grouped['self_check_kernel']['geometric_mean_speedup'] == 2
    other = dict(record, case='other_shape', candidates=[dict(c) for c in record['candidates']],
                 best=dict(record['best'], config=record['baseline'], median_ms=1.0, speedup=1.0))
    for candidate in other['candidates']:
        if candidate['config']['BLOCK'] == 1024:
            candidate['eligible'] = False
    _, grouped = result_tables([record, other])
    assert grouped['self_check_kernel']['common_config'] == record['baseline']
    assert grouped['self_check_kernel']['geometric_mean_speedup'] == 1
    return dict(ok=True, kernels=len(KERNELS), checks=['production keyword preservation',
                'geometry/stage search', 'strict pre/post bitwise rejection', 'spill/memory rejection',
                'per-case and per-kernel winners'],
                shape_audit=shape_audit())


def shape_audit():
    """Execute the real case generator with shape-only tensors; bind every JIT call."""
    from types import SimpleNamespace

    class Tensor:
        def __init__(self, shape, dtype='torch.float32', stride=None):
            self.shape, self.dtype, self.device = tuple(shape), dtype, 'cuda:0'
            self.ndim = len(self.shape)
            self._stride = stride or tuple(math.prod(self.shape[i+1:]) for i in range(self.ndim))
        def stride(self, index=None):
            return self._stride if index is None else self._stride[index]
        def numel(self):
            return math.prod(self.shape)
        def transpose(self, a, b):
            shape, strides = list(self.shape), list(self._stride)
            shape[a], shape[b] = shape[b], shape[a]
            strides[a], strides[b] = strides[b], strides[a]
            return Tensor(shape, self.dtype, tuple(strides))
        def to(self, dtype):
            return Tensor(self.shape, dtype)
        def half(self):
            return self.to('torch.float16')
        def long(self):
            return self.to('torch.int64')
        def mean(self, dim):
            return Tensor(self.shape[:dim] if dim == -1 else self.shape[:dim] + self.shape[dim+1:], self.dtype)
        def __getitem__(self, index):
            return Tensor(index.shape, self.dtype) if isinstance(index, Tensor) else self
        def __setitem__(self, index, value):
            pass
        def __add__(self, other):
            return Tensor(self.shape, self.dtype)
        __sub__ = __mul__ = __truediv__ = __floordiv__ = __add__
        def same(self, *args, **kwargs):
            return Tensor(self.shape, self.dtype)
        square = abs = clamp_min = cos = sin = copy_ = fill_ = same

    def empty(shape, *args, dtype='torch.float32', **kwargs):
        return Tensor((shape,) if isinstance(shape, int) else shape, dtype)
    def stack(tensors, dim):
        shape = list(tensors[0].shape)
        shape.insert(dim, len(tensors))
        return Tensor(shape, tensors[0].dtype)
    def cat(tensors, dim):
        shape = list(tensors[0].shape)
        shape[dim] = sum(t.shape[dim] for t in tensors)
        return Tensor(shape, tensors[0].dtype)
    torch = SimpleNamespace(float16='torch.float16', float32='torch.float32', uint8='torch.uint8',
        int64='torch.int64', randn=empty, rand=empty, empty=empty, full=empty,
        randint=lambda low, high, shape, **kw: empty(shape, dtype=kw.get('dtype', 'torch.int64')),
        arange=lambda count, **kw: Tensor((count,), 'torch.int64'),
        linspace=lambda low, high, count, **kw: Tensor((count,)), stack=stack, cat=cat,
        rsqrt=lambda t: t, promote_types=lambda a, b: 'torch.float32' if 'torch.float32' in (a, b) else a)
    triton = SimpleNamespace(cdiv=lambda n, b: (n + b - 1) // b, next_power_of_2=lambda n: 1 << (n - 1).bit_length())
    modules, signatures = {}, {}
    for path in (ROOT / 'solution/kernels').glob('*.py'):
        module = SimpleNamespace()
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in KERNELS:
                setattr(module, node.name, SimpleNamespace(__name__=node.name))
                signatures[node.name] = inspect.Signature([
                    inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for a in node.args.args])
        modules[path.stem] = module
    modules['nf4']._table = lambda device: Tensor((16,))
    seen, total, trials = set(), 0, 0
    for case in cases(torch, triton, [800, 801, 2048], modules=modules):
        name = case.kernel.__name__
        seen.add(name)
        total += 1
        for config in configurations(case, [1, 2, 3]):
            keywords = {k: v for k, v in dict(case.fixed or {}, **config).items()
                        if k not in ('num_warps', 'num_stages')}
            signatures[name].bind(*case.args, **keywords)
            assert all(value > 0 for value in case.grid(config))
            trials += 1
        if name.startswith('vocabulary_'):
            assert case.compiler == {}
        if name == 'vocabulary_statistics_kernel':
            assert case.baseline['BLOCK'] == triton.next_power_of_2(case.shape['chunk'])
        if name == 'nf4_decode_fast_kernel':
            assert case.oracle is not None and all(c['BLOCK'] == 4096 for c in case.candidates)
    assert seen == set(KERNELS)
    return dict(cases=total, configurations=trials, signatures_bound=True, kernels=len(seen))


def benchmark_nf4_gemm(torch, args, report, save):
    """EXP-002, separate from the existing bitwise launch-geometry sweep.

    Includes decoder+allocation+cuBLAS in each baseline invocation. Weights
    are genuinely bnb nested NF4, not randomly fabricated metadata. A shape
    is recommended only after every candidate passes the same fixed gate.
    This writes recommendations, never a production dispatch table.
    """
    import bitsandbytes as bnb
    from solution.kernels import nf4_gemm as fused
    from solution.kernels.nf4 import dequantize_nf4

    if bnb.__version__ != "0.50.2":
        raise RuntimeError("EXP-002 requires bitsandbytes 0.50.2")
    if torch.cuda.get_device_capability(0) != (7, 5):
        raise RuntimeError("EXP-002 approval is specific to sm75")
    report["kernels"] = ["nf4_tile_matmul_kernel"]
    report["tolerance"] = dict(rtol=0.002, atol=0.002, relative_l2=0.001)
    report["policy"] = "recommend only validated configs with candidate p90 < baseline p10"
    report["production_dispatch_modified"] = False

    def compare(actual, expected):
        delta = actual.float() - expected.float()
        result = dict(max_abs=float(delta.abs().max()),
                      relative_l2=float(delta.norm() / expected.float().norm().clamp_min(1e-30)),
                      exact=bool(torch.equal(actual, expected)), finite=bool(torch.isfinite(actual).all()))
        torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
        if not result["finite"] or result["relative_l2"] > 0.001:
            raise AssertionError(f"GEMM fixed correctness gate failed: {result}")
        return result

    def timed(call):
        begin = time.perf_counter()
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize()
        warmup_s = time.perf_counter() - begin
        samples = []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(args.repeats):
            start.record()
            output = call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
            del output
        ordered = sorted(samples)
        return dict(median_ms=statistics.median(samples), p10_ms=ordered[int((len(samples)-1)*0.1)],
                    p90_ms=ordered[int((len(samples)-1)*0.9)], samples_ms=samples, warmup_s=warmup_s)

    for n, k in ((1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)):
        weight = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.02
        packed, state = bnb.functional.quantize_4bit(
            weight, blocksize=64, compress_statistics=True, quant_type="nf4")
        del weight
        dense = dequantize_nf4(packed.view(-1, 1), state)
        native_dense = bnb.functional.dequantize_4bit(packed.view(-1, 1), state)
        if not torch.equal(dense.view(torch.uint8), native_dense.view(torch.uint8)):
            raise AssertionError("Nested decoder does not match native NF4 weights bitwise")
        del native_dense
        for m in args.rows:
            records = {}
            for transpose, biased in ((False, False), (False, True), (True, False)):
                x = torch.randn((1, m, n if transpose else k), device="cuda", dtype=torch.float16) * 0.1
                bias = torch.randn(n, device="cuda", dtype=torch.float16) * 0.01 if biased else None
                def baseline():
                    decoded = dequantize_nf4(packed.view(-1, 1), state)
                    return torch.matmul(x, decoded) if transpose else torch.nn.functional.linear(x, decoded, bias)
                key = f"nf4-gemm/M{m}/N{n}/K{k}/{'dx' if transpose else 'forward'}/bias{biased}"
                record = dict(case=key, kernel="nf4_tile_matmul_kernel", M=m, N=n, K=k,
                              transpose=transpose, bias=biased, candidates=[], ok=False)
                report["active_case"] = key
                report["cases"].append(record)
                save()
                first = time.perf_counter()
                reference = baseline()
                torch.cuda.synchronize()
                record["baseline_first_call_s"] = time.perf_counter() - first
                # Both baselines and candidates are timed in this same session;
                # no result is inferred by comparing different Kaggle hosts.
                record["baseline"] = timed(baseline)
                if not transpose:
                    # bnb may already select its own fused MMA below M=1536.
                    # A decode+cuBLAS win alone cannot authorize replacing it.
                    native = getattr(bnb.matmul_4bit, fused._MARKER, bnb.matmul_4bit)
                    record["native_bnb"] = timed(lambda: native(x, packed, state, bias=bias))
                for config in fused.CONFIGS:
                    trial = dict(config=config, correct=False)
                    try:
                        first = time.perf_counter()
                        actual = fused.nf4_matmul(x, packed, state, transpose=transpose,
                                                  bias=bias, config=config)
                        torch.cuda.synchronize()
                        trial["first_call_s"] = time.perf_counter() - first
                        trial.update(compare(actual, reference))
                        trial["correct"] = True
                        del actual
                        trial.update(timed(lambda: fused.nf4_matmul(
                            x, packed, state, transpose=transpose, bias=bias, config=config)))
                    except Exception as error:
                        trial["error"] = f"{type(error).__name__}: {error}"[:3000]
                        if any(term in str(error).lower() for term in
                               ("illegal memory access", "device-side assert", "misaligned address", "launch failure")):
                            record["candidates"].append(trial)
                            save()
                            raise
                    record["candidates"].append(trial)
                    print(json.dumps(dict(case=key, **trial)), flush=True)
                    save()
                correct = [r for r in record["candidates"] if r["correct"] and "median_ms" in r]
                if correct:
                    best = min(correct, key=lambda r: r["median_ms"])
                    record["best"] = dict(config=best["config"], median_ms=best["median_ms"],
                                          baseline_ms=record["baseline"]["median_ms"],
                                          speedup=record["baseline"]["median_ms"]/best["median_ms"])
                    record["recommend"] = best["p90_ms"] < record["baseline"]["p10_ms"]
                    if "native_bnb" in record:
                        record["recommend"] &= best["p90_ms"] < record["native_bnb"]["p10_ms"]
                    record["ok"] = True
                    records[(transpose, biased)] = best["config"]
                save()
                del x, reference, bias
            # Verify the actual optional bnb/autograd bridge in BOTH derivative
            # directions. Forced configs exist only in this benchmark process.
            if all(key in records for key in ((False, False), (False, True), (True, False))):
                saved_flag = os.environ.get("SOUP_EXP_NF4_GEMM")
                saved_shapes = dict(fused.APPROVED_SHAPES)
                try:
                    os.environ["SOUP_EXP_NF4_GEMM"] = "1"
                    fused.patch_nf4_matmul(bnb)
                    for fw, bw in ((True, False), (False, True), (True, True)):
                        fused.APPROVED_SHAPES.clear()
                        if fw:
                            fused.APPROVED_SHAPES[(m, n, k, False)] = records[(False, True)]
                        if bw:
                            fused.APPROVED_SHAPES[(m, n, k, True)] = records[(True, False)]
                        x = (torch.randn((1, m, k), device="cuda", dtype=torch.float16) * 0.1).requires_grad_()
                        other = x.detach().clone().requires_grad_()
                        bias = torch.randn(n, device="cuda", dtype=torch.float16) * 0.01
                        expected = torch.nn.functional.linear(x, dense, bias)
                        actual = bnb.matmul_4bit(other, packed, state, bias=bias)
                        grad = torch.randn_like(actual) * 0.1
                        expected.backward(grad)
                        actual.backward(grad)
                        report.setdefault("autograd", []).append(dict(M=m, N=n, K=k, forward=fw,
                            backward=bw, output=compare(actual, expected), dx=compare(other.grad, x.grad)))
                finally:
                    fused.APPROVED_SHAPES.clear()
                    fused.APPROVED_SHAPES.update(saved_shapes)
                    if saved_flag is None:
                        os.environ.pop("SOUP_EXP_NF4_GEMM", None)
                    else:
                        os.environ["SOUP_EXP_NF4_GEMM"] = saved_flag
                save()
        del dense, packed, state
        gc.collect()
    report["ok"] = bool(report["cases"]) and all(r["ok"] for r in report["cases"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--list", action="store_true", help="Print the kernel/shape plan without importing torch")
    parser.add_argument('--self-check', action='store_true', help='Run stdlib correctness/selection checks, no GPU')
    parser.add_argument("--nf4-gemm", action="store_true",
                        default=os.environ.get("SOUP_EXP_NF4_GEMM") == "1",
                        help="EXP-002: fused NF4 GEMM vs decode+cuBLAS (also selected by experiment flag)")
    parser.add_argument("--kernels", default="", help="Optional comma-separated exact kernel names")
    parser.add_argument("--rows", type=int, nargs="+", default=[800, 801, 2048])
    parser.add_argument('--stages', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--commit", default="", help="Verified archive SHA (Kaggle archive has no .git)")
    args = parser.parse_args()
    selected = set(args.kernels.split(",")) if args.kernels else set(KERNELS)
    if (selected - set(KERNELS) or args.repeats < 1 or args.warmup < 1 or any(n < 1 for n in args.rows)
            or any(n < 1 or n > 5 for n in args.stages)):
        parser.error("Unknown kernel or invalid rows/repeats/warmup")
    if args.commit and not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("--commit must be a full lowercase git SHA")
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return 0
    if args.list:
        print(json.dumps(dict(kernels=["nf4_tile_matmul_kernel"] if args.nf4_gemm else sorted(selected),
                              rows=args.rows, H=1536, I=8960, V=151936, vocabulary_chunks=[4096, 384],
                              stages=args.stages, warps=[2, 4, 8], samples=args.repeats,
                              fast_nf4_constraint='BLOCK=4096; sweep warps/stages without modifying the JIT',
                              policy="EXP-002 fixed fp16 GEMM tolerance" if args.nf4_gemm
                              else "bitwise-exact candidates only"), indent=2))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(ok=False, cases=[], kernels=sorted(selected), rows=args.rows,
                  schema=2, stages=args.stages, production_configs_modified=False,
                  commit=args.commit or None,
                  memory_policy='reject extra tensor bytes, global/profile scratch or compiler spills',
                  physical_peak_verified=False, seed=290,
                  repeats=args.repeats, warmup=args.warmup, created_utc=datetime.now(timezone.utc).isoformat(),
                  cache_policy="isolated temporary; discarded; not used by training")

    def save():
        report["best"] = {c["case"]: c["best"] for c in report["cases"] if "best" in c}
        report['table'], report['best_by_kernel'] = result_tables(report['cases'])
        destination = args.out / "bench_kernels.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)

    save()
    sys.path.insert(0, str(ROOT))
    with tempfile.TemporaryDirectory(prefix="soup-kernel-bench-") as cache:
        os.environ["TRITON_CACHE_DIR"] = cache
        try:
            import torch
            import triton

            torch.cuda.set_device(0)
            if "T4" not in torch.cuda.get_device_name(0):
                raise RuntimeError("Launch benchmark requires a Kaggle Tesla T4")
            torch.manual_seed(290)
            report.update(device=torch.cuda.get_device_name(0), torch=torch.__version__, triton=triton.__version__,
                          capability=torch.cuda.get_device_capability(0),
                          reduced_precision_reduction=repr(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction))
            if args.commit:
                report["commit"] = args.commit
            else:
                try:
                    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                              text=True, capture_output=True, check=False)
                    report["commit"] = revision.stdout.strip() if revision.returncode == 0 else None
                except FileNotFoundError:
                    report["commit"] = None
            if args.nf4_gemm:
                benchmark_nf4_gemm(torch, args, report, save)
                return 0 if report["ok"] else 1
            seen = set()
            for case in cases(torch, triton, args.rows):
                if case.kernel.__name__ not in selected:
                    continue
                report["active_case"] = case.key
                save()
                try:
                    record = benchmark_case(torch, case, args)
                except Exception as exc:
                    record = dict(case=case.key, kernel=case.kernel.__name__, shape=case.shape,
                                  baseline=case.baseline, candidates=[], ok=False,
                                  error=f'{type(exc).__name__}: {exc}'[:2000])
                    if any(word in str(exc).lower() for word in
                           ('illegal memory access', 'device-side assert', 'misaligned address', 'launch failure')):
                        record['fatal_error'] = record['error']
                report["cases"].append(record)
                seen.add(case.kernel.__name__)
                save()
                if record.get("fatal_error"):
                    raise RuntimeError(f"CUDA context failed in {case.key}: {record['fatal_error']}")
                gc.collect()
            report["missing_kernels"] = sorted(selected - seen)
            report["ok"] = bool(report["cases"]) and not report["missing_kernels"] and all(
                c["ok"] for c in report["cases"])
            report["best"] = {c["case"]: c["best"] for c in report["cases"] if "best" in c}
        except Exception:
            report["error"] = traceback.format_exc()
            print(report["error"], file=sys.stderr)
        finally:
            save()
    print(json.dumps(dict(ok=report["ok"], best=report.get("best", {})), indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
