"""Offline T4 launch sweep; never imported by solution or training.

Direct launches of all current @triton.jit entrypoints, synthetic data at real
shapes, no model/data download and no extra kernel libraries. Only candidates
whose EVERY output matches the current launch bit-for-bit are timed/ranked.
Reduction layouts can fail this strict gate even when mathematically correct;
they are rejected rather than granted looser tolerances. CE uses its actual
4096-column vocabulary chunk and 3840-column tail, not a full [N,V] buffer.
Optional dgamma partials retain 32-row boundaries so outputs are comparable.

Each candidate compiles/warms outside timing, then gets 20 CUDA-event samples
(median/p10/p90). Stateful CE is reset BEFORE the start event each iteration.
Buffers and reference snapshots are allocated outside event intervals. Results
include every rejection/error and each case's fastest exact candidate, without
changing solution code or producing a training cache. A temporary Triton cache
is discarded on exit. No clocks, allocator settings or model settings change.

Usage: python tools/kaggle/bench_kernels.py --out /kaggle/working/out
Local inspection without torch: --list (stdlib only).
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import importlib
import json
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
    "nf4_decode_kernel", "swiglu_tile_engine",
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

    def launch(self, config):
        self.kernel[self.grid(config)](*self.args, **(self.fixed or {}), **config,
                                      enable_fp_fusion=False)


def flat_grid():
    return [dict(BLOCK=b, num_warps=w) for b, w in
            ((256, 4), (512, 4), (1024, 2), (1024, 4), (2048, 4), (4096, 4), (4096, 8))]


def row_grid(block):
    return [dict(BLOCK=b, num_warps=w) for b in (block, block * 2) for w in (2, 4, 8)]


def cases(torch, triton, rows):
    ce = importlib.import_module("solution.kernels.cross_entropy")
    norm = importlib.import_module("solution.kernels.rmsnorm")
    add = importlib.import_module("solution.kernels.add_rmsnorm")
    swiglu = importlib.import_module("solution.kernels.swiglu_fused")
    nf4 = importlib.import_module("solution.kernels.nf4")
    lora = importlib.import_module("solution.kernels.lora_epilogue")
    rope = importlib.import_module("solution.kernels.rope")
    h, intermediate, vocab = 1536, 8960, 151936

    def random(*shape, dtype=torch.float32):
        return torch.randn(shape, device="cuda", dtype=dtype)

    def empty(*shape, dtype=torch.float32):
        return torch.empty(shape, device="cuda", dtype=dtype)

    def emit(key, kernel, shape, args, outputs, baseline, candidates, grid, fixed=None, reset=None):
        return Case(key, kernel, shape, tuple(args), tuple(outputs), baseline, candidates, grid, fixed, reset)

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
        yield emit(f"add-rms-bwd/N{n}", add.add_rmsnorm_backward_kernel, dict(N=n, H=h),
                   (summed, weight, inverse, gs, grad, dr, dd, h, 1, h, 1, 1, h), (dr, dd),
                   dict(BLOCK=2048, num_warps=4), row_grid(2048), lambda c: (n,),
                   fixed=dict(HAS_GS=True, HAS_GY=True, WRITE_R=True, WRITE_D=True))

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
                yield emit(f"ce-stats/N{n}/W{width}/first{first}", ce.vocabulary_statistics_kernel,
                           dict(N=n, V=vocab, chunk=width, first=first),
                           (logits, targets, state, best, width, 0 if first else 4096), (state, best),
                           dict(BLOCK=4096, num_warps=4), row_grid(4096), lambda c: (n,),
                           fixed=dict(IGNORE=-100, FIRST=first), reset=reset_state)
            dz = empty(n, width, dtype=torch.float16)
            lse = torch.full((n,), 10.0, device="cuda")
            g = random(n)
            gain = g.abs().clamp_min(1e-30) / 1024
            valid_targets = targets.clamp_min(0)  # production backward has compacted ignored rows
            yield emit(f"ce-derivative/N{n}/W{width}", ce.vocabulary_derivative_kernel,
                       dict(N=n, V=vocab, chunk=width), (logits, valid_targets, lse, g, gain, dz, width, 4096), (dz,),
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

        for dtype, cdtype in ((torch.float32, torch.float32), (torch.float16, torch.float16),
                             (torch.float16, torch.float32)):
            q = random(1, n, 12, 128, dtype=dtype).transpose(1, 2)
            k = random(1, n, 2, 128, dtype=dtype).transpose(1, 2)
            angles = random(1, n, 64)
            cos = torch.cat((angles.cos(), angles.cos()), -1).to(cdtype)
            sin = torch.cat((angles.sin(), angles.sin()), -1).to(cdtype)
            out_dtype = torch.promote_types(dtype, cdtype)
            yq, yk = empty(1, 12, n, 128, dtype=out_dtype), empty(1, 2, n, 128, dtype=out_dtype)
            gq = random(1, n, 12, 128, dtype=out_dtype).transpose(1, 2)
            gk = random(1, n, 2, 128, dtype=out_dtype).transpose(1, 2)
            dq, dk = empty(1, 12, n, 128, dtype=dtype), empty(1, 2, n, 128, dtype=dtype)
            rope_configs = [dict(ROWS=r, WIDTH=w, num_warps=warps) for r, w, warps in
                            ((1, 64, 4), (2, 64, 4), (4, 64, 2), (4, 64, 4),
                             (8, 64, 4), (16, 64, 8), (4, 128, 4))]
            for backward in (False, True):
                prefix = (gq, gk, cos, sin, dq, dk) if backward else (q, k, cos, sin, yq, yk)
                strides = (*gq.stride(), *gk.stride()) if backward else (*q.stride()[:3], *k.stride()[:3])
                yield emit(f"rope/{'bwd' if backward else 'fwd'}/T{n}/{dtype}/{cdtype}",
                           rope.rope_pair_backward_kernel if backward else rope.rope_pair_forward_kernel,
                           dict(B=1, T=n, q_heads=12, k_heads=2, D=128, qk=str(dtype), cos=str(cdtype)),
                           (*prefix, n, 1, 12, 2, 64, 1, *strides, *cos.stride()[:2], *sin.stride()[:2]),
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
            yield emit(f"nf4/{shape[0]}x{shape[1]}/nested{nested}", nf4.nf4_decode_kernel,
                       dict(weight=shape, blocksize=64, nested=nested, state2_blocksize=256),
                       (packed, quant_abs if nested else flat_abs, second, code, offset,
                        nf4._table(out.device), out, count), (out,),
                       dict(BLOCK=2048, num_warps=4), flat_grid(),
                       lambda c: (triton.cdiv(count, c["BLOCK"]),), fixed=dict(NESTED=nested))


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
    return dict(median_ms=statistics.median(samples), p10_ms=ordered[int((len(ordered) - 1) * .1)],
                p90_ms=ordered[int((len(ordered) - 1) * .9)], min_ms=min(samples),
                samples_ms=samples, warmup_s=warmup_s)


def benchmark_case(torch, case, args):
    record = dict(case=case.key, kernel=case.kernel.__name__, shape=case.shape,
                  baseline=case.baseline, candidates=[], ok=False)
    if case.reset is not None:
        case.reset()
    initial = time.perf_counter()
    case.launch(case.baseline)
    torch.cuda.synchronize()
    record["baseline_first_call_s"] = time.perf_counter() - initial
    reference = tuple(t.detach().clone() for t in case.outputs)
    configs = [case.baseline]
    configs.extend(c for c in case.candidates if c not in configs)
    for config in configs:
        trial = dict(config=config, exact=False)
        begin = time.perf_counter()
        try:
            if case.reset is not None:
                case.reset()
            case.launch(config)
            torch.cuda.synchronize()
            trial["first_validation_call_s"] = time.perf_counter() - begin
            trial["mismatch_bytes"] = bitwise(torch, case.outputs, reference)
            trial["exact"] = not any(trial["mismatch_bytes"])
            if trial["exact"]:
                trial.update(measure(torch, case, config, args.warmup, args.repeats))
            else:
                trial["rejected"] = "bitwise mismatch against current configuration"
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
    eligible = [c for c in record["candidates"] if c.get("exact") and "median_ms" in c]
    baseline = next((c for c in eligible if c["config"] == case.baseline), None)
    if eligible and baseline:
        best = min(eligible, key=lambda c: c["median_ms"])
        record["best"] = dict(config=best["config"], median_ms=best["median_ms"],
                              baseline_ms=baseline["median_ms"], speedup=baseline["median_ms"] / best["median_ms"])
        record["ok"] = True
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--list", action="store_true", help="Print the kernel/shape plan without importing torch")
    parser.add_argument("--kernels", default="", help="Optional comma-separated exact kernel names")
    parser.add_argument("--rows", type=int, nargs="+", default=[2048, 800])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--commit", default="", help="Verified archive SHA (Kaggle archive has no .git)")
    args = parser.parse_args()
    selected = set(args.kernels.split(",")) if args.kernels else set(KERNELS)
    if selected - set(KERNELS) or args.repeats < 1 or args.warmup < 1 or any(n < 1 for n in args.rows):
        parser.error("Unknown kernel or invalid rows/repeats/warmup")
    if args.commit and not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("--commit must be a full lowercase git SHA")
    if args.list:
        print(json.dumps(dict(kernels=sorted(selected), rows=args.rows, H=1536, I=8960, V=151936,
                              samples=args.repeats, policy="bitwise-exact candidates only"), indent=2))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(ok=False, cases=[], kernels=sorted(selected), rows=args.rows,
                  repeats=args.repeats, warmup=args.warmup, created_utc=datetime.now(timezone.utc).isoformat(),
                  cache_policy="isolated temporary; discarded; not used by training")

    def save():
        report["best"] = {c["case"]: c["best"] for c in report["cases"] if "best" in c}
        destination = args.out / "bench_kernels.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
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
            seen = set()
            for case in cases(torch, triton, args.rows):
                if case.kernel.__name__ not in selected:
                    continue
                report["active_case"] = case.key
                save()
                record = benchmark_case(torch, case, args)
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
