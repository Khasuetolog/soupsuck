"""Standalone T4 cold-compile diagnostic; never imported by training.

Uses a fresh temporary Triton cache and deletes it on exit. Compiler timings
wrap Triton's compiler API, while cold/warm call timings include launch/GEMM
work. No cache, warmup or compiler instrumentation reaches harness runs.
"""

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    report = {"ok": False, "compilations": [], "cases": []}

    def save():
        (args.out / "probe_compile.json").write_text(json.dumps(report, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="soup-probe-triton-") as cache:
        os.environ["TRITON_CACHE_DIR"] = cache
        try:
            import torch
            import triton
            import triton.compiler

            torch.empty(1, device="cuda")
            assert "T4" in torch.cuda.get_device_name(0), "Probe requires T4"
            report.update(torch=torch.__version__, triton=triton.__version__,
                          device=torch.cuda.get_device_name(0), cache_policy="fresh temporary, removed on exit")
            original = triton.compiler.compile

            def timed_compile(src, *positional, **kwargs):
                jit = getattr(src, "fn", None)
                fn = getattr(jit, "fn", jit)
                entry = {"kernel": getattr(fn, "__name__", str(type(src))),
                         "module": getattr(fn, "__module__", ""),
                         "options": repr(kwargs.get("options", {})), "ok": False}
                start = time.perf_counter()
                try:
                    kernel = original(src, *positional, **kwargs)
                    entry["ok"] = True
                    return kernel
                finally:
                    entry["compiler_s"] = time.perf_counter() - start
                    report["compilations"].append(entry)
                    print("compile", json.dumps(entry), flush=True)
                    save()

            # JIT binders import this API lazily; also replace existing aliases
            # on versions that import compile into runtime.jit at module load.
            for module in (triton, triton.compiler,
                           importlib.import_module("triton.compiler.compiler"),
                           importlib.import_module("triton.runtime.jit")):
                if getattr(module, "compile", None) is original:
                    module.compile = timed_compile

            from solution.kernels.cross_entropy import fused_linear_cross_entropy
            from solution.kernels.rmsnorm import rms_norm
            from solution.kernels.add_rmsnorm import add_rms_norm
            from solution.kernels.swiglu_fused import swiglu_fused_forward, swiglu_fused_backward
            from solution.kernels.lora_epilogue import lora_epilogue
            from solution.swiglu import swiglu

            n, h, v, intermediate = 2048, 1536, 151936, 8960
            report["shapes"] = {"N": n, "H": h, "V": v, "I": intermediate}

            def measure(name, call):
                record = {"name": name}
                report["cases"].append(record)
                for phase in ("cold", "warm"):
                    torch.cuda.synchronize()
                    before = len(report["compilations"])
                    start = time.perf_counter()
                    call()
                    torch.cuda.synchronize()
                    record[f"{phase}_wall_s"] = time.perf_counter() - start
                    record[f"{phase}_compiler_s"] = sum(
                        item["compiler_s"] for item in report["compilations"][before:])
                print("case", json.dumps(record), flush=True)
                save()

            weight = torch.randn(v, h, device="cuda", dtype=torch.float32) / h**0.5
            targets = torch.randint(v, (n,), device="cuda")
            targets[::13] = -100
            for dtype in (torch.float32, torch.float16):
                x = torch.randn(n, h, device="cuda", dtype=dtype, requires_grad=True)
                def head():
                    loss, _, _, _ = fused_linear_cross_entropy(x, weight, targets)
                    (loss.sum() * (128 / n)).backward()
                measure(f"head_{dtype}", head)
                del x
            del weight, targets
            gc.collect()
            torch.cuda.empty_cache()

            for dtype in (torch.float32, torch.float16):
                for train_weight in (False, True):
                    x = torch.randn(4, 512, h, device="cuda", dtype=dtype, requires_grad=True)
                    weight = torch.ones(h, device="cuda", dtype=torch.float32, requires_grad=train_weight)
                    grad = torch.randn(x.shape, device="cuda", dtype=torch.float32)
                    measure(f"rms_{dtype}_weight_grad_{train_weight}",
                            lambda: rms_norm(x, weight, 1e-6).backward(grad))
                    del x, weight, grad
            for out_dtype in (None, torch.float16):
                r = torch.randn(4, 512, h, device="cuda", requires_grad=True)
                delta = torch.randn_like(r, dtype=torch.float16, requires_grad=True)
                weight = torch.ones(h, device="cuda", requires_grad=True)
                gr = torch.randn_like(r)
                gn = torch.randn_like(r, dtype=out_dtype or torch.float32)
                def add_norm():
                    residual, normed = add_rms_norm(r, delta, weight, 1e-6, out_dtype)
                    torch.autograd.backward((residual, normed), (gr, gn))
                measure(f"add_norm_{out_dtype}", add_norm)
                del r, delta, weight, gr, gn
            for dtype in (torch.float32, torch.float16):
                gate = torch.randn(4, 512, intermediate, device="cuda", dtype=dtype)
                up = torch.randn_like(gate)
                grad = torch.randn_like(gate)
                def tile():
                    swiglu_fused_forward(gate, up)
                    swiglu_fused_backward(gate, up, grad)
                measure(f"swiglu_tile_{dtype}", tile)
                gate.requires_grad_(True)
                up.requires_grad_(True)
                measure(f"supplied_example_{dtype}", lambda: swiglu(gate, up).sum().backward())
                del gate, up, grad
            for dtype in (torch.float32, torch.float16):
                base = torch.randn(4, 512, h, device="cuda", dtype=dtype, requires_grad=True)
                adapter = torch.randn_like(base, dtype=torch.float16, requires_grad=True)
                measure(f"lora_epilogue_{dtype}", lambda: lora_epilogue(base, adapter, 2.0).sum().backward())
                del base, adapter
            own = [entry for entry in report["compilations"] if entry["module"].startswith("solution.")
                   and not entry["kernel"].startswith("example_")]
            report["own_compiler_s"] = sum(entry["compiler_s"] for entry in own)
            assert own, "Triton compiler interception captured no solution kernels"
            report["ok"] = True
        except Exception:
            report["error"] = traceback.format_exc()
            print(report["error"], file=sys.stderr)
        finally:
            save()
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
