"""EXP-004 T4 bitwise A/B check; no timing claims, no compiler-cache carry.

Run: python experiments/launch_trim_check.py --out out/launch_trim_check.json
--list inspects the plan without importing torch. A fresh temporary Triton
cache is created before torch/solution imports and discarded on process exit.
"""

import argparse
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from unittest.mock import patch


def run(report):
    import torch
    import bitsandbytes as bnb
    import solution
    from solution.tests.test_unpad import _model, _batch
    from transformers.models.qwen2 import modeling_qwen2 as module
    import solution.kernels.lora_epilogue as epilogue

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7, 5):
        raise RuntimeError("This acceptance requires a CUDA Tesla T4")
    if bnb.__version__ != "0.50.2":
        raise RuntimeError("Pinned bitsandbytes 0.50.2 required")
    os.environ.pop("SOUP_FUSED_DISABLE", None)
    os.environ["SOUP_UNPAD"] = "1"
    os.environ["SOUP_EXP_PRECOMPILE"] = "0"
    for key in ("SOUP_EXP_NF4_GEMM", "SOUP_MLP_RECOMPUTE",
                "SOUP_MLP_RECOMPUTE_LAYERS"):
        os.environ.pop(key, None)
    solution.apply(); solution.apply()
    report.update(device=torch.cuda.get_device_name(0), torch=torch.__version__, bnb=bnb.__version__)

    def bits(left, right, name):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"{name}: dtype/shape changed")
        if not torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                           right.contiguous().reshape(-1).view(torch.uint8)):
            error = (left.double() - right.double()).abs()
            raise AssertionError(f"{name}: not bitwise equal; max_abs={float(error.max())}")

    def snapshot(wrapped, raw, ids, labels, mask, initial, trim):
        if trim is None:  # exercise the actual production default
            os.environ.pop("SOUP_EXP_LAUNCH_TRIM", None)
        else:
            os.environ["SOUP_EXP_LAUNCH_TRIM"] = "1" if trim else "0"
        torch.cuda.set_rng_state(initial)
        wrapped.zero_grad(set_to_none=True)
        sdpa_calls, bases, epilogues = [], [], []
        native_sdpa = torch.nn.functional.scaled_dot_product_attention
        native_base = bnb.nn.Linear4bit.forward
        native_epilogue = epilogue.lora_epilogue

        def sdpa(q, k, v, *args, **kwargs):
            if args:
                raise AssertionError("Unexpected positional SDPA parameters")
            tensors = (q, k, v, kwargs["attn_mask"])
            sdpa_calls.append((tuple(t.detach().clone() for t in tensors),
                               tuple((t.dtype, t.shape, t.stride()) for t in tensors),
                               {key: value for key, value in kwargs.items() if key != "attn_mask"}))
            return native_sdpa(q, k, v, **kwargs)

        def base(self, *args, **kwargs):
            result = native_base(self, *args, **kwargs)
            bases.append((result, result._version))
            return result

        def observed_epilogue(result, adapter, scaling, **kwargs):
            version = result._version
            output = native_epilogue(result, adapter, scaling, **kwargs)
            assert output.data_ptr() != result.data_ptr(), "epilogue must be out-of-place"
            assert result._version == version, "epilogue mutated its base input"
            epilogues.append((result, version))
            return output

        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.nn.functional, "scaled_dot_product_attention", sdpa))
            stack.enter_context(patch.object(bnb.nn.Linear4bit, "forward", base))
            stack.enter_context(patch.object(epilogue, "lora_epilogue", observed_epilogue))
            with torch.autocast("cuda", dtype=torch.float16):
                output = wrapped(input_ids=ids, labels=labels, attention_mask=mask, use_cache=False)
            if output.logits is not None:
                raise AssertionError("Acceptance must exercise fused CE/ChunkedStep")
            step = raw._solution_fused_step
            if step is None:
                raise AssertionError("Missing detached statistics")
            output.loss.mul(128).backward()
        for tensor, version in bases + epilogues:
            assert tensor._version == version, "base result mutated after adapter or backward"
        base_pointers = {tensor.data_ptr() for tensor, _ in bases}
        aliases = sum(t.data_ptr() in base_pointers for t, _ in epilogues)
        assert aliases == (len(epilogues) if trim else 0), (trim, aliases, len(epilogues))
        values = dict(loss=output.loss.detach().clone(), token_loss=step.token_loss.clone(),
                      entropy=step.token_entropy.clone(), correct=step.correct.clone(),
                      labelled=step.labelled.clone(), rng=torch.cuda.get_rng_state(),
                      num_tokens=mask.sum(), grads={n: p.grad.detach().clone()
                        for n, p in wrapped.named_parameters() if p.requires_grad and p.grad is not None})
        return values, sdpa_calls, len(epilogues)

    for offload in (False, True):
        os.environ["SOUP_MLP_OFFLOAD"] = "1" if offload else "0"
        for seed in (1, 7):
            for lengths in ((7, 4, 1, 6), (5, 5, 5, 5)):
                wrapped, raw = _model(torch, module, "cuda", "bnb")
                torch.manual_seed(seed)
                ids, labels, mask = _batch(torch, "cuda", lengths, 9)
                initial = torch.cuda.get_rng_state()
                reference, calls, clones = snapshot(wrapped, raw, ids, labels, mask, initial, False)
                actual, changed_calls, aliases = snapshot(wrapped, raw, ids, labels, mask, initial, None)
                for key in ("loss", "token_loss", "entropy", "correct", "labelled", "rng", "num_tokens"):
                    bits(actual[key], reference[key], key)
                assert actual["grads"].keys() == reference["grads"].keys()
                for name in reference["grads"]:
                    bits(actual["grads"][name], reference["grads"][name], name)
                assert len(calls) == len(changed_calls) == len(lengths) * 2
                for original, changed in zip(calls, changed_calls):
                    assert original[1:] == changed[1:], "SDPA layouts/arguments changed"
                    for index, (left, right) in enumerate(zip(original[0], changed[0])):
                        bits(left, right, f"SDPA input {index}")
                record = dict(offload=offload, seed=seed, lengths=lengths, bitwise=True,
                              sdpa_calls=len(calls), clones_removed=aliases,
                              lora_gradient_tensors=len(actual["grads"]))
                report["cases"].append(record)
                print(json.dumps(record), flush=True)
                del wrapped, raw, reference, actual, calls, changed_calls
                torch.cuda.synchronize()
                gc.collect()
    report["ok"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out/launch_trim_check.json"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print(json.dumps(dict(flag="SOUP_EXP_LAUNCH_TRIM", cases=8, backend="bnb 0.50.2",
                              lengths=[[7, 4, 1, 6], [5, 5, 5, 5]], sequence=9,
                              seeds=[1, 7], offload=[False, True], gate="bitwise all outputs/A/B gradients/RNG/SDPA inputs"), indent=2))
        return 0
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    report = dict(ok=False, cases=[], cache_policy="fresh temporary; discarded")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="launch-trim-check-") as directory:
        os.environ["TRITON_CACHE_DIR"] = directory
        try:
            run(report)
        except Exception:
            report["error"] = traceback.format_exc()
            print(report["error"], file=sys.stderr)
        finally:
            args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["ok"] else 1


def test_launch_trim_bnb_bitwise_outputs_gradients_rng_and_aliases(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("EXP-004 bitwise acceptance needs CUDA")
    pytest.importorskip("bitsandbytes")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "launch_trim_check.json"
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--out", str(output)],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stdout + "\n" + process.stderr
    report = json.loads(output.read_text())
    assert report["ok"] and len(report["cases"]) == 8, report
    for case in report["cases"]:
        assert case["bitwise"]
        assert case["clones_removed"] == 4  # two layers, q/v adapters
        assert case["sdpa_calls"] == 8  # four rows per layer
        assert case["lora_gradient_tensors"] == 8  # A/B for four adapters


def test_launch_trim_default_and_literal_zero_opt_out(monkeypatch):
    from types import SimpleNamespace
    from solution.lora import _read_only_base_result

    class Module:
        def __init__(self):
            self._forward_hooks, self._forward_pre_hooks = {}, {}
            self._backward_hooks, self._backward_pre_hooks = {}, {}
        def parameters(self):
            return ()
    base_type = type("Linear4bit", (Module,), {"__module__": "bitsandbytes.nn.modules"})
    layer = SimpleNamespace(base_layer=base_type(), active_adapters=["default"],
        lora_A={"default": Module()}, lora_B={"default": Module()},
        lora_dropout={"default": Module()})
    result = SimpleNamespace(is_contiguous=lambda: True)
    for setting in (None, "0", "1", "other", ""):
        if setting is None:
            monkeypatch.delenv("SOUP_EXP_LAUNCH_TRIM", raising=False)
        else:
            monkeypatch.setenv("SOUP_EXP_LAUNCH_TRIM", setting)
        assert _read_only_base_result(layer, result) == (setting != "0")


if __name__ == "__main__":
    raise SystemExit(main())
