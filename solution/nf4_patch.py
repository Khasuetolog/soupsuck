"""Route both bnb 0.50.2 training dequantization paths through our NF4 decoder.

Verified source anchors, tag 0.50.2:
* autograd/_functions.py MatMul4Bit.backward -> functional.dequantize_4bit.
* backends/cuda/ops.py:899-916 _dequant_linear_fallback reconstructs nested
  absmax and calls _dequantize_4bit_impl directly, then F.linear.
* CUDA dequantize_4bit.default/.out registrations also resolve that same
  _dequantize_4bit_impl global. Replacing only functional misses the GEMM
  fallback. Replacing only the backend keeps the separate nested kernels.
https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py
https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py

The functional patch fuses nested statistics with NF4, directly into out when
provided. The backend patch reuses decoded fp32 absmax only inside a certified
nested _dequant_linear_fallback scope. Standalone flat dispatch, non-nested
states and unsupported inputs use the originals; there is no torch operator
re-registration. Kernel/compiler errors are never caught as fallback.
Registration and kernel imports are lazy so fake packages/CPU stay usable.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import importlib
import os
from types import ModuleType

from .utils import enabled

_MARKER = "_solution_fused_original"
_NESTED_BACKEND = ContextVar("solution_nf4_nested_backend", default=False)


@contextmanager
def _nested_backend_scope(nested):
    token = _NESTED_BACKEND.set(nested)
    try:
        yield
    finally:
        _NESTED_BACKEND.reset(token)


def patch_bnb_dequant():
    """Idempotently patch the pinned backend; caller owns install registration."""
    try:
        import torch
        bnb = importlib.import_module("bitsandbytes")
        functional = importlib.import_module("bitsandbytes.functional")
        backend = importlib.import_module("bitsandbytes.backends.cuda.ops")
    except (ImportError, AttributeError):
        return False
    if (not all(isinstance(m, ModuleType) for m in (bnb, functional, backend))
            or getattr(bnb, "__version__", None) != "0.50.2"
            or not callable(getattr(functional, "dequantize_4bit", None))
            or not callable(getattr(backend, "_dequantize_4bit_impl", None))
            or not callable(getattr(backend, "_dequant_linear_fallback", None))):
        return False
    original_f = functional.dequantize_4bit
    original_backend = backend._dequantize_4bit_impl
    original_linear = backend._dequant_linear_fallback

    if not hasattr(original_f, _MARKER):
        @wraps(original_f)
        def dequantize(A, quant_state=None, absmax=None, out=None, blocksize=None, quant_type="fp4"):
            arguments = dict(quant_state=quant_state, absmax=absmax, out=out,
                             blocksize=blocksize, quant_type=quant_type)
            def fallback():
                # The original functional dispatches through the patched
                # backend too. Explicitly exclude unsupported/non-nested state.
                with _nested_backend_scope(False):
                    return original_f(A, **arguments)
            if not enabled() or not isinstance(A, torch.Tensor) or not A.is_cuda:
                return fallback()
            from .kernels.nf4 import _validate_state, dequantize_nf4

            try:
                shape = _validate_state(A, quant_state, torch.float16)
            except ValueError:
                return fallback()
            if out is not None and (not isinstance(out, torch.Tensor) or out.shape != shape
                                    or out.dtype != torch.float16 or out.device != A.device
                                    or not out.is_contiguous() or out.requires_grad):
                return fallback()
            return dequantize_nf4(A, quant_state, out=out)

        setattr(dequantize, _MARKER, original_f)
        functional.dequantize_4bit = dequantize

    if not hasattr(original_backend, _MARKER):
        @wraps(original_backend)
        def dequantize_backend(A, absmax, blocksize, quant_type, dtype, out):
            if (not _NESTED_BACKEND.get() or not enabled() or blocksize != 64
                    or quant_type != "nf4" or dtype != torch.float16):
                return original_backend(A, absmax, blocksize, quant_type, dtype, out)
            from .kernels.nf4 import _validate_flat, _dequantize_flat_into

            try:
                _validate_flat(A, absmax, out)
            except ValueError:
                return original_backend(A, absmax, blocksize, quant_type, dtype, out)
            _dequantize_flat_into(A, absmax, out)
            # Match the original helper, which writes out and returns None.
            return None

        setattr(dequantize_backend, _MARKER, original_backend)
        backend._dequantize_4bit_impl = dequantize_backend

    if not hasattr(original_linear, _MARKER):
        @wraps(original_linear)
        def linear_fallback(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
                            absmax_8bit=None, absmax_code=None, absmax_offset=None):
            # Only this call site knows whether fp32 scales came from nested
            # state. Direct flat operators and non-nested GEMM stay native.
            with _nested_backend_scope(absmax_8bit is not None):
                return original_linear(A, B, shapeB, absmax, blocksize, quant_type,
                                       bias=bias, absmax_8bit=absmax_8bit,
                                       absmax_code=absmax_code, absmax_offset=absmax_offset)

        setattr(linear_fallback, _MARKER, original_linear)
        backend._dequant_linear_fallback = linear_fallback
    if os.environ.get("SOUP_EXP_NF4_GEMM") == "1":
        from .kernels.nf4_gemm import patch_nf4_matmul

        patch_nf4_matmul(bnb)
    return True
