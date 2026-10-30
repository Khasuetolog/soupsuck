"""Original Triton decoder for bnb 0.50.2 NF4/block64 + double quant/block256.

Format sources (tag 0.50.2, inspected before implementation):
https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/csrc/kernels.cu
https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/functional.py
NF4's sixteen constants below are format data, not copied kernel code. A byte
stores the earlier value in its HIGH nibble. Scale for output i is
fp32(fp32(state2.code[qabs[i//64]] * state2.absmax[(i//64)//256]) + offset).
Multiply this scale by the NF4 code in fp32, then round once to fp16. Fusion
of the nested multiply and offset addition is explicitly disabled: bnb writes
the intermediate fp32 absmax before its separate offset-add operation.

The same tile engine accepts already reconstructed fp32 scales for bnb's
backend forward-fallback path. One program decodes 2048 values/four warps;
sizes are runtime, tails are masked, and no weight/absmax-sized temporary is
needed in the nested path. A fixed 16-float format table is cached per device.
Only the narrow supported state is accepted; numerical bit equality and
performance are acceptance tests to run on T4, not claimed local results.
"""

import math

import torch
import triton
import triton.language as tl

_NF4_VALUES = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)
_TABLES = {}


@triton.jit
def nf4_decode_kernel(Packed, Abs, Scale2, Code2, Offset, Table, Out, N,
                       NESTED: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    byte = tl.load(Packed + index // 2, valid, other=0).to(tl.int32)
    nibble = tl.where((index & 1) == 0, byte >> 4, byte & 15)
    block = index // 64
    if NESTED:
        scale_code = tl.load(Abs + block, valid, other=0).to(tl.int32)
        code_value = tl.load(Code2 + scale_code, valid, other=0.0)
        scale_value = tl.load(Scale2 + block // 256, valid, other=0.0)
        decoded_scale = code_value * scale_value
        offset = tl.load(Offset)
        scale = decoded_scale + offset
    else:
        scale = tl.load(Abs + block, valid, other=0.0)
    value = tl.load(Table + nibble, valid, other=0.0)
    tl.store(Out + index, value * scale, valid)


def _table(device):
    if device not in _TABLES:
        _TABLES[device] = torch.tensor(_NF4_VALUES, device=device, dtype=torch.float32)
    return _TABLES[device]


def _tensor(t, dtype, device, minimum):
    return (isinstance(t, torch.Tensor) and t.device == device and t.dtype == dtype
            and t.is_contiguous() and t.numel() >= minimum and not t.requires_grad)


def _validate_state(packed, state, out_dtype):
    if not isinstance(packed, torch.Tensor) or not packed.is_cuda or packed.dtype != torch.uint8:
        raise ValueError("NF4 storage must be uint8 CUDA")
    if not packed.is_contiguous() or packed.ndim not in (1, 2):
        raise ValueError("NF4 storage must be contiguous and one- or two-dimensional")
    if out_dtype != torch.float16:
        raise ValueError("Only fp16 NF4 output is supported")
    if (getattr(state, "quant_type", None) != "nf4" or getattr(state, "blocksize", None) != 64
            or getattr(state, "dtype", None) != torch.float16 or not getattr(state, "nested", False)):
        raise ValueError("Expected fp16 NF4/block64 with compressed statistics")
    shape = getattr(state, "shape", None)
    if not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 2:
        raise ValueError("Expected a matrix shape in quant_state")
    if any(not isinstance(d, int) or d <= 0 for d in shape):
        raise ValueError("quant_state dimensions must be positive integers")
    n = math.prod(shape)
    blocks = triton.cdiv(n, 64)
    second = getattr(state, "state2", None)
    if (getattr(second, "blocksize", None) != 256 or getattr(second, "dtype", None) != torch.float32
            or getattr(second, "nested", False)):
        raise ValueError("Expected non-nested fp32 state2/block256")
    if (packed.numel() * 2 < n
            or not _tensor(getattr(state, "absmax", None), torch.uint8, packed.device, blocks)
            or not _tensor(getattr(second, "absmax", None), torch.float32, packed.device, triton.cdiv(blocks, 256))
            or not _tensor(getattr(second, "code", None), torch.float32, packed.device, 256)
            or not _tensor(getattr(state, "offset", None), torch.float32, packed.device, 1)
            or state.offset.numel() != 1):
        raise ValueError("Invalid packed length or nested scale/code/offset buffers")
    return tuple(shape)


def _validate_flat(packed, absmax, out):
    if (not isinstance(packed, torch.Tensor) or not packed.is_cuda or packed.dtype != torch.uint8
            or not packed.is_contiguous() or packed.ndim not in (1, 2)
            or not isinstance(out, torch.Tensor) or out.dtype != torch.float16
            or out.device != packed.device or not out.is_contiguous() or out.ndim != 2 or out.requires_grad
            or packed.numel() * 2 < out.numel()
            or not _tensor(absmax, torch.float32, packed.device, triton.cdiv(out.numel(), 64))):
        raise ValueError("Unsupported flat NF4 dequantization buffers")


def _dequantize_flat_into(packed, absmax, out):
    """Backend entry: fp32 absmax was already reconstructed by bitsandbytes."""
    _validate_flat(packed, absmax, out)
    with torch.cuda.device(packed.device):
        if out.numel():
            nf4_decode_kernel[(triton.cdiv(out.numel(), 2048),)](
                packed, absmax, absmax, absmax, absmax, _table(packed.device), out, out.numel(),
                NESTED=False, BLOCK=2048, num_warps=4, enable_fp_fusion=False)
    return out


def dequantize_nf4(packed, quant_state, out_dtype=torch.float16, *, out=None):
    """Decode directly into out (canonical state.shape), then apply the BC view."""
    shape = _validate_state(packed, quant_state, out_dtype)
    if out is not None and (not isinstance(out, torch.Tensor) or tuple(out.shape) != shape
                            or out.device != packed.device or out.dtype != out_dtype
                            or not out.is_contiguous() or out.requires_grad):
        raise ValueError("Expected a contiguous non-grad out buffer matching quant_state")
    with torch.cuda.device(packed.device):
        if out is None:
            out = torch.empty(shape, device=packed.device, dtype=out_dtype)
        nf4_decode_kernel[(triton.cdiv(out.numel(), 2048),)](
            packed, quant_state.absmax, quant_state.state2.absmax, quant_state.state2.code,
            quant_state.offset, _table(packed.device), out, out.numel(),
            NESTED=True, BLOCK=2048, num_warps=4, enable_fp_fusion=False)
    return out.t() if packed.shape[0] == 1 else out
