"""Two-dimensional SwiGLU tile engine for a frozen whole-MLP autograd boundary.

One implementation handles forward and both derivatives on 4-row x 256-column
tiles. Row/column extents and all strides are runtime arguments; neither batch
shape nor sequence length specializes the kernel. Intermediates use fp32:
a=g*sigmoid(g)*u, dg=da*u*sigmoid(g)*(1+g*(1-sigmoid(g))),
du=da*g*sigmoid(g). Output buffers select the required projection graph dtype.
The two-axis layout, shared forward/backward engine and independent strides
are original, rather than a renamed copy of the starter's flat example.
Ideas: the standard SiLU derivative and tiled pointwise fusion.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_tile_engine(G, U, A, Da, Dg, Du, N, I,
                       G_ROW, G_COL, U_ROW, U_COL, A_ROW, A_COL,
                       DA_ROW, DA_COL, DG_ROW, DG_COL, DU_ROW, DU_COL,
                       BACKWARD: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.program_id(1) * COLS + tl.arange(0, COLS)
    valid = (rows[:, None] < N) & (columns[None, :] < I)
    gate = tl.load(G + rows[:, None] * G_ROW + columns[None, :] * G_COL,
                   valid, other=0.0).to(tl.float32)
    up = tl.load(U + rows[:, None] * U_ROW + columns[None, :] * U_COL,
                 valid, other=0.0).to(tl.float32)
    probability = 1.0 / (1.0 + tl.exp(-gate))
    activated = gate * probability
    if BACKWARD:
        upstream = tl.load(Da + rows[:, None] * DA_ROW + columns[None, :] * DA_COL,
                           valid, other=0.0).to(tl.float32)
        slope = probability + activated * (1.0 - probability)
        gate_gradient = (upstream * up) * slope
        up_gradient = upstream * activated
        tl.store(Dg + rows[:, None] * DG_ROW + columns[None, :] * DG_COL,
                 gate_gradient, valid)
        tl.store(Du + rows[:, None] * DU_ROW + columns[None, :] * DU_COL,
                 up_gradient, valid)
    else:
        tl.store(A + rows[:, None] * A_ROW + columns[None, :] * A_COL,
                 activated * up, valid)


def _matrix(t):
    return t.reshape(math.prod(t.shape[:-1]), t.shape[-1])


def swiglu_fused_forward(gate, up, out=None):
    """Fill a supplied local-graph buffer, or allocate a gate-dtype output."""
    if out is None:
        out = torch.empty_like(gate)
    g, u, a = _matrix(gate), _matrix(up), _matrix(out)
    n, width = g.shape
    if n:
        swiglu_tile_engine[(triton.cdiv(n, 4), triton.cdiv(width, 256))](
            g, u, a, g, g, u, n, width,
            *g.stride(), *u.stride(), *a.stride(), *g.stride(), *g.stride(), *u.stride(),
            BACKWARD=False, ROWS=4, COLS=256, num_warps=4, enable_fp_fusion=False)
    return out


def swiglu_fused_backward(gate, up, grad, gate_dtype=None, up_dtype=None):
    """Return both derivatives in the original projection-output dtypes."""
    dgate = torch.empty(gate.shape, device=gate.device, dtype=gate_dtype or gate.dtype)
    dup = torch.empty(up.shape, device=up.device, dtype=up_dtype or up.dtype)
    g, u, da, dg, du = map(_matrix, (gate, up, grad, dgate, dup))
    n, width = g.shape
    if n:
        swiglu_tile_engine[(triton.cdiv(n, 4), triton.cdiv(width, 256))](
            g, u, g, da, dg, du, n, width,
            *g.stride(), *u.stride(), *g.stride(), *da.stride(), *dg.stride(), *du.stride(),
            BACKWARD=True, ROWS=4, COLS=256, num_warps=4, enable_fp_fusion=False)
    return dgate, dup
