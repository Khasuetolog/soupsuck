# Soup Kernel Challenge: 1.34× faster and 3.8× less memory, same loss

Our Triton kernels and runtime patches speed up and slim down the fixed
Qwen2.5-1.5B-Instruct NF4/LoRA `soup train` task on a Kaggle Tesla T4,
without changing the task, data, seed, trainer settings or the harness.

Final two-repeat jury comparison (`harness/run_task.py` ×2 + `check_rules.py`)
on solution commit **`96a890c`**:

| Measurement | Base Soup | Solution | Comparison |
| --- | ---: | ---: | --- |
| Whole `soup train` command, mean of 2 | 87.61 s (91.37 / 83.85) | **65.49 s** (65.11 / 65.87) | **1.338× speedup** |
| Training loop (64 steps), mean of 2 | 60.75 s | 39.67 s | 1.53× |
| Median step | 0.94–0.99 s | 0.51–0.52 s | ~1.9× |
| Peak GPU memory, sampled by nvidia-smi | 14,345 MiB | **3,769 MiB** | **3.806× base/solution** |
| Public held-out answer-token loss | 2.04966 | 2.04964 | **−0.001 %** vs base |
| Own Triton kernel share, steps 3 onward | — | 13.3 % (lowest run) | Check 4 PASS |
| Harness checks | — | **10/10 PASS** | `ALL CHECKS PASSED` |

The submitted commit `f1cb5ac` has `solution/` byte-identical to the measured `96a890c` (`git diff 96a890c f1cb5ac -- solution/` is empty); later commits only touch docs and `tools/`. A confirmation `full2` of `d1bd86f`, which still carried the opt-in EXP-013/015 code, also passed 10/10 (1.283×, 3,769 MiB, +0.004 %) but its solution took 69.21 s against a similar base (88.81 s); since that code bought nothing, we removed it and submit the measured `96a890c` code.

Correctness on the same T4 stack: **518 passed, 1 skipped** solution tests
(the skipped one belongs to a rejected opt-in experiment) and **79/79** Soup
regression tests with `apply()` (check 10).

**Why we report ratios, not only seconds.** Base timing drifts between Kaggle
sessions (77.7–91.4 s in selected recent sessions with identical base code). The score uses the
ratio of base to solution inside one session, so every number above comes from
one session. Whole-run optimization deltas below come from *paired* same-session
A/B rounds; microbenchmarks, estimates and historical comparisons are labelled separately.

- Public Kaggle notebook, saved version: `<KAGGLE_LINK>`
- Final submission commit: `f1cb5ac32ef7cbd1caf8455c833b67bd43992ed8`
- Submission links and team: [SUBMISSION.md](SUBMISSION.md)

## Where the time goes

Partial instrumented observation of one solution run on T4
(`tools/kaggle/start_probe.py`, commit `f81a736`, observer overhead included;
66.4 s total). The probe records `ok=false` because `import.bitsandbytes` is
missing from its phase coverage; the intervals below are observations from that
incomplete probe, not a complete attribution of startup:

| Phase | Seconds | Notes |
| --- | ---: | --- |
| Interpreter + imports before `apply()` | 5.1 | torch and early imports; excludes the initial interpreter/probe bootstrap |
| `apply()` → first training step | 18.7 | further imports including TRL/PEFT, weight load, trainer init; the 4-bit quantization first-to-last event span is 2.2 s, including gaps between events; our Triton compilation runs in a parallel child process here |
| 64 training steps | 38.3 | first step 3.4 s, median 0.49 s |
| Adapter save + process exit | 4.1 | |

The first step began 23.97 s after process launch in this instrumented run.
In the final two-repeat `full2` run on `96a890c`, wall time minus the training
loop averaged 25.8 s (25.77 / 25.87 s), including startup and shutdown.
These observed intervals can vary between runs. Displayed phase times are
rounded; interpreter/probe bootstrap and `apply()` add less than 0.1 s.
Inside the loop,
the per-step layer profile (`tools/kaggle/layer_profile.py`) attributes NF4
weight decoding 83 ms, bitsandbytes/cuBLAS GEMMs ~136 ms, MLP offload waits
57 ms, SDPA 48 ms, LoRA 30 ms and norms/RoPE 27 ms per step.

Padding (`tools/kaggle/token_stats.py`, exact Soup/TRL pipeline, seed 1):
64 batches of 4 rows contain 52,890 real tokens out of 91,856 padded positions
(**42.4 % padding**); only 33,608 positions carry labels. The unpadded decoder
skips all 42.4 %, and the head backward skips 63.4 % of positions.

## What we did

The entry point is [`solution.apply()`](solution/__init__.py).
Everything below is registered in the measured commit `96a890c`.

### Padding-free decoder within each batch

[`solution/unpad.py`](solution/unpad.py) runs the Qwen2 decoder on the packed
`[1, T, H]` tensor of real tokens of one batch (right padding removed); rows of
different batches are never mixed. Attention runs per row through SDPA with the
original causal semantics, RoPE receives the original per-row positions, and the
result is scattered back to `[B, S, H]` so TRL/Rewind labels, shift and masks are
untouched. There is one device-to-host synchronization per batch
(`metadata.tolist()`); LoRA dropout sees the same padded-shape input so its RNG
stream is unchanged. Validated against a full fp64 Qwen2+LoRA oracle.
`SOUP_UNPAD=0` disables it.

### Asynchronous MLP activation offload

[`solution/mlp.py`](solution/mlp.py) saves each layer's fp16 gate/up tensors to
pinned host memory on a side CUDA stream during forward and prefetches them back
one layer ahead during backward, guarded by CUDA events and `record_stream`
(no host synchronization). This is what lets the peak fall to 3,769 MiB without
recomputing the 8960-wide projections. `SOUP_MLP_OFFLOAD=0` keeps them on GPU.

### Out-of-process Triton compilation and prebuilt launchers

[`solution/precompile.py`](solution/precompile.py) compiles every production
kernel specialization in a child process while the parent imports libraries and
loads the model, and (EXP-010) also builds the native CPU launchers there, so the
first training step does not pay JIT cost (first step 6.5 → 2.9 s). Results go to
the run's own `TRITON_CACHE_DIR`; no cache is reused across runs.
`SOUP_EXP_PRECOMPILE=0` / `SOUP_EXP_CPU_LAUNCHERS=0` disable it.

### Frozen linear cross-entropy, entropy and predictions

[`solution/kernels/cross_entropy.py`](solution/kernels/cross_entropy.py) defines:

| `@triton.jit` name | Replaces / produces |
| --- | --- |
| `vocabulary_statistics_kernel` | Chunk softmax statistics, target logit and first argmax |
| `vocabulary_outputs_kernel` | Per-token CE, entropy and log-sum-exp |
| `vocabulary_derivative_kernel` | Softmax/CE derivative before the hidden-gradient GEMM |

**Registered:** the Qwen2 labelled-training head and TRL metrics bridge in
[`solution/patch.py`](solution/patch.py).
Instead of keeping all `[batch × sequence, vocabulary]` logits, the head visits
vocabulary chunks. Matrix multiplication remains `torch.mm`/cuBLAS. The registered implementation
uses fp16 operands and native fp16 logit output, then promotes logits
to fp32 for statistics, following the baseline bias-free AMP linear operation
without changing backend reduction flags. Measured de836e0 instead used fp32
GEMM output followed by explicit fp16 rounding; these two GEMM paths need not
be bitwise identical because cuBLAS may choose different reduction algorithms.

For each token, maintain maximum `m`, exponential sum `s`, and weighted sum `t`.
When a new chunk raises the maximum to `m_new`, rescale both sums by
`exp(m - m_new)`, then add the chunk terms:

```text
s += sum(exp(z - m_new))
t += sum(exp(z - m_new) * z)
lse = m + log(s)
loss = lse - z_target
entropy = lse - t/s
```

The implementation applies rescaling before the additions. It tracks the first
argmax across chunks, including ties, and masks vocabulary tails.
[`solution/autograd.py`](solution/autograd.py) shifts labels separately in each
row with `pad(labels, (0, 1), -100)[..., 1:]`. Ignored targets have zero per-token
loss; the final position of each row is ignored. The scalar loss is the sum
normalized by the supplied `num_items_in_batch`, or the valid-target count.

Backward recomputes chunk logits rather than saving them. The registered backward
selects valid-target rows before the backward GEMMs, then scatters dX into a
zero-filled full output; with no valid targets it returns zero dX immediately.
Forward still computes entropy and predictions for every position. Compaction
adds a nonzero synchronization outside the chunk loop. With upstream gradient
`g`, it computes `dZ = g * (softmax(z) - one_hot(target))`, casts `dZ/gain` to
fp16 for GEMM, accumulates dX in fp32, and multiplies by
`gain = max(abs(g), 1e-30)/1024`. This honors GradScaler and reduces fp16
underflow; it does not promise bitwise equality with baseline fp16 dZ.
The frozen head has no weight gradient. Excluding ignored rows before gain
computation removes the reviewed ignored-row `0 * inf` gradient path in 8f8ebbc;
dedicated edge-case tests pass on T4 in the final suite. Zero-denominator training loss retains baseline
NaN semantics.

### Qwen2 RMSNorm

[`solution/kernels/rmsnorm.py`](solution/kernels/rmsnorm.py) defines:

| `@triton.jit` name | Replaces / produces |
| --- | --- |
| `rmsnorm_forward_kernel` | RMSNorm reduction and affine output |
| `rmsnorm_input_gradient_kernel` | Input-gradient pointwise/reduction chain |
| `rmsnorm_weight_partials_kernel` | Optional partial norm-weight gradients |
| `rmsnorm_weight_finalize_kernel` | Optional second-stage norm-weight reduction |

**Registered:** real Qwen2RMSNorm instances, including input and final norms.
The last two kernels are conditional and do not run for the frozen norms of the
fixed task; their absence is an informational check 3 note.

For `r = rsqrt(mean(x.float()²) + eps)`, forward computes
`q = cast(x.float()*r, x.dtype)`, then `y = weight*q` in the promoted dtype.
The cast occurs **before** multiplying by the norm weight, matching Qwen2.
For incoming gradient `g`, input backward uses
`u = cast(g*weight, x.dtype)` and
`dx = r*(u - (x*r)*mean(u*(x*r)))`, with fp32 arithmetic around the casts.
Weight gradients use independent partial tiles and a final reduction, without
atomics. Batch-dependent sizes and strides are runtime arguments.

`out_dtype=torch.float16` writes the final promoted result rounded to half in
the same pass. Input norms use explicit decoder scopes and require fp16 CUDA
AMP plus certified q/k/v consumers. For PEFT q/v, the registered vanilla LoRA
fp16-ingress path must be ready; k uses its certified dense/bnb path. The final
norm is scoped by the causal head and requires a certified ordinary dense head
under fp16 AMP. Unsupported wrappers, hooks or consumers keep the original
promoted norm output. A custom embedding/head forward can conservatively disable
the final-norm half optimization. Standalone norms have no half-output scope.
Backward returns dX in the input dtype and retains the original promoted dtype
for the optional norm-weight gradient. Early half output changes rounding and
shared-branch gradient accumulation; it is not bitwise-equivalent training.

### Attention residual add + post-attention RMSNorm

[`solution/kernels/add_rmsnorm.py`](solution/kernels/add_rmsnorm.py) defines
`add_rmsnorm_forward_kernel` and `add_rmsnorm_backward_kernel`.
**Registered:** [`solution/decoder.py`](solution/decoder.py) replaces the real
Qwen2 decoder's attention residual addition followed by post-attention norm.

Forward forms `s = residual + attention_output` in the promoted input dtype,
then applies the Qwen2 RMSNorm cast order above. It returns both `s` for the
residual branch and the normalized MLP input. Backward combines the gradient
from the residual branch with the norm Jacobian and casts to each input dtype.
It retains `s`, the weight reference and one fp32 reciprocal RMS per row.

By default, this post-attention norm writes fp16 output to reduce activation
storage before the frozen fp16 MLP projections. The residual stream remains in
its original promoted dtype. This is a precision optimization, not a promise
of bitwise end-to-end equivalence. The consumer guard now requires certified half projections and keeps the
original promoted dtype for a dense fp32 MLP outside AMP (review P1-2).

### LoRA scale/add epilogue and fp16 ingress

[`solution/kernels/lora_epilogue.py`](solution/kernels/lora_epilogue.py) defines
`lora_scale_add_kernel` and `lora_scale_gradient_kernel`.
**Registered:** the eligible vanilla PEFT Linear4bit path through
[`solution/lora.py`](solution/lora.py).

The forward replaces separate `adapter_output * scaling` and base-result add.
It rounds the product in the adapter-output dtype **before** adding, and keeps
PEFT's optional conversion to the base dtype outside autocast. Backward casts
to the branch dtype before scaling, matching the original two operations.

The patch keeps the original base linear, A/B projections and Dropout module.
For fp16 ingress it promotes the activation to fp32 before dropout, preserving
its dispatch shape and RNG advancement relative to the aligned fp32 path;
it does not recover values lost by earlier fp16 rounding. Dropout is still
called once, after the base projection, with the original probability/order.
The current default enables eligible attention input norms to write half,
using this ingress path for q/v adapters. Preserved masks/RNG do not imply
equal dropout values or equal adapter/input gradients after early rounding. Mixed
adapters, variants and unsupported states use original PEFT.

### Frozen MLP with tiled SwiGLU and optional recomputation

[`solution/kernels/swiglu_fused.py`](solution/kernels/swiglu_fused.py) defines
`swiglu_tile_engine`, one two-dimensional engine with forward/backward variants.
**Registered:** a lazy Qwen2MLP trampoline installs
[`solution/mlp.py`](solution/mlp.py) on first CUDA use. The supplied
[`solution/swiglu.py`](solution/swiglu.py) example is not registered and its
kernels do not count toward our contribution.

For gate `g`, up projection `u`, and `p = sigmoid(g)`, the engine computes:

```text
a  = (g*p)*u
dg = (da*u)*(p + (g*p)*(1-p))
du = da*(g*p)
```

Arithmetic is fp32 within the tile, with output casts chosen for the projection
graph. Row/column tails are masked. This fuses the pointwise forward and both
activation derivatives while leaving the linear projections and their backward
policy with PyTorch/bitsandbytes.

Default **store** mode saves fp16 gate/up tensors. Optional **recompute** mode
saves fp16 input instead and rebuilds gate/up under the captured forward AMP
context during backward. It adds projection work in exchange for lower retained
activation memory. Local GradientEdges preserve projection backward without
retaining the entire activation buffer as a tensor leaf. Frozen, supported
projection types qualify; trainable projections, hooks and wrappers fall back.
Only first-order gradients are implemented. Recompute is off by default; the
default keeps fp16 gate/up and offloads them to host memory as described under
[asynchronous MLP activation offload](#asynchronous-mlp-activation-offload).

### NF4 decoder

[`solution/kernels/nf4.py`](solution/kernels/nf4.py) defines `nf4_decode_kernel`.
**Registered:** [`solution/nf4_patch.py`](solution/nf4_patch.py) patches the
bitsandbytes functional and CUDA backend dequantization paths after their
post-import dependencies are ready. The exact bitsandbytes version gate is
0.50.2. NF4 decoding is the largest own kernel in the final run (7.0 % share).
The 4-bit matrix multiplication itself remains a library operation.

The supported nested format is fp16 NF4, block size 64, with double-quantized
scales in blocks of 256. Earlier values occupy the high nibble of each byte.
For output index `i`, nested scale reconstruction is
`scale = fp32(code2[qabs[i//64]] * absmax2[(i//64)//256]) + offset`;
output is `fp16(NF4[nibble(i)] * scale)`. A second variant accepts decoded
fp32 scales, including non-nested functional calls routed through the patched
backend. FMA fusion is disabled to preserve intermediate scale rounding.
Unsupported versions, formats, devices and layouts use the original paths.
Review notes an extra allocation/copy for supported functional `out=` calls.

### Lossless storage of the tied embedding/head

[`solution/embedding.py`](solution/embedding.py) is **registered** after Soup's
`SFTTrainerWrapper._setup_transformers`. It accepts only an eligible frozen,
tied Qwen2 embedding/head after PEFT setup, with unchanged LoRA-only saving.
Every table element must satisfy `half_weight.float() == original_weight`;
the check is chunked and conversion is declined if any element fails.
The existing shared Parameter, tying, dimensions and names are preserved.
No new Triton kernel is introduced.

Only the table's storage becomes half. Embedding lookup is restored to the
original fp32 output; the dense head uses the same represented weights under
AMP. Outside AMP it temporarily reconstructs fp32 weights, so this optimization
does not promise lower eval peak memory. It is active in the final
run; no separately isolated memory or speed gain is claimed.

### Fused q/k rotary position embedding

[`solution/kernels/rope.py`](solution/kernels/rope.py) defines
`rope_pair_forward_kernel` and `rope_pair_backward_kernel`.
**Registered:** [`solution/rope.py`](solution/rope.py) replaces Qwen2's module-global
`apply_rotary_pos_emb`, which attention resolves at call time. It combines q/k
rotation, products and sum, preserving PyTorch type promotion and separate
product-rounding boundaries rather than using a fused multiply-add.
For the two half-head slices, backward computes
`dx1 = g1*cos1 + g2*sin2`, `dx2 = g2*cos2 - g1*sin1`, with cast boundaries
matching the original graph. Only cos/sin and metadata are saved.

The fast path supports even head dimensions, supported CUDA fp16/fp32 tensors,
`unsqueeze_dim=1` and non-trainable cos/sin. Other cases use plain PyTorch math;
no optional hub kernel is requested or called, including on fallback.
Sequence size is a runtime argument. Measured share in the final run: 0.92 %.

## How it plugs in

The unmodified harness calls `solution.apply()` before Soup builds its model.
Targeted post-import hooks patch Qwen2, TRL SFTTrainer, PEFT's bnb LoRA class,
bitsandbytes dequantization and Soup model setup
before the importing caller receives the module. Already loaded modules are
handled too; markers make repeated installation idempotent. Cold registration
does not import/compile the CUDA kernels or eagerly import the training stack.

The head fast path requires the exact Qwen2 causal-LM class, training with
labels, a frozen bias-free dense head, supported contiguous CUDA fp16/fp32
inputs and the fp16 head precision path. Unsupported loss options, custom loss,
label smoothing, virtual tokens, parallelism or logit selection use the
original head/trainer path. CPU, fake test modules and unsupported tensor
layouts use guarded fallbacks. Eval/generation return ordinary logits; other
eligible layer patches can still run, subject to their own precision guards.

The TRL bridge keeps the same metric names and formulas: entropy uses all
attention-mask tokens; accuracy uses shifted non-ignored labels; num_tokens
accumulates attention-mask tokens. A fresh detached Soup ChunkedStep supplies
per-token losses to the unchanged Rewind row recorder. It preserves empty-row
counts and resets per-step state between calls. Saved LoRA adapters are evaluated
by the unmodified external evaluator without importing solution.

| Experimental environment flag | Submitted default behavior in the documented snapshot | Override for comparison |
| --- | --- | --- |
| `SOUP_FUSED_DISABLE` | Unset: registered optimizations enabled | `1`: use their original paths |
| `SOUP_FUSED_NORM_FP16` | Unset: eligible input, post-attention and final norm outputs fp16 | `0`: original promoted norm-output dtype |
| `SOUP_MLP_RECOMPUTE` | Unset: store gate/up | `1`: recompute gate/up in backward |

Final measurements must use defaults with these overrides unset. Flags are for
separate experiments; no behavior is selected by a profiler, test name or
measurement phase. No ready-made kernel libraries, torch.compile or CUDA graphs
are used by this implementation.

## How to run

### Public Kaggle notebook

1. Use the repository assigned by the hackathon platform. Import
   [`templates/kaggle_notebook.ipynb`](templates/kaggle_notebook.ipynb) into Kaggle.
2. Select **GPU T4 x2**, enable Internet, and set `REPO_URL` to the repository
   URL and `COMMIT` to `f1cb5ac32ef7cbd1caf8455c833b67bd43992ed8`. The harness uses one GPU. For a private
   repository, configure the read-only `GITHUB_TOKEN` Kaggle secret as the
   template describes.
3. Run all cells with the template unchanged. Its training cell runs
   `python harness/run_task.py`: one base and one solution run, followed by the
   checks and score cells. The saved public notebook uses this one-repeat
   procedure. The two-repeat evidence reported above comes from a separate
   development `full2` run (base, solution, solution, base), reproducible with
   the local commands or development runner below.
4. Save the completed notebook version, publish it, and replace `<KAGGLE_LINK>`
   here and in SUBMISSION with the same saved-version link. Preserve result,
   checks, metrics and logs with that notebook.

### Local Linux CUDA machine

Only T4 results count for the challenge. Install dependencies before the measured
run, then execute the unchanged kit commands from the repository root:

```bash
python -m pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" \
  "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

The measured runtime was Python 3.13.15, torch 2.11.0+cu128, Triton 3.6.0 and
Tesla T4 driver 580.178.04, with the package versions above; source:
`versions` in `runs/result.json` within the `full2` evidence directory listed
under [Results and kernel-time accounting](#results-and-kernel-time-accounting).
The head's backward relies on CUDA `torch.mm` support for
`out_dtype=torch.float32`; its forward logits use fp16 GEMM output. A different runtime must be validated.
The harness sets the allocator and restricts training to one GPU; do not
substitute a different allocator or prepopulate compiler caches.

### Development runner

[`tools/kaggle/run.py`](tools/kaggle/run.py) sends committed HEAD through a
private Kaggle code dataset and collects artifacts. Install/configure Kaggle CLI
first; keep credentials outside the repository in `~/.kaggle/kaggle.json`.
`KAGGLE_BIN` selects the CLI executable and defaults to `kaggle`.

```bash
python3 tools/kaggle/run.py run --mode unit --dry-run
python3 tools/kaggle/run.py push-code
python3 tools/kaggle/run.py run --mode unit
python3 tools/kaggle/run.py run --mode solution
python3 tools/kaggle/run.py run --mode full2
```

Commit accepted changes before `push-code`: uncommitted files are not uploaded.
`unit` runs tests, `solution` produces a solution-only run and profile, and
`full2` performs the paired two-repeat comparison and checks. Inspect preserved
pytest/check exit codes; a completed Kaggle notebook alone does not imply success.
This private iteration notebook does not replace the public submission link.

## Results and kernel-time accounting

Final jury comparison: `96a890c`, two base and two solution repetitions in one
T4 session, `check_rules.py` 10/10 (evidence:
`runs_kaggle/20261031T133425573730Z_full2_96a890c_promote010/out/`, local, not
in Git). Shares are `100 * own_kernel_ms / (1000 * timed_loop_s)` for steps
3–64 of each solution run (timed loop 35.37 s and 35.99 s).

| Kernel | Run 1, ms | Run 1 share | Run 2 share |
| --- | ---: | ---: | ---: |
| `nf4_decode_kernel` | 2496.4 | 7.06 % | 6.97 % |
| `swiglu_tile_engine` | 913.1 | 2.58 % | 2.54 % |
| `lora_scale_add_kernel` | 189.8 | 0.54 % | 0.54 % |
| `add_rmsnorm_forward_kernel` | 187.6 | 0.53 % | 0.52 % |
| `rope_pair_forward_kernel` | 167.9 | 0.47 % | 0.47 % |
| `rope_pair_backward_kernel` | 158.0 | 0.45 % | 0.45 % |
| `add_rmsnorm_backward_kernel` | 150.7 | 0.43 % | 0.42 % |
| `vocabulary_statistics_kernel` | 130.4 | 0.37 % | 0.37 % |
| `vocabulary_derivative_kernel` | 108.8 | 0.31 % | 0.30 % |
| `rmsnorm_input_gradient_kernel` | 108.1 | 0.31 % | 0.30 % |
| `rmsnorm_forward_kernel` | 81.1 | 0.23 % | 0.23 % |
| `lora_scale_gradient_kernel` | 75.7 | 0.21 % | 0.21 % |
| `vocabulary_outputs_kernel` | 0.5 | 0.00 % | 0.00 % |
| **Total counted own kernels** | **4768.0** | **13.5 %** | **13.3 %** |

The total uses unrounded timings; individual rows are rounded independently.
No `example_*`, cuBLAS or library kernel time is counted.

### Optimization history

For whole-run A/B experiments, A = the default at the time and candidates use
opt-in flags, alternating order over rounds (`tools/kaggle/registry.py`, median
of paired differences, 95 % bootstrap CI). Current promotion requires paired
wall time not slower and peak memory not higher, preserved eval loss, then
`full2` + `check_rules`. Historical adoption decisions and other evidence types
are recorded explicitly below.

| Change | Δ wall, s (95 % CI) | Δ peak, MiB | Decision |
| --- | --- | ---: | --- |
| Out-of-process Triton precompile + launch trim (EXP-001/004) | −1.80 [−5.96, −1.60]; 3 pairs, `d5dc84b` | +4 (3,765 → 3,769) | historical default adoption |
| Prebuilt native CPU launchers in the compile child (EXP-010) | **−3.33 [−3.66, −2.86]**; first step 6.53 → 2.93 s | 0 | **default (`96a890c`)** |
| Parent launcher warm-up (EXP-013) | −0.41 [−1.31, +1.16]; first step −0.68 | 0 | not adopted (CI crosses 0) |
| High-priority offload copy streams (EXP-015) | −0.25 [−0.50, +0.30] | 0 | not adopted (noise) |
| Faster packed NF4 decode kernel (EXP-006) | −0.06 [−0.53, +0.25] | 0 | not adopted (noise) |
| Single autograd boundary for the LoRA pair (EXP-008) | +0.25 [−0.62, +0.55] | 0 | rejected |
| Offload gate + input, recompute up (EXP-009) | +2.06 [+0.97, +2.38] | −82 | rejected: slower |
| Keep last k MLPs on GPU (EXP-007) | not measured | estimated +70·k | not adopted: estimated memory increase |
| Own NF4×fp16 Triton GEMM (EXP-002) | microbenchmark: 4–66× slower than decode + cuBLAS per GEMM | — | rejected |
| Launch-config sweep, bit-identical outputs (EXP-011) | microbenchmark: ≤ 6.5 % per kernel; estimated < 0.2 s/run | not measured | not adopted |

Memory is a separate decision: MLP gate/up offload to pinned host memory and
the fp16 norm/head paths are what bring the peak from 14,345 to 3,769 MiB.
Historically, full MLP recomputation reached 4,123 MiB at `ce1ff09`; its
93.95 s wall time versus 73.54 s for store mode was about +20 s across separate
Kaggle sessions. This is a historical cross-session observation, not a paired
estimate of its cost. The current default is asynchronous offload.

## Limitations and review status

The optimized head targets Qwen2 labelled training with a frozen dense head on
T4/fp16. bf16, trainable heads, higher-order derivatives and unsupported model or
trainer features are outside that fast path. Layer fallbacks are separate from
the head fallback and need their own precision validation.

Attention matmul/softmax and NF4 matrix multiplication remain library
operations; only rotary position application and NF4 decoding are replaced.
Chunked CE still performs library GEMMs and Python chunk iteration. Cold
compilation and setup count toward wall time. Lossless embedding conversion
requires an exact half roundtrip and may decline on another checkpoint.

[Independent review](tasks/REVIEW.md) contains S-017, S-021 and S-023:

| Finding | Current status / evidence boundary |
| --- | --- |
| P0-1: own share below 5% | Final two-repeat `96a890c` passes check 4 at 13.3 % (lowest run). |
| P1-1: all-ignored/nonfinite upstream can produce NaN dX | Closed in code by S-018: backward exits before gain/GEMMs if no valid targets and excludes ignored rows otherwise. S-021 confirms the structure; dedicated tests pass on T4. |
| P1-2: half norm reaches dense fp32 MLP outside AMP | Closed statically in S-016/S-023 by consumer guards; unsupported MLP keeps the promoted dtype. |
| P1-023-1: input-norm half changes LoRA values and shared-gradient accumulation | Numerical limitation remains. `96a890c` checks 5–7 pass on two repeats (first-step loss 0.02 %, mean 0.01 %); this does not prove equality or hidden-row/seed robustness. Dropout masks/RNG alone do not establish numerical equivalence. |
| P2-021-1: functional NF4 `out=` allocation/copy | Remains a performance limitation; alias result is preserved. |
| P2-021-2: functional non-nested NF4 can reach own flat backend | Documented routing; unsupported-format fallback does not mean all non-nested calls avoid our kernel. |
| P2-021-3: chunk CE equality test is not whole-head bitwise proof | Remains a test-scope limitation. cuBLAS algorithms can differ with GEMM shape. |
| P2-1: description/reproducibility | Updated with the final two-repeat result; public notebook link pending. |

The final solution suite on T4 (`96a890c`): 518 passed, 1 skipped, 0 failed;
Soup regression tests 79/79 with `apply()`.

## Challenge rules retained

The full task, scoring and checks are in the
[upstream starter kit](https://github.com/MakazhanAlpamys/soup-kernel-challenge).
Only `solution/` is copied into the jury's fresh kit; our development tools and
these reported artifacts do not modify its measurement code.

| Must remain unchanged | Fixed contract |
| --- | --- |
| `task/task.yaml`, `task/prepare_data.py` | Model, dataset selection/order, batch size, max length, steps, optimizer/schedule and LoRA settings |
| `harness/` | Timing, sampled memory, evaluator, checks and scoring |
| `tests/` | Soup regression tests |
| Trainer settings and Soup outputs | Same adapter parameters/shapes, metrics, token counts and Rewind rows/losses within allowed tolerances |
| Measurement environment | One T4 GPU, harness allocator; no background GPU work; no warm compiler cache; no measurement-dependent behavior |

The task uses 256 public training rows, batch size 4 and maximum length 512,
one epoch with LoRA r=16, alpha=32 and the unchanged Soup dropout/target defaults;
source: `task/task.yaml` and the upstream kit. We do not add trainable model
parameters or load ready-made kernel libraries.

## Sources of ideas and license

These are algorithm/format references, not imported or vendored kernel libraries:

- [Milakov & Gimelshein, Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867): online max/sum updates.
- [Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009): avoiding full-vocabulary logits.
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) and [Apex normalization](https://github.com/NVIDIA/apex/tree/master/apex/normalization): linear/CE and normalization fusion ideas.
- [PEFT Linear4bit](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py): original LoRA control flow and rounding order.
- [bitsandbytes 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/tree/0.50.2): NF4 format, nibble order and nested-scale reconstruction.
- Standard SiLU/SwiGLU derivatives and RMSNorm Jacobian; our tiled kernels are written in this repository.

Code: Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
Data: [AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2)
(MIT on its card), a Kazakh translation of
[Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (CC BY-NC 4.0).
