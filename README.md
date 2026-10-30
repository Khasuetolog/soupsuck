# Soup Kernel Challenge: less memory for the same LoRA task

Our Triton kernels reduce activation and output-head memory during the fixed
Qwen2.5-1.5B-Instruct NF4/LoRA training task on a Kaggle Tesla T4.
The first complete comparison passed all ten harness checks: **70.05 s and
5,843 MiB**, versus **77.47 s and 14,345 MiB** for base Soup.

| Measurement | Base Soup | Solution | Comparison |
| --- | ---: | ---: | --- |
| Whole `soup train` command | 77.47 s | 70.05 s | 1.1059× speedup |
| Peak GPU memory, sampled by nvidia-smi | 14,345 MiB | 5,843 MiB | 2.4551× base/solution |
| Public held-out answer-token loss | 2.04964 | 2.04962 | −0.001% vs base |
| Own Triton kernel share, steps 3 onward | — | 8.6% | Check 4 PASS |
| Harness checks | — | 10/10 PASS | One paired repetition |

**Measured commit:** `de836e0264d24fbd2e184eb853e2bb81bb0f0c11`, seed 1,
64 training steps on each side. These are development results from **one**
base/solution pair, not the final two-repeat jury comparison.
The separate solution test suite reported **171 passed, 1 failed**; the Soup
regression suite used by check 10 passed all **79** tests. The remaining test
failure and review findings are described below.

Evidence root, abbreviated **E** throughout this document:
`runs_kaggle/20261030T105526661826Z_full_de836e0/out/`.
The summary numbers come from `E/runs/result.json` (`summary`); check outcomes
from `E/checks.log` and `E/runs/checks.json`; test counts from
`E/pytest_solution.log` and check 10. `E/source-manifest.json` records the full
commit and archive hashes. These local development artifacts are excluded from
Git; the final public notebook must publish the corresponding final evidence.

- Public Kaggle notebook, saved version: `<KAGGLE_LINK>`
- Final submission commit: `<FINAL_COMMIT>`
- Submission links and team: [SUBMISSION.md](SUBMISSION.md)

## What we did

The entry point is [`solution.apply()`](solution/__init__.py).
Implementation registration described here is the committed snapshot
`8f8ebbc`: it includes the measured `de836e0` implementation, an available,
unregistered NF4 decoder, and a subsequent CE update. That update uses native
fp16 logit GEMMs and compacts valid rows in backward. It has not inherited the
older run's checks or performance results; the table above remains de836e0 only.

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
vocabulary chunks. Matrix multiplication remains `torch.mm`/cuBLAS. Current
8f8ebbc uses fp16 operands and native fp16 logit output, then promotes logits
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

Backward recomputes chunk logits rather than saving them. Current 8f8ebbc
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
new GPU tests and paired measurements are still required. Zero-denominator
training loss retains baseline NaN semantics.

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
of bitwise end-to-end equivalence. The documented snapshot has an insufficient
consumer guard for ordinary fp32 MLPs outside AMP; see Limitations.

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
The measured snapshot keeps the attention input norm output fp32: fp16 ingress
support is available but was not a separate measured optimization. Mixed
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
Only first-order gradients are implemented. Recompute was not used for the
summary result and has no claimed end-to-end speed or memory result here.

### NF4 decoder: available, not registered in the measured/default entry point

[`solution/kernels/nf4.py`](solution/kernels/nf4.py) defines `nf4_decode_kernel`.
[`solution/nf4_patch.py`](solution/nf4_patch.py) provides `patch_bnb_dequant` for
the bitsandbytes functional and CUDA backend dequantization paths, but committed
`solution/patch.py` does **not** call it. It was added after measured `de836e0`.
It has no claimed measured share or contribution to the summary result.

The narrow supported format is fp16 NF4, block size 64, with double-quantized
scales in blocks of 256. Earlier values occupy the high nibble of each byte.
For output index `i`, nested scale reconstruction is
`scale = fp32(code2[qabs[i//64]] * absmax2[(i//64)//256]) + offset`;
output is `fp16(NF4[nibble(i)] * scale)`. A second variant accepts already decoded
fp32 scales. FMA fusion is disabled to preserve the intermediate scale rounding.
Format constants are data; the decoder implementation is our own. GPU equality
and integration must be validated before activating it in a final submission.

## How it plugs in

The unmodified harness calls `solution.apply()` before Soup builds its model.
Targeted post-import hooks patch Qwen2, TRL SFTTrainer and PEFT's bnb LoRA class
before the importing caller receives the module. Already loaded modules are
handled too; markers make repeated installation idempotent. Cold registration
does not import/compile the CUDA kernels or eagerly import the training stack.

The head fast path requires the exact Qwen2 causal-LM class, training with
labels, a frozen bias-free dense head, supported contiguous CUDA fp16/fp32
inputs and the fp16 head precision path. Unsupported loss options, custom loss,
label smoothing, virtual tokens, parallelism or logit selection use the
original head/trainer path. CPU, fake test modules and unsupported tensor
layouts use guarded fallbacks. Eval/generation return ordinary logits; other
eligible layer patches can still run, subject to the limitation below.

The TRL bridge keeps the same metric names and formulas: entropy uses all
attention-mask tokens; accuracy uses shifted non-ignored labels; num_tokens
accumulates attention-mask tokens. A fresh detached Soup ChunkedStep supplies
per-token losses to the unchanged Rewind row recorder. It preserves empty-row
counts and resets per-step state between calls. Saved LoRA adapters are evaluated
by the unmodified external evaluator without importing solution.

| Experimental environment flag | Submitted default behavior in the documented snapshot | Override for comparison |
| --- | --- | --- |
| `SOUP_FUSED_DISABLE` | Unset: registered optimizations enabled | `1`: use their original paths |
| `SOUP_FUSED_NORM_FP16` | Unset: post-attention norm output fp16 | `0`: original promoted norm-output dtype |
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
   URL and `COMMIT` to `<FINAL_COMMIT>`. The harness uses one GPU. For a private
   repository, configure the read-only `GITHUB_TOKEN` Kaggle secret as the
   template describes.
3. Run all cells. For the final comparison, use
   `python harness/run_task.py --repeats 2` in the training cell, then run the
   unchanged checks and score cells. This runs base, solution, solution, base.
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
`E/runs/result.json:versions`. The head relies on CUDA `torch.mm` support for
`out_dtype=torch.float32`. A different runtime must be validated.
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

The summary is the first full passing run, **one repeat at de836e0**.
`E/runs/base_1/metrics.json` reports 64 steps, training loop 53.42 s and
median step 0.8478 s; `E/runs/solution_1/metrics.json` reports 64 steps,
training loop 47.59 s and median step 0.6429 s.
Whole-command timing includes startup, cold compilation and saving the adapter.

Shares below are **training-time shares**, not percentages of total GPU kernel
time and not whole-command shares. The solution's timed steps 3–64 total
**39.556 s**. Each share is `100 * own_kernel_ms[name] / (1000 * timed_loop_s)`
from `E/runs/solution_1/metrics.json`. Base metrics have no own kernels.

| Kernel | CUDA-event time, ms | Share of timed training |
| --- | ---: | ---: |
| `rmsnorm_forward_kernel` | 144.228 | 0.3646% |
| `rmsnorm_input_gradient_kernel` | 194.386 | 0.4914% |
| `add_rmsnorm_forward_kernel` | 213.710 | 0.5403% |
| `add_rmsnorm_backward_kernel` | 254.252 | 0.6428% |
| `lora_scale_add_kernel` | 212.380 | 0.5369% |
| `lora_scale_gradient_kernel` | 108.677 | 0.2747% |
| `vocabulary_statistics_kernel` | 273.461 | 0.6913% |
| `vocabulary_outputs_kernel` | 0.637 | 0.0016% |
| `vocabulary_derivative_kernel` | 328.581 | 0.8307% |
| `swiglu_tile_engine` (both variants) | 1659.851 | 4.1962% |
| **Total counted own kernels** | **3390.163** | **8.5705%** |

Optional `rmsnorm_weight_partials_kernel` and `rmsnorm_weight_finalize_kernel`
are absent from the launch/time maps because norms are frozen. `nf4_decode_kernel`
is absent because it was not installed in this run. The example kernels were
not launched by the registered MLP. No cuBLAS or other library kernel time is
added to our share. Check 4 reports the rounded total **8.6%**, against its 5%
threshold; the jury uses the lowest eligible share across solution repetitions.

| Final two-repeat comparison | Value |
| --- | --- |
| Commit / saved public notebook | `<FINAL_COMMIT>` / `<KAGGLE_LINK>` |
| Base / solution wall time | `<FINAL_BASE_TIME>` / `<FINAL_SOLUTION_TIME>` |
| Base / solution peak memory | `<FINAL_BASE_MEMORY>` / `<FINAL_SOLUTION_MEMORY>` |
| Base / solution eval loss | `<FINAL_BASE_LOSS>` / `<FINAL_SOLUTION_LOSS>` |
| Lowest own-kernel share / all ten checks | `<FINAL_KERNEL_SHARE>` / `<FINAL_CHECKS>` |

These placeholders are deliberately unfilled: the final two-repeat result is
not the measured one-repeat result. Score points depend on other admitted teams;
we report measured ratios rather than claiming a leaderboard score.

## Limitations and review status

The optimized head targets Qwen2 labelled training with a frozen dense head on
T4/fp16. bf16, trainable heads, higher-order derivatives and unsupported model or
trainer features are outside that fast path. Layer fallbacks are separate from
the head fallback and need their own precision validation.

The implementation does not replace attention or the NF4 matrix-multiplication
kernels; projection GEMMs remain library operations. The NF4 decoder is present
but unregistered. Chunked CE still performs GEMMs and Python chunk iteration;
MLP recomputation adds projection work. Kaggle machines vary, so one pair is not
a guarantee of the same speedup on another host, seed or hidden rows.

[Independent review](tasks/REVIEW.md), scoped to de836e0, recorded:

| Finding | Status of the documented/measured snapshot |
| --- | --- |
| Own-kernel share previously below 5% | Resolved for this one measured pair: check 4 PASS at 8.6%; final two-repeat/hidden-row evidence still required |
| Fully ignored batch / nonfinite upstream can produce NaN dX through `0 * gain` | Present in measured de836e0; 8f8ebbc excludes ignored rows before backward gain/GEMMs and adds regression tests, with GPU validation still pending |
| fp16 post-attention output can reach fp32 MLP without AMP | Open and reproduced by `test_bridge_without_amp_preserves_dense_fallback`, the one failing solution test in this run |
| README still described the starter kit | Replaced by this document; final links/results remain placeholders |

Do not describe the separate solution test suite as fully passing merely because
all ten harness checks passed. Check 10 runs Soup's regression tests, not the
solution suite. The 8f8ebbc CE fix and broader norm precision changes require new paired
measurements. The review does not certify NF4, which was added later.

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
| Measurement environment | One T4 GPU, harness allocator, no background GPU work, warm compiler cache or measurement-dependent behavior |

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
